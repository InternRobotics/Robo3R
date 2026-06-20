import os
import copy
from datetime import timedelta

import hydra
from omegaconf import OmegaConf
OmegaConf.register_new_resolver('eval', eval, replace=True)
import wandb
from tqdm import tqdm
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from util.train_util import set_seed, train_one_iteration, test_one_iteration, all_reduce_data, resume_from_checkpoint, save_checkpoint, get_cosine_schedule_with_warmup
from util.data_util import build_dataloader
from robo3r.models.robo3r import Robo3R
from robo3r.models.loss import Robo3RLoss

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.backends.cuda.matmul.allow_tf32 = True


def main(cfg):
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(seconds=1800), device_id=torch.device(f"cuda:{local_rank}"))
    torch.distributed.barrier()

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    torch.set_default_dtype(torch.float32)

    # wandb init
    if rank == 0 and cfg.report_to_wandb:
        wandb.init(project=cfg.wandb_project, name=cfg.run_name)
        wandb_train_step_defined = False
        wandb_test_step_defined = False
    
    # seed
    set_seed(cfg.seed + rank)

    # dataset
    def build_dataset_config(entries, split):
        return " + ".join(
            f"ThreeDP4MWAI(split='{split}', resolution={cfg.resolution}, principal_point_centered=False, aug_crop=16, transform='colorjitter+grayscale+gaublur', data_norm_type='identity', ROOT='{entry.path}', variable_num_views=True, num_views={cfg.num_view}, num_views_min={cfg.num_view_min}, num_of_scenes={entry.num_of_scenes})"
            for entry in entries
        )

    train_dataset_config = build_dataset_config(cfg.dataset.train, 'train')
    dataloader = build_dataloader(
        dataset=train_dataset_config,
        num_workers=cfg.dataloader.num_workers,
        max_num_of_imgs_per_gpu=cfg.max_num_of_imgs_per_gpu,
    )
    test_dataset_config = build_dataset_config(cfg.dataset.test, 'test')
    test_dataloader = build_dataloader(
        dataset=test_dataset_config,
        num_workers=cfg.dataloader.num_workers,
        max_num_of_imgs_per_gpu=cfg.max_num_of_imgs_per_gpu,
    )

    # model
    model = Robo3R(fuse_robot_state=cfg.model.fuse_robot_state)
    model = model.cuda()
    resume_from_checkpoint(cfg.model.pi3_ckpt_path, model)

    # separate pred
    # model.robot_point_decoder.load_state_dict(model.point_decoder.state_dict())
    model.robot_point_head.load_state_dict(model.point_head.state_dict())
    # model.object_point_decoder.load_state_dict(model.point_decoder.state_dict())
    model.object_point_head.load_state_dict(model.point_head.state_dict())
    # model.table_point_decoder.load_state_dict(model.point_decoder.state_dict())
    model.table_point_head.load_state_dict(model.point_head.state_dict())

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if rank == 0:
        print(f'total parameters: {sum(p.numel() for p in model.parameters())/1e6}M')
        for name, module in model.named_children():
            if len(list(module.parameters())) > 0:
                print(f'{name} -- num_parameter: {sum(p.numel() for p in module.parameters())/1e6}M -- requires_grad: {next(module.parameters()).requires_grad}')
    ddp_model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    # criterion
    criterion = Robo3RLoss()
    test_criterion = Robo3RLoss()

    # optimizer
    # cfg.optimizer.lr = cfg.optimizer.lr * world_size
    grouped_param = [
        # decay, encoder
        {
            'params': [p for n, p in ddp_model.module.named_parameters() if len(p.shape) != 1 and n.startswith('encoder') and p.requires_grad],
            'weight_decay': cfg.optimizer.weight_decay,
            'lr': cfg.optimizer.lr / 10.0,
        },
        # decay, not encoder
        {
            'params': [p for n, p in ddp_model.module.named_parameters() if len(p.shape) != 1 and not n.startswith('encoder') and p.requires_grad],
            'weight_decay': cfg.optimizer.weight_decay,
            'lr': cfg.optimizer.lr,
        },
        # no decay, encoder
        {
            'params': [p for n, p in ddp_model.module.named_parameters() if len(p.shape) == 1 and n.startswith('encoder') and p.requires_grad],
            'weight_decay': 0.0,
            'lr': cfg.optimizer.lr / 10.0,
        },
        # no decay, not encoder
        {
            'params': [p for n, p in ddp_model.module.named_parameters() if len(p.shape) == 1 and not n.startswith('encoder') and p.requires_grad],
            'weight_decay': 0.0,
            'lr': cfg.optimizer.lr,
        },
    ]
    optimizer = torch.optim.AdamW(grouped_param, lr=cfg.optimizer.lr, betas=cfg.optimizer.betas)

    # lr_scheduler
    if 'step' in cfg.num_warmup:
        cfg.num_warmup = int(cfg.num_warmup.split('_')[0])
    elif 'epoch' in cfg.num_warmup:
        cfg.num_warmup = int(cfg.num_warmup.split('_')[0]) * len(dataloader) // cfg.grad_acc_step
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=cfg.num_warmup, num_training_steps=cfg.num_epoch * len(dataloader) // cfg.grad_acc_step)

    # scaler for amp
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.use_amp)
    assert cfg.amp_dtype in ['bf16', 'fp16']
    if cfg.amp_dtype == 'bf16':
        assert torch.cuda.is_bf16_supported()

    start_epoch = 0
    start_step_in_epoch = 0
    train_step = 0
    test_epoch = 0
    test_step = 0
    cur_grad_acc_step = 0

    # resume from checkpoint
    if cfg.resume_from_checkpoint is not None:
        if rank == 0:
            print(f'Loading checkpoint from {cfg.resume_from_checkpoint}')
        start_epoch, start_step_in_epoch, train_step = resume_from_checkpoint(cfg.resume_from_checkpoint, ddp_model.module, optimizer, lr_scheduler, scaler)

    if 'step' in cfg.test_freq:
        cfg.test_freq = int(cfg.test_freq.split('_')[0])
    elif 'epoch' in cfg.test_freq:
        cfg.test_freq = int(cfg.test_freq.split('_')[0]) * len(dataloader)
    if 'step' in cfg.save_checkpoint_freq:
        cfg.save_checkpoint_freq = int(cfg.save_checkpoint_freq.split('_')[0])
    elif 'epoch' in cfg.save_checkpoint_freq:
        cfg.save_checkpoint_freq = int(cfg.save_checkpoint_freq.split('_')[0]) * len(dataloader)

    # training loop
    ddp_model.train()
    for epoch in range(start_epoch, cfg.num_epoch):
        # dataloader.sampler.set_epoch(epoch)
        dataloader.dataset.set_epoch(epoch)
        dataloader.batch_sampler.set_epoch(epoch)
        tqdm_dataloader = tqdm(enumerate(dataloader), disable=rank!=0, total=len(dataloader))
        tqdm_dataloader.set_description(f'epoch {epoch} / {cfg.num_epoch - 1}')

        for step_in_epoch, batch in tqdm_dataloader:
            if epoch == start_epoch and step_in_epoch < start_step_in_epoch:
                continue

            # test
            if cfg.test and train_step % (cfg.test_freq * cfg.grad_acc_step) == 0:
                # test_dataloader.sampler.set_epoch(0)
                test_dataloader.dataset.set_epoch(0)
                test_dataloader.batch_sampler.set_epoch(0)
                test_tqdm_dataloader = tqdm(enumerate(test_dataloader), disable=rank!=0, total=len(test_dataloader))
                test_tqdm_dataloader.set_description(f'test epoch {test_epoch}')

                ddp_model.eval()

                for test_step_in_epoch, test_batch in test_tqdm_dataloader:
                    with torch.inference_mode():
                        test_metric = test_one_iteration(ddp_model, test_batch, test_criterion, cfg.use_amp, cfg.amp_dtype)

                    test_metric = all_reduce_data(test_metric, all_reduce=True)
                    if rank == 0 and cfg.report_to_wandb:
                        if not wandb_test_step_defined:
                            wandb.define_metric("test_step")
                            wandb.define_metric("test_epoch")
                            for key in test_metric.keys():
                                wandb.define_metric(f'test_step_{key}', step_metric='test_step')
                                wandb.define_metric(f'test_epoch_{key}', step_metric='test_epoch')
                            wandb_test_step_defined = True
                        wandb.log({f'test_step_{k}': v for k, v in test_metric.items()} | {'test_step': test_step})
                        if test_step_in_epoch == 0:
                            test_metric_avg = copy.deepcopy(test_metric)
                        else:
                            test_metric_avg = {k: test_metric_avg[k] + v for k, v in test_metric.items()}

                    test_step += 1

                if rank == 0 and cfg.report_to_wandb:
                    wandb.log({f'test_epoch_{k}': v / len(test_dataloader) for k, v in test_metric_avg.items()} | {'test_epoch': test_epoch})

                test_epoch += 1
                ddp_model.train()

            # save checkpoint
            if rank == 0 and cfg.save_checkpoint and train_step % (cfg.save_checkpoint_freq * cfg.grad_acc_step) == 0:
                checkpoint_path = f'{cfg.checkpoint_dir}/{cfg.run_name}/{epoch}_{step_in_epoch}.pth'
                if not os.path.exists(f'{cfg.checkpoint_dir}/{cfg.run_name}'):
                    os.makedirs(f'{cfg.checkpoint_dir}/{cfg.run_name}')
                save_checkpoint(checkpoint_path, ddp_model.module, optimizer, lr_scheduler, scaler, epoch, step_in_epoch, train_step)

            # train
            metric = train_one_iteration(ddp_model, batch, criterion, optimizer, lr_scheduler, cfg.use_amp, cfg.amp_dtype, scaler, cfg.max_grad_norm, cfg.grad_acc_step, cur_grad_acc_step, f'{cfg.checkpoint_dir}/{cfg.run_name}/{epoch}_{step_in_epoch}')
            cur_grad_acc_step = (cur_grad_acc_step + 1) % cfg.grad_acc_step
            if metric is not None:
                metric = all_reduce_data(metric, all_reduce=False)
                if rank == 0 and cfg.report_to_wandb:
                    if not wandb_train_step_defined:
                        wandb.define_metric("train_step")
                        for key in metric.keys():
                            wandb.define_metric(key, step_metric='train_step')
                        wandb_train_step_defined = True
                    wandb.log(metric | {'train_step': train_step})

            train_step += 1

    # finish
    if rank == 0 and cfg.report_to_wandb:
        wandb.finish()
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    with hydra.initialize(config_path='config'):
        import sys; overrides = sys.argv[1:]
        cfg = hydra.compose(config_name='config', overrides=overrides)
    main(cfg)
