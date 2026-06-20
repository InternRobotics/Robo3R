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

from util.train_util import set_seed, all_reduce_data, resume_from_checkpoint
from util.data_util import build_dataloader
from util.keypoint_pnp_util import *

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.backends.cuda.matmul.allow_tf32 = True

import numpy as np
from plyfile import PlyData, PlyElement
def save_pc_as_ply(pc, path):
    have_rgb = True if pc.shape[1] == 6 else False
    xyz = pc[:, :3]
    if have_rgb:
        if pc[:, 3:].max() <= 1.0:
            pc[:, 3:] = pc[:, 3:] * 255.0
        rgb = pc[:, 3:].astype(np.uint8)
        vertex_data = np.empty(pc.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                         ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    else:
        vertex_data = np.empty(pc.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_data['x'] = xyz[:, 0]
    vertex_data['y'] = xyz[:, 1]
    vertex_data['z'] = xyz[:, 2]
    if have_rgb:
        vertex_data['red'] = rgb[:, 0]
        vertex_data['green'] = rgb[:, 1]
        vertex_data['blue'] = rgb[:, 2]
    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)
def depth_to_point(depth, fx, fy, cx, cy):
    height, width = depth.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    z = depth
    point = np.zeros((height, width, 3), dtype=np.float32)

    point[:, :, 0] = (x - cx) * z / fx
    point[:, :, 1] = (y - cy) * z / fy
    point[:, :, 2] = z
    return point

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
        wandb_test_step_defined = False

    # seed
    set_seed(cfg.seed + rank)

    # dataset
    cfg.resolution = '[(630,476)]'
    data_norm_type = 'identity'
    test_dataset_config = f"ThreeDP4MWAI(split='test', resolution={cfg.resolution}, principal_point_centered=False, aug_crop=16, transform='colorjitter+grayscale+gaublur', data_norm_type='{data_norm_type}', ROOT='/shared/smartbot/yangsizhe/data/3dp4m/1kScenes_not-in-gripper_test', variable_num_views=True, num_views={cfg.num_view}, num_views_min={cfg.num_view_min}, num_of_scenes=1000 * 10) + ThreeDP4MWAI(split='test', resolution={cfg.resolution}, principal_point_centered=False, aug_crop=16, transform='colorjitter+grayscale+gaublur', data_norm_type='{data_norm_type}', ROOT='/shared/smartbot/yangsizhe/data/3dp4m/1kScenes_in-gripper_test', variable_num_views=True, num_views={cfg.num_view}, num_views_min={cfg.num_view_min}, num_of_scenes=1000 * 10)"
    test_dataloader = build_dataloader(
        dataset=test_dataset_config,
        num_workers=cfg.dataloader.num_workers,
        max_num_of_imgs_per_gpu=cfg.max_num_of_imgs_per_gpu,
    )

    # model
    from robo3r.models.robo3r import Robo3R
    model = Robo3R(fuse_robot_state=cfg.model.fuse_robot_state)
    model = model.cuda()
    resume_from_checkpoint(cfg.model.pi3_ckpt_path, model)
    resume_from_checkpoint(cfg.resume_from_checkpoint, model)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if rank == 0:
        print(f'total parameters: {sum(p.numel() for p in model.parameters())/1e6}M')
        for name, module in model.named_children():
            if len(list(module.parameters())) > 0:
                print(f'{name} -- num_parameter: {sum(p.numel() for p in module.parameters())/1e6}M -- requires_grad: {next(module.parameters()).requires_grad}')
    ddp_model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    # criterion
    from robo3r.models.loss_eval import Robo3RLoss
    test_criterion = Robo3RLoss()

    test_epoch = 0
    test_step = 0

    # test
    test_dataloader.dataset.set_epoch(0)
    test_dataloader.batch_sampler.set_epoch(0)
    test_tqdm_dataloader = tqdm(enumerate(test_dataloader), disable=rank!=0, total=len(test_dataloader))
    test_tqdm_dataloader.set_description(f'test epoch {test_epoch}')
    ddp_model.eval()
    for test_step_in_epoch, test_batch in test_tqdm_dataloader:
        # if test_step_in_epoch < 1010:
        #     continue
        ignore_keys = ["depthmap", "dataset", "label", "instance", "idx", "true_shape", "rng", "data_norm_type", "path"]
        for view in test_batch:
            for name in view.keys():
                if name in ignore_keys:
                    continue
                view[name] = view[name].cuda(non_blocking=True)

        with torch.inference_mode():
            with torch.autocast('cuda', enabled=cfg.use_amp, dtype=torch.bfloat16 if cfg.amp_dtype == 'bf16' else torch.float16):
                imgs = torch.stack([view['img'] for view in test_batch], dim=1)
                robot_qpos = torch.stack([view['robot_qpos'] for view in test_batch], dim=1)
                pred = ddp_model(imgs, other={'robot_qpos': robot_qpos})
                gt_robot_mask = torch.stack([view['robot_mask'] for view in test_batch], dim=1).bool()
                gt_object_mask = torch.stack([view['object_mask'] for view in test_batch], dim=1).bool()
                gt_table_mask = torch.stack([view['table_mask'] for view in test_batch], dim=1).bool()
                combined_robot_object_table_pts = torch.zeros_like(pred['local_points'])
                combined_robot_object_table_pts[gt_robot_mask] = pred['robot_point'][gt_robot_mask].clone()
                combined_robot_object_table_pts[gt_object_mask] = pred['object_point'][gt_object_mask].clone()
                combined_robot_object_table_pts[gt_table_mask] = pred['table_point'][gt_table_mask].clone()
                pred['local_points'] = combined_robot_object_table_pts

                abs_pose = pred['abs_pose']
                if cfg.model.pred_abs_pose_via_pnp:
                    B, N = pred['local_points'].shape[:2]
                    gt_camera_intrinsics = torch.stack([view['camera_intrinsics'] for view in test_batch], dim=1).flatten(0, 1).cpu().numpy()
                    gt_keypoint_3d = torch.stack([view['keypoint_3d'] for view in test_batch], dim=1).flatten(0, 1).cpu().numpy()
                    pred_keypoint_2d = pred['keypoint'].flatten(0, 1).cpu().numpy()
                    gt_keypoint_2d = torch.stack([view['keypoint'] for view in test_batch], dim=1).flatten(0, 1).cpu().numpy()
                    pred_keypoint_map = pred['keypoint_map'].permute(0, 1, 4, 2, 3).flatten(0, 1).cpu().numpy()
                    valid_keypoint_mask = pred_keypoint_map.reshape(*pred_keypoint_map.shape[:2], -1).max(axis=-1) >= np.array([1-1e-8, 1-1e-8, 1-1e-8, 0.9, 0.9, 1-1e-8, 1-1e-8, 1-1e-8])
                    is_in_frame = gt_keypoint_2d.sum(axis=-1) != -2.0
                    valid_keypoint_mask = valid_keypoint_mask & is_in_frame
                    for idx in range(B*N):
                        is_approx_line = if_is_approx_line(gt_keypoint_2d[idx][valid_keypoint_mask[idx]], threshold=30.0)
                        if is_approx_line:
                            valid_keypoint_mask[idx][:] = False
                        object_mask = test_batch[idx%N]['object_mask'][idx//N]
                        table_mask = test_batch[idx%N]['table_mask'][idx//N]
                        if (object_mask[gt_keypoint_2d[idx].astype(int)[-1][1], gt_keypoint_2d[idx].astype(int)[-1][0]].bool() or table_mask[gt_keypoint_2d[idx].astype(int)[-1][1], gt_keypoint_2d[idx].astype(int)[-1][0]].bool()) and (object_mask[gt_keypoint_2d[idx].astype(int)[-2][1], gt_keypoint_2d[idx].astype(int)[-2][0]].bool() or table_mask[gt_keypoint_2d[idx].astype(int)[-2][1], gt_keypoint_2d[idx].astype(int)[-2][0]].bool()):
                            valid_keypoint_mask[idx][-2:] = False
                    c2w_matrix_pnp_list = []
                    T = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
                    for idx in range(B*N):
                        if valid_keypoint_mask[idx].sum() >= 6:
                            c2w_matrix_pnp = get_c2w_via_pnp(gt_keypoint_3d[idx][valid_keypoint_mask[idx]], pred_keypoint_2d[idx][valid_keypoint_mask[idx]], gt_camera_intrinsics[idx])
                            c2w_matrix_pnp[:3, :3] = T @ c2w_matrix_pnp[:3, :3]
                            c2w_matrix_pnp[:3, 3] = T @ c2w_matrix_pnp[:3, 3]
                        else:
                            c2w_matrix_pnp = pred['abs_pose'].flatten(0, 1).cpu().numpy()[idx]
                        c2w_matrix_pnp_list.append(c2w_matrix_pnp)
                    abs_pose = torch.from_numpy(np.stack(c2w_matrix_pnp_list, axis=0)).cuda().float().reshape(B, N, 4, 4)

                pred = {
                    'local_points': combined_robot_object_table_pts,
                    'camera_poses': pred['camera_poses'],
                    'scale': pred['scale'],
                    'abs_pose': abs_pose,
                }

                with torch.autocast("cuda", enabled=False):
                    metric = test_criterion(pred, test_batch)

        test_metric = all_reduce_data(metric, all_reduce=True)

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

    # finish
    if rank == 0 and cfg.report_to_wandb:
        wandb.finish()
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    with hydra.initialize(config_path='config'):
        import sys; overrides = sys.argv[1:]
        cfg = hydra.compose(config_name='config', overrides=overrides)
    main(cfg)
