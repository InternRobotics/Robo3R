import os
import random

import numpy as np
from functools import partial
import torch
from torch.optim.lr_scheduler import LambdaLR


def all_reduce_data(data, all_reduce=True):
    if not all_reduce:
        return data
    torch.distributed.barrier()
    if isinstance(data, dict):
        keys = sorted(data.keys())
        values = torch.tensor([data[k] for k in keys], dtype=torch.get_default_dtype(), device='cuda')
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
        values /= torch.distributed.get_world_size()
        # if torch.distributed.get_rank() == 0:
        return {k: v.item() for k, v in zip(keys, values)}
    elif isinstance(data, list):
        data = torch.tensor(data, dtype=torch.get_default_dtype(), device='cuda')
        torch.distributed.all_reduce(data, op=torch.distributed.ReduceOp.SUM)
        data /= torch.distributed.get_world_size()
        # if torch.distributed.get_rank() == 0:
        return data.cpu().tolist()
    elif isinstance(data, (int, float)):
        data = torch.tensor(data, dtype=torch.get_default_dtype(), device='cuda')
        torch.distributed.all_reduce(data, op=torch.distributed.ReduceOp.SUM)
        data /= torch.distributed.get_world_size()
        # if torch.distributed.get_rank() == 0:
        return data.item()
    else:
        raise ValueError(f'Unsupported data type: {type(data)}')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_parameter_name(model, excluded_layer_type_list, excluded_layer_name_list=[], included_layer_name_list=[]):
    """
    Returns the names of the model parameters that are not inside a excluded layer.
    """
    result = []
    for name, child in model.named_children():
        child_params = get_parameter_name(child, excluded_layer_type_list, excluded_layer_name_list, included_layer_name_list)
        result += [
            f"{name}.{n}"
            for n in child_params
            if not isinstance(child, tuple(excluded_layer_type_list))
            and not any(excluded_name in f"{name}.{n}".lower() for excluded_name in excluded_layer_name_list)
            and any(include_name in f"{name}.{n}".lower() for include_name in included_layer_name_list)
        ]
    # Add model specific parameters that are not in any child
    result += [
        k for k in model._parameters.keys() if not any(excluded_name in k.lower() for excluded_name in excluded_layer_name_list) and any(include_name in k.lower() for include_name in included_layer_name_list)
    ]

    return result

def _get_cosine_schedule_with_warmup_lr_lambda(current_step, num_warmup_steps, num_training_steps):
    if current_step < num_warmup_steps:
        return 0.01 + 0.99 * current_step / max(1, num_warmup_steps)
    progress = (current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)

    return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)

def resume_from_checkpoint(checkpoint_path, model, optimizer=None, lr_scheduler=None, scaler=None):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    start_epoch = 0
    start_step_in_epoch = 0
    train_step = 0
    model.load_state_dict(checkpoint['model'], False)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        start_step_in_epoch = checkpoint['step_in_epoch']
        train_step = checkpoint['train_step']
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    if scaler is not None:
        scaler.load_state_dict(checkpoint['scaler'])

    return start_epoch, start_step_in_epoch, train_step

def get_model_state_dict(model, require_grad=True):
    state_dict = model.state_dict()
    if not require_grad:
        return state_dict
    for name, p in model.named_parameters():
        if not p.requires_grad:
            del state_dict[name]

    return state_dict

def save_checkpoint(checkpoint_path, model, optimizer=None, lr_scheduler=None, scaler=None, epoch=None, step_in_epoch=None, train_step=None, require_grad=True):
    checkpoint_dict = {
        'model': get_model_state_dict(model, require_grad),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
        'scaler': scaler.state_dict() if scaler is not None else None,
        'epoch': epoch if epoch is not None else None,
        'step_in_epoch': step_in_epoch if step_in_epoch is not None else None,
        'train_step': train_step if train_step is not None else None,
    }
    print(f'Saving checkpoint to {checkpoint_path}')
    torch.save(checkpoint_dict, checkpoint_path)

def train_one_iteration(ddp_model, batch, criterion, optimizer, lr_scheduler, use_amp, amp_dtype, scaler, max_grad_norm, grad_acc_step, cur_grad_acc_step, output_dir=None):
    ignore_keys = ["depthmap", "dataset", "label", "instance", "idx", "true_shape", "rng", "data_norm_type", "path"]
    for view in batch:
        for name in view.keys():
            if name in ignore_keys:
                continue
            view[name] = view[name].cuda(non_blocking=True)
    num_view = len(batch)
    with torch.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16 if amp_dtype == 'bf16' else torch.float16):
        imgs = torch.stack([view['img'] for view in batch], dim=1)
        robot_qpos = torch.stack([view['robot_qpos'] for view in batch], dim=1)
        pred = ddp_model(imgs, other={'robot_qpos': robot_qpos})
        with torch.autocast("cuda", enabled=False):
            loss, metric = criterion(pred, batch)
        # if num_view > 2:
        #     loss = loss * (2 / num_view)
        loss /= grad_acc_step
        if loss > 10:
            loss = loss * 0.0

    scaler.scale(loss).backward()

    if (cur_grad_acc_step + 1) % grad_acc_step == 0:
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        lr_scheduler.step()

        metric['grad_norm'] = grad_norm.item()
        for param_group_idx in range(len(optimizer.param_groups)):
            metric[f'lr_{param_group_idx}'] = optimizer.param_groups[param_group_idx]['lr']
        keys_to_delete = [k for k in metric.keys() if 'view' in k]
        for k in keys_to_delete:
            del metric[k]

        return metric

def test_one_iteration(ddp_model, batch, criterion, use_amp, amp_dtype):
    ignore_keys = ["depthmap", "dataset", "label", "instance", "idx", "true_shape", "rng", "data_norm_type", "path"]
    for view in batch:
        for name in view.keys():
            if name in ignore_keys:
                continue
            view[name] = view[name].cuda(non_blocking=True)
    num_view = len(batch)
    with torch.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16 if amp_dtype == 'bf16' else torch.float16):
        # loss, loss_details = ddp_model(batch)
        imgs = torch.stack([view['img'] for view in batch], dim=1)
        robot_qpos = torch.stack([view['robot_qpos'] for view in batch], dim=1)
        pred = ddp_model(imgs, other={'robot_qpos': robot_qpos})
        with torch.autocast("cuda", enabled=False):
            loss, metric = criterion(pred, batch)

    keys_to_delete = [k for k in metric.keys() if 'view' in k]
    for k in keys_to_delete:
        del metric[k]

    return metric
