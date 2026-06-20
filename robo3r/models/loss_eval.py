import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import *
import math
import numpy as np

from ..utils.geometry import homogenize_points, se3_inverse, depth_edge
from ..utils.alignment import align_points_scale


def weighted_mean(x: torch.Tensor, w: torch.Tensor = None, dim: Union[int, torch.Size] = None, keepdim: bool = False, eps: float = 1e-7) -> torch.Tensor:
    if w is None:
        return x.mean(dim=dim, keepdim=keepdim)
    else:
        w = w.to(x.dtype)
        return (x * w).mean(dim=dim, keepdim=keepdim) / w.mean(dim=dim, keepdim=keepdim).add(eps)

def _smooth(err: torch.FloatTensor, beta: float = 0.0) -> torch.FloatTensor:
    if beta == 0:
        return err
    else:
        return torch.where(err < beta, 0.5 * err.square() / beta, err - 0.5 * beta)

def angle_diff_vec3(v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-12):
    return torch.atan2(torch.cross(v1, v2, dim=-1).norm(dim=-1) + eps, (v1 * v2).sum(dim=-1))


class PointLoss(nn.Module):
    def __init__(self, local_align_res=4096, expected_dist_thresh=0.02):
        super().__init__()
        self.local_align_res = local_align_res

    def prepare_ROE(self, pts, mask, target_size=4096):
        B, N, H, W, C = pts.shape
        output = []
        
        for i in range(B):
            valid_pts = pts[i][mask[i]]

            if valid_pts.shape[0] > 0:
                valid_pts = valid_pts.permute(1, 0).unsqueeze(0)  # (1, 3, N1)
                # NOTE: Is is important to use nearest interpolate. Linear interpolate will lead to unstable result!
                valid_pts = F.interpolate(valid_pts, size=target_size, mode='nearest')  # (1, 3, target_size)
                valid_pts = valid_pts.squeeze(0).permute(1, 0)  # (target_size, 3)
            else:
                valid_pts = torch.ones((target_size, C), device=valid_pts.device)

            output.append(valid_pts)

        return torch.stack(output, dim=0)

    def forward(self, pred, gt):
        pred_local_pts = pred['local_points']
        gt_local_pts = gt['local_points']
        # gt_local_pts = gt['global_points']
        valid_masks = gt['valid_masks']
        details = dict()

        B, N, H, W, _ = pred_local_pts.shape

        weights_ = gt_local_pts[..., 2]
        weights_ = weights_.clamp_min(0.1 * weighted_mean(weights_, valid_masks, dim=(-2, -1), keepdim=True))
        weights_ = 1 / (weights_ + 1e-6)

        # alignment
        with torch.no_grad():
            xyz_pred_local = self.prepare_ROE(pred_local_pts.reshape(B, N, H, W, 3), valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()

            xyz_gt_local = self.prepare_ROE(gt_local_pts.reshape(B, N, H, W, 3), valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()
            xyz_weights_local = self.prepare_ROE((weights_[..., None]).reshape(B, N, H, W, 1), valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()[:, :, 0]

            S_opt_local = align_points_scale(xyz_pred_local, xyz_gt_local, xyz_weights_local)
            S_opt_local[S_opt_local <= 0] *= -1

        aligned_local_pts = S_opt_local.view(B, 1, 1, 1, 1) * pred_local_pts
        # local point loss
        local_pts_loss = F.l1_loss(aligned_local_pts[valid_masks].float(), gt_local_pts[valid_masks].float(), reduction='none') #* weights_[valid_masks].float()[..., None]
        details['local_pts_loss'] = local_pts_loss.mean()

        return details, S_opt_local


class RelativeCameraLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def rot_ang_loss(self, R, Rgt, eps=1e-6):
        """
        Args:
            R: estimated rotation matrix [B, 3, 3]
            Rgt: ground-truth rotation matrix [B, 3, 3]
        Returns:  
            R_err: rotation angular error 
        """
        residual = torch.matmul(R.transpose(1, 2), Rgt)
        trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(-1)
        cosine = (trace - 1) / 2
        R_err = torch.acos(torch.clamp(cosine, -1.0 + eps, 1.0 - eps))  # handle numerical errors and NaNs
        return R_err         # [0, 3.14]
    
    def forward(self, pred, gt, scale):
        pred_pose = pred['camera_poses']
        gt_pose = gt['camera_poses']

        B, N, _, _ = pred_pose.shape

        pred_pose_align = pred_pose.clone()
        pred_pose_align[..., :3, 3] *= scale.view(B, 1, 1)

        pred_w2c = se3_inverse(pred_pose_align)
        gt_w2c = se3_inverse(gt_pose)

        pred_w2c_exp = pred_w2c.unsqueeze(2)
        pred_pose_exp = pred_pose_align.unsqueeze(1)

        gt_w2c_exp = gt_w2c.unsqueeze(2)
        gt_pose_exp = gt_pose.unsqueeze(1)

        pred_rel_all = torch.matmul(pred_w2c_exp, pred_pose_exp)
        gt_rel_all = torch.matmul(gt_w2c_exp, gt_pose_exp)

        mask = ~torch.eye(N, dtype=torch.bool, device=pred_pose.device)

        # fxfycxcy_loss = F.l1_loss(pred['camera_fxfycxcy'], gt['camera_fxfycxcy'], reduction='mean')  # pred intrinsics

        if N == 1:  # single view input, do not compute relative pose loss
            return dict(trans_loss=torch.tensor(0.0, device=pred_pose.device), rot_loss=torch.tensor(0.0, device=pred_pose.device))  # , fxfycxcy_loss=fxfycxcy_loss

        t_pred = pred_rel_all[..., :3, 3][:, mask, ...]
        R_pred = pred_rel_all[..., :3, :3][:, mask, ...]
        
        t_gt = gt_rel_all[..., :3, 3][:, mask, ...]
        R_gt = gt_rel_all[..., :3, :3][:, mask, ...]

        trans_loss = F.l1_loss(t_pred, t_gt, reduction='none')
        trans_loss = trans_loss.norm(dim=-1, p=2).flatten(0, 1)
        rot_loss = self.rot_ang_loss(R_pred.reshape(-1, 3, 3), R_gt.reshape(-1, 3, 3))

        trans_acc_005 = (trans_loss < 0.05).float().mean()
        trans_acc_003 = (trans_loss < 0.03).float().mean()
        trans_acc_001 = (trans_loss < 0.01).float().mean()

        rot_acc_005 = (rot_loss < 0.05).float().mean()
        rot_acc_003 = (rot_loss < 0.03).float().mean()
        rot_acc_001 = (rot_loss < 0.01).float().mean()

        return dict(trans_loss=trans_loss.mean(), rot_loss=rot_loss.mean(), trans_acc_005=trans_acc_005, trans_acc_003=trans_acc_003, trans_acc_001=trans_acc_001, rot_acc_005=rot_acc_005, rot_acc_003=rot_acc_003, rot_acc_001=rot_acc_001)


class AbsoluteCameraLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def rot_ang_loss(self, R, Rgt, eps=1e-6):
        """
        Args:
            R: estimated rotation matrix [B, 3, 3]
            Rgt: ground-truth rotation matrix [B, 3, 3]
        Returns:  
            R_err: rotation angular error 
        """
        residual = torch.matmul(R.transpose(1, 2), Rgt)
        trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(-1)
        cosine = (trace - 1) / 2
        R_err = torch.acos(torch.clamp(cosine, -1.0 + eps, 1.0 - eps))  # handle numerical errors and NaNs
        return R_err         # [0, 3.14]
    
    def forward(self, pred, gt, scale):
        pred_pose = pred['abs_pose'].reshape(-1, 4, 4)  # (B, 4, 4)
        gt_pose = gt['camera_abs_pose'].reshape(-1, 4, 4)  # (B, 4, 4)

        t_pred = pred_pose[:, :3, 3]
        R_pred = pred_pose[:, :3, :3]
        
        t_gt = gt_pose[:, :3, 3]
        R_gt = gt_pose[:, :3, :3]

        trans_loss = F.l1_loss(t_pred, t_gt, reduction='none').norm(dim=-1, p=2)
        rot_loss = self.rot_ang_loss(R_pred.reshape(-1, 3, 3), R_gt.reshape(-1, 3, 3))

        trans_acc_005 = (trans_loss < 0.05).float().mean()
        trans_acc_003 = (trans_loss < 0.03).float().mean()
        trans_acc_001 = (trans_loss < 0.01).float().mean()

        rot_acc_005 = (rot_loss < 0.05).float().mean()
        rot_acc_003 = (rot_loss < 0.03).float().mean()
        rot_acc_001 = (rot_loss < 0.01).float().mean()

        return dict(abs_trans_loss=trans_loss.mean(), abs_rot_loss=rot_loss.mean(), abs_trans_acc_005=trans_acc_005, abs_trans_acc_003=trans_acc_003, abs_trans_acc_001=trans_acc_001, abs_rot_acc_005=rot_acc_005, abs_rot_acc_003=rot_acc_003, abs_rot_acc_001=rot_acc_001)


class Robo3RLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.point_loss = PointLoss()
        # self.camera_loss = RelativeCameraLoss()
        self.camera_loss = AbsoluteCameraLoss()

    def prepare_gt(self, gt):
        gt_pts = torch.stack([view['pts3d'] for view in gt], dim=1)
        B, N, H, W, _ = gt_pts.shape
        masks = torch.stack([view['valid_mask'] for view in gt], dim=1)
        poses = torch.stack([view['camera_pose'] for view in gt], dim=1)
        abs_pose = poses.clone()
        camera_intrinsics = torch.stack([view['camera_intrinsics'] for view in gt], dim=1)
        camera_fxfycxcy = torch.stack([
            camera_intrinsics[..., 0, 0] / W,  # fx
            camera_intrinsics[..., 1, 1] / H,  # fy
            camera_intrinsics[..., 0, 2] / W,  # cx
            camera_intrinsics[..., 1, 2] / H,  # cy
        ], dim=-1)

        # transform to first frame camera coordinate
        w2c_target = se3_inverse(poses[:, 0])
        gt_pts = torch.einsum('bij, bnhwj -> bnhwi', w2c_target, homogenize_points(gt_pts))[..., :3]
        poses = torch.einsum('bij, bnjk -> bnik', w2c_target, poses)

        # normalize points
        valid_batch = masks.sum([-1, -2, -3]) > 0
        if valid_batch.sum() > 0:
            B_ = valid_batch.sum()
            all_pts = gt_pts[valid_batch].clone()
            all_pts[~masks[valid_batch]] = 0
            all_pts = all_pts.reshape(B_, N, -1, 3)
            all_dis = all_pts.norm(dim=-1)

            # Note: the norm_factor will also influence the gt_local_pts
            norm_factor = torch.ones(B_, device=all_dis.device).to(all_dis.dtype)

        extrinsics = se3_inverse(poses)
        gt_local_pts = torch.einsum('bnij, bnhwj -> bnhwi', extrinsics, homogenize_points(gt_pts))[..., :3]
        
        dataset_names = gt[0]['dataset']

        return dict(
            path=[view['path'] for view in gt],
            imgs = torch.stack([view['img'] for view in gt], dim=1),
            global_points=gt_pts,
            local_points=gt_local_pts,
            valid_masks=masks,
            camera_poses=poses,
            camera_abs_pose=abs_pose,
            camera_fxfycxcy=camera_fxfycxcy,
            dataset_names=dataset_names,
            norm_factor=norm_factor,

            # pred keypoint
            keypoint_map=torch.stack([view['keypoint_map'] for view in gt], dim=1),
            keypoint=torch.stack([view['keypoint'] for view in gt], dim=1)
        )

    def normalize_pred(self, pred, gt):
        local_points = pred['local_points']
        camera_poses = pred['camera_poses']
        abs_pose = pred['abs_pose']
        B, N, H, W, _ = local_points.shape
        masks = gt['valid_masks']
        camera_poses_normalized = camera_poses.clone()
        abs_pose_normalized = abs_pose.clone()

        # normalize predict points
        all_pts = local_points.clone()
        all_pts[~masks] = 0
        all_pts = all_pts.reshape(B, N, -1, 3)
        all_dis = all_pts.norm(dim=-1)

        norm_factor = torch.ones(B, device=all_dis.device).to(all_dis.dtype)

        pred['local_points'] = local_points
        pred['camera_poses'] = camera_poses_normalized
        pred['abs_pose'] = abs_pose_normalized
        pred['norm_factor'] = norm_factor

        return pred

    def forward(self, pred, gt_raw):
        gt = self.prepare_gt(gt_raw)
        pred = self.normalize_pred(pred, gt)

        details = dict()

        # Local Point Loss
        point_loss_details, scale = self.point_loss(pred, gt)
        details.update(point_loss_details)

        # Camera Loss
        camera_loss_details = self.camera_loss(pred, gt, scale)
        details.update(camera_loss_details)

        # pred scale
        gt_scale = scale.detach() * gt['norm_factor'].detach() / pred['norm_factor'].detach()
        if 'scale' not in pred:
            pred['scale'] = torch.ones_like(gt_scale)
        scale_loss = F.l1_loss(pred['scale'].squeeze(-1).squeeze(-1).float(), gt_scale.float(), reduction='mean')
        details.update({'scale_loss': scale_loss})

        for k, v in details.items():
            details[k] = v.item()

        return details
