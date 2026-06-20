import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import *
import math
import numpy as np

from ..utils.geometry import homogenize_points, se3_inverse, depth_edge
from ..utils.alignment import align_points_scale


def widen_mask(mask, radius=2):xe
    input_tensor = mask.float()
    
    kernel_size = 2 * radius + 1
    padding = radius
    
    dilated_mask = F.max_pool2d(
        input_tensor, 
        kernel_size=kernel_size, 
        stride=1, 
        padding=padding
    )
    
    if mask.dtype == torch.bool:
        return dilated_mask > 0.5
    else:
        return dilated_mask

def widen_point_map(gt_local_pts, gt_robot_mask, radius=2):
    B, C, H, W, D = gt_local_pts.shape
    
    pts_flat = gt_local_pts.view(B * C, H, W, D).permute(0, 3, 1, 2).contiguous()
    
    mask_flat = gt_robot_mask.view(B * C, 1, H, W).contiguous()

    widened_mask_flat = widen_mask(mask_flat, radius=radius)
    
    gap_mask = widened_mask_flat & (~mask_flat)
    
    result_pts = pts_flat.clone()
    
    offsets = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0: continue
            dist_sq = dy**2 + dx**2
            offsets.append((dist_sq, dy, dx))
    
    offsets.sort(key=lambda x: x[0])
    
    is_filled = torch.zeros_like(gap_mask, dtype=torch.bool)
    
    for _, dy, dx in offsets:
        if is_filled[gap_mask].all():
            break
            
        shifted_src_mask = torch.roll(mask_flat, shifts=(dy, dx), dims=(2, 3))
        
        if dy > 0: shifted_src_mask[:, :, :dy, :] = False
        elif dy < 0: shifted_src_mask[:, :, dy:, :] = False
        if dx > 0: shifted_src_mask[:, :, :, :dx] = False
        elif dx < 0: shifted_src_mask[:, :, :, dx:] = False
        
        should_update = gap_mask & (~is_filled) & shifted_src_mask
        
        if should_update.any():
            shifted_pts = torch.roll(pts_flat, shifts=(dy, dx), dims=(2, 3))
            
            result_pts = torch.where(should_update, shifted_pts, result_pts)
            
            is_filled = is_filled | should_update

    result_pts = result_pts.permute(0, 2, 3, 1).view(B, C, H, W, D)
    
    return result_pts, widened_mask_flat.view(B, C, H, W)

def widen_point_map_no(gt_local_pts, gt_part_mask, radius=2):
    return gt_local_pts, gt_part_mask

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
    def __init__(self, local_align_res=4096):
        super().__init__()
        self.local_align_res = local_align_res
        self.criteria_local = nn.L1Loss(reduction='none')

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
    
    def noraml_loss(self, points, gt_points, mask):
        not_edge = ~depth_edge(gt_points[..., 2], rtol=0.03)
        mask = torch.logical_and(mask, not_edge)

        leftup, rightup, leftdown, rightdown = points[..., :-1, :-1, :], points[..., :-1, 1:, :], points[..., 1:, :-1, :], points[..., 1:, 1:, :]
        upxleft = torch.cross(rightup - rightdown, leftdown - rightdown, dim=-1)
        leftxdown = torch.cross(leftup - rightup, rightdown - rightup, dim=-1)
        downxright = torch.cross(leftdown - leftup, rightup - leftup, dim=-1)
        rightxup = torch.cross(rightdown - leftdown, leftup - leftdown, dim=-1)

        gt_leftup, gt_rightup, gt_leftdown, gt_rightdown = gt_points[..., :-1, :-1, :], gt_points[..., :-1, 1:, :], gt_points[..., 1:, :-1, :], gt_points[..., 1:, 1:, :]
        gt_upxleft = torch.cross(gt_rightup - gt_rightdown, gt_leftdown - gt_rightdown, dim=-1)
        gt_leftxdown = torch.cross(gt_leftup - gt_rightup, gt_rightdown - gt_rightup, dim=-1)
        gt_downxright = torch.cross(gt_leftdown - gt_leftup, gt_rightup - gt_leftup, dim=-1)
        gt_rightxup = torch.cross(gt_rightdown - gt_leftdown, gt_leftup - gt_leftdown, dim=-1)

        mask_leftup, mask_rightup, mask_leftdown, mask_rightdown = mask[..., :-1, :-1], mask[..., :-1, 1:], mask[..., 1:, :-1], mask[..., 1:, 1:]
        mask_upxleft = mask_rightup & mask_leftdown & mask_rightdown
        mask_leftxdown = mask_leftup & mask_rightdown & mask_rightup
        mask_downxright = mask_leftdown & mask_rightup & mask_leftup
        mask_rightxup = mask_rightdown & mask_leftup & mask_leftdown

        MIN_ANGLE, MAX_ANGLE, BETA_RAD = math.radians(1), math.radians(90), math.radians(3)

        loss = mask_upxleft * _smooth(angle_diff_vec3(upxleft, gt_upxleft).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
                + mask_leftxdown * _smooth(angle_diff_vec3(leftxdown, gt_leftxdown).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
                + mask_downxright * _smooth(angle_diff_vec3(downxright, gt_downxright).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
                + mask_rightxup * _smooth(angle_diff_vec3(rightxup, gt_rightxup).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD)

        loss = loss.mean() / (4 * max(points.shape[-3:-1]))

        return loss

    def forward(self, pred, gt):
        pred_local_pts = pred['local_points']
        gt_local_pts = gt['local_points']
        valid_masks = gt['valid_masks']
        details = dict()
        final_loss = 0.0

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

            # # seperate pred  # seperate pred remove original local point  # use sep pred for scale
            # gt_robot_mask = gt['robot_mask'].bool()
            # gt_object_mask = gt['object_mask'].bool()
            # gt_table_mask = gt['table_mask'].bool()
            # combined_robot_object_table_pts = torch.zeros_like(pred_local_pts)
            # combined_robot_object_table_pts[gt_robot_mask] = pred['robot_point'][gt_robot_mask].clone()
            # combined_robot_object_table_pts[gt_object_mask] = pred['object_point'][gt_object_mask].clone()
            # combined_robot_object_table_pts[gt_table_mask] = pred['table_point'][gt_table_mask].clone()
            # xyz_pred_local = self.prepare_ROE(combined_robot_object_table_pts.reshape(B, N, H, W, 3), valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()
            # xyz_gt_local = self.prepare_ROE(gt_local_pts.reshape(B, N, H, W, 3), valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()
            # xyz_weights_local = self.prepare_ROE((weights_[..., None]).reshape(B, N, H, W, 1), valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()[:, :, 0]
            # S_opt_local = align_points_scale(xyz_pred_local, xyz_gt_local, xyz_weights_local)
            # S_opt_local[S_opt_local <= 0] *= -1

            # S_opt_local = S_opt_local.clone().detach()
            # details['opt_scale_diff'] = S_opt_local.mean() - S_opt_local.mean()
            # details['S_opt_local'] = S_opt_local.mean()
            # print(S_opt_local.mean(), S_opt_local.mean(), details['opt_scale_diff'])

            # tmp = (((combined_robot_object_table_pts[0,0]-combined_robot_object_table_pts[0,0].min())/combined_robot_object_table_pts[0,0].max()).clone().detach().cpu().numpy() * 255).astype(np.uint8)
            # tmp = (gt_robot_mask[0,0].clone().detach().cpu().numpy() * 255).astype(np.uint8)
            # from PIL import Image; Image.fromarray(tmp).save('tmp/scale_debug/tmp0.png')
            # from util.io_util import save_pc_as_ply
            # save_pc_as_ply(combined_robot_object_table_pts[0,0].clone().detach().reshape(-1, 3).float().cpu().numpy(), 'tmp/scale_debug/combined_robot_object_table_pts.ply')
            # save_pc_as_ply(xyz_gt_local[0,0].clone().detach().reshape(-1, 3).float().cpu().numpy(), 'tmp/scale_debug/xyz_gt_local.ply')
            # save_pc_as_ply(xyz_weights_local[0,0].clone().detach().reshape(-1, 3).float().cpu().numpy(), 'tmp/scale_debug/xyz_weights_local.ply')

        # seperate pred remove original local point
        aligned_local_pts = S_opt_local.view(B, 1, 1, 1, 1) * pred_local_pts
        # local point loss
        local_pts_loss = self.criteria_local(aligned_local_pts[valid_masks].float(), gt_local_pts[valid_masks].float()) * weights_[valid_masks].float()[..., None]
        final_loss += local_pts_loss.mean()
        details['local_pts_loss'] = local_pts_loss.mean()

        # aligned_local_pts[~valid_masks] = 0.0
        # pred_local_pts[~valid_masks] = 0.0
        # pred_local_pts_avg_norm = pred_local_pts.norm(dim=-1).mean(dim=[-3, -2, -1])
        # gt_local_pts[~valid_masks] = 0.0
        # gt_local_pts_avg_norm = gt_local_pts.norm(dim=-1).mean(dim=[-3, -2, -1])
        # scale = gt_local_pts_avg_norm / pred_local_pts_avg_norm
        # aligned_local_pts_new = scale.view(B, 1, 1, 1, 1) * pred_local_pts
        # from util.io_util import save_pc_as_ply
        # save_pc_as_ply(aligned_local_pts_new[0,0].clone().detach().reshape(-1, 3).float().cpu().numpy(), 'tmp/scale_debug/aligned_local_pts_new.ply')
        # save_pc_as_ply(pred_local_pts[0,0].clone().detach().reshape(-1, 3).float().cpu().numpy(), 'tmp/scale_debug/pred_local_pts.ply')
        # save_pc_as_ply(aligned_local_pts[0,0].clone().detach().reshape(-1, 3).float().cpu().numpy(), 'tmp/scale_debug/aligned_local_pts.ply')
        # save_pc_as_ply(gt_local_pts[0,0].clone().detach().reshape(-1, 3).float().cpu().numpy(), 'tmp/scale_debug/gt_local_pts.ply')
        # # gt['norm_factor'] / pred['norm_factor'] * S_opt_local
        # # pred['scale'][:,0,0]
        # save_pc_as_ply(pred_local_pts[0,0].clone().detach().reshape(-1, 3).float().cpu().numpy() * pred['scale'][0][0,0].item(), 'tmp/scale_debug/pred_scaled_local_pts.ply')


        # aligned_local_pts[~valid_masks] = 0.0
        # gt_local_pts[~valid_masks] = 0.0
        # diff_map = torch.abs(aligned_local_pts.float().clone().detach() - gt_local_pts.float().clone().detach())
        # diff_map = diff_map[0, 0].sum(dim=-1)
        # diff_map /= diff_map.max()
        # diff_map = (diff_map.cpu().numpy() * 255.0).astype(np.uint8)
        # from PIL import Image; Image.fromarray(diff_map).save('diff_map.png')


        # # pred diff
        # aligned_local_pts[~valid_masks] = 0.0
        # gt_local_pts[~valid_masks] = 0.0
        # acc_diff_signed = torch.zeros_like(gt_local_pts)
        # for order_idx in range(3):
        #     pred_diff = pred['diff'][..., order_idx*(3+1):(order_idx+1)*(3+1)-1]
        #     pred_diff_act = pred['diff'][..., order_idx*(3+1)+3:(order_idx+1)*(3+1)]
        #     gt_diff_signed = (gt_local_pts.float() - aligned_local_pts.float() - acc_diff_signed).clone().detach()
        #     gt_diff = torch.abs(gt_diff_signed)
        #     gt_diff_act = gt_diff_signed[..., 2:] >= 0.0

        #     diff_loss = F.l1_loss(pred_diff[valid_masks].float(), gt_diff[valid_masks].float(), reduction='mean') * 0.1
        #     diff_act_loss = F.binary_cross_entropy(pred_diff_act[valid_masks].float(), gt_diff_act[valid_masks].float(), reduction='mean') * 0.05

        #     final_loss += diff_loss
        #     details[f'diff_loss_{order_idx}'] = diff_loss
        #     final_loss += diff_act_loss
        #     details[f'diff_act_loss_{order_idx}'] = diff_act_loss

        #     pred_diff_signed = pred_diff.clone().detach()
        #     pred_diff_signed[..., 2:][pred_diff_act.clone().detach() < 0.5] *= -1.0
        #     acc_diff_signed += pred_diff_signed

        # seperate pred
        gt_robot_mask = gt['robot_mask'].bool()
        gt_object_mask = gt['object_mask'].bool()
        gt_table_mask = gt['table_mask'].bool()
        gt_widened_robot_pts, gt_widened_robot_mask = widen_point_map_no(gt_local_pts, gt_robot_mask, radius=2)
        gt_widened_object_pts, gt_widened_object_mask = widen_point_map_no(gt_local_pts, gt_object_mask, radius=2)
        gt_widened_table_pts, gt_widened_table_mask = widen_point_map_no(gt_local_pts, gt_table_mask, radius=2)
        aligned_robot_point = S_opt_local.view(B, 1, 1, 1, 1).detach() * pred['robot_point']
        aligned_object_point = S_opt_local.view(B, 1, 1, 1, 1).detach() * pred['object_point']
        aligned_table_point = S_opt_local.view(B, 1, 1, 1, 1).detach() * pred['table_point']
        robot_point_loss = self.criteria_local(aligned_robot_point[gt_widened_robot_mask].float(), gt_widened_robot_pts[gt_widened_robot_mask].float())
        object_point_loss = self.criteria_local(aligned_object_point[gt_widened_object_mask].float(), gt_widened_object_pts[gt_widened_object_mask].float())
        table_point_loss = self.criteria_local(aligned_table_point[gt_widened_table_mask].float(), gt_widened_table_pts[gt_widened_table_mask].float())
        final_loss += robot_point_loss.mean()
        final_loss += object_point_loss.mean()
        final_loss += table_point_loss.mean()
        details['robot_point_loss'] = robot_point_loss.mean()
        details['object_point_loss'] = object_point_loss.mean()
        details['table_point_loss'] = table_point_loss.mean()
        robot_mask_loss = F.binary_cross_entropy(pred['robot_mask'].float(), gt_robot_mask.float(), reduction='mean')
        object_mask_loss = F.binary_cross_entropy(pred['object_mask'].float(), gt_object_mask.float(), reduction='mean')
        table_mask_loss = F.binary_cross_entropy(pred['table_mask'].float(), gt_table_mask.float(), reduction='mean')
        final_loss += robot_mask_loss * 0.25
        final_loss += object_mask_loss * 0.25
        final_loss += table_mask_loss * 0.25
        details['robot_mask_loss'] = robot_mask_loss
        details['object_mask_loss'] = object_mask_loss
        details['table_mask_loss'] = table_mask_loss

        # seperate pred remove original local point
        # normal loss
        normal_batch_id = [i for i in range(len(gt['dataset_names']))]
        if len(normal_batch_id) == 0:
            normal_loss = 0.0 * aligned_local_pts.mean()
        else:
            normal_loss = self.noraml_loss(aligned_local_pts[normal_batch_id], gt_local_pts[normal_batch_id], valid_masks[normal_batch_id])
            final_loss += normal_loss.mean() * 0.2 * 0.2
        details['normal_loss'] = normal_loss.mean()

        # seperate pred  # normal loss
        robot_normal_loss = self.noraml_loss(aligned_robot_point, gt_widened_robot_pts, gt_robot_mask)
        final_loss += robot_normal_loss.mean()
        details['robot_normal_loss'] = robot_normal_loss.mean()
        object_normal_loss = self.noraml_loss(aligned_object_point, gt_widened_object_pts, gt_object_mask)
        final_loss += object_normal_loss.mean()
        details['object_normal_loss'] = object_normal_loss.mean()
        table_normal_loss = self.noraml_loss(aligned_table_point, gt_widened_table_pts, gt_table_mask)
        final_loss += table_normal_loss.mean() * 0.2 * 0.2
        details['table_normal_loss'] = table_normal_loss.mean()

        return final_loss, details, S_opt_local


class CameraLoss(nn.Module):
    def __init__(self, alpha=100):
        super().__init__()
        self.alpha = alpha

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
        return R_err.mean()         # [0, 3.14]
    
    def forward(self, pred, gt, scale, cal_abs_pose=True):
        pred_pose = pred['camera_poses']
        gt_pose = gt['camera_poses']

        B, N, _, _ = pred_pose.shape

        pred_pose_align = pred_pose.clone()
        pred_pose_align[..., :3, 3] *= scale.view(B, 1, 1)

        if cal_abs_pose:
            gt_abs_pose = gt['camera_abs_pose']
            pred_abs_pose = pred['abs_pose']
            t_abs_pred = pred_abs_pose[..., :3, 3]
            t_abs_gt = gt_abs_pose[..., :3, 3]
            R_abs_pred = pred_abs_pose[..., :3, :3]
            R_abs_gt = gt_abs_pose[..., :3, :3]
            abs_trans_loss = F.l1_loss(t_abs_pred, t_abs_gt, reduction='mean')
            abs_rot_loss = self.rot_ang_loss(
                R_abs_pred.reshape(-1, 3, 3), 
                R_abs_gt.reshape(-1, 3, 3)
            )
        else:
            abs_trans_loss = torch.tensor(0.0).cuda()
            abs_rot_loss = torch.tensor(0.0).cuda()

        pred_w2c = se3_inverse(pred_pose_align)
        gt_w2c = se3_inverse(gt_pose)
        
        pred_w2c_exp = pred_w2c.unsqueeze(2)
        pred_pose_exp = pred_pose_align.unsqueeze(1)
        
        gt_w2c_exp = gt_w2c.unsqueeze(2)
        gt_pose_exp = gt_pose.unsqueeze(1)
        
        pred_rel_all = torch.matmul(pred_w2c_exp, pred_pose_exp)
        gt_rel_all = torch.matmul(gt_w2c_exp, gt_pose_exp)

        mask = ~torch.eye(N, dtype=torch.bool, device=pred_pose.device)

        fxfycxcy_loss = F.l1_loss(pred['camera_fxfycxcy'], gt['camera_fxfycxcy'], reduction='mean')  # pred intrinsics

        if N == 1:  # single view input, do not compute relative pose loss
            total_loss = 0.2 * fxfycxcy_loss + self.alpha * abs_trans_loss + abs_rot_loss
            return total_loss, dict(trans_loss=torch.tensor(0.0, device=pred_pose.device), rot_loss=torch.tensor(0.0, device=pred_pose.device), fxfycxcy_loss=fxfycxcy_loss, abs_trans_loss=abs_trans_loss, abs_rot_loss=abs_rot_loss)

        t_pred = pred_rel_all[..., :3, 3][:, mask, ...]
        R_pred = pred_rel_all[..., :3, :3][:, mask, ...]

        t_gt = gt_rel_all[..., :3, 3][:, mask, ...]
        R_gt = gt_rel_all[..., :3, :3][:, mask, ...]

        trans_loss = F.huber_loss(t_pred, t_gt, reduction='mean', delta=0.1)

        rot_loss = self.rot_ang_loss(
            R_pred.reshape(-1, 3, 3), 
            R_gt.reshape(-1, 3, 3)
        )

        total_loss = self.alpha * trans_loss + rot_loss + 0.2 * fxfycxcy_loss + self.alpha * abs_trans_loss + abs_rot_loss

        return total_loss, dict(trans_loss=trans_loss, rot_loss=rot_loss, fxfycxcy_loss=fxfycxcy_loss, abs_trans_loss=abs_trans_loss, abs_rot_loss=abs_rot_loss)


class Robo3RLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.point_loss = PointLoss()
        self.camera_loss = CameraLoss()

    def prepare_gt(self, gt):
        gt_pts = torch.stack([view['pts3d'] for view in gt], dim=1)
        B, N, H, W, _ = gt_pts.shape
        masks = torch.stack([view['valid_mask'] for view in gt], dim=1)
        poses = torch.stack([view['camera_pose'] for view in gt], dim=1)
        abs_pose = poses.clone()  # abs pose # delta pose
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

            # pred scale
            norm_factor = all_dis.sum(dim=[-1, -2]) / (masks[valid_batch].float().sum(dim=[-1, -2, -3]) + 1e-8)
            gt_pts[valid_batch] = gt_pts[valid_batch] / norm_factor[..., None, None, None, None]
            poses[valid_batch, ..., :3, 3] /= norm_factor[..., None, None]
            # Note: the norm_factor will also influence the gt_local_pts
            # norm_factor = torch.ones(B_, device=all_dis.device).to(all_dis.dtype)

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
            camera_abs_pose=abs_pose,  # abs pose # delta pose
            camera_fxfycxcy=camera_fxfycxcy,
            dataset_names=dataset_names,
            norm_factor=norm_factor,

            # seperate pred
            robot_mask=torch.stack([view['robot_mask'] for view in gt], dim=1),
            object_mask=torch.stack([view['object_mask'] for view in gt], dim=1),
            table_mask=torch.stack([view['table_mask'] for view in gt], dim=1),

            # pred keypoint
            keypoint_map=torch.stack([view['keypoint_map'] for view in gt], dim=1),
            keypoint=torch.stack([view['keypoint'] for view in gt], dim=1)
        )

    def normalize_pred(self, pred, gt):
        # # seperate pred remove original local point  # use sep pred for scale
        # combined_robot_object_table_pts = torch.zeros_like(pred['robot_point'])
        # combined_robot_object_table_pts[gt['robot_mask'].bool()] = pred['robot_point'][gt['robot_mask'].bool()].clone().detach()
        # combined_robot_object_table_pts[gt['object_mask'].bool()] = pred['object_point'][gt['object_mask'].bool()].clone().detach()
        # combined_robot_object_table_pts[gt['table_mask'].bool()] = pred['table_point'][gt['table_mask'].bool()].clone().detach()
        # pred['local_points'] = combined_robot_object_table_pts

        local_points = pred['local_points']
        camera_poses = pred['camera_poses']
        B, N, H, W, _ = local_points.shape
        masks = gt['valid_masks']
        camera_poses_normalized = camera_poses.clone()

        # normalize predict points
        all_pts = local_points.clone()
        all_pts[~masks] = 0
        all_pts = all_pts.reshape(B, N, -1, 3)
        all_dis = all_pts.norm(dim=-1)

        # pred scale
        norm_factor = all_dis.sum(dim=[-1, -2]) / (masks.float().sum(dim=[-1, -2, -3]) + 1e-8)
        local_points = local_points / norm_factor[..., None, None, None, None]
        if 'global_points' in pred and pred['global_points'] is not None:
            pred['global_points'] /= norm_factor[..., None, None, None, None]
        camera_poses_normalized[..., :3, 3] /= norm_factor.view(B, 1, 1)
        pred['robot_point'] = pred['robot_point'] / norm_factor[..., None, None, None, None]  # seperate pred
        pred['object_point'] = pred['object_point'] / norm_factor[..., None, None, None, None]
        pred['table_point'] = pred['table_point'] / norm_factor[..., None, None, None, None]
        # norm_factor = torch.ones(B, device=all_dis.device).to(all_dis.dtype)

        pred['local_points'] = local_points
        pred['camera_poses'] = camera_poses_normalized
        pred['norm_factor'] = norm_factor

        return pred

    def forward(self, pred, gt_raw):
        gt = self.prepare_gt(gt_raw)
        pred = self.normalize_pred(pred, gt)

        final_loss = 0.0
        details = dict()

        # Local Point Loss
        point_loss, point_loss_details, scale = self.point_loss(pred, gt)
        final_loss += point_loss
        details.update(point_loss_details)

        # Camera Loss
        camera_loss, camera_loss_details = self.camera_loss(pred, gt, scale)
        if camera_loss is not None:
            final_loss += camera_loss * 0.1 * 0.2 #* 0.2
        details.update(camera_loss_details)

        # pred scale
        gt_scale = scale.detach() * gt['norm_factor'].detach() / pred['norm_factor'].detach()
        scale_loss = F.huber_loss(pred['scale'].squeeze(-1).squeeze(-1).float(), gt_scale.float(), reduction='mean')
        final_loss += scale_loss * 0.1 * 5.0
        details.update({'scale_loss': scale_loss})

        # pred keypoint
        keypoint_map_loss = F.l1_loss(pred['keypoint_map'].float(), gt['keypoint_map'].permute(0, 1, 3, 4, 2).float(), reduction='mean')
        keypoint_loss = F.l1_loss(pred['keypoint'].float(), gt['keypoint'].float(), reduction='none')
        final_loss += keypoint_map_loss * 0.1
        in_frame_mask = gt['keypoint'].sum(dim=-1) > 0.0
        keypoint_loss = keypoint_loss.mean(dim=-1) * in_frame_mask
        keypoint_loss = keypoint_loss.mean()
        final_loss += keypoint_loss * 0.0003

        pred_keypoint_map = pred['keypoint_map'].permute(0, 1, 4, 2, 3)
        flat_idx = pred_keypoint_map.flatten(-2, -1).argmax(dim=-1)
        row_idx = flat_idx // pred_keypoint_map.shape[-1]
        col_idx = flat_idx % pred_keypoint_map.shape[-1]
        keypoint_via_max = torch.stack([col_idx, row_idx], dim=-1)
        keypoint_via_max_loss = F.l1_loss(keypoint_via_max.float(), gt['keypoint'].float(), reduction='none')
        keypoint_via_max_loss = keypoint_via_max_loss.mean(dim=-1) * in_frame_mask
        keypoint_via_max_loss = keypoint_via_max_loss.mean()

        details.update({'keypoint_map_loss': keypoint_map_loss, 'keypoint_loss': keypoint_loss, 'keypoint_via_max_loss': keypoint_via_max_loss})

        for k, v in details.items():
            details[k] = v.item()

        return final_loss, details
