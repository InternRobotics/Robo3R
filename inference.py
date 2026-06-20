"""Robo3R inference: reconstruct manipulation-ready 3D point clouds (robot / object /
table) in the canonical robot frame from one or more RGB views, without depth sensors
or camera calibration.

Run the bundled demo (set the checkpoint to the released franka_wQpos.pth):
    python inference.py resume_from_checkpoint=/path/to/franka_wQpos.pth

To run on your own data, edit the IMAGE_PATHS / QPOS_PATH / OUTPUT_DIR constants below.
"""
import os
import sys

import numpy as np
import torch
import hydra
from omegaconf import OmegaConf
OmegaConf.register_new_resolver('eval', eval, replace=True)

from plyfile import PlyData, PlyElement

from util.io_util import load_images
from util.train_util import resume_from_checkpoint
from util.keypoint_pnp_util import get_c2w_via_pnp  # only used by the optional PnP path

# inputs
IMAGE_PATHS = ['asset/example/rgb/0000.png']
QPOS_PATH = 'asset/example/qpos.txt'  # robot joint positions, 9-dim [7 arm + 2 finger]
OUTPUT_DIR = './output/inference'
MAX_RES = 630
# Optional camera->robot pose refinement (keypoint PnP + forward kinematics).
# Enable with `model.pred_abs_pose_via_pnp=True`; needs curobo + the FR3 model in asset/fr3.
CUROBO_CONFIG_PATH = 'asset/fr3/fr3.yaml'
CAMERA_INTRINSICS_PATHS = ['asset/example/cam_intr_0.txt']  # per-view (3, 3)


def save_pc_as_ply(pc, path):
    """Save an (N, 3) or (N, 6) [xyz(+rgb)] point cloud to a .ply file."""
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
    PlyData([PlyElement.describe(vertex_data, 'vertex')]).write(path)


def build_model(cfg):
    from robo3r.models.robo3r import Robo3R
    model = Robo3R(fuse_robot_state=cfg.model.fuse_robot_state).cuda()
    resume_from_checkpoint(cfg.model.pi3_ckpt_path, model)  # Pi3 backbone
    assert cfg.resume_from_checkpoint is not None, \
        'Set resume_from_checkpoint to the released franka_wQpos.pth checkpoint.'
    resume_from_checkpoint(cfg.resume_from_checkpoint, model)  # Robo3R head
    return model.eval()


def load_inputs():
    """Load the RGB views and robot qpos -> imgs (1, N, 3, H, W), robot_qpos (1, N, 9)."""
    views = load_images(IMAGE_PATHS, max_res=MAX_RES, image_norm='identity')
    imgs = torch.stack([view['img'] for view in views], dim=1).cuda()
    qpos = torch.from_numpy(np.loadtxt(QPOS_PATH, delimiter=',').astype(np.float32)).cuda()
    robot_qpos = qpos[None, None].expand(1, imgs.shape[1], -1).contiguous()
    return imgs, robot_qpos


def reconstruct(model, imgs, robot_qpos, cfg):
    other = {'robot_qpos': robot_qpos} if cfg.model.fuse_robot_state else None
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            return model(imgs, other=other)


def refine_pose_via_pnp(pred):
    """Optional camera->robot pose, refined from predicted 2D keypoints and the robot's
    3D keypoints (forward kinematics of qpos) via PnP, instead of the model's predicted
    `abs_pose`. Requires curobo (`pip install curobo`) and the FR3 model under asset/fr3/."""
    from util.curobo_util import KinematicsSolver

    qpos = np.loadtxt(QPOS_PATH, delimiter=',').astype(np.float32)
    solver = KinematicsSolver(CUROBO_CONFIG_PATH)
    fk = solver.ik_solver.fk(torch.from_numpy(qpos[:7]).unsqueeze(0).cuda())
    keypoint_3d = fk.links_position[0][:-1].cpu().numpy()  # (num_keypoint, 3)

    intrinsics = np.stack([np.loadtxt(p) for p in CAMERA_INTRINSICS_PATHS], axis=0)  # (N, 3, 3)
    # rescale the intrinsics from the original 640x480 capture to the 634x476 model input
    intrinsics[..., 0, 0] = intrinsics[..., 0, 0] / 640 * 634
    intrinsics[..., 1, 1] = intrinsics[..., 1, 1] / 480 * 476
    intrinsics[..., 0, 2] = intrinsics[..., 0, 2] / 640 * 634 - 2
    intrinsics[..., 1, 2] = intrinsics[..., 1, 2] / 480 * 476
    B, N = pred['abs_pose'].shape[:2]
    pred_keypoint_2d = pred['keypoint'].flatten(0, 1).cpu().numpy()
    keypoint_map = pred['keypoint_map'].permute(0, 1, 4, 2, 3).flatten(0, 1).cpu().numpy()
    confidence = keypoint_map.reshape(*keypoint_map.shape[:2], -1).max(axis=-1)
    valid = confidence >= np.array([1 - 1e-8, 1 - 1e-8, 1 - 1e-8, 0.9, 0.9, 1 - 1e-8, 1 - 1e-8, 1 - 1e-8])

    T = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])  # camera (x-forward, z-up) -> opencv
    abs_pose = pred['abs_pose'].clone()
    for i in range(B * N):
        if valid[i].sum() < 6:  # too few keypoints for a reliable PnP -> keep abs_pose
            continue
        c2w = get_c2w_via_pnp(keypoint_3d[valid[i]], pred_keypoint_2d[i][valid[i]], intrinsics[i % N])
        if c2w is None:
            continue
        c2w[:3, :3] = T @ c2w[:3, :3]
        c2w[:3, 3] = T @ c2w[:3, 3]
        abs_pose[i // N, i % N] = torch.from_numpy(c2w).to(abs_pose)
    return abs_pose


def save_point_clouds(pred, imgs, abs_pose, output_dir):
    """Save, per view, the combined robot+object+table point cloud (with RGB) in both the
    camera frame and the canonical robot frame."""
    os.makedirs(output_dir, exist_ok=True)
    scale = pred['scale'][0, 0, 0].float().item()  # metric scale of the point map

    for view_idx in range(pred['local_points'].shape[1]):
        points = torch.zeros_like(pred['local_points'][0, view_idx])  # (H, W, 3)
        mask = torch.zeros(points.shape[:2], dtype=torch.bool, device=points.device)
        for part in ('robot', 'object', 'table'):
            part_mask = pred[f'{part}_mask'][0, view_idx] > 0.5
            points[part_mask] = pred[f'{part}_point'][0, view_idx][part_mask]
            mask |= part_mask

        points_cam = points[mask].float().cpu().numpy() * scale
        rgb = imgs[0, view_idx].permute(1, 2, 0).cpu().numpy()[mask.cpu().numpy()]
        save_pc_as_ply(np.concatenate([points_cam, rgb], axis=-1),
                       f'{output_dir}/pred_pc_cam_{view_idx}.ply')

        rot = abs_pose[0, view_idx, :3, :3].float().cpu().numpy()
        trans = abs_pose[0, view_idx, :3, 3].float().cpu().numpy()
        points_robot = points_cam @ rot.T + trans
        save_pc_as_ply(np.concatenate([points_robot, rgb], axis=-1),
                       f'{output_dir}/pred_pc_robot_{view_idx}.ply')


def main(cfg):
    model = build_model(cfg)
    imgs, robot_qpos = load_inputs()
    pred = reconstruct(model, imgs, robot_qpos, cfg)
    abs_pose = refine_pose_via_pnp(pred) if cfg.model.pred_abs_pose_via_pnp else pred['abs_pose']
    np.set_printoptions(precision=6, suppress=True)
    print(f"\n=== Predicted camera->robot pose (pred_abs_pose_via_pnp={cfg.model.pred_abs_pose_via_pnp}) ===")
    for view_idx in range(abs_pose.shape[1]):
        print(f"[view {view_idx}] c2w (4x4):")
        print(abs_pose[0, view_idx].float().cpu().numpy())
    save_point_clouds(pred, imgs, abs_pose, OUTPUT_DIR)
    print(f'Saved point clouds to {OUTPUT_DIR}')


if __name__ == '__main__':
    with hydra.initialize(config_path='config'):
        overrides = sys.argv[1:]
        cfg = hydra.compose(config_name='config', overrides=overrides)
    main(cfg)
