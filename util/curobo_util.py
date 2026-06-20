import os
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from curobo.types.robot import RobotConfig
from curobo.util_file import load_yaml, get_assets_path
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.types.base import TensorDeviceType
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.types.math import Pose


def _load_robot_cfg(config_path):
    """Load `robot_cfg` from a curobo yaml, resolving relative urdf/asset paths
    against the yaml's own directory so the config is portable across machines."""
    robot_cfg = load_yaml(config_path)['robot_cfg']
    base_dir = os.path.dirname(os.path.abspath(config_path))
    kinematics = robot_cfg['kinematics']
    for key in ('urdf_path', 'asset_root_path'):
        path = kinematics.get(key)
        if path and not os.path.isabs(path):
            kinematics[key] = os.path.join(base_dir, path)
    return robot_cfg


class KinematicsSolver:
    def __init__(self, config_path):
        self.ik_solver = set_up_curobo_util(config_path)

    def solve_ik(self, ee_pose, init_qpos=None):
        ee_pose = ee_pose.tolist()
        goal = Pose.from_list(ee_pose, self.ik_solver.tensor_args)
        if init_qpos is not None:
            init_qpos = torch.from_numpy(init_qpos).to(self.ik_solver.tensor_args.device)
        
        result = self.ik_solver.solve_single(
            goal,
            retract_config=init_qpos.unsqueeze(0) if init_qpos is not None else None,
            seed_config=init_qpos.unsqueeze(0).unsqueeze(0) if init_qpos is not None else None,
        )
        if result.success:
            return result.solution[0, 0].cpu().numpy()
        
    def solve_fk(self, qpos):
        qpos = torch.from_numpy(qpos).unsqueeze(0).to(self.ik_solver.tensor_args.device)
        result = self.ik_solver.fk(qpos)

        return np.concatenate([result.ee_position.squeeze(0).cpu().numpy(), result.ee_quaternion.squeeze(0).cpu().numpy()])


def set_up_curobo_util(config_path):
    ik_config = IKSolverConfig.load_from_robot_config(
        RobotConfig.from_dict(_load_robot_cfg(config_path)),
        None,
        tensor_args=TensorDeviceType('cuda:0'),
        position_threshold=0.00001,
        rotation_threshold=0.0001,
        use_cuda_graph=True,
    )
    ik_solver = IKSolver(ik_config)
    robot_transformation = np.eye(4)
    robot_pos_offset = robot_transformation[:3, 3]
    robot_rot_offset = R.from_matrix(robot_transformation[:3, :3]).as_euler('XYZ', degrees=False)
    base_offset_joint_dict = {'pos_x_joint': robot_pos_offset[0], 'pos_y_joint': robot_pos_offset[1], 'pos_z_joint': robot_pos_offset[2], 'rot_x_joint': robot_rot_offset[0], 'rot_y_joint': robot_rot_offset[1], 'rot_z_joint': robot_rot_offset[2]}
    robot_config_dict = _load_robot_cfg(config_path)
    robot_config_dict['kinematics']['lock_joints'] = base_offset_joint_dict
    robot_cfg = RobotConfig.from_dict(robot_config_dict, ik_solver.tensor_args)
    ik_solver.kinematics.update_kinematics_config(robot_cfg.kinematics.kinematics_config)

    return ik_solver


if __name__ == '__main__':
    # Minimal forward-kinematics demo on the bundled Franka FR3 model.
    asset_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'asset')
    kinematics_solver = KinematicsSolver(os.path.join(asset_dir, 'fr3', 'fr3.yaml'))
    arm_qpos = np.load(os.path.join(asset_dir, 'example', 'qpos.npy')).astype(np.float32)[:7]
    print('FK ee pose [x, y, z, qw, qx, qy, qz]:', kinematics_solver.solve_fk(arm_qpos))
