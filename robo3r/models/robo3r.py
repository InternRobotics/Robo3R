import torch
import torch.nn as nn
from functools import partial
from copy import deepcopy

from .dinov2.layers import Mlp
from ..utils.geometry import homogenize_points
from .layers.pos_embed import RoPE2D, PositionGetter
from .layers.block import BlockRope
from .layers.attention import FlashAttentionRope
from .layers.transformer_head import TransformerDecoder, LinearPts3d, ContextTransformerDecoder
from .layers.camera_head import CameraHead
from .dinov2.hub.backbones import dinov2_vitl14, dinov2_vitl14_reg
from robo3r.models.keypoint_head import SoftArgmaxPavlo

from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_file


def freeze_all_params(modules):
    for module in modules:
        try:
            for n, param in module.named_parameters():
                param.requires_grad = False
        except AttributeError:
            # module is directly a parameter
            module.requires_grad = False

from typing import *
import itertools
class ScaleHead(nn.Sequential):
    def __init__(self, dims: Sequence[int]):
        nn.Sequential.__init__(self,
            *itertools.chain(*[
                (nn.Linear(dim_in, dim_out), nn.ReLU(inplace=True))
                    for dim_in, dim_out in zip(dims[:-2], dims[1:-1])
            ]),
            nn.Linear(dims[-2], dims[-1]),
        )

class RobotQposEncoder(nn.Module):
    def __init__(self, input_dim=9):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(128, 256)
        self.act2 = nn.GELU()
        self.fc3 = nn.Linear(256, 512)
        self.act3 = nn.GELU()
        self.fc4 = nn.Linear(512, 1024)
        self.norm = nn.LayerNorm(1024, eps=1e-6)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        x = self.fc4(x)
        x = self.norm(x)
        return x

def init_weights_zero(m):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def init_weights_identity(m):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class ZeroNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.zeros_like(x)

class Robo3R(nn.Module):
    def __init__(
            self,
            pos_type='rope100',
            decoder_size='large',
            load_vggt=True,
            freeze_encoder=True,
            use_global_points=False,
            train_conf=False,
            num_dec_blk_not_to_checkpoint=4,
            ckpt=None,
            fuse_robot_state=True,
        ):
        super().__init__()

        # ----------------------
        #        Encoder
        # ----------------------
        self.encoder = dinov2_vitl14_reg(pretrained=False)
        self.patch_size = 14
        del self.encoder.mask_token

        # ----------------------
        #  Positonal Encoding
        # ----------------------
        self.pos_type = pos_type if pos_type is not None else 'none'
        self.rope=None
        if self.pos_type.startswith('rope'): # eg rope100 
            if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(self.pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError
        

        # ----------------------
        #        Decoder
        # ----------------------
        if decoder_size == 'small':
            dec_embed_dim = 384
            dec_num_heads = 6
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'base':
            dec_embed_dim = 768
            dec_num_heads = 12
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'large':
            dec_embed_dim = 1024
            dec_num_heads = 16
            mlp_ratio = 4
            dec_depth = 36
        else:
            raise NotImplementedError
        self.decoder = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                attn_class=FlashAttentionRope,
                rope=self.rope
            ) for _ in range(dec_depth)])
        self.dec_embed_dim = dec_embed_dim

        # ----------------------
        #     Register_token
        # ----------------------
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        # ----------------------
        #  Local Points Decoder
        # ----------------------
        self.point_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.rope,
        )
        self.point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)

        # ----------------------
        #  Camera Pose Decoder
        # ----------------------
        self.camera_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,                # 8
            out_dim=512,
            rope=self.rope,
            use_checkpoint=False
        )
        self.camera_head = CameraHead(dim=512)

        # ----------------------
        #  Global Points Decoder
        # ----------------------
        self.use_global_points = use_global_points
        if use_global_points:
            self.global_points_decoder = ContextTransformerDecoder(
                in_dim=2*self.dec_embed_dim, 
                dec_embed_dim=1024,
                dec_num_heads=16,
                out_dim=1024,
                rope=self.rope,
            )
            self.global_point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)

        # For ImageNet Normalize
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

        load_vggt = False
        if load_vggt:
            vggt_weight = load_file('ckpts/VGGT-1B/model.safetensors')
            vggt_enc_weight = {k.replace('aggregator.patch_embed.', ''):vggt_weight[k] for k in list(vggt_weight.keys()) if k.startswith('aggregator.patch_embed.')}
            print("Loading vggt encoder", self.encoder.load_state_dict(vggt_enc_weight, strict=False))

            vggt_dec_weight = {k.replace('aggregator.global_blocks.', ''):vggt_weight[k] for k in list(vggt_weight.keys()) if k.startswith('aggregator.global_blocks.')}
            vggt_dec_weight1 = {}
            for k in list(vggt_dec_weight.keys()):
                idx = k.split('.')[0]
                other = k[len(idx):]
                vggt_dec_weight1[f'{int(idx)*2 + 1}{other}'] = vggt_dec_weight[k]
            vggt_dec_weight = vggt_dec_weight1 

            vggt_dec_weight_frame = {k.replace('aggregator.frame_blocks.', ''):vggt_weight[k] for k in list(vggt_weight.keys()) if k.startswith('aggregator.frame_blocks.')}
            for k in list(vggt_dec_weight_frame.keys()):
                idx = k.split('.')[0]
                other = k[len(idx):]
                vggt_dec_weight[f'{int(idx)*2}{other}'] = vggt_dec_weight_frame[k]

            print("Loading vggt decoder", self.decoder.load_state_dict(vggt_dec_weight, strict=False))

        self.train_conf = train_conf
        if train_conf:
            assert ckpt is not None
            self.conf_decoder = deepcopy(self.point_decoder)
            self.conf_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1)

            freeze_all_params([self.encoder, self.decoder, self.point_decoder, self.point_head, self.camera_decoder,  self.camera_head, self.register_token])
            if use_global_points:
                freeze_all_params([self.global_points_decoder, self.global_point_head])

        if freeze_encoder:
            print('Freezing the encoder.')
            freeze_all_params([self.encoder])

        self.num_dec_blk_not_to_checkpoint = num_dec_blk_not_to_checkpoint

        if ckpt is not None:
            checkpoint = torch.load(ckpt, weights_only=False, map_location='cpu')

            res = self.load_state_dict(checkpoint, strict=False)
            print(f'[Robo3R] Load checkpoints from {ckpt}: {res}')

            del checkpoint
            torch.cuda.empty_cache()

        self.fuse_robot_state = fuse_robot_state
        if self.fuse_robot_state:
            self.state_dim = 9  # 9 for franka
            self.robot_qpos_encoder = RobotQposEncoder(self.state_dim)
            self.robot_qpos_encoder.apply(init_weights_zero)

        # pred scale
        self.scale_token = nn.Parameter(torch.zeros(1, 1, 1024), requires_grad=True)
        torch.nn.init.normal_(self.scale_token, std=1e-6)
        self.scale_head = ScaleHead(dims=[1024, 1024, 1024, 1])

        # abs pose
        # self.abs_pose_token = nn.Parameter(torch.zeros(1, 1, 1024), requires_grad=True)
        # torch.nn.init.normal_(self.abs_pose_token, std=1e-6)
        self.abs_pose_head = ScaleHead(dims=[1024, 1024, 1024, 3+9])

        # pred intrinsics
        self.camera_token = nn.Parameter(torch.zeros(1, 1, 1024), requires_grad=True)
        torch.nn.init.normal_(self.camera_token, std=1e-6)
        self.camera_intrinsics_head = ScaleHead(dims=[1024, 1024, 1024, 4])

        # # pred diff
        # self.diff_decoder = TransformerDecoder(
        #     in_dim=3*self.dec_embed_dim, 
        #     dec_embed_dim=1024,
        #     dec_num_heads=16,
        #     out_dim=1024,
        #     rope=self.rope,
        # )
        # self.diff_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3*(3+1))

        # seperate pred
        # self.robot_point_decoder = TransformerDecoder(
        #     in_dim=2*self.dec_embed_dim, 
        #     dec_embed_dim=1024,
        #     dec_num_heads=16,
        #     out_dim=1024,
        #     rope=self.rope,
        # )
        self.robot_point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)
        self.robot_mask_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1)
        # self.object_point_decoder = TransformerDecoder(
        #     in_dim=2*self.dec_embed_dim, 
        #     dec_embed_dim=1024,
        #     dec_num_heads=16,
        #     out_dim=1024,
        #     rope=self.rope,
        # )
        self.object_point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)
        self.object_mask_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1)
        # self.table_point_decoder = TransformerDecoder(
        #     in_dim=2*self.dec_embed_dim, 
        #     dec_embed_dim=1024,
        #     dec_num_heads=16,
        #     out_dim=1024,
        #     rope=self.rope,
        # )
        self.table_point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)
        self.table_mask_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1)

        # pred keypoint
        self.num_keypoint = 8  # 8 for franka
        self.keypoint_map_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.rope,
            depth=5,
        )  # vit hidden
        self.keypoint_map_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=self.num_keypoint)

        # self.keypoint_map_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=64)
        # self.keypoint_map_act = nn.ReLU(inplace=True)
        # self.keypoint_map_conv = nn.Conv2d(64, 8, kernel_size=1, stride=1, padding=0)
    
        self.robot_keypoint_spatial_softmax = SoftArgmaxPavlo(n_keypoints=self.num_keypoint)

    def decode(self, hidden, N, H, W, other=None):
        BN, hw, _ = hidden.shape
        B = BN // N

        final_output = []

        hidden = hidden.reshape(B*N, hw, -1)

        register_token = self.register_token.repeat(B, N, 1, 1).reshape(B*N, *self.register_token.shape[-2:])

        hidden = torch.cat([register_token, hidden], dim=1)
        
        hw = hidden.shape[1]

        if self.pos_type.startswith('rope'):
            pos = self.position_getter(B * N, H//self.patch_size, W//self.patch_size, hidden.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # pred scale
        scale_token = self.scale_token.repeat(B, 1, 1)

        # # abs pose
        # abs_pose_token = self.abs_pose_token.repeat(BN, 1, 1)

        # pred intrinsics
        camera_token = self.camera_token.repeat(BN, 1, 1)
        # camera_token = None

        for i in range(len(self.decoder)):
            blk = self.decoder[i]

            if i % 2 == 0:
                pos = pos.reshape(B*N, hw, -1)
                hidden = hidden.reshape(B*N, hw, -1)

                # pred intrinsics # abs pose
                hidden = torch.cat([camera_token, hidden], dim=1)
                pos = torch.cat([torch.zeros(pos.shape[0], 1, 2).to(hidden.device).to(pos.dtype), pos], dim=1)

            else:
                pos = pos.reshape(B, N*hw, -1)
                hidden = hidden.reshape(B, N*hw, -1)

                # pred scale
                hidden = torch.cat([scale_token, hidden], dim=1)
                pos = torch.cat([torch.zeros(pos.shape[0], 1, 2).to(hidden.device).to(pos.dtype), pos], dim=1)

            if i >= self.num_dec_blk_not_to_checkpoint and self.training:
                hidden = checkpoint(blk, hidden, xpos=pos, use_reentrant=False)
            else:
                hidden = blk(hidden, xpos=pos)

            # pred intrinsics # abs pose
            if i % 2 == 0:
                camera_token = hidden[:, 0:1]
                hidden = hidden[:, 1:]
                pos = pos[:, 1:]

            # pred scale
            if i % 2 != 0:
                scale_token = hidden[:, 0:1]
                hidden = hidden[:, 1:]
                pos = pos[:, 1:]

            if i+1 in [len(self.decoder)-1, len(self.decoder)]:
                final_output.append(hidden.reshape(B*N, hw, -1))

        # return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(B*N, hw, -1)

        # pred scale # abs pose # pred intrinsics
        return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(B*N, hw, -1), scale_token, camera_token
    
    def forward(self, imgs, other=None):
        imgs = (imgs - self.image_mean) / self.image_std

        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14
        
        # encode by dinov2
        imgs = imgs.reshape(B*N, _, H, W)
        hidden = self.encoder(imgs, is_training=True)

        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]

        # vit_hidden = hidden  # vit hidden

        if self.fuse_robot_state:
            robot_qpos_feature = self.robot_qpos_encoder(other['robot_qpos'])
            hidden += robot_qpos_feature.reshape(-1, 1, 1024)

        # hidden, pos = self.decode(hidden, N, H, W)

        # pred scale # pred intrinsics # abs pose
        hidden, pos, scale_token, camera_token = self.decode(hidden, N, H, W, other)

        point_hidden = self.point_decoder(hidden, xpos=pos)
        if self.train_conf:
            conf_hidden = self.conf_decoder(hidden, xpos=pos)
        camera_hidden = self.camera_decoder(hidden, xpos=pos)

        if self.use_global_points:
            context = hidden.reshape(B, N, patch_h*patch_w+self.patch_start_idx, -1)[:, 0:1].repeat(1, N, 1, 1).reshape(B*N, patch_h*patch_w+self.patch_start_idx, -1)
            global_point_hidden = self.global_points_decoder(hidden, context, xpos=pos, ypos=pos)

        # # pred diff
        # diff_hidden = self.diff_decoder(torch.cat([hidden, point_hidden], dim=-1), xpos=pos)

        # seperate pred
        # robot_point_hidden = self.robot_point_decoder(hidden, xpos=pos)
        # object_point_hidden = self.object_point_decoder(hidden, xpos=pos)
        # table_point_hidden = self.table_point_decoder(hidden, xpos=pos)
        robot_point_hidden = point_hidden
        object_point_hidden = point_hidden
        table_point_hidden = point_hidden

        # pred keypoint
        keypoint_map_hidden = self.keypoint_map_decoder(hidden, xpos=pos[:, 0*self.patch_start_idx:])  # vit hidden

        with torch.amp.autocast(device_type='cuda', enabled=False):
            # seperate pred remove original local point
            # local points
            point_hidden = point_hidden.float()
            ret = self.point_head([point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            xy, z = ret.split([2, 1], dim=-1)
            z = torch.exp(z)
            local_points = torch.cat([xy * z, z], dim=-1)

            # # pred diff
            # diff_hidden = diff_hidden.float()
            # diff_origin = self.diff_head([diff_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            # diff = torch.zeros_like(diff_origin)
            # for order_idx in range(3):
            #     xy, z = diff_origin[..., order_idx*(3+1):(order_idx+1)*(3+1)-1].split([2, 1], dim=-1)
            #     z = torch.exp(z)
            #     diff[..., order_idx*(3+1):(order_idx+1)*(3+1)-1] = torch.cat([xy * z, z], dim=-1)
            #     diff[..., order_idx*(3+1)+3:(order_idx+1)*(3+1)] = diff_origin[..., order_idx*(3+1)+3:(order_idx+1)*(3+1)].sigmoid()

            # seperate pred
            keypoint_map_hidden = keypoint_map_hidden.float()
            robot_point_hidden = robot_point_hidden.float()
            ret = self.robot_point_head([robot_point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            xy, z = ret.split([2, 1], dim=-1)
            z = torch.exp(z)
            robot_point = torch.cat([xy * z, z], dim=-1)
            # robot_mask = self.robot_mask_head(robot_point_hidden[:, self.patch_start_idx:].permute(0, 2, 1).reshape(B*N, -1, H//14, W//14), output_shape=(H, W)).permute(0, 2, 3, 1).reshape(B, N, H, W, -1).sigmoid()
            robot_mask = self.robot_mask_head([keypoint_map_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1).sigmoid()
            object_point_hidden = object_point_hidden.float()
            ret = self.object_point_head([object_point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            xy, z = ret.split([2, 1], dim=-1)
            z = torch.exp(z)
            object_point = torch.cat([xy * z, z], dim=-1)
            # object_mask = self.object_mask_head(object_point_hidden[:, self.patch_start_idx:].permute(0, 2, 1).reshape(B*N, -1, H//14, W//14), output_shape=(H, W)).permute(0, 2, 3, 1).reshape(B, N, H, W, -1).sigmoid()
            object_mask = self.object_mask_head([keypoint_map_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1).sigmoid()
            table_point_hidden = table_point_hidden.float()
            ret = self.table_point_head([table_point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            xy, z = ret.split([2, 1], dim=-1)
            z = torch.exp(z)
            table_point = torch.cat([xy * z, z], dim=-1)
            # table_mask = self.table_mask_head(table_point_hidden[:, self.patch_start_idx:].permute(0, 2, 1).reshape(B*N, -1, H//14, W//14), output_shape=(H, W)).permute(0, 2, 3, 1).reshape(B, N, H, W, -1).sigmoid()
            table_mask = self.table_mask_head([keypoint_map_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1).sigmoid()

            # # seperate pred remove original local point
            # local_points = None

            # pred keypoint
            keypoint_map_hidden = keypoint_map_hidden.float()
            keypoint_map = self.keypoint_map_head([keypoint_map_hidden[:, self.patch_start_idx*1:]], (H, W)).reshape(B, N, H, W, -1).sigmoid()  # vit hidden
            # keypoint_map = self.keypoint_map_conv(self.keypoint_map_act(keypoint_map.reshape(B*N, H, W, -1).permute(0, 3, 1, 2))).permute(0, 2, 3, 1).reshape(B, N, H, W, -1)#.sigmoid()
            # keypoint_map = self.keypoint_map_head(keypoint_map_hidden[:, self.patch_start_idx*1:].permute(0, 2, 1).reshape(B*N, -1, H//14, W//14), output_shape=(H, W)).permute(0, 2, 3, 1).reshape(B, N, H, W, -1).sigmoid()  # vit hidden

            keypoint = self.robot_keypoint_spatial_softmax(keypoint_map.flatten(0, 1).permute(0, 3, 1, 2)).reshape(B, N, -1, 2)

            # confidence
            if self.train_conf:
                conf_hidden = conf_hidden.float()
                conf = self.conf_head([conf_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            else:
                conf = None
                
            # camera
            camera_hidden = camera_hidden.float()
            camera_poses = self.camera_head(camera_hidden[:, self.patch_start_idx:], patch_h, patch_w)
            camera_poses = camera_poses.reshape(B, N, 4, 4)

            # pred scale
            scale = self.scale_head(scale_token)

            # # delta pose
            # delta_pose = self.delta_pose_head(abs_pose_token)
            # abs_cam_trans = delta_pose[..., :3].squeeze(1)
            # abs_cam_rot = delta_pose[..., 3:].squeeze(1)
            # with torch.amp.autocast(device_type='cuda', enabled=False):
            #     abs_pose = self.camera_head.convert_pose_to_4x4(B, abs_cam_rot, abs_cam_trans, abs_cam_rot.device)

            # pred intrinsics
            camera_fxfycxcy = self.camera_intrinsics_head(camera_token)
            camera_fxfycxcy = camera_fxfycxcy.reshape(B, N, -1)

            # abs pose
            abs_pose = self.abs_pose_head(camera_token)
            abs_cam_trans = abs_pose[..., :3].flatten(0, 1)
            abs_cam_rot = abs_pose[..., 3:].flatten(0, 1)
            abs_pose = self.camera_head.convert_pose_to_4x4(B*N, abs_cam_rot, abs_cam_trans, abs_cam_rot.device)
            abs_pose = abs_pose.reshape(B, N, 4, 4)

            # # Global points
            # if self.use_global_points:
            #     global_point_hidden = global_point_hidden.float()
            #     global_points = self.global_point_head([global_point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            # else:
            #     global_points = None

            # unproject local points using camera poses
            if local_points is not None:
                points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]
            else:
                points = None

        return dict(
            points=points,
            local_points=local_points,
            conf=conf,
            camera_poses=camera_poses,
            # global_points=global_points,
            scale=scale,  # pred scale
            camera_fxfycxcy=camera_fxfycxcy,  # pred intrinsics
            # diff=diff,  # pred diff
            abs_pose=abs_pose,  # abs pose

            # seperate pred
            robot_point=robot_point,
            robot_mask=robot_mask.squeeze(-1),
            object_point=object_point,
            object_mask=object_mask.squeeze(-1),
            table_point=table_point,
            table_mask=table_mask.squeeze(-1),

            # pred keypoint
            keypoint_map=keypoint_map,
            keypoint=keypoint,
        )
