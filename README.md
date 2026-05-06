<br>
<p align="center">
  <h1 align="center"><strong>Robo3R: Enhancing Robotic Manipulation with Accurate Feed-Forward 3D Reconstruction</strong></h1>
  <p align="center">
    Sizhe Yang, Linning Xu, Hao Li, Juncheng Mu, Jia Zeng, Dahua Lin, Jiangmiao Pang
    <br>
    Shanghai AI Laboratory, The Chinese University of Hong Kong, University of Science and Technology of China, Tsinghua University
    <br>
  </p>

  <p align="center"><strong>RSS 2026</strong></p>
</p>

<div id="top" align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2602.10101-red)](https://arxiv.org/abs/2602.10101)
[![Homepage](https://img.shields.io/badge/Homepage-%F0%9F%8C%90-blue)](https://yangsizhe.github.io/robo3r/)

</div>


## 🔥 Highlight <a name="highlight"></a>

**Robo3R** enables manipulation-ready 3D reconstruction from RGB frames in real time.

By achieving accurate metric-scale 3D geometry in the canonical robot frame, Robo3R **eliminates the need for depth sensors and calibration, while improving accuracy and robustness** in challenging manipulation scenarios.

These features **lead to notable improvements in downstream applications** such as imitation learning, sim-to-real transfer, grasp synthesis, and collision-free motion planning.

<div align="center">
    <img src="./asset/teaser.jpg" alt="framework" width="90%" style="position: relative;">
</div>

## 📁 Dataset

Our curated large-scale dataset is available at [Robo3R-4M Dataset - Huggingface](https://huggingface.co/datasets/yangsizhe/Robo3R).

The dataset is generated with the **Franka FR3** robot and contains two subsets:

- **`100kScenes_dtc-objaverse_not-in-gripper`**: 100k scenes where objects are randomly placed on the tabletop.
- **`20kScenes_dtc-objaverse_in-gripper`**: 20k scenes where one object is grasped by the gripper, and the remaining objects are randomly placed on the tabletop.

The dataset is split into multiple `.tar.gz.part*` files for upload. After downloading, concatenate the parts and extract them with the following commands:

```bash
# 100kScenes_dtc-objaverse_not-in-gripper
cd 100kScenes_dtc-objaverse_not-in-gripper
cat 100kScenes_dtc-objaverse_not-in-gripper.tar.gz.part* > 100kScenes_dtc-objaverse_not-in-gripper.tar.gz
tar -xzvf 100kScenes_dtc-objaverse_not-in-gripper.tar.gz
cd ..

# 20kScenes_dtc-objaverse_in-gripper
cd 20kScenes_dtc-objaverse_in-gripper
cat 20kScenes_dtc-objaverse_in-gripper.tar.gz.part* > 20kScenes_dtc-objaverse_in-gripper.tar.gz
tar -xzvf 20kScenes_dtc-objaverse_in-gripper.tar.gz
cd ..
```

The structure of the dataset is detailed below:

```text
scene_{str(scene_idx).zfill(8)}
├── rgb
│   ├── {str(frame_idx).zfill(4)}_{str(camera_idx).zfill(2)}.jpg
│   └── ...
├── depth
│   ├── {str(frame_idx).zfill(4)}_{str(camera_idx).zfill(2)}.png
│   └── ...
├── mask
│   ├── {str(frame_idx).zfill(4)}_{str(camera_idx).zfill(2)}.png
│   └── ...
├── qpos
│   ├── {str(frame_idx).zfill(4)}.npy
│   └── ...
├── ee_pose
│   ├── {str(frame_idx).zfill(4)}.npy
│   └── ...
├── keypoint_3d
│   ├── {str(frame_idx).zfill(4)}.npy
│   └── ...
├── keypoint_2d
│   ├── {str(frame_idx).zfill(4)}_{str(camera_idx).zfill(2)}.npy
│   └── ...
└── cam_param.npy
```

**Notes:**

- **`rgb/`**: RGB images captured from each camera.
- **`depth/`**: Depth maps in metric units.
  - Background pixels have a depth value of `0`.
  - When saved as PNG, depth is scaled by `10.0 * 2**16`:
    ```python
    depth = (depth / 10.0 * 2**16).astype(np.uint16)
    from PIL import Image
    Image.fromarray(depth).save('depth.png')
    ```
- **`mask/`**: Segmentation masks. Values for *table*, *robot*, and *object* are `50`, `100`, and `150`, respectively.
- **`qpos/`**: Joint positions of the robot.
- **`ee_pose/`**: End-effector pose of the robot.
- **`keypoint_3d/`**: Coordinates of keypoints in the robot frame.
- **`keypoint_2d/`**: Projection of `keypoint_3d` onto the image plane.
- **`cam_param.npy`**: Camera intrinsics and extrinsics for all cameras.
  - Shape: `(2, num_cameras, 4, 4)`.
  - The first dimension indexes intrinsics (`[0]`) and extrinsics (`[1]`).
  - The original `(3, 3)` intrinsics matrix is padded with an extra row and column so it shares the same shape as the extrinsics, allowing both to be stored in a single array.
- **Camera axes**: `+Z` up, `+X` forward.


## 📢 The code will be released soon. Stay tuned!


## 🔗 Citation <a name="citation"></a>

If you find our work helpful, please cite:

```bibtex
@article{yang2026robo3r,
  title={Robo3R: Enhancing Robotic Manipulation with Accurate Feed-Forward 3D Reconstruction},
  author={Yang, Sizhe and Xu, Linning and Li, Hao and Mu, Juncheng and Zeng, Jia and Lin, Dahua and Pang, Jiangmiao},
  journal={arXiv preprint arXiv:2602.10101},
  year={2026}
}
```


## 📄 License <a name="license"></a>

This repository is released under the [Apache 2.0 license](./LICENSE).


## 👏 Acknowledgements <a name="acknowledgements"></a>

Our code is built upon [Pi3](https://github.com/yyfz/Pi3) and [VGGT](https://github.com/facebookresearch/vggt). We thank the authors for open-sourcing their code and for their significant contributions to the community.
