import os
import torch
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R


def if_is_approx_line(points, threshold=5.0):
    """
    判断点是否近似在一条直线上。
    points: [[x1, y1], [x2, y2], ...]
    threshold: 允许的最大偏离距离（像素）
    """
    pts = np.array(points)
    if len(pts) < 3: return True # 少于3个点直接视为共线
    x, y = pts[:, 0], pts[:, 1]
    # 技巧：如果 x 的方差比 y 小，说明近似垂直线。
    # 此时交换 x 和 y 坐标进行拟合，避免斜率无穷大问题。
    if np.std(x) < np.std(y):
        x, y = y, x
    # 1. 拟合直线 y = kx + b (一次多项式)
    k, b = np.polyfit(x, y, 1)
    # 2. 计算所有点到直线的垂直距离
    # 公式: d = |kx - y + b| / sqrt(k^2 + 1)
    distances = np.abs(k * x - y + b) / np.sqrt(k**2 + 1)
    # 3. 判断最大距离是否在阈值内
    return np.max(distances) <= threshold
# def c2w_xforward_zup_to_opencv(c2w_xzup):
#     T = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
#     c2w_cv = np.eye(4)
#     c2w_cv[:3, :3] = c2w_xzup[:3, :3] @ T.T
#     c2w_cv[:3, 3] = c2w_xzup[:3, 3]
#     return c2w_cv
def c2w_xforward_zup_to_opencv(c2w_xzup):
    T = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    c2w_cv = np.eye(4)
    c2w_cv[:3, :3] = T @ c2w_xzup[:3, :3] @ T.T
    c2w_cv[:3, 3] = T @ c2w_xzup[:3, 3]
    return c2w_cv
def draw_2d_keypoint(points, image_size, image_path='2d_keypoints.png'):
    height, width = image_size
    img = np.zeros((height, width), dtype=np.uint8)
    r = 3
    coords = np.rint(points).astype(int)
    for x, y in coords:
        if x < 0 or y < 0 or x >= width or y >= height:
            continue
        x0 = max(0, x - r)
        x1 = min(width - 1, x + r)
        y0 = max(0, y - r)
        y1 = min(height - 1, y + r)
        img[y0:y1+1, x0:x1+1] = 255
    Image.fromarray(img).save(image_path)
from pyrr import Quaternion
def convert_rvec_to_quaternion(rvec):
    """Convert rvec (which is log quaternion) to quaternion"""
    theta = np.sqrt(
        rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2]
    )  # in radians
    raxis = [rvec[0] / theta, rvec[1] / theta, rvec[2] / theta]
    # pyrr's Quaternion (order is XYZW), https://pyrr.readthedocs.io/en/latest/oo_api_quaternion.html
    quaternion = Quaternion.from_axis_rotation(raxis, theta)
    quaternion.normalize()
    return quaternion
import cv2
def solve_pnp(
    canonical_points,
    projections,
    camera_K,
    method=cv2.SOLVEPNP_EPNP,
    refinement=True,
    dist_coeffs=np.array([]),
):
    n_canonial_points = len(canonical_points)
    n_projections = len(projections)
    assert (
        n_canonial_points == n_projections
    ), "Expected canonical_points and projections to have the same length, but they are length {} and {}.".format(
        n_canonial_points, n_projections
    )
    # Process points to remove any NaNs
    canonical_points_proc = []
    projections_proc = []
    for canon_pt, proj in zip(canonical_points, projections):
        if (
            canon_pt is None
            or len(canon_pt) == 0
            or canon_pt[0] is None
            or canon_pt[1] is None
            or proj is None
            or len(proj) == 0
            or proj[0] is None
            or proj[1] is None
        ):
            continue
        canonical_points_proc.append(canon_pt)
        projections_proc.append(proj)
    # Return if no valid points
    if len(canonical_points_proc) == 0:
        return False, None, None
    canonical_points_proc = np.array(canonical_points_proc)
    projections_proc = np.array(projections_proc)
    pnp_retval, rvec, tvec = cv2.solvePnP(
        canonical_points_proc.reshape(canonical_points_proc.shape[0], 1, 3),
        projections_proc.reshape(projections_proc.shape[0], 1, 2),
        camera_K,
        dist_coeffs,
        flags=cv2.SOLVEPNP_EPNP,
    )
    tvec_0 = tvec.copy()
    rvec_0 = rvec.copy()
    if refinement:
        pnp_retval, rvec, tvec = cv2.solvePnP(
            canonical_points_proc.reshape(canonical_points_proc.shape[0], 1, 3),
            projections_proc.reshape(projections_proc.shape[0], 1, 2),
            camera_K,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
            useExtrinsicGuess=True,
            rvec=rvec,
            tvec=tvec,
        )
        if np.linalg.norm(tvec - tvec_0) > 0.1:
            pnp_retval, rvec, tvec = cv2.solvePnP(
                canonical_points_proc.reshape(canonical_points_proc.shape[0], 1, 3),
                projections_proc.reshape(projections_proc.shape[0], 1, 2),
                camera_K,
                dist_coeffs,
                flags=cv2.SOLVEPNP_SQPNP,
            )
    translation = tvec[:, 0]
    quaternion = convert_rvec_to_quaternion(rvec[:, 0])
    return pnp_retval, translation, quaternion
from scipy.ndimage.filters import gaussian_filter
def peaks_from_belief_maps(belief_map_tensor, offset_due_to_upsampling):
    assert (
        len(belief_map_tensor.shape) == 3
    ), "Expected belief_map_tensor to have shape [N x height x width], but it is {}.".format(
        belief_map_tensor.shape
    )
    # thresh_map_after_gaussian_filter specifies the minimum intensity in the belief map AFTER the gaussian filter.
    # with sigma = 3, a perfect heat map will have a max intensity of about 0.3. -- both in a 100x100 frame and a 400x400 frame
    thresh_map_after_gaussian_filter = 0.01
    sigma = 3
    all_peaks = []
    peak_counter = 0
    for j in range(belief_map_tensor.size()[0]):
        belief_map = belief_map_tensor[j].clone()
        map_ori = belief_map.cpu().data.numpy()
        map = gaussian_filter(map_ori, sigma=sigma)
        p = 1
        map_left = np.zeros(map.shape)
        map_left[p:, :] = map[:-p, :]
        map_right = np.zeros(map.shape)
        map_right[:-p, :] = map[p:, :]
        map_up = np.zeros(map.shape)
        map_up[:, p:] = map[:, :-p]
        map_down = np.zeros(map.shape)
        map_down[:, :-p] = map[:, p:]
        peaks_binary = np.logical_and.reduce(
            (
                map >= map_left,
                map >= map_right,
                map >= map_up,
                map >= map_down,
                map > thresh_map_after_gaussian_filter,
            )
        )
        peaks = zip(
            np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]  # x values
        )  # y values
        # Computing the weigthed average for localizing the peaks
        peaks = list(peaks)
        win = 5
        ran = win // 2
        peaks_avg = []
        for p_value in range(len(peaks)):
            p = peaks[p_value]
            weights = np.zeros((win, win))
            i_values = np.zeros((win, win))
            j_values = np.zeros((win, win))
            for i in range(-ran, ran + 1):
                for j in range(-ran, ran + 1):
                    if (
                        p[1] + i < 0
                        or p[1] + i >= map_ori.shape[0]
                        or p[0] + j < 0
                        or p[0] + j >= map_ori.shape[1]
                    ):
                        continue

                    i_values[j + ran, i + ran] = p[1] + i
                    j_values[j + ran, i + ran] = p[0] + j

                    weights[j + ran, i + ran] = map_ori[p[1] + i, p[0] + j]
            # if the weights are all zeros
            # then add the none continuous points
            try:
                peaks_avg.append(
                    (
                        np.average(j_values, weights=weights)
                        + offset_due_to_upsampling,
                        np.average(i_values, weights=weights)
                        + offset_due_to_upsampling,
                    )
                )
            except:
                peaks_avg.append(
                    (p[0] + offset_due_to_upsampling, p[1] + offset_due_to_upsampling)
                )
        peaks_len = min(
            len(np.nonzero(peaks_binary)[1]), len(np.nonzero(peaks_binary)[0])
        )
        peaks_with_score = [
            peaks_avg[x_] + (map_ori[peaks[x_][1], peaks[x_][0]],)
            for x_ in range(len(peaks))
        ]
        id = range(peak_counter, peak_counter + peaks_len)
        peaks_with_score_and_id = [
            peaks_with_score[i] + (id[i],) for i in range(len(id))
        ]
        all_peaks.append(peaks_with_score_and_id)
        peak_counter += peaks_len
    return all_peaks
def extract_keypoint_from_belief_map(belief_maps_batch):
    if belief_maps_batch.shape[-1] >= 400 and belief_maps_batch.shape[-2] >= 400:
        offset_due_to_upsampling = 0.0
    else:
        # heuristic based on previous work with smaller belief maps
        offset_due_to_upsampling = 0.4395
    # from PIL import Image; Image.fromarray((belief_maps_batch[0].sum(dim=0).cpu().numpy().clip(0, 1) * 255.0).astype(np.uint8)).save('pred_map.png')
    detected_kp_projs_batch = []
    for belief_maps in belief_maps_batch:
        peaks = peaks_from_belief_maps(
            belief_maps, offset_due_to_upsampling
        )
        # Determine keypoints from this
        detected_kp_projs = []
        for peak in peaks:

            if len(peak) == 1:
                detected_kp_projs.append([peak[0][0], peak[0][1]])
            else:
                if len(peak) > 1:
                    # Try to use the belief map scores
                    peak_sorted_by_score = sorted(
                        peak, key=lambda x: x[2], reverse=True
                    )
                    if (
                        peak_sorted_by_score[0][2] - peak_sorted_by_score[1][2]
                        >= 0.25
                    ):
                        # Keep the best score
                        detected_kp_projs.append(
                            [
                                peak_sorted_by_score[0][0],
                                peak_sorted_by_score[0][1],
                            ]
                        )
                    else:
                        detected_kp_projs.append([-999.999, -999.999])
                else:
                    detected_kp_projs.append([-999.999, -999.999])
        detected_kp_projs_batch.append(detected_kp_projs)
    detected_kp_projs_batch = torch.tensor(detected_kp_projs_batch).float()
    return detected_kp_projs_batch
def create_belief_map(
    image_resolution,
    # image size (width x height)
    pointsBelief,
    # list of points to draw in a 7x2 tensor
    sigma=2
    # the size of the point
    # returns a tensor of n_points x h x w with the belief maps
):
    image_width, image_height = image_resolution
    image_transpose_resolution = (image_height, image_width)
    out = np.zeros((len(pointsBelief), image_height, image_width))
    w = int(sigma * 2)
    for i_point, point in enumerate(pointsBelief):
        pixel_u = int(point[0])
        pixel_v = int(point[1])
        array = np.zeros(image_transpose_resolution)
        # TODO makes this dynamics so that 0,0 would generate a belief map.
        if (
            pixel_u - w >= 0
            and pixel_u + w + 1 < image_width
            and pixel_v - w >= 0
            and pixel_v + w + 1 < image_height
        ):
            for i in range(pixel_u - w, pixel_u + w + 1):
                for j in range(pixel_v - w, pixel_v + w + 1):
                    array[j, i] = np.exp(
                        -(
                            ((i - pixel_u) ** 2 + (j - pixel_v) ** 2)
                            / (2 * (sigma ** 2))
                        )
                    )
        out[i_point] = array
    return out
def get_c2w_via_pnp(keypoint_3d, keypoint_2d, camera_intrinsics):
    pnp_success, translation, quaternion = solve_pnp(keypoint_3d.astype(np.float32), keypoint_2d.astype(np.float32), camera_intrinsics)
    if not pnp_success:
        return None
    rot_matrix = R.from_quat(np.array(quaternion), scalar_first=False).as_matrix()
    w2c_matrix = np.eye(4)
    w2c_matrix[:3, :3] = rot_matrix
    w2c_matrix[:3, 3] = translation
    c2w_matrix = np.linalg.inv(w2c_matrix)
    return c2w_matrix
