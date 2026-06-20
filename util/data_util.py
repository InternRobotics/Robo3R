import os
from pathlib import Path
import re

import cv2
import numpy as np
from typing import List, Tuple, Union, Any
from dataclasses import dataclass
import torch
import torch.distributed as dist
import PIL
from PIL import Image
lanczos = PIL.Image.Resampling.LANCZOS
bicubic = PIL.Image.Resampling.BICUBIC
import torchvision.transforms as tvf
from scipy.spatial.transform import Rotation

import sys; sys.path.append(str(Path(__file__).parent.parent))
from util.io_util import _get_method, _load_scene_meta
from util.camera_util import CAMERA_KEYS, convert_camera_coeffs_to_pinhole_matrix, interpolate_extrinsics, interpolate_intrinsics


def load_frame(
    scene_root: Path | str,
    frame_key: int | str | float,
    modalities: str | list[str] | None = None,
    scene_meta: dict[str, Any] | None = None,
    load_intrinsics: bool = True,
    load_extrinsics: bool = True,
    fmt: str = "torch",
    interpolate: bool = False,
) -> dict[str, Any]:
    """
    Load a single frame from a scene with specified modalities.

    Args:
        scene_root (str or Path): The root directory of the scene where the data is located.
        frame_key (int or str or float): Either a string (frame name) or integer (frame index) or float (video timestamp).
        modalities (str or list[str], optional): The modality or list of modalities to load.
            If None, only basic frame information is loaded.
        scene_meta (dict, optional): Dictionary containing scene metadata. If None, it will be loaded
            from scene_meta.json in the scene_root.
        interpolate (bool, optional): Allow interpolating frames?

    Returns:
        dict: A dictionary containing the loaded frame data with the requested modalities.
    """
    scene_root = Path(scene_root)
    if scene_meta is None:
        scene_meta = _load_scene_meta(scene_root / "scene_meta.json")
    frame = get_frame(scene_meta, frame_key)
    # compact, standarized frame representation
    wai_frame = {}
    if load_extrinsics:
        extrinsics = get_extrinsics(
            scene_meta,
            frame_key,
            fmt=fmt,
        )
        if extrinsics is not None:
            wai_frame["extrinsics"] = extrinsics
    if load_intrinsics:
        camera_model = frame.get("camera_model", scene_meta.get("camera_model"))
        wai_frame["camera_model"] = camera_model
        if camera_model == "PINHOLE":
            wai_frame["intrinsics"] = get_intrinsics(scene_meta, frame_key, fmt=fmt)
        elif camera_model in ["OPENCV", "OPENCV_FISHEYE"]:
            # optional per-frame intrinsics
            for camera_key in CAMERA_KEYS:
                if camera_key in frame:
                    wai_frame[camera_key] = float(frame[camera_key])
                elif camera_key in scene_meta:
                    wai_frame[camera_key] = float(scene_meta[camera_key])
        else:
            error_message = (
                f"Camera model not supported: {camera_model} - "
                f"Please verify scene_meta.json of scene: {scene_meta['dataset_name']}/{scene_meta['scene_name']}"
            )
            print(error_message)
            raise NotImplementedError(error_message)
    wai_frame["w"] = frame.get("w", scene_meta["w"] if "w" in scene_meta else None)
    wai_frame["h"] = frame.get("h", scene_meta["h"] if "h" in scene_meta else None)
    wai_frame["frame_name"] = frame["frame_name"]
    wai_frame["frame_idx"] = get_frame_index(scene_meta, frame_key)
    wai_frame["_is_interpolated"] = frame["_is_interpolated"]

    if modalities is not None:
        if isinstance(modalities, str):
            modalities = [modalities]
        for modality in modalities:
            # Handle regex patterns in modality
            if any(char in modality for char in ".|*+?()[]{}^$\\"):
                # This is a regex pattern
                pattern = re.compile(modality)
                matching_modalities = [
                    m for m in scene_meta["frame_modalities"] if pattern.match(m)
                ]
                if not matching_modalities:
                    raise ValueError(
                        f"No modalities match the pattern: {modality} in scene: {scene_root}"
                    )
                # Use the first matching modality
                modality = matching_modalities[0]
            current_modalities = load_modality(
                scene_root, scene_meta["frame_modalities"], modality, frame, fmt=fmt
            )
            wai_frame.update(current_modalities)

    return wai_frame

def load_data(fname: str | Path, format_type: str | None = None, **kwargs) -> Any:
    """
    Loads data from a file using the appropriate method based on the file format.

    Args:
        fname (str or Path): The filename or path to load data from.
        format_type (str, optional): The format type of the data. If None, it will be inferred from the file extension if possible.
            Supported formats include: 'readable', 'scalar', 'image', 'binary', 'depth', 'normals',
            'numpy', 'ptz', 'mmap', 'scene_meta', 'labeled_image', 'mesh', 'labeled_mesh', 'caption', "latents".
        **kwargs: Additional keyword arguments to pass to the loading method.

    Returns:
        The loaded data in the format returned by the specific loading method.

    Raises:
        ValueError: If the format cannot be inferred from the file extension.
        NotImplementedError: If the specified format is not supported.
        FileExistsError: If the file does not exist.
    """
    load_method = _get_method(fname, format_type, load=True)
    return load_method(fname, **kwargs)

def crop(
    data: np.ndarray | torch.Tensor | Image.Image,
    bbox: tuple[int, int, int, int] | tuple[int, int],
) -> np.ndarray | torch.Tensor | Image.Image:
    """
    Crop data of different formats (numpy arrays, PyTorch tensors, PIL Images) to a target size.

    Args:
        data: Input data to resize (numpy.ndarray, torch.Tensor, or PIL.Image.Image)
        size: Target size as tuple (offset_height, offset_width, height, width) or tuple (height, width)

    Returns:
        Cropped data in the same format as the input
    """
    if len(bbox) == 4:
        offset_height, offset_width, target_height, target_width = bbox
    elif len(bbox) == 2:
        target_height, target_width = bbox
        offset_height, offset_width = 0, 0
    else:
        raise ValueError(f"Unsupported size length {len(bbox)}.")

    end_height = offset_height + target_height
    end_width = offset_width + target_width

    if any([sz < 0 for sz in bbox]):
        raise ValueError("Bounding box can't have negative values.")

    if isinstance(data, np.ndarray):
        if (
            max(offset_height, end_height) > data.shape[0]
            or max(offset_width, end_width) > data.shape[1]
        ):
            raise ValueError("Invalid bounding box.")
        cropped_data = data[offset_height:end_height, offset_width:end_width, ...]
        return cropped_data

    # Handle PIL images
    elif isinstance(data, Image.Image):
        if (
            max(offset_height, end_height) > data.size[1]
            or max(offset_width, end_width) > data.size[0]
        ):
            raise ValueError("Invalid bounding box.")
        return data.crop((offset_width, offset_height, end_width, end_height))

    # Handle PyTorch tensors
    elif isinstance(data, torch.Tensor):
        if data.is_nested:
            # special handling for nested tensors
            return torch.stack([crop(nested_tensor, bbox) for nested_tensor in data])
        if (
            max(offset_height, end_height) > data.shape[-2]
            or max(offset_width, end_width) > data.shape[-1]
        ):
            raise ValueError("Invalid bounding box.")
        cropped_data = data[..., offset_height:end_height, offset_width:end_width]
        return cropped_data
    else:
        raise TypeError(f"Unsupported data type '{type(data)}'.")

def load_modality_data(
    scene_root: Path | str,
    results: dict[str, Any],
    modality_dict: dict[str, Any],
    modality: str,
    frame: dict[str, Any] | None = None,
    fmt: str = "torch",
) -> dict[str, Any]:
    """
    Processes a modality by loading data from a specified path and updating the results dictionary.
    This function extracts the format and path from the given modality dictionary, loads the data
    from the specified path, and updates the results dictionary with the loaded data.

    Args:
        scene_root (str or Path): The root directory of the scene where the data is located.
        results (dict): A dictionary to store the loaded modality data and optional frame path.
        modality_dict (dict): A dictionary containing the modality information, including 'format'
            and the path to the data.
        modality (str): The key under which the loaded modality data will be stored in the results.
        frame (dict, optional): A dictionary containing frame information. If provided, that means we are loading
        frame modalities, otherwise it is scene modalities.

    Returns:
        dict: The updated results dictionary containing the loaded modality data.
    """
    modality_format = modality_dict["format"]

    # The modality is stored as a video
    if "video" in modality_format:
        assert isinstance(frame["frame_name"], float), "frame_name should be float"
        video_file = None
        if "chunks" in modality_dict:
            video_list = modality_dict["chunks"]
            # Get the correct chunk of the video
            for video_chunk in video_list:
                if video_chunk["start"] <= frame["frame_name"] <= video_chunk["end"]:
                    video_file = video_chunk
                    break
        else:
            # There is only one video (no chunks)
            video_file = modality_dict
            if "start" not in video_file:
                video_file["start"] = 0
            if "end" not in video_file:
                video_file["end"] = float("inf")
            if not (video_file["start"] <= frame["frame_name"] <= video_file["end"]):
                video_file = None

        # This timestamp is not available in any of the chunks
        if video_file is None:
            frame_name = frame["frame_name"]
            print(
                f"Modality {modality} ({modality_format}) is not available at time {frame_name}"
            )
            return results

        # Load the modality from the video
        loaded_modality = load_data(
            Path(scene_root, video_file["file"]),
            modality_format,
            frame_key=frame["frame_name"] - video_file["start"],
        )

        if "bbox" in video_file:
            loaded_modality = crop(loaded_modality, video_file["bbox"])

        if loaded_modality is not None:
            results[modality] = loaded_modality

        if frame:
            results[f"{modality}_fname"] = video_file["file"]
    else:
        modality_path = [v for k, v in modality_dict.items() if k != "format"][0]
        if frame:
            if modality_path in frame:
                fname = frame[modality_path]
            else:
                fname = None
        else:
            fname = modality_path
        if fname is not None:
            loaded_modality = load_data(
                Path(scene_root, fname),
                modality_format,
                frame_key=frame["frame_name"] if frame else None,
                fmt=fmt,
            )
            results[modality] = loaded_modality
            if frame:
                results[f"{modality}_fname"] = frame[modality_path]
    return results

def load_modality(
    scene_root: Path | str,
    modality_meta: dict[str, Any],
    modality: str,
    frame: dict[str, Any] | None = None,
    fmt: str = "torch",
) -> dict[str, Any]:
    """
    Loads modality data based on the provided metadata and updates the results dictionary.
    This function navigates through the modality metadata to find the specified modality,
    then loads the data for each modality found.

    Args:
        scene_root (str or Path): The root directory of the scene where the data is located.
        modality_meta (dict): A nested dictionary containing metadata for various modalities.
        modality (str): A string representing the path to the desired modality within the metadata,
            using '/' as a separator for nested keys.
        frame (dict, optional): A dictionary containing frame information. If provided, we are operating
        on frame modalities, otherwise it is scene modalities.

    Returns:
        dict: A dictionary containing the loaded modality data.
    """
    results = {}
    # support for nested modalities like "pred_depth/metric3dv2"
    modality_keys = modality.split("/")
    current_modality = modality_meta
    for key in modality_keys:
        try:
            current_modality = current_modality[key]
        except KeyError as err:
            error_message = (
                f"Modality '{err.args[0]}' not found in modalities metadata. "
                f"Please verify the scene_meta.json and the provided modalities in {scene_root}."
            )
            print(error_message)
            raise KeyError(error_message) from err
    if "format" in current_modality:
        results = load_modality_data(
            scene_root, results, current_modality, modality, frame, fmt=fmt
        )
    else:
        # nested modality, return last by default
        print("Nested modality, returning last by default")
        key = next(reversed(current_modality.keys()))
        results = load_modality_data(
            scene_root, results, current_modality[key], modality, frame, fmt=fmt
        )
    return results

def get_frame_index(
    scene_meta: dict[str, Any],
    frame_key: int | str | float,
    frame_index_threshold_sec: float = 1e-4,
    distance_threshold_sec: float = 2.0,
) -> int | float:
    """
    Returns the frame index from scene_meta based on name (str) or index (int) or sub-frame index (float).

    Args:
        scene_meta: Dictionary containing scene metadata
        frame_key: Either a string (frame name) or integer (frame index) or float (sub-frame index)
        frame_index_threshold_sec: A threshold for nearest neighbor clipping for indexes (in seconds).
                                   Default is 1e-4, which is 10000 fps.
        distance_th: A threshold for maximum distance between interpolated frames (in seconds).

    Returns:
        Frame index (int)

    Raises:
        ValueError: If frame_key is not a string or integer or float
    """
    if isinstance(frame_key, str):
        try:
            return scene_meta["frame_names"][frame_key]
        except KeyError as err:
            error_message = (
                f"Frame name not found: {frame_key} - "
                f"Please verify scene_meta.json of scene: {scene_meta['dataset_name']}/{scene_meta['scene_name']}"
            )
            print(error_message)
            raise KeyError(error_message) from err

    if isinstance(frame_key, int):
        return frame_key

    if isinstance(frame_key, float):
        # If exact hit
        if frame_key in scene_meta["frame_names"]:
            return scene_meta["frame_names"][frame_key]

        frame_names = sorted(list(scene_meta["frame_names"].keys()))
        distances = np.array([frm - frame_key for frm in frame_names])
        left = int(np.nonzero(distances <= 0)[0][-1])
        right = left + 1

        # The last frame or rounding errors
        if (
            left == distances.shape[0] - 1
            or abs(distances[left]) < frame_index_threshold_sec
        ):
            return scene_meta["frame_names"][frame_names[int(left)]]
        if abs(distances[right]) < frame_index_threshold_sec:
            return scene_meta["frame_names"][frame_names[int(right)]]

        interpolation_distance = distances[right] - distances[left]
        if interpolation_distance > distance_threshold_sec:
            raise ValueError(
                f"Frame interpolation is forbidden for distances larger than {distance_threshold_sec}."
            )
        alpha = -distances[left] / interpolation_distance

        return scene_meta["frame_names"][frame_names[int(left)]] + alpha

    raise ValueError(f"Frame key type not supported: {frame_key} ({type(frame_key)}).")

def get_frame(
    scene_meta: dict[str, Any],
    frame_key: int | str | float,
) -> dict[str, Any]:
    """
    Get a frame from scene_meta based on name or index.

    Args:
        scene_meta: Dictionary containing scene metadata
        frame_key: Either a string (frame name) or integer (frame index) or float (video timestamp)

    Returns:
        The frame data (dict)
    """
    frame_idx = get_frame_index(scene_meta, frame_key)
    if isinstance(frame_idx, int):
        frame = scene_meta["frames"][frame_idx]
        frame["_is_interpolated"] = False
    else:
        frame = {}
        frame["frame_name"] = frame_key
        left = int(frame_idx)  # it's floor operation
        assert left >= 0 and left < (len(scene_meta["frames"]) - 1), "Wrong index"
        frame_left = scene_meta["frames"][left]
        frame_right = scene_meta["frames"][left + 1]
        # Interpolate intrinsics and extrinsics
        frame["transform_matrix"] = interpolate_extrinsics(
            frame_left["transform_matrix"],
            frame_right["transform_matrix"],
            frame_idx - left,
        )
        frame.update(
            interpolate_intrinsics(
                frame_left,
                frame_right,
                frame_idx - left,
            )
        )
        frame["_is_interpolated"] = True
    return frame


def get_intrinsics(
    scene_meta,
    frame_key,
    fmt: str = "torch",
) -> torch.Tensor | np.ndarray | list:
    frame = get_frame(scene_meta, frame_key)
    return convert_camera_coeffs_to_pinhole_matrix(scene_meta, frame, fmt=fmt)


def get_extrinsics(
    scene_meta,
    frame_key,
    fmt: str = "torch",
) -> torch.Tensor | np.ndarray | list | None:
    frame = get_frame(scene_meta, frame_key)
    if "transform_matrix" in frame:
        if fmt == "torch":
            return torch.tensor(frame["transform_matrix"]).reshape(4, 4).float()
        elif fmt == "np":
            return np.array(frame["transform_matrix"]).reshape(4, 4)
        return frame["transform_matrix"]
    else:
        # TODO: should not happen if we enable interpolation
        return None

def load_frame(
    scene_root: Path | str,
    frame_key: int | str | float,
    modalities: str | list[str] | None = None,
    scene_meta: dict[str, Any] | None = None,
    load_intrinsics: bool = True,
    load_extrinsics: bool = True,
    fmt: str = "torch",
    interpolate: bool = False,
) -> dict[str, Any]:
    """
    Load a single frame from a scene with specified modalities.

    Args:
        scene_root (str or Path): The root directory of the scene where the data is located.
        frame_key (int or str or float): Either a string (frame name) or integer (frame index) or float (video timestamp).
        modalities (str or list[str], optional): The modality or list of modalities to load.
            If None, only basic frame information is loaded.
        scene_meta (dict, optional): Dictionary containing scene metadata. If None, it will be loaded
            from scene_meta.json in the scene_root.
        interpolate (bool, optional): Allow interpolating frames?

    Returns:
        dict: A dictionary containing the loaded frame data with the requested modalities.
    """
    scene_root = Path(scene_root)
    if scene_meta is None:
        scene_meta = _load_scene_meta(scene_root / "scene_meta.json")
    frame = get_frame(scene_meta, frame_key)
    # compact, standarized frame representation
    wai_frame = {}
    if load_extrinsics:
        extrinsics = get_extrinsics(
            scene_meta,
            frame_key,
            fmt=fmt,
        )
        if extrinsics is not None:
            wai_frame["extrinsics"] = extrinsics
    if load_intrinsics:
        camera_model = frame.get("camera_model", scene_meta.get("camera_model"))
        wai_frame["camera_model"] = camera_model
        if camera_model == "PINHOLE":
            wai_frame["intrinsics"] = get_intrinsics(scene_meta, frame_key, fmt=fmt)
        elif camera_model in ["OPENCV", "OPENCV_FISHEYE"]:
            # optional per-frame intrinsics
            for camera_key in CAMERA_KEYS:
                if camera_key in frame:
                    wai_frame[camera_key] = float(frame[camera_key])
                elif camera_key in scene_meta:
                    wai_frame[camera_key] = float(scene_meta[camera_key])
        else:
            error_message = (
                f"Camera model not supported: {camera_model} - "
                f"Please verify scene_meta.json of scene: {scene_meta['dataset_name']}/{scene_meta['scene_name']}"
            )
            print(error_message)
            raise NotImplementedError(error_message)
    wai_frame["w"] = frame.get("w", scene_meta["w"] if "w" in scene_meta else None)
    wai_frame["h"] = frame.get("h", scene_meta["h"] if "h" in scene_meta else None)
    wai_frame["frame_name"] = frame["frame_name"]
    wai_frame["frame_idx"] = get_frame_index(scene_meta, frame_key)
    wai_frame["_is_interpolated"] = frame["_is_interpolated"]

    if modalities is not None:
        if isinstance(modalities, str):
            modalities = [modalities]
        for modality in modalities:
            # Handle regex patterns in modality
            if any(char in modality for char in ".|*+?()[]{}^$\\"):
                # This is a regex pattern
                pattern = re.compile(modality)
                matching_modalities = [
                    m for m in scene_meta["frame_modalities"] if pattern.match(m)
                ]
                if not matching_modalities:
                    raise ValueError(
                        f"No modalities match the pattern: {modality} in scene: {scene_root}"
                    )
                # Use the first matching modality
                modality = matching_modalities[0]
            current_modalities = load_modality(
                scene_root, scene_meta["frame_modalities"], modality, frame, fmt=fmt
            )
            wai_frame.update(current_modalities)

    return wai_frame

def bbox_from_intrinsics_in_out(
    input_camera_matrix, output_camera_matrix, output_resolution
):
    """
    Calculate the bounding box for cropping based on input and output camera intrinsics.

    Args:
        input_camera_matrix (numpy.ndarray): Original camera intrinsics matrix
        output_camera_matrix (numpy.ndarray): Target camera intrinsics matrix
        output_resolution (tuple): Target resolution as (width, height)

    Returns:
        tuple: Crop bounding box as (left, top, right, bottom)
    """
    out_width, out_height = output_resolution
    left, top = np.int32(
        np.round(input_camera_matrix[:2, 2] - output_camera_matrix[:2, 2])
    )
    crop_bbox = (left, top, left + out_width, top + out_height)
    return crop_bbox

def colmap_to_opencv_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5

    return K

def opencv_to_colmap_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] += 0.5
    K[1, 2] += 0.5

    return K

def camera_matrix_of_crop(
    input_camera_matrix,
    input_resolution,
    output_resolution,
    scaling=1,
    offset_factor=0.5,
    offset=None,
):
    """
    Calculate the camera matrix for a cropped image.

    Args:
        input_camera_matrix (numpy.ndarray): Original camera intrinsics matrix
        input_resolution (tuple or numpy.ndarray): Original image resolution as (width, height)
        output_resolution (tuple or numpy.ndarray): Target image resolution as (width, height)
        scaling (float, optional): Scaling factor for the image. Defaults to 1.
        offset_factor (float, optional): Factor to determine crop offset. Defaults to 0.5 (centered).
        offset (tuple or numpy.ndarray, optional): Explicit offset to use. If None, calculated from offset_factor.

    Returns:
        numpy.ndarray: Updated camera matrix for the cropped image
    """
    # Margins to offset the origin
    margins = np.asarray(input_resolution) * scaling - output_resolution
    assert np.all(margins >= 0.0)
    if offset is None:
        offset = offset_factor * margins

    # Generate new camera parameters
    output_camera_matrix_colmap = opencv_to_colmap_intrinsics(input_camera_matrix)
    output_camera_matrix_colmap[:2, :] *= scaling
    output_camera_matrix_colmap[:2, 2] -= offset
    output_camera_matrix = colmap_to_opencv_intrinsics(output_camera_matrix_colmap)

    return output_camera_matrix

class ImageList:
    """
    Convenience class to apply the same operation to a whole set of images.

    This class wraps a list of PIL.Image objects and provides methods to perform
    operations on all images simultaneously.
    """

    def __init__(self, images):
        if not isinstance(images, (tuple, list, set)):
            images = [images]
        self.images = []
        for image in images:
            if not isinstance(image, PIL.Image.Image):
                image = PIL.Image.fromarray(image)
            self.images.append(image)

    def __len__(self):
        """Return the number of images in the list."""
        return len(self.images)

    def to_pil(self):
        """
        Convert ImageList back to PIL Image(s).

        Returns:
            PIL.Image.Image or tuple: Single PIL Image if list contains one image,
                                      or tuple of PIL Images if multiple images
        """
        return tuple(self.images) if len(self.images) > 1 else self.images[0]

    @property
    def size(self):
        """
        Get the size of images in the list.

        Returns:
            tuple: (width, height) of the images

        Raises:
            AssertionError: If images have different sizes
        """
        sizes = [im.size for im in self.images]
        assert all(sizes[0] == s for s in sizes), "All images must have the same size"
        return sizes[0]

    def resize(self, *args, **kwargs):
        """
        Resize all images with the same parameters.

        Args:
            *args, **kwargs: Arguments passed to PIL.Image.resize()

        Returns:
            ImageList: New ImageList containing resized images
        """
        return ImageList(self._dispatch("resize", *args, **kwargs))

    def crop(self, *args, **kwargs):
        """
        Crop all images with the same parameters.

        Args:
            *args, **kwargs: Arguments passed to PIL.Image.crop()

        Returns:
            ImageList: New ImageList containing cropped images
        """
        return ImageList(self._dispatch("crop", *args, **kwargs))

    def _dispatch(self, func, *args, **kwargs):
        """
        Apply a PIL.Image method to all images in the list.

        Args:
            func (str): Name of the PIL.Image method to call
            *args, **kwargs: Arguments to pass to the method

        Returns:
            list: List of results from applying the method to each image
        """
        return [getattr(im, func)(*args, **kwargs) for im in self.images]

def crop_image_and_other_optional_info(
    image,
    crop_bbox,
    depthmap=None,
    camera_intrinsics=None,
    additional_quantities=None,
):
    """
    Return a crop of the input view and associated data.

    Args:
        image (PIL.Image.Image or numpy.ndarray): The input image to be cropped
        crop_bbox (tuple): Crop bounding box as (left, top, right, bottom)
        depthmap (numpy.ndarray, optional): Depth map associated with the image
        camera_intrinsics (numpy.ndarray, optional): Camera intrinsics matrix
        additional_quantities (list of numpy.ndarray, optional): Additional data arrays to crop

    Returns:
        tuple: A tuple containing:
            - The cropped image
            - The cropped depth map (if provided or None)
            - Updated camera intrinsics (if provided or None)
            - List of cropped additional quantities (if provided or None)
    """
    image = ImageList(image)
    left, top, right, bottom = crop_bbox

    image = image.crop((left, top, right, bottom))
    if depthmap is not None:
        depthmap = depthmap[top:bottom, left:right]
    if additional_quantities is not None:
        additional_quantities = [
            quantity[top:bottom, left:right] for quantity in additional_quantities
        ]

    if camera_intrinsics is not None:
        camera_intrinsics = camera_intrinsics.copy()
        camera_intrinsics[0, 2] -= left
        camera_intrinsics[1, 2] -= top

    return (image.to_pil(), depthmap, camera_intrinsics, additional_quantities)

def rescale_image_and_other_optional_info(
    image,
    output_resolution,
    depthmap=None,
    camera_intrinsics=None,
    force=True,
    additional_quantities_to_be_resized_with_nearest=None,
):
    """
    Rescale the image and depthmap to the output resolution.
    If the image is larger than the output resolution, it is rescaled with lanczos interpolation.
    If force is false and the image is smaller than the output resolution, it is not rescaled.
    If force is true and the image is smaller than the output resolution, it is rescaled with bicubic interpolation.
    Depth and other quantities are rescaled with nearest interpolation.

    Args:
        image (PIL.Image.Image or np.ndarray): The input image to be rescaled.
        output_resolution (tuple): The desired output resolution as a tuple (width, height).
        depthmap (np.ndarray, optional): The depth map associated with the image. Defaults to None.
        camera_intrinsics (np.ndarray, optional): The camera intrinsics matrix. Defaults to None.
        force (bool, optional): If True, force rescaling even if the image is smaller than the output resolution. Defaults to True.
        additional_quantities_to_be_resized_with_nearest (list of np.ndarray, optional): Additional quantities to be rescaled using nearest interpolation. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - The rescaled image (PIL.Image.Image)
            - The rescaled depthmap (numpy.ndarray or None)
            - The updated camera intrinsics (numpy.ndarray or None)
            - The list of rescaled additional quantities (list of numpy.ndarray or None)
    """
    image = ImageList(image)
    input_resolution = np.array(image.size)  # (W, H)
    output_resolution = np.array(output_resolution)
    if depthmap is not None:
        assert tuple(depthmap.shape[:2]) == image.size[::-1]
    if additional_quantities_to_be_resized_with_nearest is not None:
        assert all(
            tuple(additional_quantity.shape[:2]) == image.size[::-1]
            for additional_quantity in additional_quantities_to_be_resized_with_nearest
        )

    # Define output resolution
    assert output_resolution.shape == (2,)
    scale_final = max(output_resolution / image.size) + 1e-8
    if scale_final >= 1 and not force:  # image is already smaller than what is asked
        output = (
            image.to_pil(),
            depthmap,
            camera_intrinsics,
            additional_quantities_to_be_resized_with_nearest,
        )
        return output
    output_resolution = np.floor(input_resolution * scale_final).astype(int)

    # First rescale the image so that it contains the crop
    image = image.resize(
        tuple(output_resolution), resample=lanczos if scale_final < 1 else bicubic
    )
    if depthmap is not None:
        depthmap = cv2.resize(
            depthmap,
            output_resolution,
            fx=scale_final,
            fy=scale_final,
            interpolation=cv2.INTER_NEAREST,
        )
    if additional_quantities_to_be_resized_with_nearest is not None:
        resized_additional_quantities = []
        for quantity in additional_quantities_to_be_resized_with_nearest:
            resized_additional_quantities.append(
                cv2.resize(
                    quantity,
                    output_resolution,
                    fx=scale_final,
                    fy=scale_final,
                    interpolation=cv2.INTER_NEAREST,
                )
            )
        additional_quantities_to_be_resized_with_nearest = resized_additional_quantities

    # No offset here; simple rescaling
    if camera_intrinsics is not None:
        camera_intrinsics = camera_matrix_of_crop(
            camera_intrinsics, input_resolution, output_resolution, scaling=scale_final
        )

    # Return
    return (
        image.to_pil(),
        depthmap,
        camera_intrinsics,
        additional_quantities_to_be_resized_with_nearest,
    )

def depthmap_to_camera_coordinates(depthmap, camera_intrinsics, pseudo_focal=None):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels.
    """
    camera_intrinsics = np.float32(camera_intrinsics)
    H, W = depthmap.shape

    # Compute 3D ray associated with each pixel
    # Strong assumption: there are no skew terms
    assert camera_intrinsics[0, 1] == 0.0
    assert camera_intrinsics[1, 0] == 0.0
    if pseudo_focal is None:
        fu = camera_intrinsics[0, 0]
        fv = camera_intrinsics[1, 1]
    else:
        assert pseudo_focal.shape == (H, W)
        fu = fv = pseudo_focal
    cu = camera_intrinsics[0, 2]
    cv = camera_intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z_cam = depthmap
    x_cam = (u - cu) * z_cam / fu
    y_cam = (v - cv) * z_cam / fv
    X_cam = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    # Mask for valid coordinates
    valid_mask = depthmap > 0.0

    return X_cam, valid_mask

def get_absolute_pointmaps_and_rays_info(
    depthmap, camera_intrinsics, camera_pose, **kw
):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
        - camera_pose: a 4x3 or 4x4 cam2world matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array),
        a mask specifying valid pixels,
        ray origins of absolute coordinates (HxWx3 array),
        ray directions of absolute coordinates (HxWx3 array),
        depth along ray (HxWx1 array),
        ray directions of camera/local coordinates (HxWx3 array),
        pointmap of camera/local coordinates (HxWx3 array).
    """
    camera_intrinsics = np.float32(camera_intrinsics)
    H, W = depthmap.shape

    # Compute 3D ray associated with each pixel
    # Strong assumption: pinhole & there are no skew terms
    assert camera_intrinsics[0, 1] == 0.0
    assert camera_intrinsics[1, 0] == 0.0
    fu = camera_intrinsics[0, 0]
    fv = camera_intrinsics[1, 1]
    cu = camera_intrinsics[0, 2]
    cv = camera_intrinsics[1, 2]

    # Get the rays on the unit plane
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    x_cam = (u - cu) / fu
    y_cam = (v - cv) / fv
    z_cam = np.ones_like(x_cam)
    ray_dirs_cam_on_unit_plane = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(
        np.float32
    )

    # Compute the 3d points in the local camera coordinate system
    pts_cam = depthmap[..., None] * ray_dirs_cam_on_unit_plane

    # Get the depth along the ray and compute the ray directions on the unit sphere
    depth_along_ray = np.linalg.norm(pts_cam, axis=-1, keepdims=True)
    ray_directions_cam = ray_dirs_cam_on_unit_plane / np.linalg.norm(
        ray_dirs_cam_on_unit_plane, axis=-1, keepdims=True
    )

    # Mask for valid coordinates
    valid_mask = depthmap > 0.0

    # Get the ray origins in absolute coordinates and the ray directions in absolute coordinates
    ray_origins_world = np.zeros_like(ray_directions_cam)
    ray_directions_world = ray_directions_cam
    pts_world = pts_cam
    if camera_pose is not None:
        R_cam2world = camera_pose[:3, :3]
        t_cam2world = camera_pose[:3, 3]

        # Express in absolute coordinates
        ray_origins_world = ray_origins_world + t_cam2world[None, None, :]
        ray_directions_world = np.einsum(
            "ik, vuk -> vui", R_cam2world, ray_directions_cam
        )
        pts_world = ray_origins_world + ray_directions_world * depth_along_ray

    return (
        pts_world,
        valid_mask,
        ray_origins_world,
        ray_directions_world,
        depth_along_ray,
        ray_directions_cam,
        pts_cam,
    )

@dataclass
class ImageNormalization:
    mean: torch.Tensor
    std: torch.Tensor
IMAGE_NORMALIZATION_DICT = {
    "dinov2": ImageNormalization(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])),
    "identity": ImageNormalization(mean=torch.tensor([0.0, 0.0, 0.0]), std=torch.tensor([1.0, 1.0, 1.0])),
}

def round_by(total, multiple, up=False):
    """
    Round a number to the nearest multiple of another number.
    """
    if up:
        total = total + multiple - 1
    return (total // multiple) * multiple

class BatchedMultiFeatureRandomSampler:
    def __init__(
        self, dataset, batch_size, pool_sizes, world_size=1, rank=0, drop_last=True
    ):
        self.batch_size = batch_size
        self.pool_sizes = pool_sizes if isinstance(pool_sizes, list) else [pool_sizes]

        self.total_size = round_by(len(dataset), batch_size * world_size) if drop_last else len(dataset)
        assert world_size == 1 or drop_last, ("must drop the last batch in distributed mode")

        self.world_size = world_size
        self.rank = rank
        self.epoch = None

    def __len__(self):
        return self.total_size // self.world_size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(seed=self.epoch + 777)

        sample_idxs = np.arange(self.total_size)
        rng.shuffle(sample_idxs)

        # Random feat_idxs (same across each batch)
        n_batches = (self.total_size + self.batch_size - 1) // self.batch_size

        # Generate feature indices for each feature pool
        all_feat_idxs = []
        for pool_size in self.pool_sizes:
            feat_idxs = rng.integers(pool_size, size=n_batches)
            feat_idxs = np.broadcast_to(
                feat_idxs[:, None], (n_batches, self.batch_size)
            )
            feat_idxs = feat_idxs.ravel()[: self.total_size]
            all_feat_idxs.append(feat_idxs)

        # Put them together
        idxs = np.column_stack(
            [sample_idxs] + all_feat_idxs
        )  # shape = (total_size, 1 + len(pool_sizes))

        # Distributed sampler: we select a subset of batches
        # Make sure the slice for each node is aligned with batch_size
        size_per_proc = self.batch_size * (
            (self.total_size + self.world_size * self.batch_size - 1)
            // (self.world_size * self.batch_size)
        )
        idxs = idxs[self.rank * size_per_proc : (self.rank + 1) * size_per_proc]

        yield from (tuple(idx) for idx in idxs)

class DynamicBatchedMultiFeatureRandomSampler:
    def __init__(
        self,
        dataset,
        pool_sizes,
        scaling_feature_idx=0,
        feature_to_batch_size_map=None,
        world_size=1,
        rank=0,
        drop_last=True,
    ):
        self.pool_sizes = pool_sizes if isinstance(pool_sizes, list) else [pool_sizes]
        self.scaling_feature_idx = scaling_feature_idx
        self.feature_to_batch_size_map = feature_to_batch_size_map

        self.total_size = len(dataset)

        self.world_size = world_size
        self.rank = rank
        self.epoch = None
        self.drop_last = drop_last

    def __len__(self):
        max_batch_size = max(self.feature_to_batch_size_map.values())

        # Estimate total batches using the largest batch size. This gives a lower bound on the number of batches.
        total_batches = self.total_size // max_batch_size
        if not self.drop_last and self.total_size % max_batch_size > 0:
            total_batches += 1

        return total_batches // self.world_size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        """
        Yields:
            [(sample_idx, feat_idx_1, feat_idx_2, ...), ...]
        """
        rng = np.random.default_rng(seed=self.epoch + 777)

        sample_idxs = np.arange(self.total_size)
        rng.shuffle(sample_idxs)

        target_batches_for_process = len(self)
        batches_yielded_for_process = 0

        num_sampled = 0
        batch_idx = 0
        while num_sampled < len(sample_idxs) and (batches_yielded_for_process < target_batches_for_process):
            feat_idxs = [rng.integers(pool_size) for pool_size in self.pool_sizes]
            scaling_feat = feat_idxs[self.scaling_feature_idx]
            batch_size = self.feature_to_batch_size_map.get(scaling_feat, 1)

            remaining = len(sample_idxs) - num_sampled
            if remaining < batch_size:
                if self.drop_last:
                    break
                batch_size = remaining
            batch = [tuple([sample_idxs[num_sampled + idx]] + feat_idxs) for idx in range(batch_size)]

            if len(batch) > 0 and (batch_idx % self.world_size == self.rank):
                yield batch
                batches_yielded_for_process += 1

            batch_idx += 1
            num_sampled += batch_size

class EasyDataset:
    """
    dataset1 + dataset2 ==> Concatenate datasets
    2 * dataset ==> Duplicate each element 2x
    10 @ dataset ==> Set the size to 10 (random sampling, duplicates if necessary)
    """
    def __add__(self, other):
        return CatDataset([self, other])

    def __rmul__(self, factor):
        return MulDataset(factor, self)

    def __rmatmul__(self, factor):
        return ResizedDataset(factor, self)

    def set_epoch(self, epoch):
        pass

    def make_sampler(
        self,
        batch_size=None,
        world_size=1,
        rank=0,
        drop_last=True,
        max_num_of_images_per_gpu=None,
        use_dynamic_sampler=True,
    ):
        if isinstance(self.num_views, int):
            num_of_aspect_ratios = len(self._resolutions)
            feature_pool_sizes = [num_of_aspect_ratios]
            scaling_feature_idx = 0  # Use aspect ratio as scaling feature
        elif isinstance(self.num_views, list):
            num_of_aspect_ratios = len(self._resolutions)
            num_of_num_views = len(self.num_views)
            feature_pool_sizes = [num_of_aspect_ratios, num_of_num_views]
            scaling_feature_idx = 1  # Use num_views as scaling feature
        else:
            raise ValueError(f"Bad type for {self.num_views=}, should be int or list of ints")

        if use_dynamic_sampler:
            assert max_num_of_images_per_gpu is not None

            # Create feature-to-batch-size mapping
            if isinstance(self.num_views, list):
                feature_to_batch_size_map = {}
                for num_views_idx, num_views in enumerate(self.num_views):
                    batch_size_for_multi_view_sets = max(1, max_num_of_images_per_gpu // num_views)
                    feature_to_batch_size_map[num_views_idx] = (batch_size_for_multi_view_sets)
            else:
                feature_to_batch_size_map = {0: max(1, max_num_of_images_per_gpu // self.num_views)}

            return DynamicBatchedMultiFeatureRandomSampler(
                self,
                pool_sizes=feature_pool_sizes,
                scaling_feature_idx=scaling_feature_idx,
                feature_to_batch_size_map=feature_to_batch_size_map,
                world_size=world_size,
                rank=rank,
                drop_last=drop_last,
            )
        else:
            assert batch_size is not None

            return BatchedMultiFeatureRandomSampler(
                self,
                batch_size,
                feature_pool_sizes,
                world_size=world_size,
                rank=rank,
                drop_last=drop_last,
            )

class MulDataset(EasyDataset):
    """Artifically augmenting the size of a dataset."""

    multiplicator: int

    def __init__(self, multiplicator, dataset):
        """
        Initialize a dataset that artificially augments the size of another dataset.

        Args:
            multiplicator (int): Factor by which to multiply the dataset size
            dataset (EasyDataset): The dataset to augment
        """
        assert isinstance(multiplicator, int) and multiplicator > 0
        self.multiplicator = multiplicator
        self.dataset = dataset

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The number of samples in the dataset
        """
        return self.multiplicator * len(self.dataset)

    def __repr__(self):
        """
        Get a string representation of the dataset.

        Returns:
            str: String representation showing the multiplication factor and the original dataset
        """
        return f"{self.multiplicator}*{repr(self.dataset)}"

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Args:
            idx: Index or tuple of indices to retrieve

        Returns:
            The item at the specified index from the original dataset
        """
        if isinstance(idx, tuple):
            other = idx[1:]
            idx = idx[0]
            new_idx = (idx // self.multiplicator, *other)
            return self.dataset[new_idx]
        else:
            return self.dataset[idx // self.multiplicator]

    @property
    def _resolutions(self):
        """
        Get the resolutions of the dataset.

        Returns:
            The resolutions from the original dataset
        """
        return self.dataset._resolutions

    @property
    def num_views(self):
        """
        Get the number of views used for the dataset.

        Returns:
            int or list: The number of views parameter from the original dataset
        """
        return self.dataset.num_views


class ResizedDataset(EasyDataset):
    """Artifically changing the size of a dataset."""

    new_size: int

    def __init__(self, new_size, dataset):
        """
        Initialize a dataset with an artificially changed size.

        Args:
            new_size (int): The new size of the dataset
            dataset (EasyDataset): The original dataset
        """
        assert isinstance(new_size, int) and new_size > 0
        self.new_size = new_size
        self.dataset = dataset

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The new size of the dataset
        """
        return self.new_size

    def __repr__(self):
        """
        Get a string representation of the dataset.

        Returns:
            str: String representation showing the new size and the original dataset
        """
        size_str = str(self.new_size)
        for i in range((len(size_str) - 1) // 3):
            sep = -4 * i - 3
            size_str = size_str[:sep] + "_" + size_str[sep:]
        return f"{size_str} @ {repr(self.dataset)}"

    def set_epoch(self, epoch):
        """
        Set the current epoch and generate a new random mapping of indices.

        This method must be called before using __getitem__.

        Args:
            epoch (int): The current epoch number
        """
        # This random shuffle only depends on the epoch
        rng = np.random.default_rng(seed=epoch + 777)

        # Shuffle all indices
        perm = rng.permutation(len(self.dataset))

        # Calculate how many repetitions we need
        num_repetitions = 1 + (len(self) - 1) // len(self.dataset)

        # Rotary extension until target size is met
        shuffled_idxs = np.concatenate([perm] * num_repetitions)
        self._idxs_mapping = shuffled_idxs[: self.new_size]

        # Generate the seed offset for each repetition
        # This is needed to ensure we see unique samples when we repeat a scene
        seed_offset_per_repetition = [
            np.full(len(self.dataset), i) for i in range(num_repetitions)
        ]
        seed_offset_idxs = np.concatenate(seed_offset_per_repetition)
        self._idxs_seed_offset = seed_offset_idxs[: self.new_size]

        assert len(self._idxs_mapping) == self.new_size
        assert len(self._idxs_seed_offset) == self.new_size

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Args:
            idx: Index or tuple of indices to retrieve

        Returns:
            The item at the mapped index from the original dataset

        Raises:
            AssertionError: If set_epoch has not been called
        """
        assert hasattr(self, "_idxs_mapping"), (
            "You need to call dataset.set_epoch() to use ResizedDataset.__getitem__()"
        )
        if isinstance(idx, tuple):
            other = idx[1:]
            idx = idx[0]
            self.dataset._set_seed_offset(self._idxs_seed_offset[idx])
            new_idx = (self._idxs_mapping[idx], *other)
            return self.dataset[new_idx]
        else:
            self.dataset._set_seed_offset(self._idxs_seed_offset[idx])
            return self.dataset[self._idxs_mapping[idx]]

    @property
    def _resolutions(self):
        """
        Get the resolutions of the dataset.

        Returns:
            The resolutions from the original dataset
        """
        return self.dataset._resolutions

    @property
    def num_views(self):
        """
        Get the number of views used for the dataset.

        Returns:
            int or list: The number of views parameter from the original dataset
        """
        return self.dataset.num_views


class CatDataset(EasyDataset):
    """Concatenation of several datasets"""

    def __init__(self, datasets):
        """
        Initialize a dataset that is a concatenation of several datasets.

        Args:
            datasets (list): List of EasyDataset instances to concatenate
        """
        for dataset in datasets:
            assert isinstance(dataset, EasyDataset)
        self.datasets = datasets
        self._cum_sizes = np.cumsum([len(dataset) for dataset in datasets])

    def __len__(self):
        """
        Get the length of the concatenated dataset.

        Returns:
            int: Total number of samples across all datasets
        """
        return self._cum_sizes[-1]

    def __repr__(self):
        """
        Get a string representation of the concatenated dataset.

        Returns:
            str: String representation showing all concatenated datasets joined by '+'
        """
        # Remove uselessly long transform
        return " + ".join(
            repr(dataset).replace(
                ",transform=Compose( ToTensor() Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))",
                "",
            )
            for dataset in self.datasets
        )

    def set_epoch(self, epoch):
        """
        Set the current epoch for all constituent datasets.

        Args:
            epoch (int): The current epoch number
        """
        for dataset in self.datasets:
            dataset.set_epoch(epoch)

    def __getitem__(self, idx):
        """
        Get an item from the concatenated dataset.

        Args:
            idx: Index or tuple of indices to retrieve

        Returns:
            The item at the specified index from the appropriate constituent dataset

        Raises:
            IndexError: If the index is out of range
        """
        other = None
        if isinstance(idx, tuple):
            other = idx[1:]
            idx = idx[0]

        if not (0 <= idx < len(self)):
            raise IndexError()

        db_idx = np.searchsorted(self._cum_sizes, idx, "right")
        dataset = self.datasets[db_idx]
        new_idx = idx - (self._cum_sizes[db_idx - 1] if db_idx > 0 else 0)

        if other is not None:
            new_idx = (new_idx, *other)
        return dataset[new_idx]

    @property
    def _resolutions(self):
        """
        Get the resolutions of the dataset.

        Returns:
            The resolutions from the first dataset (all datasets must have the same resolutions)

        Raises:
            AssertionError: If datasets have different resolutions
        """
        resolutions = self.datasets[0]._resolutions
        for dataset in self.datasets[1:]:
            assert tuple(dataset._resolutions) == tuple(resolutions), (
                "All datasets must have the same resolutions"
            )
        return resolutions

    @property
    def num_views(self):
        """
        Get the number of views used for the dataset.

        Returns:
            int or list: The number of views parameter from the first dataset

        Raises:
            AssertionError: If datasets have different num_views
        """
        num_views = self.datasets[0].num_views
        for dataset in self.datasets[1:]:
            assert dataset.num_views == num_views, (
                "All datasets must have the same num_views and variable_num_views parameters"
            )
        return num_views

class BaseDataset(EasyDataset):
    """
    Define all basic options.

    Usage:
        class MyDataset(BaseDataset):
            def _get_views(self, idx):
                views = []
                views.append(dict(img=, ...))
                return views
    """

    def __init__(
        self,
        num_views: int,
        variable_num_views: bool = False,
        split: str = None,
        covisibility_thres: float = None,
        resolution: Union[int, Tuple[int, int], List[Tuple[int, int]]] = None,
        principal_point_centered: bool = False,
        transform: str = None,
        data_norm_type: str = None,
        aug_crop: int = 0,
        seed: int = None,
        max_num_retries: int = 5,
        num_views_min: int = 2,
    ):
        """
        PyTorch dataset for multi-view images sampled from scenes, where the images form a single connected component.

        Args:
            num_views (int): Number of views.
            variable_num_views (bool): If True, the number of views can vary from batch to batch. The maximum number of views is num_views and minimum is 2.
                                       On by default for N-view train dataloader (hydra config).
            split (str): 'train', 'val', 'test', etc.
            covisibility_thres (float): Covisibility (%) threshold to determine if another image is a neighbor or not
            resolution (int or tuple or list of tuples): Resolution of the images
            principal_point_centered (bool): If True, the principal point is centered in the image.
            transform (str): Transform to apply to the images. Options:
            - 'colorjitter+grayscale+gaublur':
                tvf.Compose([
                    tvf.RandomApply([tvf.ColorJittter(0.3, 0.4, 0.2, 0.1)], p=0.75),
                    tvf.RandomGrayscale(p=0.05),
                    tvf.RandomApply([tvf.GaussianBlur(5, sigma=(0.1, 1.0))], p=0.05),
                ]) after ImgNorm
            - 'colorjitter': tvf.ColorJittter(0.5, 0.5, 0.5, 0.1) after ImgNorm
            - 'imgnorm': ImgNorm only
            data_norm_type (str): Image normalization type.
                                  For options, see UniCeption image normalization dict.
            aug_crop (int): Augment crop. If int greater than 0, indicates the number of pixels to increase in target resolution.
            seed (int): Seed for the random number generator.
            max_num_retries (int): Maximum number of retries for loading a different sample from the dataset, if provided idx fails.
        """
        self.num_views = num_views
        self.variable_num_views = variable_num_views
        self.num_views_min = num_views_min
        self.split = split
        self.covisibility_thres = covisibility_thres
        self._set_resolutions(resolution)
        self.principal_point_centered = principal_point_centered

        # Update the number of views if necessary and make it a list if variable_num_views is True
        if self.variable_num_views and self.num_views > self.num_views_min:
            self.num_views = list(range(self.num_views_min, self.num_views + 1))
        else:
            self.num_views = [self.num_views]

        # Initialize the image normalization type
        if data_norm_type in IMAGE_NORMALIZATION_DICT.keys():
            self.data_norm_type = data_norm_type
            image_norm = IMAGE_NORMALIZATION_DICT[data_norm_type]
            ImgNorm = tvf.Compose(
                [
                    tvf.ToTensor(),
                    tvf.Normalize(mean=image_norm.mean, std=image_norm.std),
                ]
            )
        elif data_norm_type == "identity":
            self.data_norm_type = data_norm_type
            ImgNorm = tvf.Compose([tvf.ToTensor()])
        else:
            raise ValueError(f"Unknown data_norm_type: {data_norm_type}")

        # Initialize torchvision transforms
        if transform == "imgnorm":
            self.transform = ImgNorm
        elif transform == "colorjitter":
            self.transform = tvf.Compose([tvf.ColorJitter(0.5, 0.5, 0.5, 0.1), ImgNorm])
        elif transform == "colorjitter+grayscale+gaublur":
            self.transform = tvf.Compose(
                [
                    tvf.RandomApply([tvf.ColorJitter(0.3, 0.4, 0.2, 0.1)], p=0.75),
                    tvf.RandomGrayscale(p=0.05),
                    tvf.RandomApply([tvf.GaussianBlur(5, sigma=(0.1, 1.0))], p=0.05),
                    ImgNorm,
                ]
            )
        else:
            raise ValueError(
                'Unknown transform. Available options: "imgnorm", "colorjitter", "colorjitter+grayscale+gaublur"'
            )

        # Initialize the augmentation parameters
        self.aug_crop = aug_crop

        # Initialize the seed for the random number generator
        self.seed = seed
        self._seed_offset = 0

        # Initialize the maximum number of retries for loading a different sample from the dataset, if the first idx fails
        self.max_num_retries = max_num_retries

        # Initialize the dataset type flags
        self.is_metric_scale = False  # by default a dataset is not metric scale, subclasses can overwrite this
        self.is_synthetic = False  # by default a dataset is not synthetic, subclasses can overwrite this

    def _load_data(self):
        self.scenes = []
        self.num_of_scenes = len(self.scenes)

    def __len__(self):
        "Length of the dataset is determined by the number of scenes in the dataset split"
        return self.num_of_scenes

    def get_stats(self):
        "Get the number of scenes in the dataset split"
        return f"{self.num_of_scenes} scenes"

    def __repr__(self):
        resolutions_str = "[" + ";".join(f"{w}x{h}" for w, h in self._resolutions) + "]"
        return (
            f"""{type(self).__name__}({self.get_stats()},
            {self.num_views=}
            {self.split=},
            {self.seed=},
            resolutions={resolutions_str},
            {self.transform=})""".replace("self.", "")
            .replace("\n", "")
            .replace("   ", "")
        )

    def _get_views(self, idx, num_views_to_sample, resolution):
        raise NotImplementedError()

    def _set_seed_offset(self, idx):
        """
        Set the seed offset. This is directly added to self.seed when setting the random seed.
        """
        self._seed_offset = idx

    def _set_resolutions(self, resolutions):
        assert resolutions is not None, "undefined resolution"

        if isinstance(resolutions, int):
            resolutions = [resolutions]
        elif isinstance(resolutions, tuple):
            resolutions = [resolutions]
        elif isinstance(resolutions, list):
            assert all(isinstance(res, tuple) for res in resolutions), (
                f"Bad type for {resolutions=}, should be int or tuple of ints or list of tuples of ints"
            )
        else:
            raise ValueError(
                f"Bad type for {resolutions=}, should be int or tuple of ints or list of tuples of ints"
            )

        self._resolutions = []
        for resolution in resolutions:
            if isinstance(resolution, int):
                width = height = resolution
            else:
                width, height = resolution
            assert isinstance(width, int), (
                f"Bad type for {width=} {type(width)=}, should be int"
            )
            assert isinstance(height, int), (
                f"Bad type for {height=} {type(height)=}, should be int"
            )
            self._resolutions.append((width, height))

    def _crop_resize_if_necessary(
        self,
        image,
        resolution,
        depthmap,
        intrinsics,
        additional_quantities=None,
    ):
        """
        Process an image by downsampling and cropping as needed to match the target resolution.

        This method performs the following operations:
        1. Converts the image to PIL.Image if necessary
        2. Crops the image centered on the principal point if requested
        3. Downsamples the image using high-quality Lanczos filtering
        4. Performs final cropping to match the target resolution

        Args:
            image (numpy.ndarray or PIL.Image.Image): Input image to be processed
            resolution (tuple): Target resolution as (width, height)
            depthmap (numpy.ndarray): Depth map corresponding to the image
            intrinsics (numpy.ndarray): Camera intrinsics matrix (3x3)
            additional_quantities (dict, optional): Additional image-related data to be processed
                                                   alongside the main image with nearest interpolation. Defaults to None.

        Returns:
            tuple: Processed image, depthmap, and updated intrinsics matrix.
                  If additional_quantities is provided, it returns those as well.
        """
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        # # Cropping centered on the principal point if necessary
        # if self.principal_point_centered:
        #     W, H = image.size
        #     cx, cy = intrinsics[:2, 2].round().astype(int)
        #     if cx < 0 or cx >= W or cy < 0 or cy >= H:
        #         # Skip centered cropping if principal point is outside image bounds
        #         pass
        #     else:
        #         min_margin_x = min(cx, W - cx)
        #         min_margin_y = min(cy, H - cy)
        #         left, top = cx - min_margin_x, cy - min_margin_y
        #         right, bottom = cx + min_margin_x, cy + min_margin_y
        #         crop_bbox = (left, top, right, bottom)
        #         # Only perform the centered crop if the crop_bbox is larger than the target resolution
        #         crop_width = right - left
        #         crop_height = bottom - top
        #         if crop_width > resolution[0] and crop_height > resolution[1]:
        #             image, depthmap, intrinsics, additional_quantities = (
        #                 crop_image_and_other_optional_info(
        #                     image=image,
        #                     crop_bbox=crop_bbox,
        #                     depthmap=depthmap,
        #                     camera_intrinsics=intrinsics,
        #                     additional_quantities=additional_quantities,
        #                 )
        #             )

        # Get the target resolution for re-scaling
        target_rescale_resolution = np.array(resolution)
        if self.aug_crop > 1:
            target_rescale_resolution += self._rng.integers(0, self.aug_crop)

        # High-quality Lanczos down-scaling if necessary
        image_W_H_before_rescale = image.size
        image, depthmap, intrinsics, additional_quantities = rescale_image_and_other_optional_info(
            image=image,
            output_resolution=target_rescale_resolution,
            depthmap=depthmap,
            camera_intrinsics=intrinsics,
            additional_quantities_to_be_resized_with_nearest=additional_quantities,
        )
        image_W_H_after_rescale = image.size

        # Actual cropping (if necessary)
        new_intrinsics = camera_matrix_of_crop(
            input_camera_matrix=intrinsics,
            input_resolution=image.size,
            output_resolution=resolution,
            offset_factor=0.5,
        )
        crop_bbox = bbox_from_intrinsics_in_out(
            input_camera_matrix=intrinsics,
            output_camera_matrix=new_intrinsics,
            output_resolution=resolution,
        )
        image, depthmap, new_intrinsics, additional_quantities = crop_image_and_other_optional_info(
            image=image,
            crop_bbox=crop_bbox,
            depthmap=depthmap,
            camera_intrinsics=intrinsics,
            additional_quantities=additional_quantities,
        )

        # Return the output
        if additional_quantities is not None:
            return image, depthmap, new_intrinsics, additional_quantities, {'image_W_H_before_rescale': image_W_H_before_rescale, 'image_W_H_after_rescale': image_W_H_after_rescale, 'crop_bbox': crop_bbox}
        else:
            return image, depthmap, new_intrinsics

    def _random_walk_sampling(
        self,
        scene_pairwise_covisibility,
        num_of_samples,
        max_retries=4,
        use_bidirectional_covis=True,
    ):
        """
        Randomly samples S indices from an N x N covisbility matrix by forming adjacency edges such that the resulting subgraph (given by the indices) is connected.
        If the current node has no new unvisited neighbors, backtracking occurs.
        Retries with different starting indices if the desired number of samples is not reached, excluding previously visited components.

        Args:
            scene_pairwise_covisibility : np.ndarray (mmap)
                N x N covisibility matrix for the scene, where N is the number of views in the scene.
            num_of_samples : int
                The desired number of nodes to sample (num_of_samples < N).
            max_retries : int
                The maximum number of retries with different starting indices.
            use_bidirectional_covis : bool
                Whether to compute bidirectional covisibility by averaging row and column values.
                If False, uses only row access (faster for large memory-mapped arrays).
                Defaults to True.

        Returns:
            np.ndarray
                An array of sampled indices forming a connected subgraph.
        """
        excluded_nodes = set()
        best_walk = []  # To keep track of the best walk found
        for _ in range(max_retries):
            visited = set()
            walk = []  # List to store the random walk sampling order
            stack = []  # Stack for backtracking

            # Choose a random starting index that is not in the excluded set
            all_nodes = set(range(len(scene_pairwise_covisibility)))
            available_nodes = list(all_nodes - excluded_nodes)
            if not available_nodes:
                break  # No more nodes to try
            start = self._rng.choice(available_nodes)
            walk.append(start)
            visited.add(start)
            stack.append(start)

            # Continue until we have sampled S indices or all expandable nodes are exhausted
            while len(walk) < num_of_samples and stack:
                current = stack[-1]
                # Get the pairwise covisibility for the current node
                if use_bidirectional_covis:
                    # Use bidirectional covisibility (slower for large memory-mapped arrays)
                    pairwise_covisibility = (
                        scene_pairwise_covisibility[current, :]
                        + scene_pairwise_covisibility[:, current].T
                    ) / 2
                else:
                    # Use only row access (faster for large memory-mapped arrays)
                    pairwise_covisibility = scene_pairwise_covisibility[current, :]
                # Normalize the covisibility using self covisibility
                pairwise_covisibility = pairwise_covisibility / (
                    pairwise_covisibility[current] + 1e-8
                )
                # Assign overlap score of zero to self-pairs
                pairwise_covisibility[current] = 0
                # Threshold the covisibility to get adjacency list for the current node
                adjacency_list_for_current = (
                    pairwise_covisibility > self.covisibility_thres
                ).astype(int)
                adjacency_list_for_current = np.flatnonzero(adjacency_list_for_current)
                # Get all unvisited neighbors
                candidates = [
                    idx for idx in adjacency_list_for_current if idx not in visited
                ]  # Remove visited nodes
                if candidates:
                    # Randomly select one of the unvisited overlapping neighbors
                    next_node = self._rng.choice(candidates)
                    walk.append(next_node)
                    visited.add(next_node)
                    stack.append(next_node)
                else:
                    # If no unvisited neighbor is available, backtrack
                    stack.pop()

            # Update the best walk if the current walk is larger
            if len(walk) > len(best_walk):
                best_walk = walk

            # If we have enough samples, return the result
            if len(walk) >= num_of_samples:
                return np.array(walk)

            # Add all visited nodes to the excluded set
            excluded_nodes.update(visited)

        # If all retries are exhausted and we still don't have enough samples, return the best walk found
        return np.array(best_walk)

    def _sample_view_indices(
        self,
        num_views_to_sample,
        num_views_in_scene,
        scene_pairwise_covisibility,
        use_bidirectional_covis=True,
    ):
        """
        Sample view indices from a scene based on the adjacency list and the number of views to sample.

        Args:
            num_views_to_sample (int): Number of views to sample.
            num_views_in_scene (int): Total number of views available in the scene.
            scene_pairwise_covisibility (np.ndarray): N x N covisibility matrix for the scene, where N is the number of views in the scene.
            use_bidirectional_covis (bool): Whether to compute bidirectional covisibility by averaging row and column values.
                If False, uses only row access (faster for large memory-mapped arrays).

        Returns:
            numpy.ndarray: Array of sampled view indices.
        """
        if num_views_to_sample == num_views_in_scene:
            # Select all views in the scene
            view_indices = self._rng.permutation(num_views_in_scene)
        elif num_views_to_sample > num_views_in_scene:
            # Select all views in the scene and repeat them to get the desired number of views
            view_indices = self._rng.choice(num_views_in_scene, size=num_views_to_sample, replace=True)
        else:
            # Select a subset of single component connected views in the scene using random walk sampling
            view_indices = self._random_walk_sampling(
                scene_pairwise_covisibility,
                num_views_to_sample,
                use_bidirectional_covis=use_bidirectional_covis,
            )
            # If the required num of views can't be obtained even with 4 retries, repeat existing indices to get the desired number of views
            if len(view_indices) < num_views_to_sample:
                view_indices = self._rng.choice(view_indices, size=num_views_to_sample, replace=True)

        return view_indices

    def _getitem_fn(self, idx):
        if isinstance(idx, tuple):
            if isinstance(self.num_views, int):
                idx, ar_idx = idx
                num_views_to_sample = self.num_views
            else:
                idx, ar_idx, num_views_to_sample_idx = idx
                num_views_to_sample = self.num_views[num_views_to_sample_idx]
        else:
            assert len(self._resolutions) == 1
            assert isinstance(self.num_views, int)
            ar_idx = 0
        resolution = self._resolutions[ar_idx]

        # Setup the rng
        if self.seed:  # reseed for each _getitem_fn
            # Leads to deterministic sampling where repeating self.seed and self._seed_offset yields the same multi-view set again
            # Scenes will be repeated if size of dataset is artificially increased using "N @" or "N *"
            # When scenes are repeated, self._seed_offset is increased to ensure new multi-view sets
            # This is useful for evaluation if the number of dataset scenes is < N, yet we want unique multi-view sets each iter
            self._rng = np.random.default_rng(seed=self.seed + self._seed_offset + idx)
        elif not hasattr(self, "_rng"):
            seed = torch.initial_seed()  # this is different for each dataloader process
            self._rng = np.random.default_rng(seed=seed)

        # Get the views for the given index and check that the number of views is correct
        views = self._get_views(idx, num_views_to_sample, resolution)

        for v, view in enumerate(views):
            # Store the index and other metadata
            view["idx"] = (idx, ar_idx, v)
            view["is_metric_scale"] = self.is_metric_scale
            view["is_synthetic"] = self.is_synthetic

            # Check the depth, intrinsics, and pose data (also other data if present)
            assert "camera_intrinsics" in view
            assert "camera_pose" in view
            assert np.isfinite(view["camera_pose"]).all()
            assert np.isfinite(view["depthmap"]).all()
            assert "valid_mask" not in view
            assert "pts3d" not in view
            if "non_ambiguous_mask" in view:
                assert np.isfinite(view["non_ambiguous_mask"]).all()

            # Encode the image
            width, height = view["img"].size
            view["true_shape"] = np.int32((height, width))
            view["img"] = self.transform(view["img"])
            view["data_norm_type"] = self.data_norm_type

            # Compute the pointmaps, raymap and depth along ray
            (
                pts3d,
                valid_mask,
                ray_origins_world,
                ray_directions_world,
                depth_along_ray,
                ray_directions_cam,
                pts3d_cam,
            ) = get_absolute_pointmaps_and_rays_info(**view)
            view["pts3d"] = pts3d
            view["valid_mask"] = valid_mask & np.isfinite(pts3d).all(axis=-1)
            view["depth_along_ray"] = depth_along_ray
            view["ray_directions_cam"] = ray_directions_cam
            view["pts3d_cam"] = pts3d_cam

            # Convert ambiguous mask dtype to match valid mask dtype
            if "non_ambiguous_mask" in view:
                view["non_ambiguous_mask"] = view["non_ambiguous_mask"].astype(view["valid_mask"].dtype)
            else:
                ambiguous_mask = view["depthmap"] < 0
                view["non_ambiguous_mask"] = ~ambiguous_mask
                view["non_ambiguous_mask"] = view["non_ambiguous_mask"].astype(view["valid_mask"].dtype)

            # Check shapes
            assert view["depthmap"].shape == view["img"].shape[1:]
            assert view["depthmap"].shape == view["pts3d"].shape[:2]
            assert view["depthmap"].shape == view["valid_mask"].shape
            assert view["depthmap"].shape == view["depth_along_ray"].shape[:2]
            assert view["depthmap"].shape == view["ray_directions_cam"].shape[:2]
            assert view["depthmap"].shape == view["pts3d_cam"].shape[:2]
            if "non_ambiguous_mask" in view:
                assert view["depthmap"].shape == view["non_ambiguous_mask"].shape

            # Expand the last dimennsion of the depthmap
            view["depthmap"] = view["depthmap"][..., None]

            # Append RNG state to the views, this allows to check whether the RNG is in the same state each time
            view["rng"] = int.from_bytes(self._rng.bytes(4), "big")

            # Compute and store the quaternions and translation for the camera poses
            # Notation is (x, y, z, w) for quaternions
            # This also ensures that the camera poses have a positive determinant (right-handed coordinate system)
            view["camera_pose_quats"] = Rotation.from_matrix(view["camera_pose"][:3, :3]).as_quat().astype(view["camera_pose"].dtype)
            view["camera_pose_trans"] = view["camera_pose"][:3, 3].astype(view["camera_pose"].dtype)

            assert np.isfinite(view["pts3d"]).all() and np.isfinite(view["valid_mask"]).all() and np.isfinite(view["depth_along_ray"]).all() and np.isfinite(view["ray_directions_cam"]).all() and np.isfinite(view["pts3d_cam"]).all() and np.isfinite(view["camera_pose_quats"]).all() and np.isfinite(view["camera_pose_trans"]).all()

        return views

    def __getitem__(self, idx):
        if self.max_num_retries == 0:
            return self._getitem_fn(idx)

        num_retries = 0
        while num_retries <= self.max_num_retries:
            try:
                return self._getitem_fn(idx)
            except Exception as e:
                scene_idx = idx[0] if isinstance(idx, tuple) else idx
                print(
                    f"Error in {type(self).__name__}.__getitem__ for scene_idx={scene_idx}: {e}"
                )

                if num_retries >= self.max_num_retries:
                    print(
                        f"Max retries ({self.max_num_retries}) reached, raising the exception"
                    )
                    raise e

                # Retry with a different scene index
                num_retries += 1
                if isinstance(idx, tuple):
                    # The scene index is the first element of the tuple
                    idx_list = list(idx)
                    idx_list[0] = np.random.randint(0, len(self))
                    idx = tuple(idx_list)
                else:
                    # The scene index is idx
                    idx = np.random.randint(0, len(self))
                scene_idx = idx[0] if isinstance(idx, tuple) else idx
                print(
                    f"Retrying with scene_idx={scene_idx} ({num_retries} of {self.max_num_retries})"
                )

class BlendedMVSWAI(BaseDataset):
    def __init__(
        self,
        *args,
        ROOT,
        dataset_metadata_dir,
        split,
        overfit_num_sets=None,
        sample_specific_scene: bool = False,
        specific_scene_name: str = None,
        **kwargs,
    ):
        """
        Initialize the dataset attributes.
        Args:
            ROOT: Root directory of the dataset.
            dataset_metadata_dir: Path to the dataset metadata directory.
            split: Dataset split (train, val, test).
            overfit_num_sets: If None, use all sets. Else, the dataset will be truncated to this number of sets.
            sample_specific_scene: Whether to sample a specific scene from the dataset.
            specific_scene_name: Name of the specific scene to sample.
        """
        # Initialize the dataset attributes
        super().__init__(*args, **kwargs)
        self.ROOT = '/shared/smartbot/yangsizhe/data/blendedMVS/wai'
        self.dataset_metadata_dir = dataset_metadata_dir
        self.split = split
        self.overfit_num_sets = overfit_num_sets
        self.sample_specific_scene = sample_specific_scene
        self.specific_scene_name = specific_scene_name
        self._load_data()

        # Define the dataset type flags
        self.is_metric_scale = False
        self.is_synthetic = False

    def _load_data(self):
        split_metadata_path = os.path.join(self.dataset_metadata_dir, self.split, f"blendedmvs_scene_list_{self.split}.npy")
        split_scene_list = np.load(split_metadata_path, allow_pickle=True)

        if not self.sample_specific_scene:
            self.scenes = list(split_scene_list)
        else:
            self.scenes = [self.specific_scene_name]
        self.num_of_scenes = len(self.scenes)

    def _get_views(self, sampled_idx, num_views_to_sample, resolution):
        # Get the scene name of the sampled index
        scene_index = sampled_idx
        scene_name = self.scenes[scene_index]

        # Get the metadata corresponding to the scene
        scene_root = os.path.join(self.ROOT, scene_name)
        scene_meta = load_data(
            os.path.join(scene_root, "scene_meta.json"), "scene_meta"
        )
        scene_file_names = list(scene_meta["frame_names"].keys())
        num_views_in_scene = len(scene_file_names)

        # Load the scene pairwise covisibility mmap
        covisibility_version_key = "v0"
        covisibility_map_dir = os.path.join(
            scene_root, "covisibility", covisibility_version_key
        )
        # Assumes only npy file in directory is covisbility map
        covisibility_map_name = next(f for f in os.listdir(covisibility_map_dir) if f.endswith(".npy"))
        covisibility_map_path = os.path.join(scene_root, "covisibility", covisibility_version_key, covisibility_map_name)
        pairwise_covisibility = load_data(covisibility_map_path, "mmap")

        # Get the indices of the N views in the scene
        view_indices = self._sample_view_indices(num_views_to_sample, num_views_in_scene, pairwise_covisibility)

        # Get the views corresponding to the selected view indices
        views = []
        for view_index in view_indices:
            # Load the data corresponding to the view
            view_file_name = scene_file_names[view_index]
            view_data = load_frame(
                scene_root,
                view_file_name,
                modalities=["image", "depth", "pred_mask/moge2"],
                scene_meta=scene_meta,
            )

            # Convert necessary data to numpy
            image = view_data["image"].permute(1, 2, 0).numpy()
            image = (image * 255).astype(np.uint8)
            depthmap = view_data["depth"].numpy().astype(np.float32)
            depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0)
            intrinsics = view_data["intrinsics"].numpy().astype(np.float32)
            c2w_pose = view_data["extrinsics"].numpy().astype(np.float32)
            non_ambiguous_mask = view_data["pred_mask/moge2"].numpy().astype(int)
            non_ambiguous_mask = cv2.resize(
                non_ambiguous_mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

            depthmap = np.where(non_ambiguous_mask, depthmap, 0)

            # Resize the data to match the desired resolution
            additional_quantities_to_resize = [non_ambiguous_mask]
            image, depthmap, intrinsics, additional_quantities_to_resize = (
                self._crop_resize_if_necessary(
                    image=image,
                    resolution=resolution,
                    depthmap=depthmap,
                    intrinsics=intrinsics,
                    additional_quantities=additional_quantities_to_resize,
                )
            )
            non_ambiguous_mask = additional_quantities_to_resize[0]

            # Append the view dictionary to the list of views
            views.append(
                dict(
                    img=image,
                    depthmap=depthmap,
                    camera_pose=c2w_pose,
                    camera_intrinsics=intrinsics,
                    non_ambiguous_mask=non_ambiguous_mask,
                    dataset="BlendedMVS",
                    label=scene_name,
                    instance=os.path.join("images", str(view_file_name)),
                )
            )

        return views

def c2w_xforward_zup_to_opencv(c2w_xzup):
    T = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    c2w_cv = np.eye(4)
    c2w_cv[:3, :3] = T @ c2w_xzup[:3, :3] @ T.T
    c2w_cv[:3, 3] = T @ c2w_xzup[:3, 3]
    return c2w_cv
# def c2w_xforward_zup_to_opencv(c2w_xzup):
#     T = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
#     c2w_cv = np.eye(4)
#     c2w_cv[:3, :3] = c2w_xzup[:3, :3] @ T.T
#     c2w_cv[:3, 3] = c2w_xzup[:3, 3]
#     return c2w_cv

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
    is_in_frame = np.array([True] * len(pointsBelief))

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
        else:
            is_in_frame[i_point] = False
        out[i_point] = array

    return out, is_in_frame

class ThreeDP4MWAI(BaseDataset):
    def __init__(
        self,
        *args,
        ROOT,
        split,
        num_of_scenes,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ROOT = ROOT
        self.split = split
        self.is_metric_scale = True
        self.is_synthetic = True

        self.num_of_scenes = num_of_scenes

    def _get_views(self, sampled_idx, num_views_to_sample, resolution):
        scene_name = f'scene_{str(sampled_idx // 10).zfill(8)}'
        scene_root = os.path.join(self.ROOT, scene_name)

        view_indices = self._rng.permutation(4)[:num_views_to_sample]

        views = []
        for view_index in view_indices:
            view_file_name = f'{str(sampled_idx % 10).zfill(4)}_{str(view_index).zfill(2)}'
            image = np.array(Image.open(os.path.join(scene_root, 'rgb', view_file_name + '.jpg'))).astype(np.uint8)
            depthmap = np.array(Image.open(os.path.join(scene_root, 'depth', view_file_name + '.png')), dtype=np.uint16).astype(np.float32)
            depthmap = depthmap / 2**16 * 10
            depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0)
            cam_param = np.load(f'{scene_root}/cam_param.npy').astype(np.float32)
            intrinsics = cam_param[0, view_index, :3, :3]
            opencv_c2w = c2w_xforward_zup_to_opencv(cam_param[1, view_index]).astype(np.float32)
            mask = np.array(Image.open(os.path.join(scene_root, 'mask', view_file_name + '.png'))).astype(np.uint8)
            non_ambiguous_mask = (mask != 0).astype(int)
            depthmap = np.where(non_ambiguous_mask, depthmap, 0.0).astype(np.float32)
            additional_quantities_to_resize = [non_ambiguous_mask]

            # seperate pred
            robot_mask = (mask == 100).astype(int)
            additional_quantities_to_resize.append(robot_mask)
            object_mask = (mask == 150).astype(int)
            additional_quantities_to_resize.append(object_mask)
            table_mask = (mask == 50).astype(int)
            additional_quantities_to_resize.append(table_mask)

            # Resize the data to match the desired resolution
            image, depthmap, intrinsics, additional_quantities_to_resize, image_processing_info = self._crop_resize_if_necessary(
                image=image,
                resolution=resolution,
                depthmap=depthmap,
                intrinsics=intrinsics,
                additional_quantities=additional_quantities_to_resize,
            )
            non_ambiguous_mask = additional_quantities_to_resize[0]

            robot_qpos = np.load(f'{scene_root}/qpos/{str(sampled_idx % 10).zfill(4)}.npy').astype(np.float32)  # fuse robot qpos

            # pred keypoint
            small_map = False  # small_map
            sigma = 8
            robot_keypoint = np.load(f'{scene_root}/keypoint_2d/{str(sampled_idx % 10).zfill(4)}_{str(view_index).zfill(2)}.npy').astype(np.float32)
            robot_keypoint[:, 0] = robot_keypoint[:, 0] / image_processing_info['image_W_H_before_rescale'][0] * image_processing_info['image_W_H_after_rescale'][0]
            robot_keypoint[:, 1] = robot_keypoint[:, 1] / image_processing_info['image_W_H_before_rescale'][1] * image_processing_info['image_W_H_after_rescale'][1]
            robot_keypoint[:, 0] -= image_processing_info['crop_bbox'][0]
            robot_keypoint[:, 1] -= image_processing_info['crop_bbox'][1]
            if small_map:
                robot_keypoint[:, 0] = robot_keypoint[:, 0] /14*8/2
                robot_keypoint[:, 1] = robot_keypoint[:, 1] /14*8/2
                robot_keypoint_belief_map, is_in_frame = create_belief_map(image_resolution=tuple([int(size/14*8/2) for size in image.size]), pointsBelief=robot_keypoint, sigma=2)
                robot_keypoint_belief_map = robot_keypoint_belief_map.astype(np.float32)
            else:
                robot_keypoint_belief_map, is_in_frame = create_belief_map(image_resolution=image.size, pointsBelief=robot_keypoint, sigma=sigma)
                robot_keypoint_belief_map = robot_keypoint_belief_map.astype(np.float32)
            robot_keypoint[~is_in_frame, :] = -1.0
            # image.save('rgb.png')
            # Image.fromarray((robot_keypoint_belief_map.sum(axis=0) * 255.0).astype(np.uint8)).save('robot_keypoint_belief_map_all.png')
            robot_keypoint_3d = np.load(f'{scene_root}/keypoint_3d/{str(sampled_idx % 10).zfill(4)}.npy').astype(np.float32)

            views.append(
                dict(
                    img=image,
                    depthmap=depthmap,
                    camera_pose=opencv_c2w,
                    camera_intrinsics=intrinsics,
                    non_ambiguous_mask=non_ambiguous_mask,
                    dataset="3dp4m",
                    path=f'{scene_root}/{view_file_name}',
                    label=scene_name,
                    instance=view_file_name,

                    robot_qpos=robot_qpos,  # fuse robot qpos

                    # pred keypoint
                    keypoint_map=robot_keypoint_belief_map,
                    keypoint=robot_keypoint,
                    keypoint_3d=robot_keypoint_3d,

                    # seperate pred
                    robot_mask=additional_quantities_to_resize[1],
                    object_mask=additional_quantities_to_resize[2],
                    table_mask=additional_quantities_to_resize[3],

                )
            )

        return views

class DynamicBatchDatasetWrapper:
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, batch_indices):
        if isinstance(batch_indices[0], (list, tuple)):
            return [self.dataset[idx] for idx in batch_indices]
        else:
            return self.dataset[batch_indices]

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

def get_train_data_loader(
    dataset,
    max_num_of_imgs_per_gpu,
    num_workers=8,
    drop_last=True,
    pin_memory=True,
):
    if isinstance(dataset, str):
        dataset = eval(dataset)

    batch_sampler = dataset.make_sampler(
        world_size=dist.get_world_size(),
        rank=dist.get_rank(),
        drop_last=drop_last,
        max_num_of_images_per_gpu=max_num_of_imgs_per_gpu,
        use_dynamic_sampler=True,
    )

    wrapped_dataset = DynamicBatchDatasetWrapper(dataset)

    dataloader = torch.utils.data.DataLoader(
        wrapped_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return dataloader

def build_dataloader(dataset, num_workers, max_num_of_imgs_per_gpu=None):
    dataloader = get_train_data_loader(
        dataset=dataset,
        max_num_of_imgs_per_gpu=max_num_of_imgs_per_gpu,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print('dataloader length: ', len(dataloader))

    return dataloader


if __name__ == "__main__":
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    from datetime import timedelta
    torch.distributed.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(seconds=1800), device_id=torch.device(f"cuda:{local_rank}"))
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    torch.set_default_dtype(torch.float32)
    torch.distributed.barrier()

    # train_dataset_config = "+ 140_000 @ BlendedMVSWAI(split='train', resolution=[(518, 518), (518, 392), (518, 336), (518, 294), (518, 252), (518, 168), (392, 518), (336, 518), (294, 518), (252, 518)], principal_point_centered=False, aug_crop=16, transform='colorjitter+grayscale+gaublur', data_norm_type='dinov2', ROOT='/shared/smartbot/yangsizhe/data/blendedmvs', dataset_metadata_dir='/shared/smartbot/yangsizhe/data/mapanything_meta', overfit_num_sets=None, variable_num_views=True, num_views=4, covisibility_thres=0.25)"
    train_dataset_config = "ThreeDP4MWAI(split='train', resolution=[(630, 476), (616, 476), (630, 462), (616, 462)], principal_point_centered=False, aug_crop=16, transform='colorjitter+grayscale+gaublur', data_norm_type='dinov2', ROOT='/shared/smartbot_new/yangsizhe/data/3dp4m/100k_scenes_10_frames_4_cameras_simple_texture_no_flying', variable_num_views=True, num_views=2, num_of_scenes=10000 * 10)"
    dataloader_train = build_dataloader(dataset=train_dataset_config, num_workers=8, max_num_of_imgs_per_gpu=12)

    epoch_idx = 0
    dataloader_train.dataset.set_epoch(epoch_idx)
    dataloader_train.batch_sampler.set_epoch(epoch_idx)
    torch.cuda.synchronize(); import time; t0 = time.time()
    for batch in dataloader_train:
        print(f'len(batch): {len(batch)}', f'batch[0]["img"].shape: {batch[0]["img"].shape}')
        torch.cuda.synchronize(); t1 = time.time(); print('load a batch', t1-t0); t0 = t1

    torch.distributed.destroy_process_group()
