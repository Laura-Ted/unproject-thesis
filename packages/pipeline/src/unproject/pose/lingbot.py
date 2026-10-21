"""Learned pose estimation, the backend measured against COLMAP.

Runs LingBot-Map on the Metal backend. The model is an optional extra, so
everything it needs is imported inside the functions that use it and the rest of
the pipeline works without it installed. That laziness is also load-bearing:
torch and pycolmap each bundle a copy of libomp and abort the process if both
initialise, so the two backends must never be imported together (see
`docs/reproduce.md` §6).

Three things differ from COLMAP, and they are visible in the artifact rather
than hidden here. It places every keyframe it is given, so the registered count
carries no information. It predicts a field of view per frame instead of one
camera for the take, so there is a camera per keyframe. It predicts dense depth
rather than a sparse cloud, so the cloud is back-projected and thinned.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from unproject.config import LingbotConfig, config_hash
from unproject.pose.artifacts import BackendInfo, Camera, CameraPose
from unproject.pose.backend import PoseBackendError, PoseEstimate
from unproject.pose.geometry import invert_pose, matrix_to_quaternion, quaternion_to_matrix
from unproject.pose.pointcloud import Colors, Points

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# Fixed by the patch embedding the checkpoint was trained with, so not a setting.
PATCH_SIZE = 14

INSTALL_HINT = "the lingbot backend needs its extra: uv sync --extra lingbot"


# The model package ships no stubs and is not installed by default, so these
# describe the surface this backend actually touches.
@runtime_checkable
class StreamModel(Protocol):
    """The slice of `lingbot_map.models.gct_stream.GCTStream` used here."""

    aggregator: torch.nn.Module

    def named_modules(self) -> Iterator[tuple[str, torch.nn.Module]]: ...

    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor], strict: bool) -> object: ...

    def inference_streaming(
        self,
        images: torch.Tensor,
        num_scale_frames: int,
        keyframe_interval: int,
        output_device: torch.device,
    ) -> dict[str, torch.Tensor]: ...


class LingbotBackend:
    """Estimates poses with LingBot-Map, converting into the project convention.

    The axes are OpenCV's on both sides, so no basis change is needed, but the
    direction has to be reversed: see `poses_from_extrinsics`.
    """

    def __init__(self, config: LingbotConfig, *, config_dir: Path) -> None:
        self._config = config
        self._config_dir = config_dir

    def describe(self) -> BackendInfo:
        # The weights revision rather than the package version: upstream has
        # published every commit as 0.1.0, and it is the weights that decide the
        # poses. The installed package version goes in the run manifest.
        return BackendInfo(
            name="lingbot",
            version=self._config.revision[:12],
            config_hash=config_hash(self._config),
        )

    def estimate(self, image_paths: list[Path], keyframe_indices: list[int]) -> PoseEstimate:
        if len(image_paths) != len(keyframe_indices):
            raise PoseBackendError(
                f"{len(image_paths)} images against {len(keyframe_indices)} keyframe indices"
            )
        if not image_paths:
            raise PoseBackendError("no keyframes to estimate poses for")

        _require_extra()
        device = _select_device()
        logger.info("running %d keyframes through lingbot-map on %s", len(image_paths), device)

        images = _load_images(image_paths, self._config.image_size)
        model = _build_model(self._config, device, self._config_dir)
        predictions = _infer(model, images, self._config, device)
        return _collect_estimate(predictions, image_paths, keyframe_indices, self._config)


def model_space_geometry(width: int, height: int, image_size: int) -> tuple[int, int, int]:
    """Where a keyframe lands after the model's resize-and-crop.

    Returns the resized width and height and how many rows the centre crop takes
    off the top. The height is rounded to a whole number of patches, so the two
    axes are not scaled by quite the same factor and the intrinsics have to be
    undone per axis rather than by one scale.
    """
    if width <= 0 or height <= 0:
        raise PoseBackendError(f"keyframe has a degenerate size {width}x{height}")
    resized_width = image_size
    resized_height = round(height * (resized_width / width) / PATCH_SIZE) * PATCH_SIZE
    resized_height = max(resized_height, PATCH_SIZE)
    crop_top = max(0, (resized_height - image_size) // 2)
    return resized_width, resized_height, crop_top


def intrinsics_to_keyframe(
    intrinsics: NDArray[np.float64], *, width: int, height: int, image_size: int
) -> tuple[float, float, float, float]:
    """Re-express a model-space 3x3 K at the keyframe's own resolution.

    COLMAP reports intrinsics at the resolution of the images it was given, so
    this backend has to as well. A focal length measured in the model's cropped
    518-pixel space is not comparable with one measured on the keyframe.
    """
    resized_width, resized_height, crop_top = model_space_geometry(width, height, image_size)
    scale_x = resized_width / width
    scale_y = resized_height / height
    return (
        float(intrinsics[0, 0]) / scale_x,
        float(intrinsics[1, 1]) / scale_y,
        float(intrinsics[0, 2]) / scale_x,
        (float(intrinsics[1, 2]) + crop_top) / scale_y,
    )


def backproject(
    depth: NDArray[np.float64],
    intrinsics: NDArray[np.float64],
    quaternion: NDArray[np.float64],
    translation: NDArray[np.float64],
) -> Points:
    """Lift one depth map into world space.

    The extrinsics are world-to-camera, `x_cam = R x_world + t`, so the way back
    is `x_world = R^T (x_cam - t)`.
    """
    height, width = depth.shape
    columns, rows = np.meshgrid(
        np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64)
    )
    focal_x, focal_y = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    centre_x, centre_y = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    camera = np.stack(
        [
            (columns - centre_x) / focal_x * depth,
            (rows - centre_y) / focal_y * depth,
            depth,
        ],
        axis=-1,
    ).reshape(-1, 3)
    # Row vectors, so right-multiplying by R applies R transposed.
    rotation = quaternion_to_matrix(quaternion)
    world: Points = (camera - translation.reshape(3)) @ rotation
    return world


def select_confident(
    confidence: NDArray[np.float64], *, percentile: float, budget: int
) -> NDArray[np.int64]:
    """Indices of the depths worth keeping, at most `budget` of them.

    Above the confidence percentile, then thinned by an even stride rather than a
    random draw, so a run reproduces without having to carry a seed for it.
    """
    flat = confidence.reshape(-1)
    threshold = float(np.percentile(flat, percentile))
    keep: NDArray[np.int64] = np.flatnonzero(flat >= threshold)
    if keep.size <= budget:
        return keep
    stride = keep.size / budget
    picks = (np.arange(budget, dtype=np.float64) * stride).astype(np.int64)
    return keep[picks]


def _require_extra() -> None:
    """Fail before anything heavy loads if the extra is not installed.

    Deliberately a spec lookup rather than an import: torch cannot share a
    process with pycolmap, so the path that reports the extra missing must not
    be the path that loads torch.
    """
    if importlib.util.find_spec("lingbot_map") is None:
        raise PoseBackendError(INSTALL_HINT)


def _select_device() -> torch.device:
    try:
        import torch
    except ImportError as error:
        raise PoseBackendError(INSTALL_HINT) from error

    if torch.backends.mps.is_available():
        return torch.device("mps")
    logger.warning("no Metal backend available, falling back to CPU; expect this to be slow")
    return torch.device("cpu")


def _checkpoint_path(config: LingbotConfig, config_dir: Path) -> Path:
    """Fetch the pinned weights and check they are the ones the config names."""
    # The project runs every model locally and reports nothing upstream. The hub
    # client otherwise pings home on import, so this is set before it loads
    # rather than left to an environment the operator has to remember.
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise PoseBackendError(INSTALL_HINT) from error

    cache_dir = (config_dir / config.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = Path(
        str(
            hf_hub_download(
                repo_id=config.repo_id,
                filename=config.checkpoint,
                revision=config.revision,
                cache_dir=str(cache_dir),
            )
        )
    )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    if digest.hexdigest() != config.sha256:
        raise PoseBackendError(
            f"{path} hashes to {digest.hexdigest()}, but the config pins {config.sha256}"
        )
    return path


def _build_model(config: LingbotConfig, device: torch.device, config_dir: Path) -> StreamModel:
    import torch

    try:
        from lingbot_map.models.gct_stream import GCTStream
    except ImportError as error:
        raise PoseBackendError(INSTALL_HINT) from error

    built = GCTStream(
        img_size=config.image_size,
        patch_size=PATCH_SIZE,
        enable_3d_rope=True,
        kv_cache_sliding_window=config.kv_cache_sliding_window,
        kv_cache_scale_frames=config.num_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        # FlashInfer is CUDA-only. This path is maintained upstream rather than a
        # dormant fallback, and it is what the memory figures were measured on.
        use_sdpa=True,
        camera_num_iterations=config.camera_num_iterations,
    )

    checkpoint = torch.load(
        _checkpoint_path(config, config_dir), map_location="cpu", weights_only=False
    )
    built.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
    del checkpoint

    _narrow_complex_buffers(built)
    built = built.to(device).eval()
    if config.dtype == "bfloat16":
        # Only the trunk. The heads upcast their own inputs and are meant to run
        # in single precision.
        built.aggregator = built.aggregator.to(dtype=torch.bfloat16)
    return cast(StreamModel, built)


def _narrow_complex_buffers(model: StreamModel) -> None:
    """Bring any complex128 tensor down to complex64.

    Metal has no complex128 kernels, and the rotary embedding builds its
    frequency table in double precision. The computation is left alone and only
    the stored table is narrowed, so the frequencies keep the precision they were
    computed at. This is the one upstream incompatibility that has to be handled;
    everything else on the inference path runs as shipped.
    """
    import torch

    for name, submodule in model.named_modules():
        for attribute, value in vars(submodule).items():
            if isinstance(value, torch.Tensor) and value.dtype == torch.complex128:
                setattr(submodule, attribute, value.to(torch.complex64))
                logger.debug("narrowed %s.%s to complex64 for Metal", name, attribute)


def _load_images(image_paths: list[Path], image_size: int) -> torch.Tensor:
    """Preprocess with the model's own loader, so nothing drifts from training."""
    try:
        from lingbot_map.utils.load_fn import load_and_preprocess_images
    except ImportError as error:
        raise PoseBackendError(INSTALL_HINT) from error

    return cast(
        "torch.Tensor",
        load_and_preprocess_images(
            [str(path) for path in image_paths],
            mode="crop",
            image_size=image_size,
            patch_size=PATCH_SIZE,
        ),
    )


def _infer(
    model: StreamModel,
    images: torch.Tensor,
    config: LingbotConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Stream the keyframes through the model.

    Deliberately outside any autocast context. The trunk is already cast and its
    input is cast to match, which leaves the heads in single precision without
    depending on an autocast device string that upstream hardcodes to CUDA.
    """
    import torch

    dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float32
    with torch.no_grad():
        predictions = model.inference_streaming(
            images.to(device=device, dtype=dtype),
            num_scale_frames=config.num_scale_frames,
            keyframe_interval=1,
            output_device=torch.device("cpu"),
        )
    # The Metal allocator holds on to freed blocks, and it is that footprint the
    # OS pages against rather than the live tensors.
    if device.type == "mps":
        torch.mps.empty_cache()
    return predictions


def _collect_estimate(
    predictions: dict[str, torch.Tensor],
    image_paths: list[Path],
    keyframe_indices: list[int],
    config: LingbotConfig,
) -> PoseEstimate:
    try:
        from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
    except ImportError as error:
        raise PoseBackendError(INSTALL_HINT) from error

    model_height, model_width = (int(size) for size in predictions["depth"].shape[2:4])
    decoded = pose_encoding_to_extri_intri(
        predictions["pose_enc"].float().cpu(), (model_height, model_width)
    )
    extrinsics: NDArray[np.float64] = decoded[0][0].double().numpy()
    intrinsics: NDArray[np.float64] = decoded[1][0].double().numpy()
    if extrinsics.shape[0] != len(keyframe_indices):
        raise PoseBackendError(
            f"model returned {extrinsics.shape[0]} poses for {len(keyframe_indices)} keyframes"
        )

    sizes = _keyframe_sizes(image_paths)
    cameras: list[Camera] = []
    for position, keyframe_index in enumerate(keyframe_indices):
        width, height = sizes[position]
        focal_x, focal_y, centre_x, centre_y = intrinsics_to_keyframe(
            intrinsics[position], width=width, height=height, image_size=config.image_size
        )
        # One camera per keyframe: the field of view is predicted per frame, so
        # there is no single intrinsic set to share the way COLMAP shares one.
        cameras.append(
            Camera(
                camera_id=keyframe_index,
                model="PINHOLE",
                width=width,
                height=height,
                fx=focal_x,
                fy=focal_y,
                cx=centre_x,
                cy=centre_y,
            )
        )

    poses = poses_from_extrinsics(extrinsics, keyframe_indices)
    points, colors = _cloud_from_depth(predictions, intrinsics, poses, config)
    return PoseEstimate(
        cameras=cameras,
        poses=poses,
        points=points,
        colors=colors,
        # The model minimises nothing comparable to a reprojection error, and a
        # placeholder would put a meaningless number in a results table.
        mean_reprojection_error_px=None,
    )


def poses_from_extrinsics(
    extrinsics: NDArray[np.float64], keyframe_indices: list[int]
) -> list[CameraPose]:
    """Turn the model's `[R|t]` into the project's stored pose.

    The model's extrinsics are **camera-to-world**: `t` is where the camera sits,
    and `R` turns camera axes into world axes. This project stores the opposite
    direction, so each pose is inverted here.

    Upstream's own prose disagrees with itself about this - `pose_enc.py` calls
    the output "camera from world", and `demo.py` then inverts it again to get
    what it labels c2w - so it was settled by measurement instead. On both
    Mip-NeRF 360 garden and Tanks and Temples truck, each against the COLMAP
    reconstruction the dataset ships, inverting takes the mean rotation error
    from 173 degrees to 1.1 and the position error from 6.4% of scene extent to
    0.8%. Reading it the other way is not a small error; it points every camera
    backwards.
    """
    poses: list[CameraPose] = []
    for position, keyframe_index in enumerate(keyframe_indices):
        camera_to_world = matrix_to_quaternion(extrinsics[position, :3, :3])
        quaternion, translation = invert_pose(camera_to_world, extrinsics[position, :3, 3])
        poses.append(
            CameraPose(
                keyframe_index=keyframe_index,
                camera_id=keyframe_index,
                quaternion_wxyz=cast(
                    "tuple[float, float, float, float]", tuple(quaternion.tolist())
                ),
                translation=cast("tuple[float, float, float]", tuple(translation.tolist())),
            )
        )
    return poses


def _keyframe_sizes(image_paths: list[Path]) -> list[tuple[int, int]]:
    """Read each keyframe's size without decoding it."""
    try:
        from PIL import Image, ImageOps
    except ImportError as error:
        raise PoseBackendError(INSTALL_HINT) from error

    sizes: list[tuple[int, int]] = []
    for path in image_paths:
        with Image.open(path) as handle:
            # The model's loader applies this before resizing, so the size the
            # intrinsic conversion works from has to account for it too.
            oriented = ImageOps.exif_transpose(handle)
            sizes.append((int(oriented.width), int(oriented.height)))
    return sizes


def _cloud_from_depth(
    predictions: dict[str, torch.Tensor],
    intrinsics: NDArray[np.float64],
    poses: list[CameraPose],
    config: LingbotConfig,
) -> tuple[Points | None, Colors | None]:
    """Back-project the depth maps into the cloud stage [4] initialises from.

    Thinned per frame rather than over the whole sequence. Together the dense
    maps are tens of millions of points, and an even share per frame also spreads
    the cloud over the whole take instead of over whichever frames happened to
    come out most confident.
    """
    depth = predictions["depth"].float().numpy()[0, ..., 0].astype(np.float64)
    confidence = predictions["depth_conf"].float().numpy()[0].astype(np.float64)
    images = predictions["images"].float().numpy()[0]

    frame_count = int(depth.shape[0])
    budget = max(1, config.max_points // frame_count)

    chunks: list[Points] = []
    tints: list[Colors] = []
    for index in range(frame_count):
        keep = select_confident(
            confidence[index], percentile=config.depth_confidence_percentile, budget=budget
        )
        if keep.size == 0:
            continue
        # The stored pose, not the raw one: `backproject` expects the
        # world-to-camera direction this project keeps.
        world = backproject(
            depth[index],
            intrinsics[index],
            np.asarray(poses[index].quaternion_wxyz, dtype=np.float64),
            np.asarray(poses[index].translation, dtype=np.float64),
        )
        colors = (images[index].transpose(1, 2, 0).reshape(-1, 3) * 255.0).clip(0, 255)
        chunks.append(world[keep])
        tints.append(colors[keep].astype(np.uint8))

    if not chunks:
        return None, None
    return np.concatenate(chunks, axis=0), np.concatenate(tints, axis=0)
