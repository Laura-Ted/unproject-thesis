"""COLMAP pose estimation, the reference the learned backend is measured against."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, cast, runtime_checkable

import numpy as np
import pycolmap
from numpy.typing import NDArray

from unproject.config import ColmapConfig, config_hash
from unproject.ingest.types import SourceKind
from unproject.pose.artifacts import BackendInfo, Camera, CameraPose
from unproject.pose.backend import PoseBackendError, PoseEstimate
from unproject.pose.geometry import normalize_quaternion
from unproject.pose.pointcloud import Colors, Points

logger = logging.getLogger(__name__)

DATABASE_DIRNAME = "colmap"


# pycolmap is a pybind11 extension with no stubs. These describe the surface we
# actually read, so the backend type-checks and the dependency is documented.
# `test_pose_colmap.py` asserts the real classes still satisfy them.
@runtime_checkable
class ColmapRotation(Protocol):
    @property
    def quat(self) -> NDArray[np.float64]:
        """Eigen coefficient order, which is (x, y, z, w)."""


@runtime_checkable
class ColmapRigid3d(Protocol):
    @property
    def rotation(self) -> ColmapRotation: ...

    @property
    def translation(self) -> NDArray[np.float64]: ...


@runtime_checkable
class ColmapCamera(Protocol):
    @property
    def camera_id(self) -> int: ...

    @property
    def model_name(self) -> str: ...

    @property
    def width(self) -> int: ...

    @property
    def height(self) -> int: ...

    @property
    def focal_length_x(self) -> float: ...

    @property
    def focal_length_y(self) -> float: ...

    @property
    def principal_point_x(self) -> float: ...

    @property
    def principal_point_y(self) -> float: ...

    @property
    def params(self) -> NDArray[np.float64]: ...

    def extra_params_idxs(self) -> Sequence[int]: ...


@runtime_checkable
class ColmapImage(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def camera_id(self) -> int: ...

    @property
    def has_pose(self) -> bool: ...

    def cam_from_world(self) -> ColmapRigid3d:
        """A method, unlike the other members here, which are properties."""


@runtime_checkable
class ColmapPoint3D(Protocol):
    @property
    def xyz(self) -> NDArray[np.float64]: ...

    @property
    def color(self) -> NDArray[np.uint8]: ...


class ColmapReconstruction(Protocol):
    @property
    def cameras(self) -> Mapping[int, ColmapCamera]: ...

    @property
    def images(self) -> Mapping[int, ColmapImage]: ...

    @property
    def points3D(self) -> Mapping[int, ColmapPoint3D]:  # noqa: N802 - matches pycolmap
        ...

    def compute_mean_reprojection_error(self) -> float: ...


class ColmapBackend:
    """Runs COLMAP's incremental pipeline over the keyframes.

    Feature extraction and matching run on CPU; there is no CUDA on the target
    machine, so this is minutes rather than seconds on a real scene.
    """

    def __init__(self, config: ColmapConfig, *, work_dir: Path, source_kind: SourceKind) -> None:
        self._config = config
        self._work_dir = work_dir
        self._source_kind = source_kind

    def describe(self) -> BackendInfo:
        return BackendInfo(
            name="colmap",
            version=str(pycolmap.__version__),
            config_hash=config_hash(self._config),
        )

    def estimate(self, image_paths: list[Path], keyframe_indices: list[int]) -> PoseEstimate:
        if len(image_paths) != len(keyframe_indices):
            raise PoseBackendError(
                f"{len(image_paths)} images against {len(keyframe_indices)} keyframe indices"
            )
        if not image_paths:
            raise PoseBackendError("no keyframes to estimate poses for")

        image_dir = image_paths[0].parent
        if any(path.parent != image_dir for path in image_paths):
            raise PoseBackendError("COLMAP needs every keyframe under one directory")

        by_name = {
            path.name: index for path, index in zip(image_paths, keyframe_indices, strict=True)
        }
        reconstruction = self._reconstruct(image_dir, sorted(by_name))
        return _collect_estimate(reconstruction, by_name)

    def _reconstruct(self, image_dir: Path, image_names: list[str]) -> ColmapReconstruction:
        colmap_dir = self._work_dir / DATABASE_DIRNAME
        colmap_dir.mkdir(parents=True, exist_ok=True)
        database = colmap_dir / "database.db"
        # A database left over from an interrupted run would be matched against
        # rather than rebuilt, and the result would depend on run history.
        database.unlink(missing_ok=True)

        logger.info("extracting features from %d keyframes on CPU", len(image_names))
        pycolmap.extract_features(
            database_path=database,
            image_path=image_dir,
            image_names=image_names,
            camera_mode=(
                pycolmap.CameraMode.SINGLE
                if self._config.single_camera
                else pycolmap.CameraMode.AUTO
            ),
            reader_options=pycolmap.ImageReaderOptions(camera_model=self._config.camera_model),
            device=pycolmap.Device.cpu,
        )

        matcher = self.resolve_matcher()
        logger.info("matching with the %s strategy", matcher)
        if matcher == "sequential":
            pycolmap.match_sequential(database_path=database, device=pycolmap.Device.cpu)
        else:
            pycolmap.match_exhaustive(database_path=database, device=pycolmap.Device.cpu)

        output = colmap_dir / "sparse"
        output.mkdir(parents=True, exist_ok=True)
        reconstructions = pycolmap.incremental_mapping(
            database_path=database, image_path=image_dir, output_path=output
        )
        if not reconstructions:
            raise PoseBackendError(
                "COLMAP registered no images; the scene may have too little texture, "
                "too little camera motion, or keyframes too far apart to match"
            )

        # Disconnected pieces of a scene come back as separate reconstructions.
        # The largest is the one to report; the others are counted as unregistered
        # rather than merged, since merging would need a scale we do not have.
        largest = max(reconstructions.values(), key=lambda model: len(model.images))
        if len(reconstructions) > 1:
            logger.warning(
                "COLMAP produced %d disconnected reconstructions; keeping the largest (%d images)",
                len(reconstructions),
                len(largest.images),
            )
        return cast(ColmapReconstruction, largest)

    def resolve_matcher(self) -> str:
        if self._config.matcher != "auto":
            return self._config.matcher
        # Video frames arrive in order, so neighbours are the likely matches and
        # sequential matching is both faster and better conditioned. An image
        # directory carries no such promise.
        return "sequential" if self._source_kind == "video" else "exhaustive"


def _collect_estimate(
    reconstruction: ColmapReconstruction, by_name: dict[str, int]
) -> PoseEstimate:
    cameras = [
        Camera(
            camera_id=int(camera.camera_id),
            model=str(camera.model_name),
            width=int(camera.width),
            height=int(camera.height),
            fx=float(camera.focal_length_x),
            fy=float(camera.focal_length_y),
            cx=float(camera.principal_point_x),
            cy=float(camera.principal_point_y),
            distortion=[float(camera.params[index]) for index in camera.extra_params_idxs()],
        )
        for camera in reconstruction.cameras.values()
    ]

    poses: list[CameraPose] = []
    for image in reconstruction.images.values():
        if not image.has_pose:
            continue
        keyframe_index = by_name.get(str(image.name))
        if keyframe_index is None:
            logger.warning("COLMAP returned an unexpected image name %r", image.name)
            continue
        quaternion, translation = _pose_of(image)
        poses.append(
            CameraPose(
                keyframe_index=keyframe_index,
                camera_id=int(image.camera_id),
                quaternion_wxyz=quaternion,
                translation=translation,
            )
        )
    poses.sort(key=lambda pose: pose.keyframe_index)

    points, colors = _points_of(reconstruction)
    return PoseEstimate(
        cameras=cameras,
        poses=poses,
        points=points,
        colors=colors,
        mean_reprojection_error_px=float(reconstruction.compute_mean_reprojection_error()),
    )


def _pose_of(
    image: ColmapImage,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float]]:
    """Read one image's world-to-camera pose into the project's ordering.

    `cam_from_world` is already world-to-camera, so no inversion is needed, but
    Eigen stores the quaternion as (x, y, z, w) and this project stores
    (w, x, y, z). Getting that reordering wrong yields a valid-looking rotation.
    """
    rigid = image.cam_from_world()
    x, y, z, w = (float(value) for value in rigid.rotation.quat)
    quaternion = normalize_quaternion(np.array([w, x, y, z]))
    translation = np.asarray(rigid.translation, dtype=np.float64).reshape(3)
    return (
        cast(tuple[float, float, float, float], tuple(quaternion.tolist())),
        cast(tuple[float, float, float], tuple(translation.tolist())),
    )


def _points_of(reconstruction: ColmapReconstruction) -> tuple[Points | None, Colors | None]:
    entries = list(reconstruction.points3D.values())
    if not entries:
        return None, None
    positions = np.array([point.xyz for point in entries], dtype=np.float64)
    colors = np.array([point.color for point in entries], dtype=np.uint8)
    return positions, colors
