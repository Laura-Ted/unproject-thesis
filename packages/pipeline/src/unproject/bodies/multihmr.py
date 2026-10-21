"""Body estimation with Multi-HMR 2.

Everything torch touches is imported inside the function that needs it: torch
and pycolmap each bundle libomp and cannot share a process, and the CLI reaches
both, so this module has to stay importable without torch (docs/reproduce.md §6).

The model is single shot, so every person in a frame comes out of one forward
pass and there is no crop to manage. It predicts metric camera-space positions
and its own field of view.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import math
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from unproject.bodies.artifacts import MESH_INFLUENCES, BodyModelInfo, EstimatorInfo
from unproject.bodies.device import DeviceError, select_device
from unproject.config import BodiesConfig, KnownIntrinsics, config_hash

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# Fixed by the checkpoint: the image is resized so its longest side is this, then
# padded to a whole number of patches. Not a setting, so not in the config.
IMAGE_SIZE = 768
PATCH_SIZE = 16

INSTALL_HINT = "the bodies stage needs its extra: uv sync --extra multihmr"

# Length of the shape vector the head predicts. Fixed by the checkpoint.
_SHAPE_COUNT = 11

FloatArray = NDArray[np.float32]

# A frame to estimate: a file on disk, or RGB pixels already in memory.
type FrameInput = Path | NDArray[np.uint8]


class BodyEstimationError(Exception):
    """Raised when the estimator cannot produce parameters for a frame."""


@dataclass(frozen=True, slots=True)
class PersonEstimate:
    """One person in one frame, in the project's convention.

    Plain dataclass rather than a model: it carries numpy arrays and is consumed
    immediately by the stage, which is what validates and writes.
    """

    track_id: int
    confidence: float
    box_xyxy: tuple[float, float, float, float]
    # (bones, 4, 4), as the model emits them. Not reduced to quaternions: a 4x4
    # carries a translation per bone and it is not established that dropping it
    # is safe.
    bone_poses: FloatArray
    rest_bone_poses: FloatArray
    shape: FloatArray
    root_translation: FloatArray
    joints_3d: FloatArray


@dataclass(frozen=True, slots=True)
class Topology:
    """The body's shape in the sense a renderer means it.

    Constant across every person and every frame, so it is fetched once. The
    skinning is cut to the four strongest bones per vertex, which is what a GPU
    skins and what three.js accepts; `reduce_influences` has what that costs.
    """

    faces: NDArray[np.uint16]
    skin_indices: NDArray[np.uint16]
    skin_weights: FloatArray
    # Which of the model's vertices survived, so a rest pose can be cut to
    # match. Anny keeps vertices no triangle references, and they wander: on
    # this clip twelve of them moved by up to two metres while everything drawn
    # stayed within one. Nothing renders them, so they are dropped.
    kept_vertices: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class FrameEstimate:
    """Everything found in one frame, with the camera it was found against."""

    # fx, fy, cx, cy at the *source* frame's resolution, not the model's.
    intrinsics: tuple[float, float, float, float]
    people: list[PersonEstimate]


# multihmr2 ships no stubs and is installed only with the extra, so these
# describe the surface this module touches.
class PersonOutputLike(Protocol):
    """The slice of `multihmr2.models...decoder.PersonOutput` used here."""

    num_person: int
    conf: torch.Tensor
    shape: torch.Tensor
    j3d: torch.Tensor
    j2d: torch.Tensor
    transl_pelvis: torch.Tensor
    bone_poses: torch.Tensor
    rest_bone_poses: torch.Tensor
    trackfeat: torch.Tensor | None


class DecoderOutputLike(Protocol):
    """The slice of `multihmr2.models...decoder.DecoderOutput` used here."""

    K: torch.Tensor
    persons: PersonOutputLike


class BodyModelLike(Protocol):
    """The slice of `anny.Anny` used here."""

    bone_count: int
    bone_parents: list[int]
    # Which bones pull on each vertex, and how hard. (vertices, influences).
    vertex_bone_weights: torch.Tensor
    vertex_bone_indices: torch.Tensor
    # Anny's shape space is interpretable: gender, age, muscle, weight, height,
    # proportions. The estimator predicts these, not PCA coefficients.
    phenotype_labels: list[str]

    def get_triangular_faces(self) -> torch.Tensor: ...

    def get_phenotype_blendshape_coefficients(self, **phenotypes: torch.Tensor) -> torch.Tensor: ...

    def get_rest_model(self, blendshape_coeffs: torch.Tensor) -> dict[str, torch.Tensor]: ...


class ModelLike(Protocol):
    """The slice of upstream's `MultiHMR` module used here.

    `forward` is `encoder(x)` then `full_body_decoder(z, K=z.K, ...)`, with the
    camera passed between the two as an argument. That seam is where a known
    camera goes in.
    """

    @property
    def encoder(self) -> Callable[[torch.Tensor], EncoderOutputLike]: ...

    @property
    def full_body_decoder(self) -> Callable[..., list[DecoderOutputLike]]: ...


class EncoderOutputLike(Protocol):
    K: torch.Tensor
    feat: torch.Tensor


class SessionLike(Protocol):
    """The slice of `multihmr2.api.InferenceSession` used here."""

    img_size: int
    body_model_world: BodyModelLike
    model: ModelLike

    def __call__(
        self,
        img: NDArray[np.uint8] | str,
        conf_thresh: float,
        dist_thresh_nms: float,
    ) -> DecoderOutputLike: ...


class TrackerLike(Protocol):
    """The slice of `multihmr2.tracker.FeatPelvisTracker` used here."""

    def reset(self) -> None: ...

    def track_next_frame(
        self,
        time: int,
        trackfeat: torch.Tensor,
        pelvis_xyz: torch.Tensor,
        pelvis_ijn: torch.Tensor,
        pelvis_ori: torch.Tensor,
    ) -> torch.Tensor: ...


def model_space_geometry(width: int, height: int) -> tuple[float, int, int]:
    """How a source frame is mapped into the model's input space.

    Returns the scale and the near-side padding. Upstream resizes so the longest
    side is `IMAGE_SIZE`, then centre-pads to a whole number of patches, so
    undoing it needs the padding as well as the scale.

    Mirrors `nonsquare_and_resize_image_without_K` including its rounding: half a
    pixel here is a wrong principal point in the artifact.
    """
    if width <= 0 or height <= 0:
        raise BodyEstimationError(f"frame has a degenerate size {width}x{height}")
    scale = min(IMAGE_SIZE / width, IMAGE_SIZE / height)
    resized_width = round(width * scale)
    resized_height = round(height * scale)
    padded_width = math.ceil(resized_width / PATCH_SIZE) * PATCH_SIZE
    padded_height = math.ceil(resized_height / PATCH_SIZE) * PATCH_SIZE
    return scale, (padded_width - resized_width) // 2, (padded_height - resized_height) // 2


def model_input_size(width: int, height: int) -> tuple[int, int, float]:
    """The tensor the model sees for a source frame: padded width, padded
    height, and the source-to-model scale.

    What a benchmark should record beside the source size (ADR-0005 WP2):
    two sources of different resolution can reach the model as the same
    tensor, and two of the same resolution but different shape as different
    ones, so the cost is decided here.
    """
    scale, _, _ = model_space_geometry(width, height)
    padded_width = math.ceil(round(width * scale) / PATCH_SIZE) * PATCH_SIZE
    padded_height = math.ceil(round(height * scale) / PATCH_SIZE) * PATCH_SIZE
    return padded_width, padded_height, scale


def intrinsics_to_source(
    intrinsics: NDArray[np.float64], *, width: int, height: int
) -> tuple[float, float, float, float]:
    """Re-express the model's 3x3 K at the source frame's own resolution.

    A focal length measured in the model's padded 768-pixel space is not
    comparable with one measured on the frame the operator supplied.
    """
    scale, pad_left, pad_top = model_space_geometry(width, height)
    return (
        float(intrinsics[0, 0]) / scale,
        float(intrinsics[1, 1]) / scale,
        (float(intrinsics[0, 2]) - pad_left) / scale,
        (float(intrinsics[1, 2]) - pad_top) / scale,
    )


def intrinsics_to_model(known: KnownIntrinsics) -> NDArray[np.float64]:
    """Express a source-resolution camera in the model's padded input space.

    The exact inverse of `intrinsics_to_source`, and pinned to it by a test: the
    camera the estimator is told and the camera the artifact records must be
    the same numbers.
    """
    scale, pad_left, pad_top = model_space_geometry(known.width, known.height)
    return np.array(
        [
            [known.fx * scale, 0.0, known.cx * scale + pad_left],
            [0.0, known.fy * scale, known.cy * scale + pad_top],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def points_to_source(
    points: NDArray[np.float64], *, width: int, height: int
) -> NDArray[np.float64]:
    """Map model-space 2D points back onto the source frame."""
    scale, pad_left, pad_top = model_space_geometry(width, height)
    mapped = points.copy()
    mapped[..., 0] = (mapped[..., 0] - pad_left) / scale
    mapped[..., 1] = (mapped[..., 1] - pad_top) / scale
    return mapped


def reduce_influences(
    weights: FloatArray, indices: NDArray[np.int64], keep: int = MESH_INFLUENCES
) -> tuple[FloatArray, NDArray[np.uint16]]:
    """Keep the strongest few bones per vertex, renormalised.

    Anny lets twelve bones pull on a vertex and a GPU skins four. Renormalising
    matters more than the choice: a vertex that lost half its weight and was not
    renormalised would collapse halfway to the origin rather than deform a
    little.

    Measured against the model's own mesh, four influences move the median
    vertex by a thousandth of a millimetre and the worst by 30 mm, on a body
    1700 mm tall.
    """
    if weights.shape != indices.shape:
        raise BodyEstimationError(f"{weights.shape} weights against {indices.shape} indices")
    order = np.argsort(-weights, axis=1)[:, :keep]
    kept = np.take_along_axis(weights, order, axis=1)
    # A vertex with no influence at all would divide by zero; it stays at rest.
    total = np.maximum(kept.sum(axis=1, keepdims=True), 1e-8)
    return (
        (kept / total).astype(np.float32),
        np.take_along_axis(indices, order, axis=1).astype(np.uint16),
    )


def drop_unused_vertices(
    faces: NDArray[np.int64], vertex_count: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Keep only vertices some triangle refers to, and reindex the faces.

    Returns the surviving vertex indices and the faces rewritten against them.
    """
    kept = np.unique(faces)
    if kept.size == 0:
        raise BodyEstimationError("the body has no faces")
    if int(kept.max()) >= vertex_count:
        raise BodyEstimationError(f"a face refers to vertex {int(kept.max())} of {vertex_count}")
    lookup = np.full(vertex_count, -1, dtype=np.int64)
    lookup[kept] = np.arange(kept.size, dtype=np.int64)
    return kept, lookup[faces]


def box_from_joints(
    joints_2d: NDArray[np.float64], *, width: int, height: int
) -> tuple[float, float, float, float]:
    """Bounding box of a person, from their projected joints.

    There is no detector crop to report: the model is single shot and finds
    people as peaks in a heatmap. Clipped, since a joint off the edge still
    projects somewhere.
    """
    if joints_2d.size == 0:
        raise BodyEstimationError("cannot derive a box from an empty joint set")
    x_min = float(np.clip(joints_2d[:, 0].min(), 0.0, width))
    y_min = float(np.clip(joints_2d[:, 1].min(), 0.0, height))
    x_max = float(np.clip(joints_2d[:, 0].max(), 0.0, width))
    y_max = float(np.clip(joints_2d[:, 1].max(), 0.0, height))
    return x_min, y_min, x_max, y_max


class EstimatorSession:
    """A loaded model and its tracker, following one continuous sequence.

    The tracker carries identities from one frame to the next, so a session is
    tied to a single sequence: frames must arrive in the order they were shot,
    and a new sequence needs a new session.
    """

    def __init__(
        self,
        session: SessionLike,
        tracker: TrackerLike,
        device: torch.device,
        body_model: BodyModelInfo,
    ) -> None:
        self._session = session
        self._tracker = tracker
        self._device = device
        self._body_model = body_model
        self._image_size = float(session.img_size)
        self._seen = 0
        self._topology: Topology | None = None

    @property
    def device(self) -> torch.device:
        """Where the model lives. A clock around `frame` is a clock on this."""
        return self._device

    @property
    def body_model(self) -> BodyModelInfo:
        return self._body_model

    def topology(self) -> Topology:
        """Faces and skinning, reduced to what a GPU can use and what is drawn.

        Kept once built: it is the same for every person and every frame, and
        rebuilding it (copying the weights off the GPU and sorting them) takes
        about three seconds.
        """
        if self._topology is not None:
            return self._topology
        body = self._session.body_model_world
        weights, indices = reduce_influences(
            body.vertex_bone_weights.detach().float().cpu().numpy(),
            body.vertex_bone_indices.detach().cpu().numpy(),
        )
        faces = body.get_triangular_faces().detach().cpu().numpy().astype(np.int64)
        kept, remapped = drop_unused_vertices(faces, len(weights))
        self._topology = Topology(
            faces=remapped.astype(np.uint16),
            skin_indices=indices[kept],
            skin_weights=weights[kept],
            kept_vertices=kept,
        )
        return self._topology

    def rest_bone_poses(self, shape: FloatArray) -> FloatArray:
        """The bind pose for one person's shape.

        Paired with `rest_vertices` from the same shape. Mixing a rest mesh from
        one shape with bind bones from another is what turns a 1 mm error into a
        15 mm one; they belong together.
        """
        rest = self._rest_model(shape)["rest_bone_poses"]
        return cast(FloatArray, rest.squeeze(0).detach().float().cpu().numpy())

    def rest_vertices(self, shape: FloatArray) -> FloatArray:
        """The unposed body for one person's shape.

        Two public calls and no skinning: the posed vertices would be thrown
        away, and the path that computes them runs through NVIDIA Warp, which
        has no Metal backend.
        """
        rest = self._rest_model(shape)["rest_vertices"]
        return cast(FloatArray, rest.squeeze(0).detach().float().cpu().numpy())

    def _rest_model(self, shape: FloatArray) -> dict[str, torch.Tensor]:
        import torch

        body = self._session.body_model_world
        device = body.vertex_bone_weights.device
        row = torch.as_tensor(shape, dtype=torch.float32, device=device).reshape(1, -1)
        named = {label: row[:, index] for index, label in enumerate(body.phenotype_labels)}
        return body.get_rest_model(body.get_phenotype_blendshape_coefficients(**named))

    def frame(
        self, image: FrameInput, config: BodiesConfig, *, width: int, height: int
    ) -> FrameEstimate:
        """Estimate one frame, advancing the tracker.

        `image` is a path or an RGB array. Arrays go through upstream's own
        preprocessing, so the live path and the offline path cannot drift.
        """
        known = config.intrinsics
        if known is not None and (width, height) != (known.width, known.height):
            raise BodyEstimationError(
                f"the configured camera is for {known.width}x{known.height} frames and "
                f"this one is {width}x{height}. A focal length is only meaningful at "
                "the resolution it was measured at; the frames or the camera are wrong."
            )
        prediction = _infer(self._session, image, config)
        estimate = _collect_frame(
            prediction,
            tracker=self._tracker,
            time_index=self._seen,
            width=width,
            height=height,
            image_size=self._image_size,
        )
        self._seen += 1
        # The Metal allocator does not hand freed blocks back, and it is the
        # driver figure the OS pages against.
        if self._device.type == "mps" and self._seen % 50 == 0:
            import torch

            torch.mps.empty_cache()
        return estimate


class MultiHmrEstimator:
    """Runs Multi-HMR 2 over a sequence of frames, in order."""

    def __init__(self, config: BodiesConfig, *, config_dir: Path) -> None:
        self._config = config
        self._config_dir = config_dir
        self._body_model: BodyModelInfo | None = None
        self._loaded: SessionLike | None = None

    def describe(self) -> EstimatorInfo:
        return EstimatorInfo(
            name="multihmr2",
            checkpoint_sha256=self._config.sha256,
            backbone_repo=self._config.backbone_repo,
            backbone_ref=self._config.backbone_ref,
            config_hash=config_hash(self._config),
        )

    def body_model(self) -> BodyModelInfo:
        """The skeleton the parameters drive, read off the model that produced them.

        Only known once the model is built, so this is valid after `open` (which
        `estimate` calls first) rather than before.
        """
        if self._body_model is None:
            raise BodyEstimationError("the body model is only known once the estimator has run")
        return self._body_model

    def open(self) -> EstimatorSession:
        """Return a session for one sequence, loading the model if needed.

        The model is kept and reused; the tracker is not. Loading costs about
        twenty seconds cold and four warm, while a tracker holds identities and
        so belongs to exactly one sequence.
        """
        _require_extra()
        device = _select_device()
        if self._loaded is None:
            self._loaded = _build_session(self._config, device, self._config_dir)
            self._body_model = _describe_body_model(self._loaded, shape_count=_SHAPE_COUNT)
        assert self._body_model is not None
        return EstimatorSession(self._loaded, _build_tracker(), device, self._body_model)

    def estimate(self, frame_paths: Sequence[Path]) -> Iterator[FrameEstimate]:
        """Yield one estimate per frame, in order.

        A generator, not a list: upstream's `infer_video` accumulates every
        frame's output and reaches gigabytes on a clip this length.
        """
        if not frame_paths:
            raise BodyEstimationError("no frames to estimate bodies for")

        opened = self.open()
        logger.info("running %d frames through multi-hmr 2", len(frame_paths))
        for path in frame_paths:
            width, height = _frame_size(path)
            yield opened.frame(path, self._config, width=width, height=height)


def _describe_body_model(session: SessionLike, *, shape_count: int) -> BodyModelInfo:
    from importlib import metadata

    body = session.body_model_world
    try:
        version = metadata.version("anny")
    except metadata.PackageNotFoundError as error:  # pragma: no cover - extra is present
        raise BodyEstimationError(INSTALL_HINT) from error

    return BodyModelInfo(
        name="anny",
        version=version,
        bone_count=int(body.bone_count),
        shape_count=shape_count,
        bone_parents=[int(parent) for parent in body.bone_parents],
    )


def _require_extra() -> None:
    """Fail before anything heavy loads if the extra is not installed.

    A spec lookup rather than an import: the path that reports the extra missing
    must not be the path that loads torch.
    """
    if importlib.util.find_spec("multihmr2") is None:
        raise BodyEstimationError(INSTALL_HINT)


def _select_device() -> torch.device:
    """CUDA, then Metal, then the CPU; or what `UNPROJECT_DEVICE` insists on.

    `bodies.device` decides and this only turns its refusals into the stage's
    own error, so a caller sees one exception type for "cannot estimate".
    """
    try:
        return select_device()
    except ImportError as error:
        raise BodyEstimationError(INSTALL_HINT) from error
    except DeviceError as error:
        raise BodyEstimationError(str(error)) from error


def ensure_checkpoint(config: BodiesConfig, config_dir: Path) -> Path:
    """Fetch the pinned weights if absent and check their digest; the path.

    What the first job of a suite would do anyway, offered ahead of time so a
    preflight can spend the network before the clock rather than during it.
    """
    return _checkpoint_path(config, config_dir)


def checkpoint_status(
    config: BodiesConfig, config_dir: Path
) -> tuple[Literal["ok", "warn", "fail"], str]:
    """Whether the pinned weights are on disk and are the pinned ones.

    Reads and hashes; never fetches. Absent is a warning - the first model
    build fetches them - and a digest that does not match is a failure, since
    the build would refuse them.
    """
    path = (config_dir / config.cache_dir).resolve() / "multihmr2.pt"
    if not path.is_file():
        return "warn", f"{path} absent; fetched from {config.checkpoint_url} at the first build"
    digest = _sha256(path)
    if digest != config.sha256:
        return "fail", f"{path} hashes to {digest[:12]}…, the config pins {config.sha256[:12]}…"
    return "ok", f"{path}: digest matches"


def _checkpoint_path(config: BodiesConfig, config_dir: Path) -> Path:
    """Fetch the pinned weights and check they are the ones the config names.

    Fetched here rather than by upstream, which shells out to `wget`; macOS does
    not ship one.
    """
    cache_dir = (config_dir / config.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Upstream asserts the stem is exactly this before it will load the file.
    path = cache_dir / "multihmr2.pt"

    if not path.is_file():
        logger.info("fetching %s -> %s", config.checkpoint_url, path)
        _download(config.checkpoint_url, path)

    digest = _sha256(path)
    if digest != config.sha256:
        raise BodyEstimationError(f"{path} hashes to {digest}, but the config pins {config.sha256}")
    return path


def _download(url: str, destination: Path) -> None:
    """Stream a file to disk, leaving nothing behind if it fails part way."""
    import urllib.request

    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(url) as response, partial.open("wb") as handle:
            while block := response.read(1 << 22):
                handle.write(block)
    except OSError as error:
        partial.unlink(missing_ok=True)
        raise BodyEstimationError(f"could not fetch {url}: {error}") from error
    partial.replace(destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_session(config: BodiesConfig, device: torch.device, config_dir: Path) -> SessionLike:
    """Load the model onto the chosen device.

    `init_hmr_session` is not used: it picks CUDA or CPU and would never choose
    Metal, silently costing about an order of magnitude; the choice is made in
    `bodies.device` and handed in.
    """
    try:
        from multihmr2.api import InferenceSession
    except ImportError as error:
        raise BodyEstimationError(INSTALL_HINT) from error

    checkpoint = _checkpoint_path(config, config_dir)
    with _pinned_backbone(config):
        built = InferenceSession(checkpoint, device=device)
    session = cast(SessionLike, built)
    if config.intrinsics is not None:
        session.model = _KnownCameraModel(session.model, intrinsics_to_model(config.intrinsics))
    return session


class _KnownCameraModel:
    """Upstream's model with the camera it predicts replaced by one it is told.

    The encoder predicts a field of view from the image and the decoder turns
    it into a focal length that divides every apparent size into a distance.
    Nothing else in the network reads the camera, so substituting it here
    changes exactly one thing: where the bodies are put, not what they look
    like. The output's `K` is the substituted one, so the artifact records the
    camera that was actually used.
    """

    def __init__(self, model: ModelLike, intrinsics: NDArray[np.float64]) -> None:
        self._model = model
        self._intrinsics = intrinsics

    def __call__(self, x: torch.Tensor, **kwargs: object) -> list[DecoderOutputLike]:
        import torch

        encoded = self._model.encoder(x)
        told = torch.as_tensor(self._intrinsics, dtype=encoded.K.dtype, device=encoded.K.device)
        return self._model.full_body_decoder(encoded, K=told.expand_as(encoded.K), **kwargs)

    @property
    def encoder(self) -> Callable[[torch.Tensor], EncoderOutputLike]:
        return self._model.encoder

    @property
    def full_body_decoder(self) -> Callable[..., list[DecoderOutputLike]]:
        return self._model.full_body_decoder


@contextmanager
def _pinned_backbone(config: BodiesConfig) -> Iterator[None]:
    """Hold torch.hub to a fixed commit of the backbone repository.

    The encoder calls `torch.hub.load('facebookresearch/dinov3', ...)`, which
    fetches source from a branch and executes it: unpinned code, even though the
    weights come from the checkpoint. The repository is named inside upstream's
    encoder rather than passed in, so redirecting the call is the only lever.
    Put back afterwards so nothing else inherits it.
    """
    try:
        import torch
    except ImportError as error:
        raise BodyEstimationError(INSTALL_HINT) from error

    pinned = f"{config.backbone_repo}:{config.backbone_ref}"
    original: Callable[..., object] = torch.hub.load

    def load_pinned(repo_or_dir: str, *args: object, **kwargs: object) -> object:
        # Only the repository the encoder asks for; anything else is left alone.
        target = pinned if repo_or_dir == config.backbone_repo else repo_or_dir
        return original(target, *args, **kwargs)

    torch.hub.load = load_pinned
    try:
        yield
    finally:
        torch.hub.load = original


def _build_tracker() -> TrackerLike:
    try:
        from multihmr2.tracker import FeatPelvisTracker
    except ImportError as error:
        raise BodyEstimationError(INSTALL_HINT) from error

    tracker = FeatPelvisTracker()
    tracker.reset()
    return cast(TrackerLike, tracker)


def _infer(session: SessionLike, image: FrameInput, config: BodiesConfig) -> DecoderOutputLike:
    # Upstream takes a path or an RGB array and preprocesses either the same
    # way, so nothing here reimplements the resize the model was trained with.
    return session(
        str(image) if isinstance(image, Path) else image,
        conf_thresh=config.confidence_threshold,
        dist_thresh_nms=config.nms_distance_metres,
    )


def _frame_size(path: Path) -> tuple[int, int]:
    """Read a frame's size without decoding it."""
    try:
        from PIL import Image
    except ImportError as error:
        raise BodyEstimationError(INSTALL_HINT) from error

    with Image.open(path) as handle:
        return int(handle.width), int(handle.height)


def _collect_frame(
    prediction: DecoderOutputLike,
    *,
    tracker: TrackerLike,
    time_index: int,
    width: int,
    height: int,
    image_size: float,
) -> FrameEstimate:
    """Turn one prediction into the stage's own types, and assign identities."""
    persons = prediction.persons
    intrinsics = intrinsics_to_source(
        prediction.K.detach().float().cpu().double().numpy(), width=width, height=height
    )

    if persons.num_person == 0:
        return FrameEstimate(intrinsics=intrinsics, people=[])

    track_ids = _assign_tracks(
        persons, tracker=tracker, time_index=time_index, image_size=image_size
    )

    confidences = persons.conf.detach().float().cpu().numpy().reshape(-1)
    shapes = persons.shape.detach().float().cpu().numpy()
    translations = persons.transl_pelvis.detach().float().cpu().numpy()
    bone_poses = persons.bone_poses.detach().float().cpu().numpy()
    rest_bone_poses = persons.rest_bone_poses.detach().float().cpu().numpy()
    joints_3d = persons.j3d.detach().float().cpu().numpy()
    joints_2d = points_to_source(
        persons.j2d.detach().float().cpu().double().numpy(), width=width, height=height
    )

    people = [
        PersonEstimate(
            track_id=int(track_ids[position]),
            # Clipped: rounding can put the reported probability just over one.
            confidence=float(np.clip(confidences[position], 0.0, 1.0)),
            box_xyxy=box_from_joints(joints_2d[position], width=width, height=height),
            bone_poses=bone_poses[position].astype(np.float32),
            rest_bone_poses=rest_bone_poses[position].astype(np.float32),
            shape=shapes[position].astype(np.float32),
            root_translation=translations[position].astype(np.float32),
            joints_3d=joints_3d[position].astype(np.float32),
        )
        for position in range(persons.num_person)
    ]
    return FrameEstimate(intrinsics=intrinsics, people=people)


def _assign_tracks(
    persons: PersonOutputLike,
    *,
    tracker: TrackerLike,
    time_index: int,
    image_size: float,
) -> NDArray[np.int64]:
    """Carry identities across frames, the way upstream's video path does.

    Fed normalised 2D pelvis position and log inverse depth rather than raw
    metres, so near and far people compare on the same scale.
    """
    import torch

    if persons.trackfeat is None:
        raise BodyEstimationError("the model returned no tracking features")

    pelvis_xyz = persons.transl_pelvis.clone().float()
    pelvis_ijn = (
        torch.cat(
            [
                persons.j2d[:, 0] / image_size,
                torch.log(1 / pelvis_xyz[:, 2:3].clamp_min(1e-5)),
            ],
            dim=-1,
        )
        .clone()
        .float()
    )
    identities = tracker.track_next_frame(
        time_index,
        persons.trackfeat.clone().float(),
        pelvis_xyz,
        pelvis_ijn,
        persons.bone_poses[:, 0, :3, :3].clone().float(),
    )
    if len(identities) != persons.num_person:
        raise BodyEstimationError(
            f"tracker returned {len(identities)} ids for {persons.num_person} people"
        )
    return cast("NDArray[np.int64]", identities.detach().cpu().numpy().astype(np.int64))
