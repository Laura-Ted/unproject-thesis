"""Tests for the deterministic parts of the bodies estimator.

The model is not asserted on. The geometry around it is: mapping out of the
model's padded input space and back is where a wrong answer looks plausible.
"""

from __future__ import annotations

import subprocess
import sys
from typing import cast

import numpy as np
import pytest

from unproject.bodies.artifacts import BodyModelInfo
from unproject.bodies.multihmr import (
    IMAGE_SIZE,
    PATCH_SIZE,
    BodyEstimationError,
    ModelLike,
    SessionLike,
    TrackerLike,
    box_from_joints,
    drop_unused_vertices,
    intrinsics_to_model,
    intrinsics_to_source,
    model_input_size,
    model_space_geometry,
    points_to_source,
    reduce_influences,
)
from unproject.config import BodiesConfig, KnownIntrinsics


def _intrinsics(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def test_the_stage_loads_neither_torch_nor_pycolmap_on_import() -> None:
    """The two cannot share a process, and this stage is on the CLI's import path.

    Each bundles its own libomp and the second to initialise aborts the process
    (see `openmp_guard.py`), so the estimator imports torch inside the functions
    that need it.
    """
    probe = (
        "import sys; import unproject.bodies.stage; "
        "print('pycolmap' in sys.modules, 'torch' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False False"


def test_the_cli_still_loads_neither() -> None:
    """Adding the bodies verb must not drag torch into the COLMAP path."""
    probe = (
        "import sys; import unproject.cli; print('pycolmap' in sys.modules, 'torch' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False False"


def test_weights_and_backbone_are_both_pinned() -> None:
    """The checkpoint digest alone does not pin the run: the backbone is code."""
    config = BodiesConfig()
    assert len(config.sha256) == 64
    assert len(config.backbone_ref) == 40
    assert ":" not in config.backbone_repo


def test_a_landscape_frame_keeps_its_aspect_and_needs_no_padding() -> None:
    # 1280x720 scales by exactly 0.6 to 768x432, and 432 is 27 whole patches.
    scale, pad_left, pad_top = model_space_geometry(1280, 720)
    assert scale == pytest.approx(0.6)
    assert (pad_left, pad_top) == (0, 0)


def test_the_longest_side_reaches_the_model_size() -> None:
    for width, height in ((1280, 720), (720, 1280), (1000, 1000), (1920, 1080)):
        scale, _, _ = model_space_geometry(width, height)
        assert round(max(width, height) * scale) == IMAGE_SIZE


def test_padding_never_exceeds_a_patch() -> None:
    """Padding only fills out the last partial patch.

    Only the near-side value is returned, since that is what the inverse
    subtracts; an odd remainder leaves the extra pixel on the far side.
    """
    for width, height in ((1280, 713), (1000, 999), (1920, 817), (768, 768), (640, 481)):
        _, pad_left, pad_top = model_space_geometry(width, height)
        assert 0 <= pad_left < PATCH_SIZE
        assert 0 <= pad_top < PATCH_SIZE


def test_a_degenerate_frame_size_is_refused() -> None:
    with pytest.raises(BodyEstimationError):
        model_space_geometry(0, 720)


def test_the_principal_point_comes_back_to_the_image_centre() -> None:
    # What the model actually returned for a 1280x720 frame of the crowd clip.
    model_k = _intrinsics(3906.16, 3906.16, 384.0, 216.0)
    fx, fy, cx, cy = intrinsics_to_source(model_k, width=1280, height=720)
    assert cx == pytest.approx(640.0)
    assert cy == pytest.approx(360.0)
    # 0.6 scale, so the focal length grows by the same factor it was shrunk by.
    assert fx == pytest.approx(3906.16 / 0.6)
    assert fy == pytest.approx(3906.16 / 0.6)


def test_projecting_gives_the_same_point_in_either_space() -> None:
    """The invariant that catches a wrong scale or a forgotten pad.

    If `intrinsics_to_source` and `points_to_source` ever disagree, every box in
    the artifact is quietly offset.
    """
    rng = np.random.default_rng(seed=0)
    model_k = _intrinsics(3906.16, 3906.16, 384.0, 216.0)

    for width, height in ((1280, 720), (1080, 1920), (1000, 999)):
        fx, fy, cx, cy = intrinsics_to_source(model_k, width=width, height=height)
        source_k = _intrinsics(fx, fy, cx, cy)

        # In front of the camera: z > 0 is what this convention means.
        points = rng.uniform(-3.0, 3.0, size=(32, 3))
        points[:, 2] = rng.uniform(2.0, 40.0, size=32)

        normalised = points / points[:, 2:3]
        in_model = (model_k @ normalised.T).T[:, :2]
        in_source = (source_k @ normalised.T).T[:, :2]

        mapped = points_to_source(in_model, width=width, height=height)
        assert np.allclose(mapped, in_source, atol=1e-9)


def test_the_convention_puts_x_right_y_down_and_z_in_front() -> None:
    """Pins the axes, established by measuring against the model itself."""
    k = _intrinsics(1000.0, 1000.0, 640.0, 360.0)
    centre = np.array([[0.0, 0.0, 10.0]])
    right = np.array([[1.0, 0.0, 10.0]])
    below = np.array([[0.0, 1.0, 10.0]])

    def project(points: np.ndarray) -> list[float]:
        projected = (k @ (points / points[:, 2:3]).T).T[:, :2]
        return [float(value) for value in projected[0]]

    assert project(right)[0] > project(centre)[0], "+x must move right in the image"
    assert project(below)[1] > project(centre)[1], "+y must move down in the image"


def test_a_box_is_clipped_to_the_frame() -> None:
    """A joint off the edge still projects somewhere."""
    joints = np.array([[-50.0, -20.0], [1400.0, 900.0], [600.0, 300.0]])
    x_min, y_min, x_max, y_max = box_from_joints(joints, width=1280, height=720)
    assert (x_min, y_min) == (0.0, 0.0)
    assert (x_max, y_max) == (1280.0, 720.0)


def test_a_box_spans_the_joints_it_was_given() -> None:
    joints = np.array([[100.0, 50.0], [300.0, 400.0], [200.0, 90.0]])
    assert box_from_joints(joints, width=1280, height=720) == (100.0, 50.0, 300.0, 400.0)


def test_an_empty_joint_set_has_no_box() -> None:
    with pytest.raises(BodyEstimationError):
        box_from_joints(np.zeros((0, 2)), width=1280, height=720)


def test_a_missing_extra_is_reported_rather_than_crashing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the extra the stage must say which one, not fail on an import.

    The check is a spec lookup rather than an import for a reason: the path that
    reports the extra missing must not be the path that loads torch.
    """
    import importlib.util

    from unproject.bodies import multihmr

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(BodyEstimationError, match="uv sync --extra multihmr"):
        multihmr._require_extra()


def test_influences_are_cut_to_the_strongest() -> None:
    weights = np.array([[0.1, 0.5, 0.2, 0.05, 0.15]], dtype=np.float32)
    indices = np.array([[10, 11, 12, 13, 14]], dtype=np.int64)
    _, bones = reduce_influences(weights, indices, keep=3)
    # 0.5, 0.2, 0.15 - the 0.1 and 0.05 go.
    assert sorted(bones[0].tolist()) == [11, 12, 14]


def test_what_is_kept_still_sums_to_one() -> None:
    """A vertex that lost half its weight would otherwise collapse halfway to
    the origin rather than deform a little."""
    weights = np.array([[0.3, 0.3, 0.2, 0.1, 0.1]], dtype=np.float32)
    indices = np.array([[0, 1, 2, 3, 4]], dtype=np.int64)
    kept, _ = reduce_influences(weights, indices, keep=2)
    assert kept.sum() == pytest.approx(1.0, abs=1e-6)


def test_a_vertex_no_bone_pulls_on_is_left_alone() -> None:
    # Dividing by its total would be a division by zero; it simply stays put.
    kept, _ = reduce_influences(
        np.zeros((1, 5), dtype=np.float32), np.zeros((1, 5), dtype=np.int64), keep=4
    )
    assert kept.sum() == pytest.approx(0.0)


def test_bone_indices_come_back_as_shorts() -> None:
    # There are 163 bones and 19158 vertices; both fit, and halving the buffers
    # is worth the cast.
    _, bones = reduce_influences(
        np.eye(4, dtype=np.float32), np.arange(16).reshape(4, 4).astype(np.int64), keep=2
    )
    assert bones.dtype == np.uint16


def test_mismatched_weights_and_indices_are_refused() -> None:
    with pytest.raises(BodyEstimationError, match="against"):
        reduce_influences(np.zeros((3, 5), dtype=np.float32), np.zeros((3, 4), dtype=np.int64))


def test_vertices_no_triangle_uses_are_dropped() -> None:
    """Anny keeps vertices nothing renders, and they wander.

    On the reference clip twelve of them moved by up to two metres while every
    vertex that is actually drawn stayed within one. Keeping them would put a
    two-metre outlier in a table describing a body 1700 tall.
    """
    faces = np.array([[0, 2, 4], [2, 4, 5]], dtype=np.int64)
    kept, remapped = drop_unused_vertices(faces, vertex_count=8)
    assert kept.tolist() == [0, 2, 4, 5]
    # The faces now index the kept list, not the original.
    assert remapped.tolist() == [[0, 1, 2], [1, 2, 3]]


def test_a_face_pointing_past_the_end_is_refused() -> None:
    with pytest.raises(BodyEstimationError, match="refers to vertex"):
        drop_unused_vertices(np.array([[0, 1, 9]], dtype=np.int64), vertex_count=3)


def test_a_body_with_no_faces_is_refused() -> None:
    with pytest.raises(BodyEstimationError, match="no faces"):
        drop_unused_vertices(np.zeros((0, 3), dtype=np.int64), vertex_count=5)


# --- a camera the estimator is told -------------------------------------------


def _known(width: int = 1920, height: int = 1080) -> KnownIntrinsics:
    return KnownIntrinsics(width=width, height=height, fx=1103.5, fy=1101.4, cx=942.1, cy=528.4)


@pytest.mark.parametrize(
    ("width", "height"), [(1920, 1080), (2048, 2048), (1080, 1920), (1000, 999)]
)
def test_a_known_camera_survives_the_round_trip_to_model_space(width: int, height: int) -> None:
    """What the estimator is told and what the artifact records must be the
    same numbers. The decoder hands back the K it was given, and the stage maps
    it to source resolution; if the two mappings disagreed, the artifact would
    claim a camera nobody supplied."""
    known = _known(width, height)

    fx, fy, cx, cy = intrinsics_to_source(intrinsics_to_model(known), width=width, height=height)

    assert (fx, fy, cx, cy) == pytest.approx((known.fx, known.fy, known.cx, known.cy))


def test_in_model_space_the_focal_length_is_scaled_and_the_centre_is_padded() -> None:
    """A 1280x700 frame becomes 768 wide and 420 tall, then padded to 432 - a
    multiple of the 16-px patch - with 6 px above: the focal length shrinks by
    0.6 and the vertical centre gains those 6 px. Forgetting the pad is the
    mistake this pins."""
    known = KnownIntrinsics(width=1280, height=700, fx=1000.0, fy=990.0, cx=640.0, cy=350.0)

    k = intrinsics_to_model(known)

    assert k[0, 0] == pytest.approx(1000.0 * 0.6)
    assert k[1, 1] == pytest.approx(990.0 * 0.6)
    assert k[0, 2] == pytest.approx(640.0 * 0.6)
    assert k[1, 2] == pytest.approx(350.0 * 0.6 + 6.0)
    assert k[2, 2] == 1.0


class _FakeEncoded:
    def __init__(self, k: object) -> None:
        self.K = k
        self.feat = None


class _FakeModel:
    """Records what the decoder was handed, which is the whole question."""

    def __init__(self) -> None:
        import torch

        self.predicted = torch.tensor([[[999.0, 0.0, 384.0], [0.0, 999.0, 224.0], [0.0, 0.0, 1.0]]])
        self.decoder_saw: object = None
        self.decoder_kwargs: dict[str, object] = {}

    def encoder(self, x: object) -> _FakeEncoded:
        return _FakeEncoded(self.predicted)

    def full_body_decoder(self, *args: object, **kwargs: object) -> list[object]:
        self.decoder_saw = kwargs.pop("K")
        self.decoder_kwargs = kwargs
        return []


def test_the_wrapper_hands_the_decoder_the_camera_it_was_told() -> None:
    """Upstream's forward is encoder, then decoder with `K=z.K`. The wrapper
    keeps both halves and replaces only what passes between them."""
    torch = pytest.importorskip("torch")
    from unproject.bodies.multihmr import _KnownCameraModel

    inner = _FakeModel()
    told = intrinsics_to_model(_known())

    wrapped = _KnownCameraModel(cast(ModelLike, inner), told)
    wrapped(torch.zeros(1, 3, 448, 768), conf_thresh=0.4, lowres=False)

    seen = inner.decoder_saw
    assert isinstance(seen, torch.Tensor)
    assert seen.shape == (1, 3, 3)
    assert np.allclose(seen[0].numpy(), told)
    assert not np.allclose(seen[0].numpy(), inner.predicted[0].numpy())
    assert inner.decoder_kwargs == {"conf_thresh": 0.4, "lowres": False}


def test_a_frame_of_the_wrong_size_is_refused_before_inference() -> None:
    """A focal length is only meaningful at the resolution it was measured at."""
    from unproject.bodies.multihmr import EstimatorSession

    torch = pytest.importorskip("torch")

    class _Session:
        img_size = 768
        body_model_world = None
        model = None

        def __call__(self, *args: object, **kwargs: object) -> object:
            raise AssertionError("inference must not run on a frame of the wrong size")

    # The guard runs before the tracker or the body model are touched, so the
    # session is the only collaborator that needs to exist; the rest are cast.
    session = EstimatorSession(
        cast(SessionLike, _Session()),
        tracker=cast(TrackerLike, object()),
        device=torch.device("cpu"),
        body_model=cast(BodyModelInfo, object()),
    )
    config = BodiesConfig(intrinsics=_known(1920, 1080))

    with pytest.raises(BodyEstimationError, match="1920x1080"):
        session.frame(np.zeros((1080, 1920, 3), dtype=np.uint8), config, width=1280, height=720)


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [
        # 16:9 at 1920: long edge to 768, and 432 is already 27 patches.
        (1920, 1080, (768, 432, 0.4)),
        # Half resolution reaches the model as the same tensor: the factor is
        # not the tensor size.
        (960, 540, (768, 432, 0.8)),
        # Letterboxed to square: same scale as 1920x1080, 336 more rows.
        (1920, 1920, (768, 768, 0.4)),
        (320, 240, (768, 576, 2.4)),
    ],
)
def test_model_input_size_is_what_the_benchmark_records(
    width: int, height: int, expected: tuple[int, int, float]
) -> None:
    padded_width, padded_height, scale = model_input_size(width, height)

    assert (padded_width, padded_height) == expected[:2]
    assert scale == pytest.approx(expected[2])
