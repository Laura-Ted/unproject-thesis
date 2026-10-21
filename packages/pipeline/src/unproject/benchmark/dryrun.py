"""An estimator that answers instantly and loads nothing.

The point of a benchmark is the model, so this measures nothing worth reporting.
It exists so the orchestration around the model - frame selection, repetitions,
warm-up, the result file, the summary - can be exercised and tested on a machine
with no weights, which is every machine in CI.
"""

from __future__ import annotations

import numpy as np

from unproject.bodies.artifacts import EstimatorInfo
from unproject.bodies.multihmr import FloatArray, FrameEstimate, FrameInput, PersonEstimate
from unproject.config import BodiesConfig, config_hash

# Recognisable in a result file as "this did not measure a model".
DRY_RUN_CHECKPOINT = "0" * 64


class DryRunSession:
    """Returns one person per frame, with arrays of the right shape."""

    def __init__(self, bone_count: int) -> None:
        self._bone_count = bone_count

    @property
    def device(self) -> None:
        """Nothing to synchronise: there is no model."""
        return None

    def frame(
        self, image: FrameInput, config: BodiesConfig, *, width: int, height: int
    ) -> FrameEstimate:
        return FrameEstimate(
            intrinsics=(float(width), float(width), width / 2.0, height / 2.0),
            people=[self._person()],
        )

    def _person(self) -> PersonEstimate:
        zeros = np.zeros((self._bone_count, 4, 4), dtype=np.float32)
        return PersonEstimate(
            track_id=0,
            confidence=1.0,
            box_xyxy=(0.0, 0.0, 1.0, 1.0),
            bone_poses=zeros,
            rest_bone_poses=zeros,
            shape=np.zeros(11, dtype=np.float32),
            root_translation=np.zeros(3, dtype=np.float32),
            joints_3d=np.zeros((self._bone_count, 3), dtype=np.float32),
        )

    def rest_vertices(self, shape: FloatArray) -> FloatArray:
        return np.zeros((1, 3), dtype=np.float32)

    def rest_bone_poses(self, shape: FloatArray) -> FloatArray:
        return np.zeros((self._bone_count, 4, 4), dtype=np.float32)


class DryRunEstimator:
    """Stands in for `MultiHmrEstimator` without importing torch."""

    def __init__(self, config: BodiesConfig, *, bone_count: int = 4) -> None:
        self._config = config
        self._bone_count = bone_count

    def open(self) -> DryRunSession:
        return DryRunSession(self._bone_count)

    def describe(self) -> EstimatorInfo:
        return EstimatorInfo(
            name="multihmr2",
            checkpoint_sha256=DRY_RUN_CHECKPOINT,
            backbone_repo=self._config.backbone_repo,
            backbone_ref=self._config.backbone_ref,
            config_hash=config_hash(self._config),
        )
