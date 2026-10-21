"""Which joint is which, written down instead of assumed.

MuPoTS and Anny describe a body differently. Comparing them means declaring, for
every joint being measured, where it comes from on each side — by name, with the
anatomical claim recorded, and with the body-model version it was written
against.

The rule this file exists to enforce: **never infer a correspondence from
matching array lengths or similar-looking indices.** Two skeletons with the same
joint count are not thereby the same skeleton, and an evaluation built on a
guessed mapping produces confident numbers about the wrong joints.

So every failure here is closed: a rig other than the mapped one, a wrong joint
count, a duplicate target, a missing required joint or a non-finite value stops
the evaluation rather than degrading it.
"""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path
from typing import Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from unproject.ground_truth.dataset import BoolArray, FloatArray, GroundTruthError

MAPPING_SCHEMA_VERSION = 1

# Where a joint on the common skeleton comes from: straight out of the release,
# or computed from several of its joints.
type Derivation = Literal["direct", "midpoint"]


class JointSource(BaseModel):
    """One side of one joint's correspondence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Names, never indices. An index is silently valid after a release reorders
    # its skeleton; a name fails loudly.
    names: list[str] = Field(min_length=1)
    derivation: Derivation = "direct"

    @model_validator(mode="after")
    def _check_derivation(self) -> Self:
        if self.derivation == "direct" and len(self.names) != 1:
            msg = f"a direct joint takes exactly one source name, got {self.names}"
            raise ValueError(msg)
        if self.derivation == "midpoint" and len(self.names) < 2:
            msg = f"a midpoint needs at least two source names, got {self.names}"
            raise ValueError(msg)
        return self


class CommonJoint(BaseModel):
    """One joint of the common skeleton, and where each side finds it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    # Why this is anatomically the same point on both sides. Prose, deliberately:
    # the claim is anatomical and cannot be checked by code.
    provenance: str = Field(min_length=1)
    mupots: JointSource
    anny: JointSource


class JointMapping(BaseModel):
    """A versioned correspondence between two skeletons."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=MAPPING_SCHEMA_VERSION, ge=1)
    identifier: str = Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")
    description: str = ""
    # What this mapping was written against. A rig with another bone count is
    # refused (`check_skeleton`); the package version and checkpoint are
    # recorded, not compared.
    anny_package_version: str = Field(min_length=1)
    anny_bone_count: int = Field(gt=0)
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    joints: list[CommonJoint] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_schema_version(self) -> Self:
        if self.schema_version != MAPPING_SCHEMA_VERSION:
            msg = (
                f"mapping schema_version {self.schema_version} is not readable by "
                f"this build (expected {MAPPING_SCHEMA_VERSION})"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_targets_are_unique(self) -> Self:
        names = [joint.name for joint in self.joints]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            msg = f"the common skeleton names {duplicates} more than once"
            raise ValueError(msg)
        return self

    @property
    def common_names(self) -> tuple[str, ...]:
        return tuple(joint.name for joint in self.joints)

    def source_names(self, side: Literal["mupots", "anny"]) -> set[str]:
        """Every released joint name this mapping reads on one side."""
        return {
            name
            for joint in self.joints
            for name in (joint.mupots if side == "mupots" else joint.anny).names
        }


def load_mapping(path: Path) -> tuple[JointMapping, str]:
    """Parse a mapping and return it with the SHA-256 of the file it came from.

    The hash goes into the evaluation descriptor: a result computed under one
    correspondence must not be comparable with a result computed under another
    just because both say "mapping v1".
    """
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise GroundTruthError(f"cannot read joint mapping {path}: {error}") from error

    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
        raise GroundTruthError(f"{path} is not valid TOML: {error}") from error

    try:
        mapping = JointMapping.model_validate(data)
    except ValueError as error:
        raise GroundTruthError(f"{path} failed validation:\n{error}") from error

    return mapping, hashlib.sha256(raw).hexdigest()


def apply_mapping(
    mapping: JointMapping,
    side: Literal["mupots", "anny"],
    joints: FloatArray,
    names: tuple[str, ...],
    valid: BoolArray,
) -> tuple[FloatArray, BoolArray]:
    """Reduce one side's joints to the common skeleton.

    A derived joint is valid only when every joint it is derived from is valid:
    a midpoint of a known hip and an unannotated one is not a pelvis, it is half
    an answer.
    """
    index = {name: position for position, name in enumerate(names)}
    missing = sorted(mapping.source_names(side) - set(index))
    if missing:
        raise GroundTruthError(
            f"mapping {mapping.identifier!r} reads {side} joint(s) {missing}, which "
            f"this release does not provide. Its joints are {sorted(index)}. Fix the "
            "mapping against the released names; do not match by position."
        )

    common = np.zeros((len(mapping.joints), 3), dtype=np.float64)
    common_valid = np.zeros(len(mapping.joints), dtype=np.bool_)
    for position, joint in enumerate(mapping.joints):
        source = joint.mupots if side == "mupots" else joint.anny
        rows = [index[name] for name in source.names]
        if not bool(valid[rows].all()):
            continue
        common[position] = joints[rows].mean(axis=0)
        common_valid[position] = True
    return common, common_valid
