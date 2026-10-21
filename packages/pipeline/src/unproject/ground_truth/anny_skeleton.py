"""The bone labels the estimator's Anny model actually has, in its own order.

Kept behind a function with a lazy import because everything else in this
package is arrays and files: the metrics must stay importable, and testable,
without torch.

Why the labels are read from the model rather than written down here. There are
two lists that look right and are not:

- `rig.default.json` in the `anny` package has 163 entries, matching the bone
  count, but they are in alphabetical order rather than model order;
- a default `Anny()` gives 104 bones, in which `upperarm01.L` sits at index 48
  rather than 50.

Either would produce a full set of plausible numbers about the wrong joints.
The only authority is the model `multihmr2` itself builds, so that is what this
constructs - with the same arguments, taken from `multihmr2.api`.
"""

from __future__ import annotations

from functools import cache

from unproject.ground_truth.dataset import GroundTruthError


@cache
def anny_bone_labels() -> tuple[str, ...]:
    """The ordered bone labels of the 163-bone model the estimator poses.

    Cached because building the model reads several cached tensors and every
    sequence of an evaluation asks for the same answer.
    """
    try:
        import anny
    except ImportError as error:  # pragma: no cover - depends on the install
        raise GroundTruthError(
            "the anny package is required to read predicted joint names. Install "
            "the bodies extra, or run gt-summary, which needs no body model."
        ) from error

    # Exactly the call in multihmr2.api.InferenceSession.__init__. Different
    # arguments give a different skeleton with a different bone order.
    model = anny.create_fullbody_model(
        local_changes=True,
        pose_parameterization="root_relative_world",
        remove_unattached_vertices=False,
    )
    return tuple(str(label) for label in model.bone_labels)


def check_skeleton(bone_count: int, *, expected: int) -> None:
    """Refuse an artifact posed by a skeleton other than the mapped one.

    The mapping records the bone count it was written against. A release that
    changed the rig would keep every index valid and move what each one means.
    """
    if bone_count != expected:
        raise GroundTruthError(
            f"this bodies artifact has {bone_count} bones and the joint mapping was "
            f"written against {expected}. The mapping's indices would still be valid "
            "and would mean different joints. Re-source the mapping against the rig "
            "this artifact was produced with."
        )
