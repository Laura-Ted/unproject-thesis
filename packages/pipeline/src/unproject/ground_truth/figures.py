"""The three pictures a ground-truth evaluation is read from.

Drawn from the observations file, never from anything held aside, so a figure
cannot disagree with the table beside it.

Deterministic on purpose (fixed SVG hash salt, no date), as in `unproject.rq3.figures`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from unproject.ground_truth.config import MetricThresholds
from unproject.ground_truth.dataset import CANONICAL_CONVENTION
from unproject.ground_truth.observations import Observations

if TYPE_CHECKING:
    from matplotlib.figure import Figure

DEPTH_FIGURE = "depth-bias.svg"
JOINT_FIGURE = "per-joint-error.svg"
FOCAL_FIGURE = "focal-ratio-vs-depth-bias.svg"

# Fixed so two runs over the same observations produce the same bytes.
_HASH_SALT = "unproject-ground-truth"

_FRAME = f"{CANONICAL_CONVENTION}: +Z away from the camera"


def write_figures(
    observations: Observations, thresholds: MetricThresholds, directory: Path
) -> list[Path]:
    """Draw what there is data for. Returns what was written.

    A figure with nothing to plot is skipped rather than drawn empty, as in
    `unproject.rq3.figures`.
    """
    import matplotlib

    # Before pyplot: this writes files and never opens a window, and the default
    # backend would go looking for a display.
    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = _HASH_SALT

    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, figure in (
        (DEPTH_FIGURE, _depth_bias(observations)),
        (JOINT_FIGURE, _per_joint(observations, thresholds)),
        (FOCAL_FIGURE, _focal_ratio(observations)),
    ):
        if figure is None:
            continue
        path = directory / name
        figure.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None})
        _close(figure)
        written.append(path)
    return written


def _depth_bias(observations: Observations) -> Figure | None:
    """Signed depth error against the distance the person actually was at.

    The figure the depth claim rests on. A cloud centred on zero would say the
    estimator is unbiased; a cloud that climbs with distance would say the error
    is proportional, which is what a wrong focal length produces; a flat cloud
    above zero would say it is a constant offset. The three are not the same
    finding and the scatter is what separates them.
    """
    import matplotlib.pyplot as plt

    if observations.depth_m.size == 0:
        return None

    depth = observations.depth_m
    error = observations.root_error_xyz_mm[:, 2]
    figure, axes = plt.subplots(figsize=(7.0, 4.0))
    axes.scatter(depth, error, s=6, alpha=0.25, color="#4878a8", linewidths=0.0)
    axes.axhline(0.0, color="#333333", linewidth=1.0)
    # Both readings drawn, so the picture answers which one it is rather than
    # leaving it to the eye.
    axes.axhline(
        float(error.mean()),
        color="#a84848",
        linestyle="--",
        linewidth=1.0,
        label=f"constant offset: {error.mean():+.0f} mm",
    )
    proportion = float((error / (depth * 1000.0)).mean())
    span = np.array([depth.min(), depth.max()])
    axes.plot(
        span,
        span * 1000.0 * proportion,
        color="#48a848",
        linewidth=1.2,
        label=f"proportional: {proportion:+.1%} of true distance",
    )
    axes.set_xlabel("annotated distance to the camera (m)")
    axes.set_ylabel("predicted minus annotated depth (mm)")
    axes.set_title(f"Depth error against true distance\n{_FRAME}", fontsize=9)
    axes.legend(fontsize=8)
    figure.tight_layout()
    return figure


def _per_joint(observations: Observations, thresholds: MetricThresholds) -> Figure | None:
    """Root-relative error per joint, with the PCK radius drawn on it.

    Root-relative rather than absolute, because absolute error is dominated by
    where the person was put and would show the same number at every joint.
    """
    import matplotlib.pyplot as plt

    from unproject.ground_truth.metrics import per_joint

    if not observations.joint_valid.any():
        return None

    means, present = per_joint(observations.root_relative_mm, observations.joint_valid)
    names = [name for name, seen in zip(observations.joint_names, present, strict=True) if seen]
    values = means[present]
    if not names:
        return None

    figure, axes = plt.subplots(figsize=(7.0, 4.0))
    axes.barh(range(len(names)), values, color="#4878a8")
    axes.axvline(
        thresholds.pck_radius_mm,
        color="#a84848",
        linestyle="--",
        linewidth=1.0,
        label=f"PCK radius: {thresholds.pck_radius_mm:.0f} mm",
    )
    axes.set_yticks(range(len(names)), names, fontsize=8)
    axes.invert_yaxis()
    axes.set_xlabel("mean root-relative error (mm)")
    axes.set_title("Articulation error per joint of the common skeleton", fontsize=9)
    axes.legend(fontsize=8)
    figure.tight_layout()
    return figure


def _focal_ratio(observations: Observations) -> Figure | None:
    """Per-sequence depth bias against how wrong the model's focal length was.

    The mechanism figure. In a monocular fit, depth and focal length trade off
    directly: a body of plausible size imaged at a given number of pixels is
    placed further away when the assumed focal length is longer. So if the depth
    error is the focal error, the points sit on the line where relative bias
    equals ratio minus one. Points off that line are error the focal length does
    not explain, and are the thing to go and look at.

    One point per sequence because the annotation-derived camera is one per
    sequence; plotting people would draw the same twenty points thousands of
    times.
    """
    import matplotlib.pyplot as plt

    if len(observations.sequence_names) < 2:
        return None

    ratios: list[float] = []
    biases: list[float] = []
    names: list[str] = []
    for index, name in enumerate(observations.sequence_names):
        rows = observations.sequence_index == index
        if not rows.any():
            continue
        depth_mm = observations.depth_m[rows] * 1000.0
        biases.append(float((observations.root_error_xyz_mm[rows, 2] / depth_mm).mean()))
        ratios.append(
            float(
                observations.sequence_fx_model_estimated[index]
                / observations.sequence_fx_annotation_derived[index]
            )
        )
        names.append(name)
    if len(names) < 2:
        return None

    x = np.array(ratios) - 1.0
    y = np.array(biases)
    figure, axes = plt.subplots(figsize=(6.0, 5.0))
    span = np.array([min(x.min(), y.min()), max(x.max(), y.max())]) * 1.1
    axes.plot(
        span, span, color="#a84848", linestyle="--", linewidth=1.0, label="bias = focal error"
    )
    axes.axhline(0.0, color="#cccccc", linewidth=0.8)
    axes.axvline(0.0, color="#cccccc", linewidth=0.8)
    axes.scatter(x, y, s=28, color="#4878a8", zorder=3)
    for name, px, py in zip(names, x, y, strict=True):
        axes.annotate(name, (px, py), fontsize=7, xytext=(3, 3), textcoords="offset points")
    r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else float("nan")
    axes.set_xlabel("model-estimated / annotation-derived focal length - 1")
    axes.set_ylabel("mean signed depth error / true distance")
    axes.set_title(
        f"Depth bias against focal-length error, one point per sequence (r = {r:+.2f})",
        fontsize=9,
    )
    axes.legend(fontsize=8)
    figure.tight_layout()
    return figure


def _close(figure: Figure) -> None:
    import matplotlib.pyplot as plt

    plt.close(figure)
