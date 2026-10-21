"""Testing what the depth residual is, without deciding in advance that it is.

ADR-0005 WP5. The camera experiment left a residual: with annotation-derived
intrinsics every sequence is placed too close, by a roughly constant fraction
of the distance. A proportional error is what a wrong *body size* produces in
a monocular fit, and reading the residual as "the body-size prior" is a
hypothesis: this module is where it is tested rather than repeated.

**The prediction, stated before the data.** Monocular depth is focal length
times metric size over apparent size: `z = f * S / s`. Given the annotation-
derived focal length, the model's depth is wrong by exactly the ratio of the
size it assumed to the size the person is: `z_pred / z_true = S_pred / S_true`.
The body-scale proxy - bilateral femur plus tibia, declared in
`metrics.scale_proxy` and recorded per matched person at evaluation time -
measures both sides on the same joints. So, per person, with
`r = proxy_pred / proxy_annot` and `y = (z_pred - z_true) / z_true`:

    H1:  y = r - 1        (slope 1, intercept 0, on the annotation-derived camera)

Under the model-estimated camera the focal error multiplies in: the predicted
relative error is `f_est / f_derived * r - 1`, with the ratio per sequence.
Both are one regression of `y` on the predicted value `p`, and the reading
rule below is fixed here, not after the plot:

| slope of y on p | R² | reading |
|---|---|---|
| within 0.7-1.3 | >= 0.5 | H1 supported: the proxy accounts for the residual |
| > 0.3 | >= 0.2 | partial: the proxy tracks the residual and something else is in it |
| otherwise | | unsupported: the residual is not body scale as this proxy measures it |

Everything else in the output is **post-hoc** and labelled: the split of the
residual by frame shape, and the vertical root offset per sequence, which
ADR-0005 WP5 leaves as an open limit unless the root definitions can be
aligned without an arbitrary choice.

An association consistent with H1 does not prove the prior *causes* the
residual - it says the residual is proportional to the size mismatch this
proxy sees, which is what the hypothesis predicts and what a competing
explanation would have to match. The text uses that level of certainty.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from unproject.ground_truth.dataset import FloatArray, GroundTruthError
from unproject.ground_truth.evaluate import read_evaluation
from unproject.ground_truth.mupots import image_size
from unproject.ground_truth.observations import Observations

if TYPE_CHECKING:
    from matplotlib.figure import Figure

HYPOTHESES_SCHEMA_VERSION = 1
HYPOTHESES_FILENAME = "hypotheses.json"
PER_SEQUENCE_FILENAME = "hypotheses-per-sequence.csv"
FIGURE_FILENAME = "scale-ratio-vs-depth-error.svg"

# The reading rule, as declared in the module docstring.
SUPPORTED_SLOPE = (0.7, 1.3)
SUPPORTED_R2 = 0.5
PARTIAL_SLOPE = 0.3
PARTIAL_R2 = 0.2

# Bootstrap over matched people for the slope and intercept intervals. Seeded,
# so the file regenerates byte for byte.
BOOTSTRAP_ROUNDS = 1000
BOOTSTRAP_SEED = 20260915

_HASH_SALT = "unproject-ground-truth-hypotheses"

type Reading = Literal["supported", "partial", "unsupported", "not_testable"]
type AnalysisKind = Literal["confirmatory", "post_hoc"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Fit(_Frozen):
    """A least-squares line of y on p, with what a bootstrap says about it."""

    people: int
    slope: float | None
    intercept: float | None
    slope_ci_low: float | None
    slope_ci_high: float | None
    intercept_ci_low: float | None
    intercept_ci_high: float | None
    pearson_r: float | None
    spearman_rho: float | None
    r_squared: float | None
    residual_sd: float | None
    residual_abs_median: float | None
    # Medians of the raw quantities, so the reader sees the effect size in
    # the units it was measured in.
    y_median: float | None
    p_median: float | None
    ratio_median: float | None


class SequenceFit(_Frozen):
    sequence: str
    shape: Literal["square", "wide"]
    fit: Fit
    focal_ratio: float | None
    y_median: float | None
    root_y_bias_mm: float | None


class ShapeSplit(_Frozen):
    """Post-hoc: the residual about the overall fit of y on p, by frame shape."""

    kind: AnalysisKind = "post_hoc"
    square_people: int
    wide_people: int
    square_residual_median: float | None
    wide_residual_median: float | None
    # Why the split is confounded, and which run is the controlled test of the
    # shape effect, per WP5.
    note: str


class VerticalOffset(_Frozen):
    kind: AnalysisKind = "post_hoc"
    root_y_bias_mm: float | None
    root_y_bias_median_mm: float | None
    per_sequence_sd_mm: float | None
    note: str


class Hypotheses(_Frozen):
    schema_version: int = Field(default=HYPOTHESES_SCHEMA_VERSION, ge=1)
    evaluation_id: str
    camera_source: str
    kind: AnalysisKind = "confirmatory"
    hypothesis: str
    prediction: str
    reading_rule: str
    proxy: str
    fit: Fit
    reading: Reading
    per_sequence: list[SequenceFit]
    shape_split: ShapeSplit
    vertical_offset: VerticalOffset
    bootstrap_rounds: int
    bootstrap_seed: int
    notes: list[str] = Field(default_factory=list)


def _straight_line(x: FloatArray, y: FloatArray) -> tuple[float, float]:
    """Slope and intercept of y on x by least squares."""
    design = np.stack([x, np.ones_like(x)], axis=1)
    (slope, intercept), *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(slope), float(intercept)


def _rank(values: FloatArray) -> FloatArray:
    """Average ranks, for Spearman without scipy."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    ranks[order] = np.arange(1, values.shape[0] + 1, dtype=np.float64)
    # Ties share their mean rank.
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if counts.max() > 1:
        sums = np.zeros(unique.shape[0])
        np.add.at(sums, inverse, ranks)
        ranks = sums[inverse] / counts[inverse]
    return ranks


def fit_line(p: FloatArray, y: FloatArray, *, rounds: int, seed: int) -> Fit:
    """The regression of y on the predicted value p, with bootstrap intervals."""
    keep = np.isfinite(p) & np.isfinite(y)
    p, y = p[keep], y[keep]
    people = int(p.shape[0])
    if people < 3 or float(np.std(p)) == 0.0:
        none = None
        return Fit(
            people=people,
            slope=none,
            intercept=none,
            slope_ci_low=none,
            slope_ci_high=none,
            intercept_ci_low=none,
            intercept_ci_high=none,
            pearson_r=none,
            spearman_rho=none,
            r_squared=none,
            residual_sd=none,
            residual_abs_median=none,
            y_median=_round(float(np.median(y))) if people else none,
            p_median=_round(float(np.median(p))) if people else none,
            ratio_median=none,
        )
    slope, intercept = _straight_line(p, y)
    residual = y - (slope * p + intercept)
    total = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - float(np.sum(residual**2)) / total if total > 0.0 else float("nan")
    # A constant y has no correlation with anything; NaN, reported as null.
    with np.errstate(invalid="ignore", divide="ignore"):
        pearson = float(np.corrcoef(p, y)[0, 1])
        spearman = float(np.corrcoef(_rank(p), _rank(y))[0, 1])

    rng = np.random.default_rng(seed)
    slopes = np.empty(rounds)
    intercepts = np.empty(rounds)
    for index in range(rounds):
        draw = rng.integers(0, people, people)
        slopes[index], intercepts[index] = _straight_line(p[draw], y[draw])
    return Fit(
        people=people,
        slope=_round(slope),
        intercept=_round(intercept),
        slope_ci_low=_round(float(np.percentile(slopes, 2.5))),
        slope_ci_high=_round(float(np.percentile(slopes, 97.5))),
        intercept_ci_low=_round(float(np.percentile(intercepts, 2.5))),
        intercept_ci_high=_round(float(np.percentile(intercepts, 97.5))),
        pearson_r=_round(pearson),
        spearman_rho=_round(spearman),
        r_squared=_round(r_squared),
        residual_sd=_round(float(np.std(residual))),
        residual_abs_median=_round(float(np.median(np.abs(residual)))),
        y_median=_round(float(np.median(y))),
        p_median=_round(float(np.median(p))),
        ratio_median=_round(float(np.median(p + 1.0))),
    )


def reading_of(fit: Fit) -> Reading:
    """Apply the reading rule. Fixed before the data; see the module docstring."""
    if fit.slope is None or fit.r_squared is None:
        return "not_testable"
    if SUPPORTED_SLOPE[0] <= fit.slope <= SUPPORTED_SLOPE[1] and fit.r_squared >= SUPPORTED_R2:
        return "supported"
    if fit.slope > PARTIAL_SLOPE and fit.r_squared >= PARTIAL_R2:
        return "partial"
    return "unsupported"


def predicted_relative_error(observations: Observations) -> tuple[FloatArray, FloatArray]:
    """Per person: the H1 prediction `p` and the observed relative error `y`.

    `p = focal_ratio(sequence) * proxy_pred / proxy_annot - 1`. Under the
    annotation-derived camera the focal ratio is 1 by construction and `p` is
    the scale ratio minus one.
    """
    if not observations.has_scale_proxy:
        raise GroundTruthError(
            "this evaluation was written before the scale proxy existed (observations "
            "schema 1). Rescore the same bodies with the current code - no inference is "
            "needed - and run the analysis on the new evaluation."
        )
    ratio = observations.scale_proxy_predicted_m / observations.scale_proxy_annotated_m
    focal = (
        observations.sequence_fx_model_estimated / observations.sequence_fx_annotation_derived
    )[observations.sequence_index]
    p = focal * ratio - 1.0
    depth_mm = observations.depth_m * 1000.0
    with np.errstate(divide="ignore", invalid="ignore"):
        y = observations.root_error_xyz_mm[:, 2] / depth_mm
    return p, y


def analyse(directory: Path) -> Hypotheses:
    """The confirmatory fit over the whole evaluation, per sequence, and the
    post-hoc diagnostics. Reads the evaluation and nothing else."""
    descriptor, observations = read_evaluation(directory)
    p, y = predicted_relative_error(observations)
    fit = fit_line(p, y, rounds=BOOTSTRAP_ROUNDS, seed=BOOTSTRAP_SEED)

    per_sequence: list[SequenceFit] = []
    residual = y - ((fit.slope or 0.0) * p + (fit.intercept or 0.0))
    square_rows = np.zeros(p.shape[0], dtype=np.bool_)
    for index, name in enumerate(observations.sequence_names):
        rows = observations.sequence_index == index
        shape = _shape_of(name)
        if shape == "square":
            square_rows |= rows
        derived = float(observations.sequence_fx_annotation_derived[index])
        estimated = float(observations.sequence_fx_model_estimated[index])
        per_sequence.append(
            SequenceFit(
                sequence=name,
                shape=shape,
                fit=fit_line(p[rows], y[rows], rounds=BOOTSTRAP_ROUNDS // 10, seed=BOOTSTRAP_SEED),
                focal_ratio=_round(estimated / derived) if derived else None,
                y_median=_round(float(np.median(y[rows]))) if rows.any() else None,
                root_y_bias_mm=(
                    _round(float(observations.root_error_xyz_mm[rows, 1].mean()))
                    if rows.any()
                    else None
                ),
            )
        )

    finite = np.isfinite(residual)
    square = square_rows & finite
    wide = ~square_rows & finite
    y_bias = [row.root_y_bias_mm for row in per_sequence if row.root_y_bias_mm is not None]
    return Hypotheses(
        evaluation_id=descriptor.evaluation_id,
        camera_source=descriptor.camera_source,
        hypothesis=(
            "the residual depth error after the intrinsics are supplied is a body-scale "
            "effect: the model places a person at the distance its assumed body size implies"
        ),
        prediction=(
            "per matched person, relative depth error y = focal_ratio * (proxy_pred / "
            "proxy_annot) - 1, i.e. slope 1 and intercept 0 of y on that prediction"
        ),
        reading_rule=(
            f"supported if slope in [{SUPPORTED_SLOPE[0]}, {SUPPORTED_SLOPE[1]}] and R^2 >= "
            f"{SUPPORTED_R2}; partial if slope > {PARTIAL_SLOPE} and R^2 >= {PARTIAL_R2}; "
            "unsupported otherwise; declared before the data"
        ),
        proxy="bilateral femur + tibia from the common skeleton, metres (metrics.scale_proxy)",
        fit=fit,
        reading=reading_of(fit),
        per_sequence=per_sequence,
        shape_split=ShapeSplit(
            square_people=int(square.sum()),
            wide_people=int(wide.sum()),
            square_residual_median=(
                _round(float(np.median(residual[square]))) if square.any() else None
            ),
            wide_residual_median=_round(float(np.median(residual[wide]))) if wide.any() else None,
            note=(
                "post-hoc and confounded: TS1-TS5 differ from TS6-TS20 in scene and actors as "
                "well as shape. The controlled test is a letterbox_square evaluation of the "
                "same wide sequences compared with gt-compare; this split does not replace it."
            ),
        ),
        vertical_offset=VerticalOffset(
            root_y_bias_mm=_round(float(observations.root_error_xyz_mm[:, 1].mean()))
            if observations.root_error_xyz_mm.shape[0]
            else None,
            root_y_bias_median_mm=_round(float(np.median(observations.root_error_xyz_mm[:, 1])))
            if observations.root_error_xyz_mm.shape[0]
            else None,
            per_sequence_sd_mm=_round(float(np.std(y_bias))) if len(y_bias) > 1 else None,
            note=(
                "descriptive only: MuPoTS's pelvis and Anny's root are different anatomical "
                "definitions and no alignment between them is established; the offset is an "
                "open limit, not a finding (ADR-0005 WP5.5)"
            ),
        ),
        bootstrap_rounds=BOOTSTRAP_ROUNDS,
        bootstrap_seed=BOOTSTRAP_SEED,
        notes=[
            "an association consistent with H1 does not prove the prior is the cause; it says "
            "the residual is proportional to the size mismatch this proxy measures",
            f"{int((~np.isfinite(p)).sum())} matched people had no proxy on one side and are "
            "excluded from the fit; they stay in every other table",
        ],
    )


def _shape_of(sequence: str) -> Literal["square", "wide"]:
    try:
        width, height = image_size(sequence)
    except GroundTruthError:
        return "wide"
    return "square" if width == height else "wide"


def write_hypotheses(hypotheses: Hypotheses, observations_dir: Path, out: Path) -> list[Path]:
    """The JSON, the per-sequence CSV and the figure. Deterministic."""
    out.mkdir(parents=True, exist_ok=True)
    as_json = out / HYPOTHESES_FILENAME
    as_json.write_text(hypotheses.model_dump_json(indent=2) + "\n", encoding="utf-8")

    as_csv = out / PER_SEQUENCE_FILENAME
    with as_csv.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "sequence",
                "shape",
                "people",
                "focal_ratio",
                "y_median",
                "ratio_median",
                "slope",
                "intercept",
                "r_squared",
                "root_y_bias_mm",
            ]
        )
        for row in hypotheses.per_sequence:
            writer.writerow(
                [
                    row.sequence,
                    row.shape,
                    row.fit.people,
                    _cell(row.focal_ratio),
                    _cell(row.y_median),
                    _cell(row.fit.ratio_median),
                    _cell(row.fit.slope),
                    _cell(row.fit.intercept),
                    _cell(row.fit.r_squared),
                    _cell(row.root_y_bias_mm),
                ]
            )

    _, observations = read_evaluation(observations_dir)
    figure = _figure(observations, hypotheses)
    as_svg = out / FIGURE_FILENAME
    figure.savefig(as_svg, format="svg", bbox_inches="tight", metadata={"Date": None})
    _close(figure)
    return [as_json, as_csv, as_svg]


def _figure(observations: Observations, hypotheses: Hypotheses) -> Figure:
    """Observed relative depth error against the H1 prediction, per person,
    with the identity the hypothesis predicts and the line that was fitted."""
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = _HASH_SALT
    import matplotlib.pyplot as plt

    p, y = predicted_relative_error(observations)
    keep = np.isfinite(p) & np.isfinite(y)
    figure, axes = plt.subplots(figsize=(6.0, 5.0))
    axes.scatter(p[keep], y[keep], s=4, alpha=0.2, color="#4878a8", linewidths=0.0)
    if keep.any():
        span = np.array([float(np.percentile(p[keep], 1)), float(np.percentile(p[keep], 99))])
        axes.plot(span, span, color="#a84848", linestyle="--", linewidth=1.0, label="H1: y = p")
        fit = hypotheses.fit
        if fit.slope is not None and fit.intercept is not None:
            axes.plot(
                span,
                fit.slope * span + fit.intercept,
                color="#48a848",
                linewidth=1.2,
                label=f"fit: slope {fit.slope:.2f}, R² {fit.r_squared or 0.0:.2f}",
            )
        if span[0] < span[1]:
            axes.set_xlim(span[0], span[1])
        low, high = float(np.percentile(y[keep], 1)), float(np.percentile(y[keep], 99))
        if low < high:
            axes.set_ylim(low, high)
    axes.axhline(0.0, color="#cccccc", linewidth=0.8)
    axes.axvline(0.0, color="#cccccc", linewidth=0.8)
    axes.set_xlabel("predicted relative depth error: focal ratio * scale ratio - 1")
    axes.set_ylabel("observed relative depth error: (z_pred - z_annot) / z_annot")
    axes.set_title(
        f"Body-scale hypothesis, {hypotheses.camera_source} camera: {hypotheses.reading}\n"
        f"{hypotheses.fit.people} matched people; axes clipped to the 1st-99th percentiles",
        fontsize=9,
    )
    axes.legend(fontsize=8)
    figure.tight_layout()
    return figure


def _close(figure: Figure) -> None:
    import matplotlib.pyplot as plt

    plt.close(figure)


def _round(value: float) -> float | None:
    return None if not np.isfinite(value) else round(float(value), 6)


def _cell(value: float | None) -> str:
    return "" if value is None else f"{value:g}"


__all__ = [
    "BOOTSTRAP_ROUNDS",
    "BOOTSTRAP_SEED",
    "FIGURE_FILENAME",
    "HYPOTHESES_FILENAME",
    "PER_SEQUENCE_FILENAME",
    "Fit",
    "Hypotheses",
    "analyse",
    "fit_line",
    "predicted_relative_error",
    "reading_of",
    "write_hypotheses",
]
