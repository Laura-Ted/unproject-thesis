# 2026-09-15 - WP5, turning three readings of the residual into tests

WP5 of ADR-0005. The residual after the intrinsics are supplied has been
called "the body-size prior" in every note since 2026-09-11. That is a
hypothesis. This commit pre-registers it, with the analysis and the reading
rule fixed before any data is seen, builds the controlled test of the
square-frame reading, and labels what is left as post-hoc or open. **No
analysis was run on real observations here** - the thesis evaluations are on
the author's machine and predate the scale proxy - so this is the design and
its tests on synthetic data.

## H1, body scale: a prediction with a slope

The physics gives a number to test, not just a direction. Monocular depth is
`z = f · S / s`; with the focal length supplied, the depth is wrong by the
ratio of the size the model assumed to the size the person is. WP1 recorded
that ratio's ingredients per matched person - bilateral femur plus tibia on
both sides, from the common skeleton, declared in the protocol before any
analysis. So per person the prediction is `y = f · r − 1`, with `y` the
relative depth error, `r` the proxy ratio, and `f` the sequence's focal ratio
(1 under the annotation-derived camera). A regression of `y` on that
prediction should have slope 1 and intercept 0 if the residual is what the
notes say it is.

The reading rule is in `hypotheses.py` and in the protocol's new §4.6:
supported if the slope is within 0.7–1.3 and R² ≥ 0.5; partial if slope > 0.3
and R² ≥ 0.2; unsupported otherwise. Bootstrap intervals over people, 1000
rounds, seed 20260915; Pearson and Spearman; the residual after the fit; the
same per sequence. `unproject gt-hypotheses <evaluation>` writes
`hypotheses.json`, a per-sequence CSV and the figure, deterministically, and
refuses a schema-1 evaluation with the way out (rescore the same bodies; no
inference).

The tests build the three cases the rule has to tell apart - a residual that
*is* `r − 1`, one that ignores `r`, one that half-tracks it - and check the
reading, the intervals, the focal ratio's entry under the model camera, that
a missing proxy leaves the fit rather than being zeroed, and that the
bootstrap is seeded. Whether the real residual is supported, partial or
unsupported is a question this commit makes answerable and does not answer.

The text the analysis licenses is bounded in the module docstring and the
protocol: an association consistent with H1 shows the residual is
proportional to the size mismatch this proxy measures. It does not prove the
prior is the cause.

## H2, square frames: the same content, transformed

TS1–TS5 sit 12 % too close where TS6–TS20 sit 5 % too close, and the two
groups differ in scene, actors and aspect ratio at once. The ADR forbids
reading the split as an aspect-ratio effect, and `gt-hypotheses` reports it
labelled post-hoc and confounded.

The controlled test is `frame_transform = "letterbox_square"` on the
ground-truth config, shipped as
`experiments/ground-truth/mupots-3d-letterbox-square.toml`: every wide
sequence is padded with black rows to 1920×1920 before the estimator sees it
(lossless PNG under `artifacts/_derived/`, released per sequence with
`--release-frames`), the 2D annotations and the derived principal point move
down by the padding, and the 3D annotations, the actors and the intrinsics are
the plain treatment's. The transform is part of the evaluation identity and
of the per-sequence artifact name (`-letterbox-square`), so it can never share
a directory with the plain run. `gt-compare` then reads the two per sequence.
If the bias of TS6–TS20 moves, the aspect-ratio path did it; if not, the
square sequences' offset is about those sequences. Cost: about an hour of
inference for fifteen sequences; the five square ones pass through unchanged.

## H3, the vertical offset: left open, on purpose

The root Y bias grows when the principal point is supplied. MuPoTS's pelvis
and Anny's root are different definitions and any alignment would be a
choice; `gt-hypotheses` reports the offset per sequence, labelled post-hoc
and descriptive, and the protocol leaves it as an open limit.

## Runbook for the author

After WP4's final runs:

```shell
uv run unproject gt-hypotheses artifacts/ground-truth/<final treatment>
uv run unproject gt-hypotheses artifacts/ground-truth/<final control>
uv run unproject gt-evaluate experiments/ground-truth/mupots-3d-letterbox-square.toml --final --release-frames
uv run unproject gt-compare artifacts/ground-truth/<final treatment> artifacts/ground-truth/<letterbox> \
    --label-a wide --label-b letterboxed
```

The readings go into a dated note as they come out. If H1 reads unsupported
or partial, the thesis text about the residual is reduced to what the
distribution supports - a proportional, sequence-independent offset of
measured size - per ADR-0005 §6, and the hypothesis is reported as tested and
not supported. That is a result.

## Verified here

18 new tests for the fit, the reading rule, the prediction under both
cameras, the schema-1 refusal, the ranks, the seeded bootstrap, the analysis
on a synthetic evaluation (deterministic files, post-hoc labels), and the
transform (padding, the moved annotations and principal point, the untouched
3D, black rows written and reused, a square sequence passed through, a wide
one padded and releasable, the transform in the identity and the artifact
name); one CLI test end to end. Full gate from the clean environment.
