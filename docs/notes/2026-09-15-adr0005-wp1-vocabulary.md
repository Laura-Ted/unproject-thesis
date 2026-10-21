# 2026-09-15 - WP1, saying what the numbers are

WP1 of ADR-0005. Schema 2 of the ground-truth contract: every value that
selected an algorithm, a camera or a metric is now named by what it is, and a
v1 artifact still reads. Labels change; numbers do not.

## What was wrong with the names

Three words were doing work they had not earned.

**`official`** named the greedy 40-pixel matcher, and the CLI printed
"official 3DPCK_rel" above a number that uses 13 of the 14 official PCK joints,
takes root-relative error on `annot3` where the released evaluator uses
`univ_annot3`, and aggregates its own way. Reproducing the association does
not make the metric official. The result is the **Unproject protocol on
MuPoTS-3D**, a MuPoTS-compatible project metric, and it now has an identifier
that says so: `unproject_mupots_14j_v1`.

**`released`** named the second camera condition. MuPoTS publishes no
calibration; the camera is recovered by least squares from the released 2D
and 3D annotations, per sequence, with a residual. It is **annotation-derived**
intrinsics, and the config, the artifact suffix, the CSV columns, the figure
axis and the docs all say that now.

**`association_gate = 0.1`** sat in both thesis configs beside a matcher that
never read it. The greedy matcher's radius is fixed by its algorithm. A
setting that does nothing must not be recordable as if it did.

## What changed

Config schema 2 (`GroundTruthConfig`):

```
matcher          official  -> mupots_greedy_2d_40px
                 native    -> normalized_global_assignment
camera           estimated -> model_estimated
                 released  -> annotation_derived
metric_protocol  new; "unproject_mupots_14j_v1"
thresholds.association_gate   required by the global matcher, refused by the greedy one
```

A schema-1 file is refused with that table in the message, not read under the
old names. The shipped configs are migrated;
`mupots-3d-released-camera.toml` is now
`mupots-3d-annotation-derived-camera.toml`, and the per-sequence artifact
suffix is `-annotation-derived-camera`. The CLI's `--matcher` takes the new
values and re-validates the config, so the gate rule cannot be bypassed by an
override.

Descriptor schema 2 (`evaluation.json`) repeats `metric_protocol`,
`matching` (matcher plus the one parameter it read and its unit) and
`camera_source` at the top level, and replaces `body_artifact_hashes` with
`body_artifacts`, each carrying the commit and dirty state read from that
run's `manifest.json`. A score now traces to the code that inferred the
bodies, not only to the code that scored them; WP4's final-run gate reads
both. `final` is false unless a final run wrote it. A v1 descriptor is
upgraded on read, its notes recorded in `migration`, and `gt-summary` prints
them.

Observations schema 2 (`observations.npz`) renames
`sequence_fx_released`/`sequence_fx_estimated` to
`sequence_fx_annotation_derived`/`sequence_fx_model_estimated`, adds a
`schema_version` marker, and records per matched person a body-scale proxy on
both sides - bilateral femur plus tibia, in metres, from the common skeleton,
declared in the protocol before any analysis reads it (WP5). A v1 file loads
under the new names with the proxy NaN and `has_scale_proxy = False`.

`per-sequence.csv` columns: `fx_annotation_derived`, `fx_model_estimated`,
`focal_ratio_estimated_over_derived`.

The evaluator checks the mapping against the protocol it claims: 14 joints
and a pelvis for `unproject_mupots_14j_v1`; a different joint set under the
same identifier is refused.

Docs: README, architecture, reproduce and the ground-truth protocol use the
new vocabulary; the protocol's §4.1 no longer calls the headline official and
gains a frozen vocabulary table (§7a) and the proxy definition (§4.2). The
nine notes that used the old words keep them, under a preamble that says how
to read them. The README's claims about the residual, the square sequences
and the vertical offset are downgraded to hypotheses, per ADR-0005 §3.5.

## What did not change

The matchers pair the same people, the metrics compute the same errors, the
figures draw the same points. 632 + 18 tests pass; the new ones cover the
rename table, the conditional gate, the v1 descriptor and observations
migrations, the protocol-mapping check and the scale proxy (a length that does
not move with placement; NaN, not shorter, when a leg joint is missing).

Evaluation ids will change on rescoring, because the config content changed.
That is expected and is what WP4's comparison tool verifies: same arrays, new
labels.

## Not done here

The two thesis evaluations on the author's machine are still schema 1. They
read back correctly - the upgrade path is tested on a hand-written v1
descriptor of the treatment run - and WP4 regenerates them from a clean
commit under schema 2.
