# Ground-Truth Evaluation Protocol

**Status:** Draft — frozen section by section, see §7
**Created:** 2026-09-08
**Governs:** RQ3 of `docs/adr/ADR-0002-live-body-capture-from-monocular-video.md`,
as amended by `docs/adr/ADR-0004-public-ground-truth-evaluation.md`
**Supersedes:** the manually measured corridor study in `docs/experiments/protocol.md` §2.1

This is the operational document for ADR-0004. The ADR decides; this says what to
run, in what order, with which values held fixed. Nothing here contains results,
and no result may be recorded in this file.

Changing anything below after a measurement has been taken requires a dated note
in `docs/notes/` saying what changed, why, and which results were collected
under which version.

---

## 1. What each dataset is allowed to answer

The two datasets are reported separately and never pooled into one score.

| | MuPoTS-3D | Anny-One |
|---|---|---|
| Kind | real video, capture-derived ground truth | synthetic images, synthetic ground truth |
| Role | **primary** accuracy result | secondary compatibility diagnostic |
| Answers | detection coverage, 3D joint accuracy, camera-space placement, depth error, temporal stability | frame geometry, camera estimation, native Anny pose and shape |
| Cannot answer | anything about the Anny mesh or phenotype parameters | real-world accuracy, temporal behaviour, domain transfer |

"Capture-derived ground truth" and "synthetic ground truth" are the terms used
whenever the distinction matters. Neither is a claim of metaphysical certainty:
the first came from a multi-view markerless capture process with its own error,
the second is exact only with respect to the scene the renderer built.

---

## 2. Environment

The measuring machine, and the same one the RQ1/RQ2 benchmark names:

| | |
|---|---|
| Machine | Apple MacBook Pro, M3 Pro, 18 GB unified memory |
| OS | macOS 26.6.2 (build 25G83) |
| Python | 3.12, pinned in `.python-version` |
| Backend | PyTorch MPS; a CPU fallback must be labelled as such |

Every evaluation additionally records its own environment. The table is the
declared target, not a substitute for that record.

---

## 3. Acquisition, which is the author's to do

**MuPoTS-3D, concretely.** The project page offers `mupots-3d-eval.zip`, 10 KB —
the licence, the official MATLAB evaluator, and a download script. The dataset
itself is `MultiPersonTestSet.zip`, about 6 GB, and the script's `wget` line is
**commented out on purpose**. Uncommenting it is the acceptance gesture, and it
is the author's to make. Extract to `datasets/mupots-3d/`.

The licence is non-commercial research use, forbids redistribution, and requires
citing Mehta et al. 3DV 2018 and Mehta et al. 3DV 2017. Read it before running
anything: `docs/notes/2026-09-09-mupots-release-inspected.md` summarises it but
is not a substitute for it.

Per ADR-0004 §4.1, no one accepts a licence or fetches a gated file on the
author's behalf. The author:

1. reads and accepts each official licence;
2. obtains the files from the official provider named in §8;
3. places them under `datasets/mupots-3d/` or `datasets/anny-one/`;
4. keeps whatever proof of release version or download date reproducibility needs.

The project supplies verification and indexing commands, never a downloader that
walks around an access gate.

Raw dataset files are immutable inputs. Nothing in this repository renames,
rewrites or repairs them. Every derived frame, normalised annotation, cache or
prediction goes under `artifacts/`.

**Not committed, ever:** images, videos, annotations, extracted frames, or any
remote from which they could be recovered. Committed: configuration, small
non-sensitive manifests, official file identifiers, hashes, mappings, and
instructions the licence permits.

---

## 4. What is measured

### 4.1 MuPoTS-3D — the headline

> **Corrected 2026-09-15 (ADR-0005 WP1).** An earlier version of this section
> labelled 3DPCK<sub>rel</sub> *Official*. It is not: the mapping covers 13 of
> the 14 official PCK joints, the root-relative error is taken on `annot3`
> rather than `univ_annot3`, and only the association reproduces the released
> evaluator. Everything below is the **Unproject protocol on MuPoTS-3D**, a
> MuPoTS-compatible project metric with identifier `unproject_mupots_14j_v1`.
> The word *official* is reserved, per ADR-0005 §3, for a result that uses
> exactly the released evaluator's joints, scale field, association and
> aggregation with a numerical cross-check on the same input; no such result
> exists in this project, and none is claimed.

On all 20 official test sequences, per-sequence results preserved, under
protocol `unproject_mupots_14j_v1`:

| Metric | Definition |
|---|---|
| 3DPCK<sub>rel</sub> | percentage of common joints within 150 mm **after root alignment**, on `annot3`. Project protocol. |
| 3DPCK<sub>abs</sub> | percentage of common joints within 150 mm in camera coordinates, **no alignment**. Project protocol. |

The released evaluator computes a root-relative variant **only**, over its own
14 joints on `univ_annot3`: it subtracts the pelvis from both sides before
taking any error. `3DPCKabs` comes from later work, not from the dataset
authors. Neither number here is that evaluator's, and ADR-0004 §6.2's
assumption that the headline would be official is superseded; the reading of
the release is in `docs/notes/2026-09-09-mupots-release-inspected.md`.

The official PCK covers 14 of the 17 joints — head_top, neck, shoulders,
elbows, wrists, hips, knees, ankles — and an AUC over 0–150 mm in 5 mm steps.
The common skeleton here maps 13 of those 14 plus the pelvis; `head_top` has
no Anny counterpart, and every table says so.

Both coverage modes are reported: all annotated people (recall), and only those
matched to a prediction (every accuracy figure).

The two metrics answer different questions and neither substitutes for the other:
`3DPCKrel` is pose articulation with global placement removed, `3DPCKabs` is
where the body actually sits in the scene.

Implementing the official protocol in full is optional (ADR-0005 WP4). If it
is done it uses `univ_annot3`, all 14 official joints including `head_top`, the
published aggregation and a numerical cross-check against the reference
evaluator, and it is reported as a distinct row under its own identifier,
never merged with this one.

### 4.2 MuPoTS-3D — project diagnostics

Reported separately from the headline, never mixed into it:

- MPJPE in millimetres over matched valid common joints, and root-relative MPJPE
- root translation MAE and **signed** bias for X, Y and Z
- depth absolute error and signed bias, by ground-truth distance bin, with
  median, interquartile range and outlier count beside the mean (ADR-0005 WP4)
- precision, recall, F1, false positives, false negatives at the frozen threshold
- per-joint and per-sequence breakdowns
- per matched person, a **body-scale proxy** on both sides: the bilateral sum
  of femur and tibia lengths - hip→knee plus knee→ankle, left and right - in
  metres, from the common skeleton, NaN when any of the six joints is invalid.
  Declared here on 2026-09-15, before any analysis read it, for the
  hypothesis in ADR-0005 WP5 that the residual depth error after supplying
  the intrinsics is a body-size effect. The proxy uses only direct joints
  that both releases define the same way; it is a length, so it does not move
  with where the person is placed.

### 4.3 MuPoTS-3D — temporal

Only for consecutive frames where a ground-truth identity is valid and stable and
both frames matched:

- MPJVE, mean per-joint velocity error, mm/s
- root velocity error by X, Y, Z
- coverage: eligible pairs, evaluated pairs, pairs lost to detection or
  association gaps

**Gaps are not bridged.** No velocity is computed across an interval larger than
the frozen maximum. Multi-HMR track IDs are never manually joined, and ID-switch
claims are omitted entirely unless released identity continuity is verified.

Raw and viewer-smoothed predictions stay separate and use exactly the same
eligible pairs.

### 4.4 Anny-One

After the §6 classification, on the frozen subset only:

- detection precision, recall, F1
- common Anny-joint MPJPE, root-relative MPJPE, PCK at declared thresholds
- root X/Y/Z error and depth bias
- focal length or field-of-view error, **only** where released and predicted
  camera fields are definitionally compatible
- per-parameter shape error, **only** for parameters whose names, units, ranges
  and ordering are verified identical

Pose-matrix geodesic error and per-vertex error are optional and gated: see
ADR-0004 §6.4. If any gate fails, joint-level evaluation is the complete
deliverable and that is a result, not unfinished work.

### 4.6 Hypotheses about the residual — pre-registered 2026-09-15

ADR-0005 WP5. The camera experiment (§4.1, the `annotation_derived` condition)
leaves a residual depth error of roughly −5 % of distance on every sequence.
Three readings of it circulate in the notes. Each is a hypothesis here, and
the analysis that tests each is fixed before it is run.

**H1 — body scale (confirmatory).** A monocular fit places a person at
`z = f · S / s`, focal length times metric size over apparent size. With the
annotation-derived focal length, the depth is wrong by the ratio of the size
the model assumed to the size the person is. The proxy of §4.2 measures both
sides on the same joints, so per matched person, with `r = proxy_pred /
proxy_annot`, `y = (z_pred − z_annot) / z_annot` and, under the model-estimated
camera, the sequence's focal ratio `f = fx_model_estimated / fx_annotation_derived`
(1 by construction under the derived camera):

```text
prediction   y = f · r − 1        slope 1, intercept 0 of y on p = f·r − 1
analysis     least squares of y on p over all matched people with a proxy on
             both sides; Pearson r, Spearman ρ, R²; 1000-round seeded bootstrap
             (seed 20260915) for the slope and intercept intervals; the same
             fit per sequence; the residual after the fit
reading      supported   if slope ∈ [0.7, 1.3] and R² ≥ 0.5
             partial     if slope > 0.3 and R² ≥ 0.2
             unsupported otherwise
```

Run on the `annotation_derived` evaluation first, which is where the camera
term is absent; then on the `model_estimated` one, where `f` carries it.
`unproject gt-hypotheses <evaluation>` writes `hypotheses.json`, a
per-sequence CSV and the figure. An association consistent with H1 shows the
residual is proportional to the size mismatch this proxy measures; it does not
prove the prior is the cause, and the text says so.

**H2 — square frames (controlled test, not the TS1–TS5 split).** The five
square sequences carry a larger residual than the fifteen wide ones, and
differ from them in scene and actors as well as shape. Comparing the two
groups cannot separate those. The controlled test transforms the *same*
content: `experiments/ground-truth/mupots-3d-letterbox-square.toml`
(`frame_transform = "letterbox_square"`) pads every wide sequence with black
rows to a square, moves the 2D annotations and the derived principal point
down by the padding, leaves the 3D, the actors and the intrinsics untouched,
and is compared per sequence with the plain `annotation_derived` run by
`unproject gt-compare`. If the depth bias of TS6–TS20 moves, the aspect-ratio
path did it; if it does not, the square sequences' extra offset is about those
sequences and stays a limitation. The residual median of TS1–TS5 against
TS6–TS20 after the H1 fit is still reported by `gt-hypotheses`, labelled
**post-hoc** and confounded.

**H3 — vertical offset (open limit).** The root Y bias grows when the
principal point is supplied. MuPoTS's pelvis and Anny's root are different
anatomical definitions and no alignment between them is established without an
arbitrary choice; the offset is reported per sequence, labelled post-hoc and
descriptive, and left as an open limit per ADR-0005 WP5.5.

No threshold above is revised after the data. An analysis not listed here is
post-hoc and is labelled so wherever it appears.

### 4.5 Reporting rules that apply to everything

Every metric carries dataset, release, subset, sequence and sample counts;
ground-truth people, predictions, matches, false negatives and false positives;
the number of valid joints behind the value; units, alignment, threshold and
aggregation; the confidence threshold and association gate.

**Recall accompanies every accuracy number.** A model that finds one easy person
out of ten must never read as highly accurate.

Micro and macro aggregation are both kept and never conflated. Skewed continuous
errors report median and p90 or the interquartile range; means survive only where
a named standard metric requires one.

---

## 5. Frozen values

These are held fixed from the moment the first model output is inspected. All are
_to be frozen_ until the release files confirm the official values — see §7.

| | Value | Status |
|---|---|---|
| Confidence threshold | the stage [B] default from the experiment config | inherited, frozen |
| PCK radius | 150 mm | frozen, from the official protocol |
| Matcher, headline | `mupots_greedy_2d_40px`: the released evaluator's greedy association, 40 px per axis, fixed by the algorithm | frozen |
| Matcher, diagnostic | `normalized_global_assignment`: minimum-cost assignment on common visible 2D joint distance normalised by the image diagonal, gate `association_gate` (fraction of the diagonal), released 2D box distance or IoU as fallback when common 2D joints are absent | frozen |
| Camera conditions | `model_estimated` (control) and `annotation_derived` (treatment: intrinsics recovered by least squares from the released 2D and 3D; not a published calibration) | frozen |
| Metric protocol | `unproject_mupots_14j_v1` | frozen 2026-09-15 |
| Depth bins | 2.5, 3.5, 4.5, 6.0 m, from the annotated distribution (p5 2.37, median 3.62, p95 5.94) | frozen 2026-09-10 |
| Max temporal interval | _to be decided from the released frame rate_ | open |
| Anny-One subset size | 300 images, minimum acceptable pilot 60 | frozen |
| Seed | recorded in the config | frozen |

**Ground-truth 3D distance is never used to choose a match** for a metric that
later claims 3D accuracy. Association is 2D.

---

## 6. The training-overlap gate

Before any Anny-One result is classified, the official Multi-HMR 2 paper,
checkpoint card, repository and author documentation are inspected and the known
training and model-selection datasets recorded.

- evaluated images may have participated in training or selection →
  **in-domain compatibility diagnostic**
- an official held-out split is documented and provably unused → that split may
  be named as such
- evidence ambiguous → the conservative first label

**Current status: ambiguous, therefore in-domain compatibility diagnostic.**
What was checked and what it said is in
`docs/notes/2026-09-08-ground-truth-sources.md`. The full paper PDF has not been
consulted; doing so is the remaining way to close this gate properly.

This gate never blocks running Anny-One. It blocks presenting the result as
independent generalisation evidence.

---

## 7. Open gates

Nothing below can be settled without the released files in hand. Each is a named
gate, not a guess, and each must be closed in a dated note with a file-level or
official source before the adapter that depends on it is written.

**MuPoTS-3D** — mostly closed on 2026-09-09 by reading the official evaluator.
See `docs/notes/2026-09-09-mupots-release-inspected.md`.

| Gate | Status |
|---|---|
| Joint names, count, ordering | **Closed.** 17-joint `relevant` set, listed in the note. |
| Layout | **Closed.** `TS1`–`TS20`, jpg frames, `annot.mat` + `occlusion.mat`. |
| Image size | **Closed.** TS1–5 are 2048×2048; TS6–20 are 1920×1080. |
| Validity semantics | **Closed.** `isValidFrame` per person per frame. |
| Official evaluator behaviour | **Closed.** Root-relative only; 150 mm; greedy 2D matching at 40 px. |
| Licence | **Closed.** Non-commercial research, no redistribution, two citations required. |
| Units of the 3D annotations | **Closed 2026-09-10.** Millimetres, read off the files. |
| The joint mapping | **Closed 2026-09-10.** 14 joints, both sides sourced, in `mappings/mupots-3d-to-anny-v1.toml`. head_top, head and spine are excluded, so the reported PCK covers 13 of the official 14 and every table must say so. |
| Person identity across frames | **Open.** The evaluator re-matches every frame rather than trusting an id. |
| **Are all people in frame annotated?** | **Closed 2026-09-10. No.** TS9 has three people in every frame and two annotation slots; the third is fully visible and unoccluded. No precision or false-positive figure can come from MuPoTS - only recall. See `docs/notes/2026-09-10-mupots-pilot-ts9.md`. |
| **Is a non-universal `annot3` released?** | **Closed 2026-09-10. Yes.** See below. |

That last gate governed whether the primary benchmark could answer the primary
question, and the answer is yes. Every person carries **four** fields:
`annot2`, `annot3`, `univ_annot3`, `isValidFrame`.

Measured across all 20 sequences, 8,370 frames, 20,760 valid annotated people:
the **pelvis is bit-identical** between `annot3` and `univ_annot3` — maximum
difference 0.0000 mm — while the other joints differ by a median of 89.4 mm.
`univ_annot3` normalises skeleton *size* and leaves the root exactly where it
was.

So:

- **absolute placement and depth** are available in either field, and the
  depth claim this thesis rests on can be measured;
- **per-joint absolute error** uses `annot3`, the subject's true scale;
- the official root-relative PCK uses `univ_annot3` because normalising limb
  length is what removes body size from a pose comparison.

Reported depths run 1.67 m to 7.12 m, median 3.62 m.

**Anny-One**

- annotation schema: pose, shape, camera, scene, person, view, segmentation
- whether annotations can be obtained without every RGB archive (122 GB total)
- the `scenes_with_issues.txt` exclusion list format
- Anny package version, bone count and parameter semantics behind the release,
  and whether they match what the pinned checkpoint emits

**Both**

- licence terms, specifically whether a metadata-only fixture may be redistributed

---

## 7a. Vocabulary, frozen 2026-09-15

Every table, figure, README and derived text follows ADR-0005 §3. In
particular:

| Say | Never say | Because |
|---|---|---|
| Unproject protocol on MuPoTS-3D; MuPoTS-compatible project metric; `unproject_mupots_14j_v1` | official 3DPCK, official protocol | 13 of 14 official joints, `annot3`, project aggregation |
| `mupots_greedy_2d_40px` reproduces the released evaluator's association | official matcher | reproducing the pairing does not make the metric official |
| annotation-derived intrinsics; `annotation_derived` | released camera, true camera, camera told | recovered by least squares from annotations; MuPoTS ships no calibration |
| model-estimated camera; `model_estimated` | guessed camera (in tables) | it is the model's own prediction |
| effect of the intervention on the intrinsics, under this protocol | the camera fixes the depth | one protocol, one dataset |
| body-scale, square-frame and vertical-offset **hypotheses** | the residual *is* the body prior | until WP5's analyses say otherwise |
| smoothness, noise attenuation | accuracy gain from the filter | the filter has no ground truth |
| accuracy over matched people, with coverage and counts | accuracy | a matched-only number without recall misleads |

## 8. Official sources

Only these. Documentation must never point a reader at an unofficial mirror.

- Multi-HMR 2 — <https://github.com/naver/multi-hmr2>, paper arXiv:2606.14841
- MuPoTS-3D — <https://vcai.mpi-inf.mpg.de/projects/SingleShotMultiPerson/>,
  Mehta et al., *Single-Shot Multi-Person 3D Pose Estimation From Monocular RGB*,
  3DV 2018
- Anny-One — <https://europe.naverlabs.com/research/vision-foundation-models-for-human-understanding/anny-one/>
- Anny — <https://github.com/naver/anny>

Official release files, licences and evaluator documentation take precedence over
any secondary description, including this document. A discrepancy found during
implementation is recorded before code or metrics change.

---

## 9. Commands

```shell
uv run unproject gt-evaluate experiments/ground-truth/mupots-3d.toml
uv run unproject gt-evaluate experiments/ground-truth/mupots-3d-annotation-derived-camera.toml
uv run unproject gt-evaluate experiments/ground-truth/mupots-3d-letterbox-square.toml
uv run unproject gt-evaluate experiments/ground-truth/anny-one.toml
uv run unproject gt-summary artifacts/ground-truth/<evaluation-id>
uv run unproject gt-hypotheses artifacts/ground-truth/<evaluation-id>
uv run unproject gt-compare artifacts/ground-truth/<a> artifacts/ground-truth/<b>
```

`gt-evaluate` validates the dataset before loading model weights, resumes
complete per-sequence body artifacts, and never silently overwrites a finished
result. `gt-summary` runs no inference and needs no weights.

A result that enters the thesis is run with `--final` (ADR-0005 WP4): the
working tree must be clean, every sequence must be scored, and every body
artifact's manifest must record a clean tree, or the run is refused before it
loads a model. `gt-summary --regenerate` rewrites every derived view from the
observations, and `gt-compare` shows two evaluations to be the same numbers
under different labels, or draws the paired per-sequence effect of an
intervention. Depth errors are reported as a distribution beside the mean,
with outliers counted and kept (§4.2).

`unproject rq3` remains for the older uncontrolled analyses. It is not the
ground-truth benchmark and must not be presented as one.
