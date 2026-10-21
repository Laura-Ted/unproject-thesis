# 2026-09-11 - MuPoTS-3D, all twenty sequences

> **Terminology, added 2026-09-15 (ADR-0005 WP1).** This note predates the
> schema-2 vocabulary and keeps its original wording. Read it with these
> substitutions: the metric it calls *official 3DPCK* is the **Unproject
> protocol on MuPoTS-3D** (`unproject_mupots_14j_v1`, a MuPoTS-compatible
> project metric: 13 of the 14 official PCK joints plus pelvis, root-relative on
> `annot3`); the matcher it calls *official* is `mupots_greedy_2d_40px`, which
> reproduces the released evaluator's association and nothing else; the camera
> it calls *released*, *true* or *told* is **annotation-derived** intrinsics,
> recovered by least squares from the released 2D and 3D annotations, and the
> one it calls *guessed* or *estimated* is **model-estimated**. Evaluation ids
> quoted here are schema-1 ids; the same bodies rescored under schema 2 land
> under new ids. The numbers are unchanged.

WP4 of ADR-0004. The primary real-video benchmark, complete. Per the ADR this
is the minimum complete thesis result; everything after it is reinforcement.

```shell
uv run unproject gt-evaluate experiments/ground-truth/mupots-3d.toml
uv run unproject gt-summary artifacts/ground-truth/1c289e996789
```

> Later the same day the evaluation config gained a `camera` field for the
> experiment in `2026-09-11-the-camera-experiment.md`, which changed this
> evaluation's id to `a1a82f0677ec`. Rescored from the same bodies; every
> number below is identical, and the id in the table headers is the old one.

Inference ran 22:59 to 00:28 on 2026-09-10/11, 89 minutes on the M3 Pro, at
code revision `49c2dea` with a clean tree. 8,370 frames, 0.72 s per frame. The
scoring was then regenerated from the cached body artifacts with the figure
and columns added below; every value was identical, which is the acceptance
criterion "rerunning aggregation performs no inference and produces identical
values", met.

Identity: checkpoint `fd2d9ab7010a`, mapping `eb66b465b801`, annotation
inventory `25e98ea86b61`. Nothing excluded.

The ADR asks that this note separate what was observed from what it is taken
to mean from what does not fit. It does, in that order.

---

## Observations

```
coverage [official matcher]: 20190 of 20760 annotated people scored (97.3% recall)
                             over 8370 frame(s); 14551 prediction(s) unmatched
  official 3DPCK_rel   56.87 % micro    56.87 % macro   (282660 joints, 20190 people)
  project  3DPCK_abs   16.99 % micro    16.99 % macro
  MPJPE                352.1 mm       root-relative  141.2 mm
  root MAE mm         X   87.8  Y   99.6  Z  293.0
  root bias mm        X   -29.6  Y   +81.3  Z   +72.6
```

**Per sequence.** Root-relative error is nearly the same everywhere. Absolute
depth bias is not.

| seq | size | recall | PCK<sub>rel</sub> | PCK<sub>abs</sub> | MPJPE<sub>rel</sub> mm | **Z bias mm** | fx released | fx estimated |
|---|---|---|---|---|---|---|---|---|
| TS1 | 2048² | 1.000 | 51.6 | 3.2 | 151.9 | **−404** | 1503 | 1511 |
| TS2 | 2048² | 0.946 | 52.8 | 13.6 | 148.2 | **−259** | 1507 | 1563 |
| TS3 | 2048² | 0.994 | 53.7 | 19.3 | 150.2 | **−235** | 1493 | 1609 |
| TS4 | 2048² | 0.995 | 54.8 | 1.9 | 155.7 | **−430** | 1492 | 1513 |
| TS5 | 2048² | 0.967 | 61.8 | 20.9 | 133.1 | **−199** | 1495 | 1561 |
| TS6 | 1920×1080 | 1.000 | 66.8 | 0.0 | 130.6 | **+429** | 1133 | 1291 |
| TS7 | 1920×1080 | 1.000 | 44.3 | 14.1 | 148.4 | **+332** | 1105 | 1271 |
| TS8 | 1920×1080 | 0.942 | 57.9 | 30.9 | 140.3 | **+227** | 1124 | 1233 |
| TS9 | 1920×1080 | 1.000 | 62.1 | 0.0 | 134.4 | **+542** | 1103 | 1346 |
| TS10 | 1920×1080 | 1.000 | 60.4 | 0.2 | 135.9 | **+423** | 1103 | 1291 |
| TS11 | 1920×1080 | 0.999 | 51.9 | 25.3 | 145.6 | +142 | 1114 | 1230 |
| TS12 | 1920×1080 | 1.000 | 54.4 | 5.3 | 148.5 | +203 | 1114 | 1237 |
| TS13 | 1920×1080 | 0.950 | 56.5 | 3.8 | 145.3 | **−355** | 1103 | 1027 |
| TS14 | 1920×1080 | 0.851 | 63.4 | 49.6 | 136.5 | +60 | 1124 | 1152 |
| TS15 | 1920×1080 | 1.000 | 50.1 | 25.6 | 152.5 | +1 | 1101 | 1190 |
| TS16 | 1920×1080 | 0.989 | 63.1 | 25.4 | 131.0 | −64 | 1138 | 1145 |
| TS17 | 1920×1080 | 1.000 | 56.6 | 22.0 | 136.1 | −197 | 1138 | 1105 |
| TS18 | 1920×1080 | 0.963 | 59.1 | 8.1 | 135.0 | −85 | 1192 | 1192 |
| TS19 | 1920×1080 | 0.995 | 61.0 | 7.0 | 136.1 | **+487** | 1104 | 1305 |
| TS20 | 1920×1080 | 0.944 | 57.9 | 1.4 | 135.7 | +69 | 1191 | 1223 |

Root-relative MPJPE spans 130.6-155.7 mm across the twenty. Z bias spans
−430 to +542 mm. Within a sequence the sign is consistent - TS9's 1,002 people
were all too far, TS4's 599 all too close - and across sequences it flips.

**Focal length.** `fx released` is recovered from the annotations by
least-squares (sub-pixel residual on every sequence). `fx estimated` is the
median of what the model produced per frame. Across the twenty, the correlation
between (estimated/released − 1) and the relative Z bias is **r = +0.84**.
Splitting by image size:

| sequences | slope | intercept | residual sd | r |
|---|---|---|---|---|
| TS6-TS20, 1920×1080 | +0.80 | −2.8 % | 1.9 % | **+0.96** |
| TS1-TS5, 2048×2048 | +0.60 | **−11.9 %** | 1.6 % | +0.69 |

`figures/focal-ratio-vs-depth-bias.svg` is this table as a picture: one point
per sequence, the fifteen wide ones on a line, the five square ones on a second
line below it.

**Per joint** (root-relative, mm): hips 111-113, shoulders 120-122, neck 141,
elbows 149-161, ankles 165-171, wrists 169-183, knees 184-188. The left/right
gap seen on TS9+TS10 alone (wrist 139 vs 191) closed to 169 vs 183 over the
full set.

**Coverage.** 570 annotated people unmatched (2.7%), concentrated in TS14
(279 of them, recall 0.851). 14,551 predictions unmatched, of which TS15 alone
holds 4,523 - it has 6,810 predictions for 2,287 annotated people.

---

## Interpretations

**The pose is roughly right and roughly the same everywhere; the placement is
wrong by an amount set by the camera.** That is what a root-relative error
varying by 25 mm and an absolute bias varying by 970 mm across the same twenty
sequences means. A model property would be constant across cameras. This is
not.

**The mechanism is the focal length, and for the fifteen wide sequences it is
nearly the whole story.** Multi-HMR is monocular and invents its own
intrinsics. In a monocular fit, a body of plausible size that covers a given
number of pixels is placed further away when the assumed focal length is
longer; the ratio of focal lengths is the ratio of depths. On TS6-TS20 the
relative depth bias follows the focal error with r = +0.96 and a residual
scatter of 1.9%. The slope of 0.80 rather than 1.00 says the model's focal
guess is partly compensated by something else - a body-size prior, most
likely - but four fifths of the bias is the camera.

This changes what the thesis can say. The pilot on TS9+TS10 found "+502 mm,
never too close" and the WP3 note deliberately refused to call it either a
constant or a proportion. It was neither. It was **one camera's** focal error,
and the full run shows other cameras producing the opposite sign. A number
quoted from two sequences would have been wrong in kind, not just in size,
which is the reason ADR-0004 §9 insists on all twenty.

**A consequence worth stating plainly.** If the bias is the focal length, then
supplying the true focal length - which a calibrated camera, a phone's EXIF, or
a known lens all provide - should remove most of it. That is a fixable error,
and the fix is outside the model. Whether Multi-HMR can be given intrinsics
rather than guessing them is an implementation question, not a research one.

**3DPCK<sub>abs</sub> 17.0% against 3DPCK<sub>rel</sub> 56.9%** is the same
gap the pilot showed, less extreme because across twenty cameras the biases
partly cancel. The gap is the thesis's premise measured: articulation and
placement are different competencies and the model has much more of one than
the other.

**Recall 97.3% is a detection result; precision is not computable here.** The
14,551 unmatched predictions are dominated by unannotated people - TS15's
three-to-one ratio is not a model hallucinating 4,500 bodies - and the WP2
note established that MuPoTS annotates a subset of who is in frame. So the
unmatched count is reported and not interpreted.

---

## Anomalies

**The five square sequences sit 12% too close, beyond what their focal error
explains.** TS1-TS5 have focal ratios of 1.00-1.08, which on the wide
sequences' line would predict a bias between −3% and +4%. They show −7% to
−14%. Their own fit has an intercept of −11.9%. Something about these
sequences, other than the focal length, pulls every person closer.

Not resolved here. Candidates, none tested:

1. *Subject scale.* TS1-TS5 are the indoor studio sequences with different
   actors. If they are larger than the body-size prior, a monocular fit places
   them closer. A 12% height difference is a lot for adults, so this alone
   seems short.
2. *Square input handling.* The model resizes to a 768 px long side and pads
   to a multiple of the patch size; a square image is the one case with no
   padding on either axis. `model_space_geometry` mirrors upstream's code path
   and the 2D association worked (recall 1.000 on TS1), so the *image-plane*
   geometry is right. That leaves the depth-from-scale step.
3. *A principal-point or aspect assumption* in how the model turns its focal
   guess into a translation. The recovered optical centre for TS1 is at the
   image centre, so the *released* camera is not the odd one.

The right next step is small: hold the focal length fixed and vary only the
image aspect on a synthetic person of known size, and see whether the
predicted depth moves. If it does, it is (2) or (3) and it is a bug worth
reporting upstream. If not, it is (1) and it is a limitation.

**TS14 recall 0.851**, the lowest by a margin. 279 annotated people
unmatched. Not looked at. Might be occlusion, might be people at the frame
edge. Recorded.

**Outliers at +4 m.** The per-person depth scatter has a handful of points
near +4,000 mm - a person placed at 9 m who was annotated at 5 m. With 20,190
people a few gross mis-associations are expected, and the 40 px official
matcher does not gate on plausibility. They are in `observations.npz` and
counted in every mean; a robust statistic would discount them, and the ADR
says to report medians for skewed errors, which the summary does not yet do.

**The evaluation descriptor for the regenerated scoring records
`git_dirty: true`**, because the columns and figure were added after the
inference run and the scoring re-ran on the uncommitted tree. Inference
itself was at `49c2dea`, clean. Fixed by committing and regenerating once
more; the values do not change.

---

## What the thesis can now claim, and what it cannot

Can: on a public multi-person 3D benchmark with released ground truth, a
current monocular body model recovers articulation to ~140 mm root-relative
regardless of camera, and recovers absolute placement to a bias that ranges
over nearly a metre depending on the camera, with the focal-length error
accounting for r = 0.96 of that range on the fifteen sequences of one aspect
ratio.

Cannot: a single "the model is X mm too far" number. There is no such number;
there is a per-camera one. Cannot claim precision. Cannot yet claim a velocity
metric - the frame rate is still unsourced. Cannot yet explain the square
sequences.

## Open

- The square-sequence offset, above.
- Frame rate, so MPJVE exists.
- Medians alongside means for the depth error, per §6.5.
- Whether Multi-HMR can accept intrinsics, which decides whether the bias is
  a fix or a finding.
