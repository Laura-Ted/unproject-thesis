# 2026-09-11 - Telling the estimator the camera

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

The experiment the full run asked for. Two evaluations that differ in one
line of configuration:

| | config | evaluation | camera |
|---|---|---|---|
| control | `mupots-3d.toml` | `a1a82f0677ec` | the model's own guess |
| treatment | `mupots-3d-released-camera.toml` | `2585bc32e4d9` | recovered from the annotations, per sequence |

Same checkpoint, same mapping, same thresholds, same matcher, same 20
sequences, same 20,190 people. Inference for the treatment ran 15:30-17:04 at
`27e0ec5`; the control's bodies are the ones from the night before, rescored
under the new config schema (the `camera` field changed the evaluation id from
`1c289e996789`; every number is identical).

How the camera gets in: in upstream's decoder, `dist = K[0,0] / exp(net)` -
distance is the focal length divided by an apparent size the network predicts
from features that never see K. So substituting K between encoder and decoder
changes where the bodies are put and nothing else. The control test for that
claim is the first row below.

---

## Observations

```
                        control (guessed)    treatment (told)
3DPCK_rel                  56.87 %              56.87 %        identical
root-relative MPJPE       141.2 mm             141.2 mm        identical
3DPCK_abs                  16.99 %              26.97 %
MPJPE                     352.1 mm             283.6 mm
root MAE   X / Y / Z    87.8 / 99.6 / 293.0   60.6 / 133.1 / 258.5
root bias  X / Y / Z   -29.6 / +81.3 / +72.6  -0.9 / +126.1 / -195.2
recall                     97.3 %               97.3 %
```

**Per sequence, Z bias in millimetres.** `ratio` is the model's guessed focal
over the released one, from the control.

| seq | size | ratio | guessed | told | | seq | size | ratio | guessed | told |
|---|---|---|---|---|---|---|---|---|---|---|
| TS1 | 2048² | 1.005 | −404 | −422 | | TS11 | wide | 1.104 | +142 | −345 |
| TS2 | 2048² | 1.037 | −259 | −346 | | TS12 | wide | 1.111 | +203 | −395 |
| TS3 | 2048² | 1.078 | −235 | −399 | | TS13 | wide | 0.931 | −355 | −88 |
| TS4 | 2048² | 1.014 | −430 | −461 | | TS14 | wide | 1.025 | +60 | −29 |
| TS5 | 2048² | 1.044 | −199 | −337 | | TS15 | wide | 1.081 | +1 | −325 |
| TS6 | wide | 1.140 | +429 | −374 | | TS16 | wide | 1.007 | −64 | −79 |
| TS7 | wide | 1.151 | +332 | −172 | | TS17 | wide | 0.971 | −197 | −121 |
| TS8 | wide | 1.097 | +227 | −7 | | TS18 | wide | 1.001 | −85 | −73 |
| TS9 | wide | 1.220 | **+542** | −146 | | TS19 | wide | 1.183 | **+487** | −92 |
| TS10 | wide | 1.170 | +423 | −153 | | TS20 | wide | 1.027 | +69 | −22 |

| Z bias across the 20 sequences | guessed | told |
|---|---|---|
| mean | +34 mm | −219 mm |
| standard deviation | **299 mm** | **152 mm** |
| range, worst to best | **972 mm** | **454 mm** |
| wide only, sd | 255 mm | 129 mm |
| square only, mean | −305 mm | −393 mm |

**By true distance, treatment only.** Signed depth bias as a fraction of the
annotated distance: −7.5 %, −5.0 %, −4.4 %, −5.2 %, −6.9 % across the five
bins from under 2.5 m to over 6 m.

**Sign.** In the control, 12 sequences were too far and 8 too close. In the
treatment, **all 20 are too close**.

---

## Interpretations

**The focal length was the between-camera variation, and it is gone.** The
three sequences the model had placed half a metre too far - TS9, TS19, TS6,
with guessed focals 14-22 % too long - come back to −146, −92 and −374 mm.
The spread across cameras halves, by standard deviation and by range. What
was a per-camera number is now, to a first approximation, one number.

**The pose did not move, which is what proves the intervention was clean.**
3DPCK<sub>rel</sub> and root-relative MPJPE are identical to the last decimal.
If the wrapper had disturbed anything but the camera, they could not be. This
is the first row of the table and the reason the rest of it can be believed.

**What is left is a constant proportion, not a constant offset.** With the
true camera every sequence is too close, and by 4-7 % of the distance
regardless of how far away the person is. A constant offset would show up as
a fixed number of millimetres shrinking as a fraction with distance. This
does not shrink. A fixed proportion is what a wrong *body size* produces: the
model assumes a body of some height, sees it cover some pixels, and with the
focal length now correct the only free quantity left is how big the body is.
Too close means the assumed body is smaller than the actors.

So the depth error has two parts and they have now been separated:

1. a **camera** part, per sequence, ranging over a metre, removed by
   supplying the intrinsics;
2. a **body-scale** part, roughly −5 %, the same everywhere, which the camera
   cannot touch because it is not about the camera.

**The first part is fixable outside the model and the second is not.** A
calibrated camera, a phone's EXIF, a known lens: all supply the first. The
second is the estimator's prior and would need either a taller assumed body
or a per-subject height, which a single image does not contain.

**3DPCK<sub>abs</sub> 17 → 27 % is the practical size of the fix on this
protocol.** Smaller than the halving of the bias would suggest, because the
150 mm radius is tight against a residual 5 % of 3-4 m, which is 150-200 mm.
The radius was chosen for articulation, not placement, and reads placement
harshly.

---

## Anomalies

**The square sequences did not improve. They got worse.** TS1-TS5 went from
a mean of −305 mm to −393 mm. Their guessed focal was already within 1-8 % of
the released one, so there was little for the correction to do on Z - and the
released principal point moved them further. Their residual is −12 %, against
−5 % for the wide sequences, exactly the gap the full-run note measured as
the intercept difference. **The camera does not explain it.** That leaves the
body-scale reading, which would need the five indoor actors to be about 7 %
taller than the outdoor ones relative to the prior - or something in how a
square input is handled that a focal length does not reach. The synthetic test
proposed yesterday still stands and is now the right next step, because the
cheap explanation has been ruled out.

**Y bias went up, +81 → +126 mm.** The released principal point sits at
cy ≈ 528 on the wide sequences, 12 px above the image centre the model
assumes, and using it moved every person's Y by a consistent amount in the
wrong direction. Either the annotation's pelvis and Anny's root differ in
height by a few centimetres - a definitional offset that the correct camera
exposes rather than causes - or the recovered cy is absorbing something else.
Not resolved. X bias, by contrast, went from −30 mm to −1 mm.

**The treatment's focal-ratio figure is degenerate**, every point at ratio
1.000, because the ratio compares the artifact's camera to the released one
and here they are the same by construction. It is correct and uninformative.
The comparison that matters is the paired one above, built from the two
`per-sequence.csv` files; no single-evaluation figure can draw it.

**26 fewer unmatched predictions** (14,551 → 14,525). Moving the bodies moves
their projections slightly; a handful of 40 px matches flipped. Noise.

---

## What the thesis can now say

Yesterday: the depth error correlates with the focal error at r = 0.96.

Today: supply the focal length and the between-camera error is gone, the
pose is untouched, and what remains is a uniform −5 % that is the model's
body-size prior. The correlation was the cause.

Stated as a claim: *a monocular body estimator's absolute depth error on
MuPoTS-3D decomposes into a per-camera term, up to ±0.5 m, that vanishes
when the camera is known, and a per-model term of about −5 % of distance that
does not.*

## Open

- The square sequences' extra −7 %. The camera is now excluded; test the
  aspect-ratio path with a synthetic body.
- The Y offset. Compare Anny's root height against the MuPoTS pelvis on a
  standing frame.
- Whether the −5 % is the Anny rest-pose height. Measurable: the median
  predicted body height over 20,190 people against the median annotated one.
