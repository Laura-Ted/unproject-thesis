# 2026-09-10 - The MuPoTS pilot on TS9, and two things it found

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

ADR-0004 §9 WP2's review gate: run one sequence end to end and inspect the
numerical invariants before any metric exists.

```shell
uv run unproject ground-truth-bodies experiments/ground-truth/mupots-3d.toml --sequence TS9
```

> Superseded the same day by `gt-evaluate --sequence TS9`, which runs the
> same stage [B] path and then scores it; see
> `2026-09-10-first-ground-truth-numbers.md`. The numbers below are
> unaffected - the body artifact is reused, not recomputed.

```
fitted TS9: 1503 detection(s) over 501 frame(s), 3 identities, against 1002 annotated
       released camera fx=1103.5 fy=1101.4 cx=942.1 cy=528.4, residual 0.217 px over 17034 joints
       artifacts/mupots-TS9/bodies/5dfdda5b0305
```

## The invariants that had to hold, and did

**Frames line up.** 501 annotated, 501 estimated. `run_sequence` refuses when
they do not, because that is the one error a later metric cannot see: every
frame would still be scored against *some* annotation, just not its own, and the
result would look like an error rather than an exception.

**The released 3D and 2D describe the same camera.** Fitting a pinhole to all
17,034 annotated joints leaves 0.217 px RMS, worst single joint 1.54 px, median
per-person worst 0.68 px. Sub-pixel over ten thousand points is not something a
wrong unit or a flipped axis survives — this is the check that replaces looking
at a rendered skeleton.

**The image-size table reads the right way round.** The recovered optical centre
is (942.1, 528.4). The frame is 1920x1080, whose centre is (960, 540). Had the
official `[rows cols]` table been read as `(width, height)`, the sequence would
have been treated as 1080x1920 and the centre would have landed nowhere near.

**Depths are plausible.** Annotated pelvis depth over the sequence runs
3.09-4.16 m, median 3.66 m, across 1,002 annotated people. Metres, not
millimetres, and not the 1e-3 of a silent double conversion.

## Finding 1: the estimator invents a focal length 22% too long

Multi-HMR 2 is monocular and estimates its own intrinsics. Over TS9 it produced
fx between 1221.9 and 1423.0, against a released 1103.5 — a median ratio of
**1.220**.

Depth and focal length trade off directly in a monocular fit: a person of
plausible size, imaged at a given number of pixels, is placed further away when
the assumed focal length is longer. So this predicts a systematic overestimate
of absolute distance, of roughly that factor, before a single metric is
computed. That is precisely the quantity this thesis exists to measure, and
seeing it fall out of the pilot is the first independent sign that the depth
jitter measured in RQ2 was measuring the right thing.

It is a prediction, not a measurement. The measurement needs person-to-person
matching, which is WP3, and is deliberately not done here so the number is not
arrived at by looking for the one that was expected.

## Finding 2: MuPoTS does not annotate everyone in frame

TS9's `annot.mat` is 501x2. Three people are detected in every one of the 501
frames, all at confidence 1.00, as three stable tracks at image x ≈ 990, 1300
and 1655, with median box heights 323, 402 and 615 px. The two annotated people
sit at x ≈ 1030 and x ≈ 1600. The middle one is annotated nowhere.

Looking at `img_000000.jpg` settles it: a woman in a dark coat walks through the
centre of the shot, fully visible, unoccluded, closer to the camera than one of
the two annotated subjects. She is a person, not a false positive.

Slot counts across the release:

| slots | sequences |
|-------|-----------|
| 2 | TS1-TS6, TS8-TS12 |
| 3 | TS7, TS13-TS20 |

8,370 frames and 20,760 annotated people in total. The count is the number of
*annotated* subjects, and TS9 proves it is not the number of people present.

**Consequence, and it is not small.** No precision, false-positive rate or
detection-accuracy figure can be computed from MuPoTS. An unmatched prediction
here means "the release did not annotate this person" at least as often as it
means "the model was wrong", and there is nothing in the data that separates the
two. Any such number would be measuring the annotation policy.

This also explains something in the official evaluator that had looked like an
odd choice: `EVALUATION_MODE = 1` scores only ground-truth people that a
prediction matched. That is not leniency, it is the authors accommodating their
own annotation policy.

ADR-0004 §6 asks for detection metrics alongside the pose metrics. For MuPoTS
they must be reported as recall only - of the annotated people, how many were
found - and the report must say why the other half of the confusion matrix is
absent. The controlled RQ3 recordings, where the number of people is known
because the author counted them, remain the only place precision can be claimed.

## Still open

The frame rate. `experiments/ground-truth/mupots-estimator.toml` declares 30 fps
because ingest requires a positive rate for a directory of jpgs, with a comment
saying it is not sourced. Nothing above depends on it. MPJVE does, and stays
unreported until the rate comes from the 3DV 2018 paper or the authors.
