# 2026-09-10 - The first numbers against a public ground truth

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

WP3 of ADR-0004: matching, metrics, and the artifact they are written into. Run
over two sequences, TS9 and TS10 - 752 frames, 1,504 annotated people - as the
pilot before the full twenty.

```shell
uv run unproject gt-evaluate experiments/ground-truth/mupots-3d.toml --sequence TS9 --sequence TS10
uv run unproject gt-summary artifacts/ground-truth/2afbfbac6372
```

```
coverage [official matcher]: 1504 of 1504 annotated people scored (100.0% recall)
                             over 752 frame(s); 871 prediction(s) unmatched
  official 3DPCK_rel   61.56 % micro    61.56 % macro   (21056 joints, 1504 people)
  project  3DPCK_abs    0.10 % micro     0.10 % macro
  MPJPE                538.4 mm       root-relative  134.9 mm
  root MAE mm         X   94.7  Y   86.8  Z  502.2
  root bias mm        X   -94.4  Y   +85.7  Z  +502.2
```

## What the two headline numbers mean, and why they are so far apart

**3DPCK<sub>rel</sub> 61.6%** is the official protocol: both skeletons moved to
a common pelvis, then joints counted within 150 mm. It asks whether the pose is
right.

**3DPCK<sub>abs</sub> 0.1%** is the same 150 mm radius with nothing aligned. It
asks whether the person is in the right place. Almost no joint passes, and the
reason is one number: the root sits **502 mm too far away**, every time.

A 62% and a 0.1% from the same predictions on the same frames is the clearest
statement of the thesis's premise that has come out of this project. A model
can know what a body is doing and still be half a metre wrong about where it is.

## The bias has a sign, and it never changes it

X is biased -94 mm, Y +86 mm, Z **+502 mm**. On Z the mean absolute error
(502.2) and the mean signed bias (+502.2) are the same number to a tenth of a
millimetre, which means **not one of the 1,504 people was predicted too close**.
This is not scatter. It is an offset.

## What cannot yet be concluded, and why the full run is needed

WP2 predicted this from the focal length: Multi-HMR estimated fx ≈ 1345 against
a released 1103, 22% long, and depth trades directly against focal length in a
monocular fit. A wrong focal produces a bias **proportional** to distance.

`figures/depth-bias.svg` plots the signed error against true distance with both
readings drawn on it: a constant +502 mm line and a proportional +13.8% line.
**Over this data they are indistinguishable.** TS9 and TS10 span 3.1 m to 4.3 m,
and no scatter over a 1.2 m range can separate a constant offset from a
proportion of the distance.

The full release spans 1.67 m to 7.12 m. That is more than a factor of four, and
the two lines diverge by roughly a metre across it. So the question "is
Multi-HMR's depth error a fixed offset or a fixed percentage?" is answerable,
just not yet, and the answer is worth the run: a proportional error is a camera
calibration problem with a known fix, and a constant one is not.

Recorded here so the 13.8% figure is not quoted later as if it had been
measured. It has not been.

## Left and right, checked rather than assumed

The per-joint table showed the right side consistently worse than the left -
right wrist 191 mm against left wrist 139 mm, right knee PCK 22% against left
knee 47%. Four independent joints all leaning the same way is what a swapped
left/right convention looks like, so it was tested rather than explained away.

A probe mapping with the Anny side's `.L` and `.R` exchanged, everything else
identical:

| | shipped mapping | left/right swapped |
|---|---|---|
| 3DPCK<sub>rel</sub> | **61.56 %** | 10.51 % |
| root-relative MPJPE | **134.9 mm** | 367.2 mm |

Swapping makes it six times worse. The mapping is right and the asymmetry is in
the data - TS9 and TS10 have subjects turning, and one side spends more time
occluded. The probe files were deleted; this table is what they were for.

## Why this 61.6% is not the published number

It should not be compared directly with Multi-HMR's reported MuPoTS score, for
two reasons that are both in the method rather than in the model:

1. **13 of the official 14 PCK joints.** Anny has no crown-of-the-skull bone, so
   `head_top` is not mapped (see the mapping file's own notes).
2. **The predicted joints are Anny bone origins, not Human3.6M joint centres.**
   "Where the shin begins" and "the knee joint as H36M defines it" are close but
   not identical, and a systematic few centimetres per joint is a large share of
   a 134.9 mm root-relative error.

Neither touches the absolute depth result, which is about the pelvis, and the
pelvis is the one joint both sides name explicitly.

## Two matchers, and a bug they exposed

ADR-0004 §5.6 asks for the released evaluator's association *and* a project
one. Both are implemented:

- **official** - greedy, each ground-truth person in source order takes the
  prediction with the most joints within 40 px on each axis. Reproduced rather
  than improved, including its known weakness: an early person can take a
  prediction a later, closer person needed. A unit test pins that case.
- **native** - minimum-cost bipartite assignment on distance normalised by the
  image diagonal, so the gate means the same thing at 2048x2048 as at 1920x1080.

On TS9 and TS10 they agree exactly. The people are well separated, which is
where greedy and optimal cannot differ.

Getting there exposed a real bug. Both matchers first wrote to the **same
output directory**: the choice lived in a CLI flag, so it never reached the
identity hash, and running the second silently claimed to be the first. The same
was true of `--sequence`: one sequence and all twenty produced the same
evaluation id. Both are now part of the identity - the matcher by moving into
the config file where the project's reproducibility rules say a parameter belongs, the sequence list
by being hashed with it.

## What is in the artifact

```
artifacts/ground-truth/<id>/
  evaluation.json     descriptor, identity, coverage, written last
  observations.npz    one row per matched person, one per frame
  per-sequence.csv    derived
  per-joint.csv       derived
  figures/            derived
```

`observations.npz` is the source of truth and holds no aggregates and no object
arrays. The CSVs and figures regenerate from it byte for byte - checked, not
assumed, both by a test and by rerunning the evaluation and comparing the SVGs.

## The command surface changed

`ground-truth-bodies`, added in WP2, is gone. ADR-0004 §7.3 fixes the surface at
`gt-evaluate` and `gt-summary`, and `gt-evaluate --sequence` does what the old
command did and then scores it. The WP2 note's command line no longer runs; its
numbers are unaffected, since `gt-evaluate` reuses the same body artifacts.

`gt-summary` reads a finished evaluation with no dataset, no weights and no
inference, which is what makes a result checkable by someone who cannot run it.

## Still open

- **The full twenty sequences.** WP4. It is what separates the constant offset
  from the proportional one.
- **The frame rate**, so MPJVE can be computed at all. Unchanged from WP2.
- **Detection precision** stays impossible on MuPoTS: 871 unmatched predictions
  over these two sequences, and TS9 alone accounts for 501 of them - one real,
  unannotated person, once per frame. See the WP2 note.
