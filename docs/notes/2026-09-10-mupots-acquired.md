# 2026-09-10 - MuPoTS-3D is here, and the decisive gate is closed

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

The dataset was downloaded at the author's explicit instruction. The licence
obligations — non-commercial use, no redistribution, and citing Mehta et al.
3DV 2018 and 3DV 2017 — attach to the use, not to who ran the command, and are
the author's either way.

## Acquisition record

Per ADR-0004 §4.3:

| | |
|---|---|
| File | `MultiPersonTestSet.zip` |
| Source | `http://gvv.mpi-inf.mpg.de/projects/SingleShotMultiPerson/` |
| Bytes | 5,642,949,208 |
| sha256 | `0f5efe136e3692420c904b08c52fa42c7e70a27f955fbb2e7067909f848db136` |
| Acquired | 2026-09-10 |
| Contents | 8,431 files: 8,370 jpg frames, 20 `annot.mat`, 20 `occlusion.mat` |
| Extracted to | `datasets/mupots-3d/MultiPersonTestSet/`, 5.3 GB, gitignored |

No official checksum is published, so the hash above is locally computed and
identifies what was actually obtained rather than what was meant to be.

The archive was deleted after extraction; the machine had 16 GB free and both
copies would not have left comfortable room.

## The gate that decided the thesis

The open question was whether MuPoTS ships anything other than `univ_annot3`,
the *universal* coordinates with the skeleton rescaled to a canonical size. If
it did not, the primary benchmark could support root-relative accuracy but not
the absolute depth claim this thesis is about, and ADR-0004 §10's fallback
would have applied.

**It does.** Every annotated person carries four fields:

```
annot2   (2, 17)   pixels
annot3   (3, 17)   millimetres, subject scale
univ_annot3  (3, 17)   millimetres, canonical skeleton
isValidFrame
```

And the relationship between the two 3D fields is sharper than expected.
Measured over all 20 sequences, 8,370 frames and 20,760 valid annotated people:

| | |
|---|---|
| Pelvis difference, `annot3` vs `univ_annot3` | **0.0000 mm**, maximum |
| Other joints | median 89.4 mm, maximum 163.7 mm |

`univ_annot3` normalises skeleton **size** and leaves the **root exactly where
it was**. Not a rescaling of the whole scene, as the name had suggested.

So:

- **absolute placement and depth** are measurable, from either field;
- **per-joint absolute error** should use `annot3`, the subject's true scale;
- the official root-relative PCK uses `univ_annot3` because normalising limb
  length is precisely what takes body size out of a pose comparison.

That last point is worth keeping for the thesis: it explains *why* the standard
benchmark metric is built the way it is, rather than treating it as arbitrary.

## What the release actually contains

Twenty sequences, 8,370 frames, 20,760 valid annotated people. The annotation
array is (frames × person slots), with 2 slots in TS1–TS12 and 3 in most of
TS13–TS20; `isValidFrame` marks which slots hold a real person in that frame.

| | frames | slots | valid people |
|---|---:|---:|---:|
| TS1–TS5 | 201–401 | 2 | 402–802 each |
| TS11 | 701 | 2 | 1,402 |
| TS15 | 801 | 3 | 2,287 |
| TS18 | 126 | 3 | 378 |

The 17 joints are the `relevant` set the evaluator declares, and index 14
(0-based) is the pelvis, matching the evaluator's own root alignment.

MAT format 5.0, created 2018-02-01 — plain `scipy.io.loadmat`, not HDF5, and no
pickle anywhere. ADR-0004 §5.7's warning about deserialising untrusted formats
does not bite here.

## Depth bins, now from data rather than round numbers

Pelvis depth runs **1.67 m to 7.12 m**, median 3.62, p5 2.37, p95 5.94. The
placeholder bins of 2/4/6/8 m would have left the outermost nearly empty and
lumped half the dataset into one. Frozen instead at **2.5, 3.5, 4.5, 6.0 m**,
which splits the actual distribution.

## Still open

The frame rate. The release ships numbered jpgs with no stated rate, so the
maximum temporal interval for velocity metrics is still a placeholder at 0.2 s.
It has to be settled before any MPJVE number is reported.
