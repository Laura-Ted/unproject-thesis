# 2026-09-09 - What is actually in the MuPoTS-3D release

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

Downloaded and read `mupots-3d-eval.zip` from the official project page — 10,853
bytes, sha256 `5a1083c5e96cbdaeccc9b14a661712369324f08fc5dd87b12c52c328b12c1171`.

That file is not the dataset. It is the licence, a download script, and the
**official MATLAB evaluator**, which turns out to answer most of the open gates
in `docs/experiments/ground-truth-protocol.md` §7 — and to contradict one
assumption in ADR-0004.

Nothing from the archive is committed. The licence forbids redistribution, so
what follows is facts and hashes, which §4.3 permits.

## The licence, which is the author's to accept

`license.txt`, MPI for Informatics, 2018. The parts that bind this project:

- **non-commercial research only**, and that extends to "methods and models that
  make use of the provided Software in any way";
- **redistribution is not permitted** — "publication, redistribution and
  (re)selling of the software, of modifications, extensions, and derivatives of
  it… are not permitted";
- **two citations are required** in any resulting publication: Mehta et al. 3DV
  2018, and Mehta et al. 3DV 2017.

`README.txt` adds that results must not be published in non-scientific
literature, and asks users to "respect our actors and not to use the data for any
distasteful manipulations".

An undergraduate thesis is non-commercial research, so the intended use fits. The
citation requirement is not optional and belongs in the bibliography now, not at
the end.

**Consequence for ADR-0004 §8.2:** no metadata-only fixture may be redistributed.
The stated fallback applies — an equivalent synthetic fixture matching the
documented schema, with hashes of the real files kept for local integration
testing.

**Consequence for §5.6:** the official evaluator is readable and its behaviour
can be pinned by the archive hash above, but it cannot be vendored into this
repository. Porting it and pinning the source hash is the licence-compatible
route.

## How the data is actually obtained

`get_mupots-3d.sh` is four lines, and three of them are **commented out on
purpose**:

```
echo "Please ensure you have about 6GB free. Uncomment the lines below to proceed"
#source_path="http://gvv.mpi-inf.mpg.de/projects/SingleShotMultiPerson"
#if [ ! -f "./MultiPersonTestSet.zip" ]; then
#  wget "$source_path/MultiPersonTestSet.zip"
#fi
```

Uncommenting is the affirmative acceptance gesture. That is the author's action
under ADR-0004 §4.1 and it has not been performed. The dataset is
`MultiPersonTestSet.zip`, about 6 GB.

## Gates now closed

**Layout.** `MultiPersonTestSet/TS1/` … `TS20/`, `.jpg` frames, with `annot.mat`
and `occlusion.mat` per sequence.

**Image sizes, and there are two.** TS1–TS5 are 2048×2048; TS6–TS20 are
1920×1080. A single hardcoded resolution would have been wrong for three quarters
of the dataset.

**Annotation fields.** `annot2` (2D), `univ_annot3` (3D), `isValidFrame`.

**The joint set.** 28 joints are defined; evaluation uses the 17-joint
`'relevant'` set, "Human3.6m joints in CPM order":

| # | joint | # | joint | # | joint |
|---|---|---|---|---|---|
| 1 | head_top | 7 | left_elbow | 13 | left_knee |
| 2 | neck | 8 | left_wrist | 14 | left_ankle |
| 3 | right_shoulder | 9 | right_hip | 15 | **pelvis** |
| 4 | right_elbow | 10 | right_knee | 16 | spine |
| 5 | right_wrist | 11 | right_ankle | 17 | head |
| 6 | left_shoulder | 12 | left_hip | | |

This is the MuPoTS half of the joint mapping, sourced. It covers everything
ADR-0004 §5.5 said to start from — pelvis, shoulders, elbows, wrists, hips,
knees, ankles, neck, head — and the Anny half is still open.

**PCK.** Threshold 150 mm, confirming the frozen value. An AUC is also computed
over 0–150 mm in 5 mm steps. The PCK joint groups cover **14** of the 17 joints:
head, neck, shoulders, elbows, wrists, hips, knees, ankles. Pelvis, spine and
head(17) are in the pose but not in any PCK group.

**Matching.** 2D, greedy rather than optimal: for each ground-truth person, the
prediction sharing the most joints within **40 pixels on each axis**, counting
only joints visible in both. Threshold hardcoded at the call site.

**Two official modes**, confirming what ADR-0004 §6.2 expected:
`EVALUATION_MODE = 0` evaluates all annotated people, `1` only those matched to a
prediction.

## The finding that changes the plan

The official evaluator computes **3DPCK<sub>rel</sub> only**. Both sides are
root-centred before any error is taken:

```
pred_pose_3d{k} = pred_pose_3d{k} - repmat(pred_pose_3d{k}(:,15), 1, 17);
P = gt_pose_3d{k}(:,1:num_joints) - repmat(gt_pose_3d{k}(:,15), 1, num_joints);
```

Joint 15 is the pelvis. There is no `3DPCKabs` in this release — that metric
comes from later work (Moon et al., RootNet), not from the dataset authors.

ADR-0004 §6.2 says to "implement and report the official protocol first,
including… 3DPCKabs". **That assumption is wrong**, and §13 says a discrepancy is
recorded before code or metrics change. So it is recorded here.

It does not damage the plan; it clarifies it:

- **3DPCK<sub>rel</sub>** is the official headline and stays comparable with
  published numbers;
- **3DPCK<sub>abs</sub>, root error and depth bias** — the metrics this thesis
  actually exists for — are the project's own, and must be labelled as such
  rather than as the official protocol.

There is a second, sharper problem underneath. The evaluator reads
**`univ_annot3`** — *universal* coordinates, which in the MPI-INF-3DHP family
means the skeleton has been rescaled to a canonical size. Scale normalisation is
precisely what an absolute-depth measurement must not have applied to it: a body
rescaled to a standard height and then placed to match its image would sit at a
different distance.

Whether the release also carries a non-universal `annot3` is **not known** and
cannot be known until `annot.mat` is opened. Two outcomes:

- **`annot3` is present** — absolute depth evaluation proceeds on it, and the
  distinction between the two fields goes in the thesis;
- **only `univ_annot3` is present** — MuPoTS can support root-relative accuracy
  and 2D-consistency, but **not** the absolute depth claim, and the fallback in
  ADR-0004 §10 applies: keep the headline metrics, omit the unsupported
  diagnostic, and say so.

This is now the single most important thing to check the moment the data lands.
It decides whether the primary benchmark can answer the primary question.

## What was not done

`get_mupots-3d.sh` was not run and no licence was accepted. The 6 GB archive is
not on this machine.
