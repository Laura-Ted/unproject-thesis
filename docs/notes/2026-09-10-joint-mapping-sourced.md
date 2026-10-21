# 2026-09-10 - The joint mapping, and two wrong answers that looked right

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

WP2 task 3 of ADR-0004: the explicit MuPoTS-to-Anny correspondence. Both sides
are now sourced and the file is written. The dataset is still not downloaded;
none of this needed it.

Worth recording carefully, because this is the file ADR-0004 §5.5 exists to
protect, and the route to it passed two plausible wrong answers.

## The MuPoTS side

From `mpii_get_joints.m` in the official evaluator archive. Evaluation uses the
17-joint `relevant` set, "Human3.6m joints in CPM order": head_top, neck, the
two shoulders, elbows, wrists, hips, knees, ankles, then pelvis, spine, head.

## The Anny side, and the first wrong answer

`anny` 0.6.0 ships `data/mpfb2/rigs/standard/rig.default.json`, and it holds
exactly **163** entries — the same count as our artifact's `joints_3d`. It is
tempting to read the mapping straight out of it.

It would have been wrong twice. The JSON is a dict in alphabetical order
(`breast.L`, `breast.R`, `clavicle.L`…), not the model's order. And the model
does not use that rig unmodified.

## The second wrong answer

`Anny()` instantiated with its defaults gives **104** bone labels with index 0 =
`root` — a reduced rig. Close enough in shape to look right, and wrong: in the
104-bone ordering `upperarm01.L` is index 48, in the real one it is 50. Every
arm joint would have been off by two, producing a complete, confident, wrong
evaluation.

## The right answer

The ordering that matters is the one the loaded checkpoint carries. Read off
`body_model_world` from an opened session: **163 labels, index 0 = `root`**,
`upperleg01.L` at 2, `lowerleg01.L` at 4, `foot.L` at 6, `upperarm01.L` at 50,
`lowerarm01.L` at 52, `wrist.L` at 54, `neck01` at 100, `head` at 103.

## Which end of a bone is a joint

Upstream computes `j3d = output["bone_poses"][:, :, :3, -1]` — the translation
column of each bone's posed transform, which is the bone's **origin**, not its
tail. So a limb joint is named by the bone that *starts* at it: the knee is
where the shin begins, not where the thigh ends.

That reading was not taken on trust. Medians over 3,350 real detections in
`artifacts/pedestrians/bodies/4f4f09628526`:

| segment | measured |
|---|---|
| upperleg01.L → lowerleg01.L | 41.9 cm |
| lowerleg01.L → foot.L | 40.0 cm |
| lowerarm01.L → wrist.L | 23.2 cm |
| root → head | 71.7 cm |

A femur, a tibia, a forearm, and a trunk. Reading the wrong end of each bone
would not produce a skeleton with human proportions, so this is a measurement
rather than an assumption — which is what ADR-0004 §5.4 asks for.

## What the mapping contains

Fourteen joints: pelvis, plus the thirteen official PCK joints that map without
ambiguity. Three are deliberately absent and the file says so:

- **head_top** — Anny has no crown-of-skull bone. This one *is* in an official
  PCK group, so the reported PCK covers 13 of the official 14 and every table
  must say so.
- **head** — Anny's `head` origin is the base of the skull; whether MuPoTS means
  the same point is not established. Not in a PCK group.
- **spine** — Anny has spine01 through spine05 and which one MuPoTS means is a
  guess. §5.5 admits spine landmarks only when both definitions match.

Verified end to end: the mapping loads, hashes to `eb66b465b80132f1…`, and
reduces a 163-row array to 14 common joints with every one valid.

## Not done

The dataset is still not on this machine, so the adapter that reads `annot.mat`
is not written. That needs the file to confirm the MAT layout, and a decision
about a `ground-truth` optional extra for the parser, which ADR-0004 §5.7
governs.
