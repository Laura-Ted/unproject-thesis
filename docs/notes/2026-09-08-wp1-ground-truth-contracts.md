# 2026-09-08 - WP0 and WP1 of ADR-0004

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

Contracts and validation for the ground-truth evaluation. No adapter, no metric,
no dataset — those are WP2 onward and the first of them is behind a review gate.

## Where it stops, and why

ADR-0004 §9 puts a review gate at the end of WP0: **the author confirms lawful
local availability**. Neither dataset is on this machine, so WP0 tasks 3 to 5 —
inventorying the real layout and confirming the annotation schemas — could not
start, and WP2 cannot start either.

What could be done was done: the protocol, the source note, and the whole of WP1,
which ADR-0004 requires to run without datasets or weights anyway.

## The gates that are actually open

Written down in `docs/experiments/ground-truth-protocol.md` §7 rather than
guessed at. The ones that will bite first:

- MuPoTS native joint names, count, ordering, units and coordinate frame, and
  whether the release carries camera-space coordinates directly;
- whether Anny-One annotations can be had without all 122 GB of RGB archives;
- whether the `joints_3d` rows a `BodiesArtifact` holds carry the Anny bone names
  in the Anny order.

That last one looks obvious and is not. The `anny` 0.6.0 package declares
`skeleton_type = "cmu_mb"` with CMU MotionBuilder names, and a finished run
confirms 163 bones and shape count 11. But those rows come out of **Multi-HMR 2**,
not out of the `anny` package, and nobody has checked that the two agree. So the
joint mapping is committed as a `.toml.template` with no joints in it.

That is the rule ADR-0004 §5.5 exists for: never infer a correspondence from
matching array lengths or similar-looking indices. Two skeletons with the same
joint count are not the same skeleton.

## The training-overlap gate stays open, conservatively

The Multi-HMR 2 repository does not document its training data. The paper
abstract does not either. Anny, Anny-One and Multi-HMR 2 all come from NAVER and
the checkpoint outputs Anny meshes, so it would be surprising if Anny-One were
not in the mix — but surprise is not evidence.

§4.5 says ambiguous means **in-domain compatibility diagnostic**, and that is the
label. Reading the full paper is the way to close it properly, and it is worth
doing before WP5. It costs little to leave open: it changes what an Anny-One
result may be called, not whether it may be run.

## What WP1 built

Five modules under `unproject/ground_truth/`, all of which type-check strictly
and none of which import torch — there is a test for that, in a subprocess,
because this process has certainly imported it already. A reader checking whether
a dataset is present should not pay three gigabytes to be told a file is missing.

The design choice worth recording: **every check fails closed, and every message
is written for a person.** The failure mode this evaluation exists to avoid is
not a crash, it is a plausible wrong number. Millimetres read as metres, a
root-relative annotation read as camera-space, a joint mapped by position, a
person quietly dropped between annotation and average — none of those look wrong
in a table.

So:

- a record declaring any convention but `camera_relative_opencv_metres` is
  refused, with the instruction to test the conversion against a synthetic
  transform rather than by looking at a rendered skeleton;
- a joint further than 200 m from the camera is refused as "what a millimetre
  annotation looks like when it is not converted";
- a mapping whose source names the release does not provide is refused with the
  release's actual names listed and "do not match by position";
- `Coverage` refuses to validate unless matched plus missed equals the annotated
  count. A model that finds one easy person in ten is representable; a person
  vanishing between the two is not.

Missing a dataset names the official page in the error, because ADR-0004 §4.1
puts acquisition on the author and an error they cannot act on is worse than
none.

## One thing deliberately not deduplicated

`inventory.py` is not in the module list ADR-0004 §5.2 sketches. The §4.3
integrity record is ten fields plus directory walking and hashing, and folding it
into `dataset.py` — described there as "small shared normalized types" — would
have made that module neither small nor about types. One extra module beats one
misleading one.

## Not done, and not attempted

No licence accepted, nothing downloaded, no adapter, no metric, no dataset file
read. The mapping template has no joints in it and the two evaluation configs
carry `release = "unset: record what was downloaded, and when"`, which is a
placeholder that will fail loudly rather than quietly.
