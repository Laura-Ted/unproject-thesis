# 2026-09-08 - What the official sources actually say

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

WP0 of ADR-0004, tasks 1 and 2. Everything below was read from an official page
today. Where a page did not answer, that is recorded as not answered rather than
filled in from memory — the whole point of this note is that the adapters are
built on sources, not on recollection.

Neither dataset is on this machine yet, so tasks 3 to 5 of WP0 — inventorying the
actual layout and confirming the annotation schemas — cannot start. Those are the
gates listed in the protocol §7.

## MuPoTS-3D

From the project page and the surrounding literature:

- 20 sequences, 5 indoor and 15 outdoor, over 8,000 frames
- 2–3 people per sequence, interacting
- **static camera**, which is this project's own condition and the reason this
  dataset was chosen over 3DPW or EMDB
- ground truth from a multi-view markerless motion-capture process
- described as suitable for evaluation in both person-centric and **camera-
  centric** coordinates, which is the half this project needs

The camera-centric part is why MuPoTS is the primary benchmark here. Most 3D pose
benchmarks align the root before measuring, which deletes exactly the error this
thesis is about.

The metric this project has been computing by hand has a name in the literature:
**MRPE**, mean root position error, usually reported per axis. `3DPCKabs` is the
same family. That means the numbers can sit beside published ones instead of
standing alone.

**Not confirmed from an official source:** the licence text, the current download
URL, the archive size, the native joint names and ordering, the units, and
whether the released annotation files carry camera-space coordinates directly or
require a documented transform. The project page says the datasets are made
"publicly available for research purposes" without a formal licence document, and
the file linked from it is 10 KB — evaluation scripts, not the data.

## Anny-One

From the official dataset page:

- 780,964 RGB images across 22,951 synthetic scenes
- 8 tar archives, roughly 12–17 GB each, **about 122 GB in total**
- `AnnyOne_LICENSE.txt` ships with the release; the page does not reproduce its
  terms
- no registration requirement mentioned; direct download links are given
- per scene: camera parameters across multiple viewpoints, per-person 2D
  visibility masks, bounding boxes, and 3D pose and shape in the Anny parametric
  model
- segmentation masks were added on 2026-02-18
- an official exclusion list exists: `scenes_with_issues.txt`, scenes with
  annotation issues, filtered in the same 2026-02-18 update
- a Python visualisation script is provided for parsing annotations

The 122 GB is the reason ADR-0004 §4.4 forbids evaluating the whole release and
fixes a 300-image subset. It also raises a question the protocol lists as a gate:
whether the annotations can be obtained without pulling every RGB archive.

The provided parsing script is worth reading before it is trusted. ADR-0004 §5.7
forbids deserialising an untrusted pickle just because an official sample does.

## The training-overlap gate, and why it stays open

ADR-0004 §4.5 says an Anny-One result cannot be called independent generalisation
evidence unless the checkpoint's training and model-selection data are known.

Checked today:

- **the Multi-HMR 2 repository** — does not document the training datasets. It
  mentions Anny-One as "a synthetic dataset of 780K+ multi-person and multi-view
  images with Anny ground-truth meshes" but does not say whether it was used for
  training or validation.
- **the paper abstract, arXiv:2606.14841** — silent on datasets. It does say the
  method "achieves state-of-the-art pelvis-centered performance while
  substantially improving detection accuracy and **metric 3D localization**",
  which is worth noting on its own: metric localisation is precisely what RQ3
  measures.

Not checked: the full paper PDF or HTML.

So the evidence is ambiguous, and §4.5 is explicit about what that means —
**in-domain compatibility diagnostic**, the conservative label. It is worth being
plain about why the suspicion exists: Anny, Anny-One and Multi-HMR 2 all come
from NAVER, and the checkpoint outputs Anny meshes. It would be surprising if
Anny-One were not in the training mix.

Reading the full paper is the one remaining way to close this properly, and it is
worth doing before WP5. Until then the conservative label stands, and it costs
little: it changes what the result may be called, not whether it may be run.

## One gate narrowed: the Anny skeleton

The installed `anny` 0.6.0 ships a rest pose declaring `skeleton_type = "cmu_mb"`
with CMU MotionBuilder bone names — `LeftArm`, `LeftForeArm`, `LeftShoulder`,
`LeftUpLeg`, `LHipJoint` and so on. A finished run confirms the rest: Anny 0.6.0,
**163 bones**, 11 shape parameters, checkpoint `fd2d9ab7…`.

That narrows the Anny half of the joint mapping but does not close it. The names
above come from the `anny` package; the `joints_3d` rows in a `BodiesArtifact`
come from **Multi-HMR 2**, and that those rows carry those names in that order is
exactly the sort of thing that looks obvious and is worth one check. It has not
been checked.

So `experiments/ground-truth/mappings/mupots-3d-to-anny-v1.toml.template` is
committed unfilled, with a `.template` suffix so `validate-config` cannot pick it
up and nothing can load it by accident. Filling it is WP2 work and needs the
MuPoTS release in hand.

## What was not done, and why

No licence was accepted and nothing was downloaded. ADR-0004 §4.1 puts both on
the author, and that is the right place for them: accepting terms on someone
else's behalf is not a thing to automate.

## Correcting one earlier claim

An earlier session suggested MuPoTS-3D might need "an email to the MPI" for
access. That was speculation on my part, not something the project page says. The
page says the data is publicly available for research purposes. What the actual
access route is remains one of the open gates, and the author will find out by
following the official link rather than by asking me.
