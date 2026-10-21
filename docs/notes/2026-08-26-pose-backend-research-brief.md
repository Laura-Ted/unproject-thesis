> **Resolved.** This brief was written with the backend choice still open. It
> was answered by LingBot-Map, which is what `backend = "lingbot"` runs today
> (`experiments/fixture-lingbot.toml`). The question below is closed; the
> document is kept as the record of how it was asked.

# Research Brief — Learned Camera-Pose Model for Apple Silicon

**Date of writing:** 2026-08-26
**Audience:** a researcher with no prior knowledge of this project
**Status:** open question, blocking

You are being given this document and nothing else. Everything you need is here.
Do not assume familiarity with the project, and do not ask me questions — produce
the deliverable described in §8.

---

## 1. What I am asking you to do

Find the **best currently available neural network model that estimates camera
poses from a set of photographs, and that actually runs on Apple Silicon.**

"Currently available" means **released or meaningfully updated between roughly
October 2025 and today, 2026-08-26**. See §7 — recency is a hard requirement, not
a preference.

I already have two candidates (§6). Your job is to determine whether something
better has appeared since, and to verify any candidate against the criteria in §5
using the method in §4. **A recommendation without that verification is useless
to me** — the last two candidates I picked by reading papers both turned out to be
unusable on this hardware, and I only found out by inspecting their dependency
manifests.

---

## 2. Project context

### 2.1 What the system is

A pipeline that reconstructs a navigable 3D scene from ordinary 2D video. It is
the practical component of an undergraduate computer science thesis. The output
is a **3D Gaussian Splatting** representation (a scene represented as millions of
small 3D blobs, renderable in real time), viewable in a web browser.

The pipeline has six stages. Each writes typed files to disk; the next reads them.

```
[1] Ingest      video file or image folder -> individual frames
[2] Preprocess  discard blurred and near-duplicate frames -> keyframes
[3] Pose        estimate where each keyframe was shot from -> poses.json   <-- YOUR SCOPE
[4] Train       fit the 3D Gaussian Splatting model        -> scene.ply
[5] Evaluate    score quality on held-out views            -> metrics.json
[6] Serve       stream the result to a web viewer
```

Stages 1 and 2 are built and working. Stage 3 is half built (see §2.3). Stages 4
to 6 are not started.

### 2.2 The research question

The thesis does not claim "I built a 3D converter". The measurable question is:

> How does reconstruction quality degrade when camera poses are predicted by a
> **learned model** instead of computed by **classical Structure-from-Motion**,
> across scenes of varying texture density and camera motion?

This is why stage 3 has **two interchangeable backends**:

- **the classical baseline**, COLMAP — already implemented and working
- **the learned model** — this is what you are searching for

Everything else in the pipeline is held constant. The pose backend is the only
variable. So the model you recommend is not a convenience: it is one of the two
things the entire thesis measures.

### 2.3 What is already built

Stage 3 has a working COLMAP backend (via `pycolmap`), a stable output contract,
and an interface that a second backend plugs into. The plumbing is done. What is
missing is the learned model itself.

The output contract that any backend must fill (this is what the model's output
gets converted into, so read it as "what I need out of the model"):

```
per camera:   focal length x/y, principal point x/y, image size, distortion
per image:    rotation (quaternion) + translation, world-to-camera
scene:        a sparse 3D point cloud, used to initialise stage [4]
diagnostics:  how many images the backend managed to place, out of how many given
```

---

## 3. Hard constraints — non-negotiable

### 3.1 Hardware

- **Apple MacBook Pro, M3 Pro, 18 GB unified memory, macOS.**
- **There is no NVIDIA GPU and no CUDA on this machine, and there will not be one.**
- Peak memory budget for a single pipeline stage: about **12 GB**, leaving room
  for the OS and an editor.
- Cloud GPUs are not an option: the footage may contain identifiable people and
  must not leave the machine. All inference runs locally. This is a hard
  requirement, not a preference.

**This constraint is the entire difficulty of the task.** The 3D reconstruction
and computer-vision ecosystem is overwhelmingly CUDA-first. Most models that look
suitable are not, and the incompatibility is usually buried in a transitive
dependency rather than stated anywhere in the README.

Acceptable execution paths, in order of preference:

1. **Apple MLX** — Apple's own array framework, native to Apple Silicon. If a
   model has an MLX port, that is the best possible answer. Check for this.
2. **PyTorch with the MPS backend** — `torch.device("mps")`. This is the normal
   answer and what I expect.
3. **CoreML** — acceptable if a converted model exists.
4. **CPU only** — acceptable only if inference on ~50 images finishes in minutes,
   not hours.

### 3.2 Software environment

- Python **3.12**.
- Package manager is `uv`.
- The pipeline itself uses **numpy 2.x** and **OpenCV 5**. A model pinned to
  `numpy<2` is not automatically disqualified, because I can run it in a separate
  environment and call it as an external process, but **flag it clearly** as it
  adds work.

### 3.3 Licensing

**Ignore licensing entirely.** Do not evaluate it, do not mention it, do not let
it influence the ranking. It is not a consideration for this decision.

---

## 4. The verification method you must apply

This is the most important section. Papers, README badges and model cards
routinely claim portability that does not survive contact with a Mac. Verify from
**dependency manifests and source code**, not from prose.

### 4.1 Read the dependency manifest

Find `requirements.txt`, `pyproject.toml`, `setup.py`, or `environment.yml` in
the model's repository. Read the actual file, from the raw source, e.g.:

```
https://raw.githubusercontent.com/OWNER/REPO/main/requirements.txt
```

**Disqualifying or high-risk dependencies:**

| Package | Why it matters |
|---|---|
| `xformers` | CUDA attention kernels. Publishes **no macOS wheel at all**. Fatal. |
| `flash-attn` | Same category, worse. Fatal. |
| `triton` | NVIDIA GPU compiler. Fatal unless optional. |
| `nvidia-*`, `cuda-*`, `cupy` | Self-evident. Fatal. |
| `apex`, `deepspeed` | NVIDIA training stacks. Fatal. |
| `gsplat` | CUDA-only rasteriser. Fatal. |
| `pytorch3d`, `mmcv`, `detectron2` | Compile CUDA extensions. Usually fatal. |
| `numpy<2` | Not fatal, but flag it — see §3.2. |

If any appears, check whether it is a **hard** dependency or an **optional
extra** (`[project.optional-dependencies]`, or `extra == "..."` markers). An
optional extra is fine; a hard dependency is fatal.

### 4.2 Confirm wheel availability for macOS ARM

For any dependency you are unsure about, check what files it publishes on PyPI:

```
https://pypi.org/pypi/PACKAGE/VERSION/json
```

Look at the filenames in `urls`. You need one matching `macosx_*_arm64.whl`.

**Critical warning, learned the hard way:** a dependency resolver reporting
success does **not** prove installability. `uv pip install --dry-run` will happily
resolve to a source distribution that it would then fail to compile. I made
exactly this mistake and reported a model as viable when it was not. Only the
presence of an actual `macosx_*_arm64` wheel, or code with no compiled
extensions, counts as evidence.

### 4.3 Read the attention implementation

Open the model's attention module (commonly `models/`, `layers/`, `modules/`).

- `F.scaled_dot_product_attention` → **portable**, native PyTorch, works on MPS.
- `xformers.ops.memory_efficient_attention` → **not portable**.
- `flash_attn_func` or similar → **not portable**.
- A hand-written CUDA kernel loaded via `torch.utils.cpp_extension` → **not portable**.

A model can declare `xformers` in requirements while never importing it on the
inference path — check the code, not just the manifest.

### 4.4 Grep for CUDA assumptions in the source

Search the repository for `cuda`, `.half()`, `bfloat16`, `autocast`,
`CUDAExtension`, `.cu` files.

Distinguish two very different cases:

- **Cosmetic:** `device = "cuda" if torch.cuda.is_available() else "cpu"`, or
  `torch.cuda.amp.autocast(...)`. These are a handful of lines I can change.
  **Do not disqualify a model for these** — just say they are present.
- **Structural:** a compiled `.cu` kernel, a `CUDAExtension` in `setup.py`, or an
  op with no CPU/MPS fallback. **Fatal.**

### 4.5 Look for evidence someone already ran it on Apple Silicon

Search GitHub issues, discussions, and forks for `MPS`, `Apple Silicon`, `M1`,
`M2`, `M3`, `M4`, `macOS`. A fork that adds MPS support is strong positive
evidence and a useful implementation reference. An open issue titled "does not
work on Mac" with no resolution is strong negative evidence.

Also search specifically for an **MLX port** — Apple's framework. Rare, but the
best possible outcome if it exists.

### 4.6 Check the weight size and memory behaviour

- Model weight file size, from the HuggingFace API:
  `https://huggingface.co/api/models/ORG/MODEL`
- **MPS does not reliably support float16 autocast for large vision
  transformers**, so assume float32 and roughly double the on-disk size for the
  memory estimate.
- Ask specifically: **how does memory scale with the number of input images?**
  Models that apply global attention across all views at once grow superlinearly
  and hit a ceiling at a few dozen images. This matters enormously to me — see
  §5, criterion 4.

---

## 5. Ranking criteria, in priority order

1. **It runs on Apple Silicon.** Verified per §4. A model that fails here scores
   zero regardless of everything else. This is not negotiable and not a
   trade-off against quality.

2. **It outputs camera poses, not just depth.** Many recent models predict
   excellent monocular depth and no extrinsics at all. Those cannot be used:
   they do not answer the research question. The model must produce **camera
   extrinsics for each input image**. Intrinsics and a point cloud are strongly
   desirable, because I need them too and would otherwise have to derive them.

3. **Recency.** See §7.

4. **How many images it can handle at once, on 18 GB.** Concretely: how many
   photographs can go in per run? This is not a performance detail, it is
   methodological. My classical baseline handles hundreds of images. If the
   learned model handles forty, then the two backends are being fed different
   scenes and the comparison is compromised. I need a real number here, or an
   honest statement that it is unknown.

5. **Pose accuracy** on standard benchmarks, if reported. Useful for choosing
   between two candidates that both pass criteria 1 to 4. Never a reason to
   recommend one that fails them.

6. **Is it a recognised reference point?** The thesis compares against this
   model, so a widely used one makes the comparison legible to a committee.
   Download counts, citations, and whether other papers benchmark against it are
   reasonable proxies. This is a tiebreaker, not a requirement.

---

## 6. Already evaluated — do not repeat this work

### 6.1 Rejected, with the evidence

**Depth Anything 3** (`depth-anything-3` on PyPI, v0.1.1). This was the project's
original choice and it is **not installable on this machine**:

- `xformers` is a **hard** dependency. xformers publishes wheels only for
  `manylinux_2_28_x86_64` and `win_amd64`. There is no macOS wheel of any kind,
  so installation falls back to compiling CUDA kernels, which fails.
- It also pins `numpy<2`, conflicting with the pipeline's numpy 2.x.
- Its `all` and `gs` extras pull `gsplat`, which is CUDA-only.

**Apple Depth Pro** (`apple/ml-depth-pro`). Predicts metric depth only. **No
camera poses.** Structurally cannot serve as a pose backend. Rejected on
criterion 2, not on portability.

**DUSt3R / MASt3R** (`naver/dust3r`, `naver/mast3r`). Dependencies are otherwise
clean, but they build optional CUDA kernels for rotary embeddings and rely on a
global alignment optimisation step. Worth revisiting only if the candidates below
fail.

### 6.2 Current best candidates — your baseline to beat

**VGGT** — `facebookresearch/vggt`, weights `facebook/VGGT-1B`. Meta and Oxford,
CVPR 2025.

- Predicts intrinsics, extrinsics, depth maps and point maps in one forward pass.
- Dependencies: `torch`, `torchvision`, `numpy`, `Pillow`, `huggingface_hub`,
  `einops`, `safetensors`. Nothing CUDA-specific.
- `vggt/layers/attention.py` sets `XFORMERS_AVAILABLE = False` and uses
  `F.scaled_dot_product_attention`. Portable.
- CUDA appears only in device strings and `torch.cuda.amp.autocast` — cosmetic.
- Weights 4.68 GB. A community fork `jmanhype/vggt-mps` reports float32-only on
  MPS.
- ~433,000 downloads; the de facto reference for feed-forward reconstruction.

**MapAnything** — `facebookresearch/map-anything`, weights
`facebook/map-anything`. Meta, 2025.

- Same category of output. `xformers` appears only as an optional extra.
- Newer and less established: ~35,000 downloads.

**These are the bar.** If nothing newer beats them, say so plainly and recommend
one of them — that is a perfectly acceptable answer. Do not invent a novelty to
appear productive.

---

## 7. Recency — read this carefully

**Today is 2026-08-26.**

This field moves extremely fast. A model from early 2025 is, in practical terms,
several generations old. VGGT is CVPR 2025 and I already consider it dated; it is
listed above as a baseline to beat, not as the answer.

**Concentrate your search on roughly October 2025 through August 2026.** Consider
anything older only if nothing newer survives §4.

Suggested places to look, non-exhaustively:

- arXiv listings under cs.CV for 2026, particularly anything citing or
  benchmarking against VGGT
- CVPR 2026, ICLR 2026, ECCV 2026 accepted papers
- HuggingFace models sorted by recency, tagged for pose estimation, 3D
  reconstruction, or structure-from-motion
- "awesome" lists and survey papers on feed-forward 3D reconstruction, checking
  their most recent entries
- Papers that position themselves as successors to VGGT, DUSt3R or MASt3R

Watch for a specific failure mode: a 2026 paper with impressive numbers, a GitHub
repository containing only a README, and no released weights or inference code.
**A model I cannot download and run is not a candidate.** Confirm that weights
and inference code actually exist and are downloadable.

---

## 8. What to give me back

A single report, structured as follows.

### 8.1 A comparison table

| Model | Released | Outputs poses? | Apple Silicon | Max images @18GB | Weights | Verdict |
|---|---|---|---|---|---|---|

Populate the "Apple Silicon" column with a verdict **plus the evidence that
produced it**, e.g. "yes — `F.scaled_dot_product_attention`, no CUDA-only deps in
`requirements.txt`", or "no — hard `flash-attn` dependency, no macOS wheel".

### 8.2 A single recommendation

One model, with the reasoning. State plainly which criteria in §5 it satisfies
and which it does not.

### 8.3 The evidence, per claim

For each portability claim, give the **URL and the specific file or line** you
based it on. I will re-check these. A claim I cannot trace to a source is one I
will discard.

### 8.4 Runner-up

One alternative, and the condition under which I should prefer it.

### 8.5 What you could not determine

State it explicitly. "I could not find how memory scales with image count" is a
useful and honest answer. A guess presented as a fact is not, and has already
cost me a week on this project.

### 8.6 Integration notes

For the recommended model: how to install it on macOS ARM with Python 3.12, how
to load the weights, roughly what its inference call looks like, and which
CUDA-specific lines would need changing for MPS.

---

## 9. Vocabulary, if any of this is unfamiliar

| Term | Meaning |
|---|---|
| **Pose** | Where a camera was and which way it pointed, in world space |
| **Extrinsics** | The numeric form of a pose: a rotation plus a translation |
| **Intrinsics** | The lens: focal length, principal point, distortion |
| **SfM** | Structure-from-Motion; the classical, non-learned way to recover poses |
| **COLMAP** | The standard SfM implementation; my baseline |
| **3DGS** | 3D Gaussian Splatting; the scene representation this pipeline produces |
| **Keyframe** | A frame selected from a video as carrying new information |
| **MPS** | Metal Performance Shaders; PyTorch's GPU backend on Apple Silicon |
| **MLX** | Apple's own array/ML framework, native to Apple Silicon |
| **Feed-forward reconstruction** | Predicting 3D geometry in a single network pass, without per-scene optimisation |
