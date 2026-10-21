# ADR-0001 — Project Context and Baseline Architecture

**Status:** Accepted
**Type:** Foundational (supersedes nothing; all later ADRs build on this one)
**Audience:** Contributors working on this repository

> **Note for contributors:** This document is the single source of truth for project
> context. Read it fully before proposing code, dependencies, or design changes.
> If a request conflicts with a constraint listed here, surface the conflict
> instead of silently working around it.

---

## 1. Project Overview

### 1.1 What this project is

A software system that reconstructs 3D representations of real scenes from 2D
video input. The system is the practical component of an undergraduate
(triennale) computer science thesis.

The system has two input modes:

- **Local video files** — the primary development and evaluation mode.
- **University IP cameras via API** — the target production mode, integrated
  later behind the same abstraction.

The output is a navigable 3D scene (3D Gaussian Splatting representation),
plus per-frame depth maps, exposed through a web viewer.

### 1.2 What this project is *not*

Explicitly out of scope. Do not propose work in these directions:

- Continuous real-time 3D reconstruction of moving subjects (4D Gaussian
  Splatting, dynamic scene SLAM). This is an open research problem and is not
  achievable within the constraints below.
- Text-to-3D or generative asset creation (TRELLIS, Hunyuan3D, and similar).
  The project reconstructs *real* geometry from *real* footage.
- Single-image mesh generation as a primary deliverable. It may appear as a
  comparison baseline only.
- Production deployment, multi-tenancy, authentication hardening, or horizontal
  scaling. This is a research prototype with a clean architecture, not a
  product.

### 1.3 Research question

The thesis is not "I built a 3D converter". The measurable question is:

> How does reconstruction quality degrade when camera poses are predicted by a
> learned model (Depth Anything 3) instead of computed by classical
> Structure-from-Motion (COLMAP), across scenes of varying texture density and
> camera motion?

Every architectural decision below is justified against this question. Code that
does not serve either (a) producing reconstructions or (b) measuring them is
scope creep.

---

## 2. Glossary

Terms used throughout the codebase, without further explanation.

| Term | Meaning |
|---|---|
| **3DGS** | 3D Gaussian Splatting — scene represented as millions of anisotropic 3D Gaussians; renders in real time |
| **SfM** | Structure-from-Motion — classical pipeline recovering camera poses and a sparse point cloud from images |
| **Pose** | A camera's position and orientation in world space (extrinsics) |
| **Intrinsics** | Per-camera optical parameters: focal length, principal point, distortion coefficients |
| **Keyframe** | A frame selected from a video as sufficiently distinct from its neighbours to contribute new information |
| **Splat** | A single 3D Gaussian primitive |
| **NVS** | Novel View Synthesis — rendering the scene from a viewpoint not present in the input |
| **PSNR / SSIM / LPIPS** | Image similarity metrics used to score reconstruction quality against held-out frames |
| **DA3** | Depth Anything 3 |

---

## 3. Hard Constraints

These are non-negotiable. Any proposal violating them must be rejected.

### 3.1 Hardware

- Development and execution target: **Apple MacBook Pro, M3 Pro, 18 GB unified
  memory, macOS**.
- **There is no NVIDIA GPU and no CUDA available.** This eliminates a large
  fraction of the 3D reconstruction ecosystem. Before suggesting any library,
  verify it runs on Metal / MPS / WebGPU or on CPU.
- Peak memory budget for any single pipeline stage: **~12 GB**, leaving headroom
  for the OS and editor.

### 3.2 Known-incompatible libraries

Do not suggest these. They are CUDA-only or Linux-only and have been evaluated
and rejected:

- `gsplat` (nerfstudio-project) — actively maintained and excellent, but the
  rasterizer is CUDA-only.
- `nerfstudio` splatfacto path — depends on `gsplat`.
- Original INRIA `diff-gaussian-rasterization` — CUDA-only.
- `TRELLIS` / `TRELLIS.2` — requires an NVIDIA GPU; upstream tests Linux only.

These may still be *cited* in the thesis as related work. They must not enter
`pyproject.toml`.

### 3.3 Licensing

The thesis code will be published. Track the license of every dependency.

- **OpenSplat is AGPLv3**, which is copyleft and viral. If it is linked into our
  own code and that code is distributed, our code must also be AGPLv3. It is
  currently used only as an *external process* for comparison, never as a
  library import. Do not change this without a new ADR.
- Prefer MIT / Apache-2.0 dependencies for anything in the core pipeline.

### 3.4 Data and privacy

University cameras may record identifiable people. Until an ADR covers this:

- No footage from university cameras is committed to the repository.
- No footage leaves the local machine (no cloud inference APIs, no telemetry).
- All models run locally. This is a hard requirement, not a preference.

---

## 4. Decision: Technology Stack

### 4.1 Reconstruction core

| Concern | Choice | Rationale |
|---|---|---|
| 3DGS training | **Brush** | Cross-platform via WebGPU + the Burn framework; no CUDA dependency; runs natively on Apple Silicon; quality competitive with `gsplat` |
| Depth + pose estimation | **Depth Anything 3 (DA3)** | Predicts depth, camera intrinsics/extrinsics, and 3D Gaussians from arbitrary views; the streaming variant handles long video within our memory budget |
| Reference SfM | **COLMAP / pycolmap** | Classical baseline the research question measures DA3 against; also the fallback when DA3 poses are unusable |
| Image ops | **OpenCV** | Calibration, undistortion, keyframe selection, format conversion |
| Comparison trainer | **OpenSplat** (optional) | Second 3DGS implementation with Metal support; used as an external binary for cross-validation only, never imported (see §3.3) |

### 4.2 Application layer

| Layer | Choice | Rationale |
|---|---|---|
| Pipeline runtime | **Python 3.11+** | The ML ecosystem lives here; no realistic alternative |
| API service | **FastAPI** | Async-native, generates an OpenAPI schema we derive TypeScript types from |
| Frontend | **React + TypeScript (strict)** | Matches the author's primary competence; strict mode enforced, `any` disallowed |
| 3D rendering | **Three.js via React Three Fiber** | Declarative, composable, integrates with React state; mature `.ply` / splat loaders |
| Job progress | **WebSocket** | Training takes minutes; HTTP request/response is the wrong shape |
| Python packaging | **uv** | Fast, lockfile-based, reproducible |
| JS packaging | **pnpm** | Workspace support for the monorepo |

### 4.3 Rejected alternatives

- **Everything in Python (Gradio/Streamlit UI).** Faster to prototype, but
  discards the author's frontend competence and produces a weaker system-design
  chapter. Rejected.
- **Everything in TypeScript (Brush compiled to WASM in-browser).** Attractive
  and technically possible, but DA3 has no mature browser runtime, and debugging
  ML in WASM is hostile. Rejected as the primary path; possibly revisited for a
  demo build.
- **Cloud GPU (Colab, RunPod) for training.** Removes the hardware constraint,
  but conflicts with §3.4 privacy requirements and makes the "runs on consumer
  hardware" claim untestable. Rejected.

---

## 5. System Architecture

### 5.1 Pipeline stages

Stages are **decoupled and independently resumable**. Each writes typed
artifacts to disk; the next reads them. A failure in stage 4 must never require
re-running stage 1.

```
[1] Ingest        FrameSource -> raw frames + metadata
[2] Preprocess    undistort, resize, keyframe selection -> keyframes/
[3] Pose          DA3 *or* COLMAP -> poses.json (identical schema for both)
[4] Train         Brush -> scene.ply
[5] Evaluate      PSNR/SSIM/LPIPS on held-out frames -> metrics.json
[6] Serve         API streams artifacts to the React viewer
```

The pose stage is the experiment's variable. **The schema of `poses.json` must
be identical regardless of which backend produced it** — this is what makes the
comparison valid. Treat that schema as a stable contract.

### 5.2 Input abstraction

All input flows through one interface, so that switching from local video to
university cameras touches exactly one implementation:

```python
class FrameSource(Protocol):
    """Yields frames with timestamps from any source.

    Implementations should be lazy: callers may stop early, and the whole
    video is not expected to fit in memory.
    """

    def frames(self) -> Iterator[Frame]: ...
    def metadata(self) -> SourceMetadata: ...
```

Implementations, in the order they are built:

1. `VideoFileSource` — local video. **This is the current development path.**
2. `ImageDirectorySource` — pre-extracted frames, useful for public datasets.
3. `RTSPCameraSource` — university IP cameras. Deferred; the interface exists so
   this is an additive change, not a refactor.

### 5.3 Repository layout

```
/apps
  /web              React + TS viewer
  /api              FastAPI orchestrator
/packages
  /pipeline         Python: ingest, preprocess, pose, train, evaluate
/experiments        One config file per experiment run (versioned)
/docs
  /adr              This file and its successors
  /notes            Lab notebook: dated observations, failures, dead ends
/datasets           Gitignored; tracked with DVC
/artifacts          Gitignored; pipeline outputs
```

### 5.4 Configuration

- Every experiment is fully described by a **versioned config file**. No
  hardcoded parameters, no magic numbers in function bodies.
- Every run records: config hash, git commit SHA, dependency versions, random
  seeds, wall-clock timings, peak memory, and resulting metrics.
- A run that cannot be reproduced from its config file is a bug.

---

## 6. Conventions for Contributors

### 6.1 Code style

**TypeScript**
- Strict mode. `any` is not permitted; use `unknown` and narrow.
- Functional components with hooks. No class components.
- Named exports for shared modules.
- `async`/`await`; no chained `.then()`.
- Descriptive names over abbreviations.
- Keep components focused and composable; extract repeated logic into hooks.

**Python**
- Full type hints; `mypy --strict` on `/packages/pipeline`.
- Pydantic models for anything crossing a process or file boundary.
- `async`/`await` in the API layer; the pipeline itself is synchronous and runs
  in a worker.

**Both**
- Comments are written in **English**, kept short, and added only where intent
  is non-obvious. Prefer explaining *why*, not *what*.
- Comments are collaborative in tone — they suggest and propose rather than
  dictate.
- **Never delete existing comments.** Never remove `debugger` statements.

### 6.2 Testing

- Deterministic modules are tested: keyframe selection, pose parsing, coordinate
  conversions, config validation, artifact schemas.
- Model training is not unit-tested. Its correctness is established by the
  evaluation metrics, not by assertions.
- A regression suite runs the full pipeline on one small fixture scene and
  asserts that metrics stay within a tolerance band.

### 6.3 Documentation discipline

- Each significant decision gets its own ADR in `/docs/adr`, numbered
  sequentially. An ADR is immutable once accepted — supersede it, do not edit
  it.
- `/docs/notes` is a dated lab notebook. **Record failures and dead ends**, not
  just successes; the thesis needs them, and reconstructing them later is
  impossible.

### 6.4 Working agreement

- **Verify before suggesting.** Library recommendations must be checked against
  §3.1 and §3.2. The 3D reconstruction ecosystem is overwhelmingly CUDA-first,
  and plausible-sounding suggestions are frequently incompatible with this
  hardware.
- **Do not silently substitute.** If the requested approach cannot work, say so
  and explain why rather than implementing something adjacent.
- **Preserve the experiment.** Changes that alter the `poses.json` contract, the
  evaluation metrics, or the held-out frame split invalidate prior results.
  Flag these explicitly and require confirmation.
- **Prefer clarity over cleverness.** This code is read by a thesis committee.
  A clever one-liner that needs a paragraph of explanation is a net loss.
- **No over-engineering.** No abstraction layers for hypothetical future needs.
  `FrameSource` exists because two concrete implementations are planned; nothing
  else gets that treatment without justification.

---

## 7. Evaluation Protocol

Defined before implementation, so results cannot be retrofitted to a
conclusion.

**Held-out split.** A fixed fraction of frames is excluded from training and
used only for evaluation. The split is deterministic given the config seed.

**Metrics.**
- Reconstruction quality: PSNR, SSIM, LPIPS on held-out views.
- Pose accuracy: rotation and translation error of DA3 poses against COLMAP
  poses, on scenes where COLMAP converges.
- Cost: wall-clock time and peak memory, per stage.

**Scene variables.** Scenes are captured or selected to vary along axes the
research question cares about: texture density (textured vs. flat surfaces),
lighting stability, camera motion (orbit vs. linear vs. handheld), and scene
scale (object vs. room).

**Reporting.** Every claim in the thesis maps to a run ID in
`/experiments`. Aggregate numbers without traceable runs are not acceptable.

---

## 8. Consequences

**Positive**
- The system runs entirely on the author's laptop, keeping the privacy
  requirement satisfiable and the feedback loop tight.
- The stage decoupling means the pose backend can be swapped without touching
  training, which is exactly what the research question requires.
- The `FrameSource` abstraction means camera integration is additive work, so
  development is never blocked on external access.

**Negative / accepted trade-offs**
- Excluding CUDA excludes the most mature tooling. Expect rougher edges, thinner
  documentation, and more time spent on environment setup than a Linux/NVIDIA
  project would need.
- Brush is younger than `gsplat` and its output format may diverge from other
  splat viewers as it evolves. Pin the version and record it per run.
- The Python/TypeScript split adds a process boundary and a serialization
  contract that a single-language prototype would not need. This is accepted for
  the system-design value it provides.

**Neutral**
- Scene scale is bounded by 18 GB of unified memory. Large outdoor captures may
  need reduced resolution or splat-count caps. This is a documented limitation,
  not a defect.

---

## 9. Open Questions

Tracked here until resolved by a subsequent ADR.

- Should DA3 poses be used raw, or refined with a bundle-adjustment pass before
  training? A hybrid may outperform both endpoints and would strengthen the
  thesis, but it complicates the comparison.
- Fixed university cameras have static, known-ish geometry. Does that make
  one-time calibration strictly better than per-run pose estimation, and does it
  make the research question less interesting in that setting?
- What is the minimum viable definition of the "real-time" component? Current
  working assumption: per-frame depth inference streamed to the viewer, with
  full 3DGS reconstruction remaining an offline job. This should be stated
  explicitly in the thesis to avoid overclaiming.
- Does the evaluation need a mesh-based comparison (photogrammetry via Meshroom)
  as a third baseline, or do two pose backends suffice?

---

## 10. References

- Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*,
  ACM TOG 2023 — the foundational 3DGS paper.
- Lin et al., *Depth Anything 3: Recovering the Visual Space from Any Views*,
  arXiv:2511.10647 — DA3.
- Schönberger & Frahm, *Structure-from-Motion Revisited*, CVPR 2016 — COLMAP.
- Brush — https://github.com/ArthurBrussee/brush
- Depth Anything 3 — https://github.com/ByteDance-Seed/Depth-Anything-3
- OpenSplat — https://github.com/pierotofy/OpenSplat
