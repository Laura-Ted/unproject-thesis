# ADR-0002 — Local Multi-Person 3D Body Capture as the Thesis Direction

**Status:** Accepted
**Date:** 2026-09-04
**Type:** Foundational research-direction and delivery decision
**Audience:** Thesis author, supervisors, and contributors
**Supersedes:** ADR-0001 where it defines full static-scene reconstruction and 3D Gaussian Splatting as the primary thesis objective
**Preserves:** ADR-0001 constraints on local execution, Apple Silicon, privacy, reproducibility, typed artifacts, strict typing, licensing, and documentation discipline

> **Note for contributors:** Read this ADR completely before proposing or
> making changes. It is the current source of truth for thesis scope, research
> questions, architecture, experimental method, delivery order, and quality
> gates. If a request conflicts with it, identify the conflict instead of
> silently implementing an adjacent solution.

---

## 1. Decision Summary

The thesis will study:

> **The feasibility and limitations of local multi-person 3D body capture from
> monocular video on Apple Silicon, supported by an interactive and reproducible
> software system.**

The primary path is:

```text
video file or browser camera
              |
              v
      local Multi-HMR 2 inference
              |
              +--> estimated camera intrinsics
              +--> tracked 3D joints
              +--> body pose and shape parameters
              +--> reconstructable body meshes
              |
              v
  typed artifacts or versioned WebSocket messages
              |
              v
    synchronised React and Three.js viewer
```

The earlier scene-reconstruction path remains in the repository as preliminary
work:

```text
ingest -> preprocess -> COLMAP or LingBot-Map pose -> pose evaluation
```

It motivates the change in direction: near-static monocular cameras and dynamic
human subjects violate the assumptions required by Structure-from-Motion. It is
not the primary implementation path and receives no new feature work unless
needed to reproduce or explain that finding.

Stage [4], 3D Gaussian Splatting training, is deferred. It is neither a thesis
deliverable nor a prerequisite for evaluating body capture.

---

## 2. Context and Rationale

ADR-0001 defined an end-to-end static-scene reconstruction system based on
camera-pose estimation followed by 3D Gaussian Splatting. That was a reasonable
starting hypothesis, but experiments exposed a mismatch between the intended
footage and the method:

- crowd footage is dominated by independently moving subjects;
- the reference camera may move too little to provide triangulation baseline;
- COLMAP can register too few useful views in that setting;
- learned camera-pose estimation does not remove the missing geometric evidence;
- the training stage is not implemented;
- mature reconstruction tooling remains strongly CUDA-oriented;
- completing pose research, training, rendering, and reconstruction evaluation
  would exceed the scope of one undergraduate thesis.

The repository now contains a more coherent implemented path. Multi-HMR 2
estimates several people in one forward pass, associates detections over time,
produces a parametric body representation, and supports both offline artifacts
and interactive streaming. The viewer renders raw or smoothed skeletons and
meshes beside the source footage.

Preliminary observations also expose a researchable limitation: reconstructed
bodies may appear plausible from the source viewpoint while their depth varies
substantially over time or becomes implausible from an off-axis viewpoint.
Measuring this limitation is more useful than hiding it.

The contribution is therefore a systems and empirical-evaluation contribution,
not a novel neural architecture.

---

## 3. Project Definition

### 3.1 Objective

Design, implement, and evaluate a system that:

1. accepts local video files and browser camera streams;
2. performs all inference on the local Apple Silicon machine;
3. estimates and tracks multiple 3D human bodies;
4. stores reproducible offline results in typed artifacts;
5. transports live results through a versioned protocol;
6. visualises source footage and reconstructed bodies in synchronisation;
7. measures computational feasibility, scaling behaviour, temporal stability,
   and geometric limitations.

### 3.2 What this project is not

This project is not:

- a new body-reconstruction model;
- a clinical or safety-critical measurement system;
- a biometric identity system;
- a production surveillance product;
- a cloud inference service;
- a general-purpose 3D or 4D reconstruction platform;
- a photorealistic scene-reconstruction thesis;
- evidence that monocular video uniquely determines metric 3D geometry.

### 3.3 Contribution boundary

Subject to experimental evidence, the thesis may claim:

- feasibility on the exact tested Apple Silicon hardware;
- measured cold-start, latency, throughput, memory, and transport costs;
- measured relationships between cost and selected input/output properties;
- measured temporal instability and controlled positional error;
- measured effects and trade-offs of post-processing;
- reproducibility of offline runs and derived results;
- privacy benefits of processing without cloud inference.

The thesis must not claim without separate evidence:

- generalisation to other hardware, scenes, populations, or camera systems;
- ground-truth 3D reconstruction;
- calibrated confidence probabilities;
- verified real-world identity from a track identifier;
- improved geometric accuracy merely because output looks smoother;
- real-time inference merely because rendering remains interactive;
- fitness for medical, occupational-safety, policing, or biometric decisions.

Use **interactive streaming** for progressive results. Use **real-time
inference** only if sustained measured processing meets the declared source
rate under the reported configuration.

---

## 4. Research Questions

### RQ1 — Feasibility

> Can multi-person 3D body capture run entirely locally on the tested Apple
> Silicon hardware with usable interactive behaviour and bounded resources?

Measure:

- model cold-start time;
- warm per-frame latency, including median, p90, and p95;
- sustained processed frames per second;
- peak process memory;
- Metal allocated and driver memory when exposed by the installed PyTorch API;
- end-to-end overhead relative to inference-only execution;
- processed, skipped, and dropped frame counts;
- age of the latest displayed camera estimate;
- artifact size and network volume for skeleton and mesh modes.

Every result must identify the machine, operating system, Python and package
versions, model checksums, source digest, effective configuration, and Git state.

### RQ2 — Scaling Behaviour

> Which observable properties of the input and requested output determine
> computation, memory, and transport cost?

Use a small pre-registered matrix covering:

- low, medium, and high visible-person counts;
- 16:9 and square inputs;
- at least two resolutions of otherwise identical content;
- skeleton and mesh output modes.

Report distributions, not only averages. Treat comparisons as observational
when more than one variable changes. Do not infer that person count caused a
difference when aspect ratio, resolution, or scene content also changed.

### RQ3 — Stability and Geometric Limitations

> How stable and geometrically credible are reconstructed bodies over time,
> especially along the monocular depth axis?

Measure:

- 2D reprojection consistency;
- frame-to-frame lateral and depth displacement for stationary subjects;
- apparent lateral and depth velocity during controlled movement;
- within-track variation of estimated body shape;
- track fragmentation and manually verifiable ID switches;
- lateral and depth error in a small scene with measured floor positions;
- stability and response delay before and after smoothing.

Raw results are primary. Smoothed results must be reported separately and must
never overwrite raw artifacts.

### 4.1 Scope filter

Before starting any activity, answer:

> Which research question, reproducibility requirement, or final-demonstration
> failure does this address, and what evidence will show completion?

If there is no precise answer, the activity does not enter the thesis.

---

## 5. Hard Constraints

### 5.1 Hardware and execution

- Primary execution target: Apple Silicon on macOS.
- No NVIDIA GPU and no CUDA are available.
- All inference and evaluation involving private footage run locally.
- Approximate peak memory budget per stage: 12 GB unless the final machine
  measurement establishes a different safe limit.
- CPU fallback may exist, but performance claims must identify the backend.
- Before adding a dependency, verify macOS, ARM64, Python, memory, and licensing
  compatibility.

ADR-0001 names an M3 Pro with 18 GB unified memory, while later repository notes
report measurements on an M4 Pro. This discrepancy must be resolved before
final experiments. The benchmark records the exact machine actually used; no
hardware identity is inferred from old prose.

### 5.2 Privacy and ethics

- No identifiable footage is committed to Git.
- No footage is sent to a cloud model or telemetry service.
- Participants provide informed consent before controlled recordings.
- Collect only footage needed for the declared experiment.
- Do not publish identifiable raw footage without explicit permission.
- Prefer numerical outputs and appropriately anonymised figures in the public
  repository.
- Store no unnecessary participant attributes.

### 5.3 Licensing and supply chain

- The repository is intended for publication; every direct dependency needs a
  recorded compatible licence.
- Prefer permissive MIT, BSD, or Apache-2.0 dependencies.
- Copyleft or research-only components require an explicit compatibility review
  and a new ADR before they enter the distributed core.
- Model weights, source repositories, and backbones must be pinned by immutable
  revision and, where practical, checksum.
- A downloaded checkpoint is verified before deserialisation.
- No dependency is added by branch name, floating tag, or `latest` reference.
- CUDA-only dependencies may be discussed as related work but do not enter the
  runtime graph.

The OpenSplat licensing discussion from ADR-0001 remains relevant historical
context, but OpenSplat and Brush are not part of the primary body-capture path.

### 5.4 Time and scope

- This is an undergraduate thesis, not a product programme.
- Reliability and evidence take priority over new visible features.
- Work packages are completed in order and stopped at review boundaries.
- No work package authorises a rewrite.

---

## 6. Architecture

### 6.1 Repository boundaries

```text
apps/api                 local FastAPI service and live transport
apps/web                 React/TypeScript viewer and Three.js renderer
packages/pipeline        model adapter, stages, artifacts, metrics, and CLI
experiments              versioned experiment configurations
docs/adr                 accepted architectural decisions
docs/experiments         pre-registered protocols and reproduction guides
docs/notes               dated observations, failures, and dead ends
docs/figures             reproducibly generated thesis figures
datasets                 local inputs; ignored
artifacts                local derived results; ignored
models                   pinned local weights; ignored
uploads                  private operator inputs; ignored
```

Do not introduce a database, message broker, second backend service, workflow
engine, or cloud component. The supported application remains one local API
process and one browser client.

### 6.2 Pipeline stages

The repository retains these stages:

```text
[1] Ingest       video or image directory -> lossless frames and metadata
[2] Preprocess   resize and keyframe selection -> keyframes
[3] Pose         COLMAP or LingBot-Map -> camera poses and point cloud
[4] Train        deferred; no thesis implementation
[5] Evaluate     existing comparison of pose runs
[B] Bodies       ingest frames -> tracked body parameters and mesh assets
[6] Serve        artifacts and local live inference
[V] Viewer       synchronised source and interactive reconstruction
```

Stage [B] is a sibling of stage [2]. It consumes every selected ingest frame
because temporal capture must not discard frames merely for being visually
similar. Stages [1]–[3] and [5] remain stable preliminary work.

### 6.3 Input abstraction

`FrameSource` remains the boundary for local files and image directories.
Browser-camera input arrives through the live protocol because it is already a
stream of decoded frames. Do not add RTSP or university-camera support unless
it becomes necessary for a declared experiment; the abstraction does not make
that integration a thesis requirement.

### 6.4 One inference implementation

Offline and live paths must share:

- model loading;
- preprocessing;
- confidence filtering;
- coordinate conversion;
- body-model interpretation;
- tracker semantics;
- mesh reconstruction.

Streaming-specific frame selection and transport remain explicit. Do not copy
model logic into the API or benchmark layer.

### 6.5 Local concurrency model

- All work touching the Metal model runs on one dedicated worker thread.
- Each sequence owns a separate tracker session.
- Camera input is latest-only and reports dropped frames.
- File input preserves frame order and does not silently drop frames.
- Disconnect and shutdown must cooperatively terminate readers and tasks.
- Multiple clients may be serialised; they must not create an invisible,
  unbounded inference backlog.

### 6.6 Stable coordinate and time conventions

- Body output uses camera-relative OpenCV coordinates in metres: +X right, +Y
  down, +Z forward.
- The Three.js conversion remains isolated and tested.
- Recorded-video timestamps are source-relative seconds.
- Camera timestamps use elapsed monotonic time.
- Intrinsics are recorded at source-frame resolution.
- Smoothing is viewer-side interpretation and never changes raw artifacts.

### 6.7 Local security boundary

The supported deployment is loopback-only. The API defaults to `127.0.0.1`.
Documentation must warn that binding to a network interface exposes artifact
files, uploaded footage, and inference without authentication. Production
authentication, authorisation, and multi-tenancy are non-goals.

---

## 7. Reproducibility and Data Contracts

### 7.1 Configuration

- Each experiment is described by a versioned configuration file.
- Relative input paths resolve against the declaring configuration file.
- Configuration models reject unknown keys.
- Algorithmic constants that affect results belong in configuration or a
  versioned schema, not in an undocumented manual step.
- A configuration must be sufficient to recreate a run when its local input
  and pinned weights are available.

### 7.2 Artifact identity

An artifact may be reused only when its complete computational identity matches:

- effective stage configuration;
- upstream artifact identity;
- source-content digest;
- model/checkpoint/backbone identity where applicable;
- artifact schema version.

A source path alone is not an input identity. Replacing a file or an image in a
directory at the same path must produce a different ingest identity and must
invalidate dependent results.

### 7.3 Transactional completion

- A descriptor is the completion marker and is published last.
- New output is staged outside the currently reusable artifact.
- Final publication uses same-filesystem atomic replacement where possible.
- A failed overwrite preserves either the previous complete artifact or no
  reusable artifact.
- A valid-looking descriptor must never point to partial or truncated files.
- Readers verify referenced file existence and expected byte size before reuse.
- Existing local artifacts are not deleted or migrated without user approval.

### 7.4 Typed boundaries

- Pydantic models define Python file and wire boundaries.
- Zod schemas validate browser-visible JSON.
- Binary sidecars declare names, endianness, dtype, shape, offsets, and lengths.
- Cross-references and uniqueness constraints are validated.
- Schema changes are versioned, tested, documented, and explicitly approved.
- Backward incompatibility is acceptable only when regeneration is documented.

### 7.5 Manifests

Each measured run records at least:

- experiment and stage;
- effective configuration hash;
- source digest;
- Git commit and dirty state;
- dependency versions;
- model and backbone identities;
- random seed;
- hardware and execution backend;
- start and finish timestamps;
- wall-clock duration;
- available memory measurements;
- success, failure, and exclusion reason.

### 7.6 Raw and derived results

- Raw observations are immutable inputs to analysis.
- Derived summaries, CSV files, tables, and figures are regenerated from raw
  observations.
- JSON or JSON Lines is the machine-readable source of truth.
- Generated CSV is for inspection and thesis tooling, not independent editing.
- Aggregation must be deterministic and must not run inference.

---

## 8. Experimental Method

### 8.1 Pre-registration

Before final data collection, version a protocol defining:

- exact hardware and software environment;
- scenes and selection rationale;
- independent and controlled variables;
- warm-up and repetition policy;
- metrics, units, and calculation methods;
- failure and exclusion rules;
- aggregation method;
- planned tables and figures;
- privacy and consent handling.

After final measurement begins, protocol changes require a dated note describing
what changed, why, and which results were collected under each version.

### 8.2 Timing measurements

- Measure cold start separately from steady state.
- Use at least three measured repetitions when runtime permits.
- Use a monotonic high-resolution clock.
- Do not include warm-up samples in steady-state aggregates.
- Preserve individual observations.
- Report sample count, median, a tail percentile, and range or interquartile
  range.
- Report failures and zero-detection frames.
- Do not report precision unsupported by the clock or sample size.

### 8.3 RQ2 comparison matrix

Keep the matrix minimal and interpretable:

| Factor | Required levels |
|---|---|
| Visible people | Low, medium, high |
| Aspect ratio | 16:9, square |
| Resolution | Two levels from identical content |
| Output | Skeleton, mesh |

Prefer controlled transformations of the same source. When natural scenes are
used for person-count levels, label the comparison observational and report
confounders.

### 8.4 Controlled geometric case study

Use the smallest scene that answers RQ3:

- fixed camera and recording settings;
- measured floor markers at several lateral and depth positions;
- one consenting participant; add a second only for a declared occlusion or
  tracking question;
- stationary intervals at known positions;
- one simple measured walking path;
- recorded uncertainty for floor placement, camera placement, and annotation.

This is a controlled case study, not a general accuracy benchmark.

### 8.5 Missing data and post-processing

- Missing detections remain missing in accuracy calculations.
- Interpolation used only for visual playback is not ground truth.
- Track fragments are reported rather than manually joined without a record.
- Raw and smoothed measurements use identical intervals and definitions.
- Smoothing is evaluated for both stability benefit and response cost.

### 8.6 Threats to validity

The final analysis must discuss:

- one hardware family and the exact tested machine;
- one primary pretrained body estimator;
- a small set of purpose-built scenes;
- uncertainty of manually placed and annotated ground truth;
- occlusion and tracking errors;
- confidence scores not being calibrated probabilities;
- missing multi-view depth constraints;
- visual plausibility not implying metric correctness;
- possible coupling between scene content and performance factors.

---

## 9. Required Deliverables and Non-Goals

### 9.1 Required deliverables

1. Reliable offline body-capture execution.
2. Reliable local video and camera demonstration.
3. Versioned experiment and benchmark definitions.
4. Machine-readable raw measurements.
5. Deterministic aggregation into thesis-ready summaries.
6. A controlled RQ3 case study.
7. Reproducible tables and figures.
8. Architecture, protocol, limitations, and reproduction documentation.
9. Automated checks that do not require model weights or private footage.
10. A concise recorded demonstration for presentation and portfolio use.

### 9.2 Non-goals

- 3D Gaussian Splatting implementation or training;
- new model development or fine-tuning;
- broad multi-model benchmarking;
- dynamic 4D scene reconstruction;
- multi-camera fusion;
- SLAM or bundle-adjustment research;
- hand, face, or expression capture;
- avatar retargeting or Blender/Unity/Unreal export;
- mobile or cloud applications;
- authentication, multi-tenancy, or production deployment;
- medical, ergonomic, or safety-critical validation;
- a large custom dataset;
- unrelated visual redesign.

These are future-work opportunities, not unfinished thesis requirements.

---

## 10. Engineering Standards

### 10.1 Python

- Python remains pinned to the supported 3.12 range.
- All signatures are typed and pass mypy strict.
- No bare `Any` or type suppression hides a real mismatch.
- Pydantic models guard process and file boundaries.
- The pipeline is synchronous; blocking API work runs outside the event loop.
- Exceptions preserve actionable context and avoid exposing internal details to
  remote clients.

### 10.2 TypeScript and React

- TypeScript strict mode remains enabled.
- Use `unknown` plus narrowing instead of `any`.
- Shared modules use named exports.
- Effects own and clean up the resources they create.
- Only the latest asynchronous user action may update visible session state.
- The render loop remains independent of model inference cadence.
- Runtime JSON is parsed rather than asserted.

### 10.3 Tests

- Every bug fix starts with a regression test demonstrating the failure.
- Deterministic geometry, selection, schemas, conversions, aggregation, and
  state transitions are unit-tested.
- Model-backed tests are separate, explicitly marked, and never part of normal
  offline CI.
- No unit test downloads weights, contacts external services, or requires
  private data.
- API lifecycle tests cover cancellation and disconnect behaviour.
- Frontend tests cover stale asynchronous completion and cleanup.
- Artifact tests cover valid, corrupted, missing, and incompatible data.

### 10.4 Documentation

- Accepted ADRs are immutable; later changes supersede them with a new ADR.
- Dated notes record failures and dead ends as well as successes.
- Documentation distinguishes implemented, planned, and future work.
- Commands, filenames, schemas, and local links must match the repository.
- Code comments explain non-obvious intent and coordinate or concurrency
  constraints, not syntax.

### 10.5 Change discipline

- Make the smallest coherent change satisfying one work package.
- Avoid speculative abstractions and unrelated refactors.
- Preserve user changes and existing comments.
- Do not delete datasets, model weights, artifacts, or results without explicit
  approval.
- A runtime dependency or accepted schema change requires prior approval.

---

## 11. Quality Gates

The supported full verification sequence is:

```shell
uv run ruff check .
uv run mypy
uv run pytest
pnpm format:check
pnpm compile
pnpm lint
pnpm test
pnpm build
uv run unproject validate-config experiments/
```

Additional gates:

- generated artifacts, datasets, uploads, model weights, caches, coverage
  outputs, and build outputs remain untracked;
- `pnpm-lock.yaml` is the authoritative JavaScript lockfile;
- no test requires network access after dependencies are installed;
- a documentation-only change runs formatting/link checks appropriate to the
  files changed rather than model inference;
- final verification results are reported exactly, including skipped or
  unavailable checks.

If a sandbox prevents `uv` from using its global cache, use the existing
virtual environment for verification. Do not change repository configuration
to work around the sandbox.

---

## 12. Ordered Implementation Plan

Complete one work package, verify it, report it, and stop for review before
starting the next. A work package should normally map to one commit or pull
request.

### WP0 — Consolidate the source of truth

**Supports:** all RQs and project governance

Tasks:

1. Version ADR-0001 under `docs/adr` as the historical baseline without
   rewriting its accepted content.
2. Keep this ADR as the current direction decision.
3. Update the README so body capture is the primary thesis path and scene
   reconstruction is preliminary work.
4. Correct stale filenames, including references to `bodies.npz` where the code
   writes `bodies.bin`.
5. Move still-valid notes from ignored temporary storage into `docs/notes`.
6. Remove or replace broken links and references to nonexistent commands.
7. Reconcile the M3 Pro/M4 Pro documentation discrepancy without guessing.
8. Add `docs/experiments/protocol.md` with explicit `Draft` status and no
   invented results.

Acceptance criteria:

- all repository-local documentation links resolve;
- current commands and artifact names match code;
- historical and current decisions are clearly distinguished;
- implemented, planned, deferred, and future work are labelled;
- no prior ADR is silently altered.

### WP1 — Make artifact identity and publication trustworthy

**Supports:** reproducibility and every reported result

Tasks:

1. Include the source-content fingerprint in ingest identity.
2. Propagate that identity through dependent stage hashes.
3. Stage new output outside the reusable target artifact.
4. Publish completion transactionally.
5. Preserve the previous complete artifact when overwrite fails.
6. Validate referenced frame and sidecar existence and expected size before
   reuse.
7. Reject interrupted and incompatible artifacts with actionable diagnostics.

Required regression cases:

- a different video at the same path yields a different ingest identity;
- changing one image in a directory yields a different identity;
- identical content and configuration reuse existing output;
- downstream identities change after source-content changes;
- mid-run failure cannot expose partial output as complete;
- failed overwrite leaves the previous complete artifact usable;
- missing or truncated sidecars prevent reuse.

Stop condition:

- do not automatically delete or migrate existing local artifacts. Document
  regeneration and request approval for destructive migration.

### WP2 — Make live lifecycle and concurrency reliable

**Supports:** RQ1 and final-demo reliability

Tasks:

1. Add cooperative cancellation to the threaded video reader.
2. Prevent producers from blocking indefinitely on a full queue after client
   disconnect.
3. Deterministically consume cancelled tasks and reader exceptions.
4. Close the estimator executor through FastAPI lifespan handling.
5. Preserve the single dedicated Metal worker.
6. Define and bound concurrent-session backlog.
7. Return stable user-facing errors while retaining detailed local logs.

Required regression cases:

- disconnect while the producer is blocked;
- disconnect during inference;
- malformed camera frame;
- video-reader failure;
- application shutdown;
- two sessions cannot execute model work concurrently;
- a disconnected session does not damage the next session.

Acceptance criteria:

- no decoder, task, worker, or MediaStream-equivalent server resource remains
  owned by a closed session;
- camera input stays latest-only and reports dropped frames;
- file input remains ordered and lossless with respect to configured stride.

### WP3 — Remove frontend asynchronous state races

**Supports:** RQ1 and final-demo reliability

Tasks:

1. Introduce request ownership or cancellation for run loading, mesh loading,
   upload, topology loading, and camera start.
2. Prevent stale completions from changing `session`, `bodies`, `busy`, or
   `error` state.
3. Stop a camera stream obtained after its request was superseded.
4. Ignore WebSocket events after session close.
5. Preserve source synchronisation and raw/smoothed controls.

Required regression cases:

- run A resolves after run B;
- mesh A resolves after switching to run B;
- teardown occurs during upload;
- camera permission resolves after teardown;
- a closed socket emits a late message.

Acceptance criteria:

- only the latest user action owns visible state and resources;
- stale network work is aborted where possible and ignored otherwise;
- no camera track remains active after teardown.

### WP4 — Add reproducible performance measurement

**Supports:** RQ1 and RQ2

Tasks:

1. Define minimal versioned benchmark configuration and result schemas.
2. Reuse the production estimator path.
3. Record per-run and per-frame timing, warm-up status, people detected, input
   geometry, output mode, dropped frames, process memory, and available Metal
   memory counters.
4. Record environment, Git state, source digest, model identities, and effective
   configuration.
5. Add deterministic aggregation producing summary JSON and generated CSV.
6. Provide a fixture or dry-run path that tests orchestration without loading
   weights.

Design constraints:

- introduce one small CLI surface, not a general benchmark framework;
- do not duplicate inference code;
- do not require the browser except for an explicit end-to-end transport test;
- never mix cold-start and steady-state samples.

Acceptance criteria:

- one documented command creates each raw benchmark result;
- one documented command regenerates summaries without inference;
- raw observations are append-only or content-addressed;
- aggregation is deterministic and unit-tested;
- failures and exclusions remain visible.

### WP5 — Implement RQ3 metrics and controlled annotations

**Supports:** RQ3

Tasks:

1. Finalise the controlled-scene protocol before final recording.
2. Define a small typed annotation format for measured positions and labelled
   stationary/motion intervals.
3. Compute raw lateral and depth position error.
4. Compute temporal displacement and apparent velocity by axis.
5. Compute within-track shape variation and track fragmentation.
6. Compute 2D reprojection consistency from recorded intrinsics.
7. Evaluate smoothing with the same metrics plus response delay.
8. Generate machine-readable summaries and publication-ready figures from the
   same raw data.

Acceptance criteria:

- every schema and figure names units and coordinate convention;
- missing detections are not silently interpolated into accuracy results;
- raw and smoothed measurements remain distinguishable;
- annotation uncertainty is recorded;
- analysis is deterministic and tested with synthetic cases.

### WP6 — Pre-register and execute final experiments

**Supports:** RQ1, RQ2, and RQ3

Tasks:

1. Freeze protocol and experiment configurations.
2. Record the exact benchmark machine and resolve earlier hardware ambiguity.
3. Execute the minimal RQ1/RQ2 matrix with declared warm-up and repetitions.
4. Execute the controlled RQ3 case study.
5. Preserve raw outputs and write dated notes for failures or exclusions.
6. Generate final tables and figures only through analysis commands.
7. Review every conclusion against section 3.

Acceptance criteria:

- every reported value traces to configuration and raw observations;
- aggregation reruns produce identical output;
- failed and anomalous runs are visible;
- conclusions separate observation, interpretation, and speculation;
- no post-hoc protocol change is left undocumented.

### WP7 — Delivery hardening

**Supports:** reproducibility and portfolio quality

Tasks:

1. Add CI for Python and frontend checks that need neither weights nor private
   footage.
2. Verify setup and artifact viewing from a clean checkout.
3. Document download size, cold-start behaviour, privacy boundary, and measured
   performance without unsupported promises.
4. Add a concise architecture diagram and reproduction guide.
5. Record a short demo showing source view, off-axis view, raw and smoothed
   output, skeletons, and meshes.
6. Remove accidental generated files or non-authoritative lockfiles only after
   confirming they are not user work.

Acceptance criteria:

- CI passes from a clean checkout after ordinary dependency installation;
- the documented default command starts the application;
- completed artifacts remain viewable without optional model dependencies;
- missing inference extras produce actionable installation guidance;
- repository status remains clean after the documented workflow;
- no private footage or model weight is committed.

---

## 13. Priorities and Time-Boxing

Priority order:

1. trustworthy artifacts and raw measurements;
2. reliable offline and live demonstrations;
3. controlled RQ3 evidence;
4. deterministic analysis and figures;
5. documentation and CI;
6. optional presentation polish.

If time becomes constrained:

- keep skeleton mode before optional mesh polish;
- keep offline experiments before camera UX polish;
- keep raw measurement export before chart styling;
- keep one controlled participant before adding participants;
- keep median and tail latency before optional counters;
- remove optional UI work;
- never remove source digests, raw observations, units, failure reporting,
  protocol versioning, or the raw/smoothed distinction.

---

## 14. Definition of Done

The implementation and evaluation are complete when:

- WP0–WP7 acceptance criteria are satisfied, except optional items explicitly
  removed through documented time-boxing;
- all quality gates pass;
- final experiments follow the frozen protocol;
- every thesis claim maps to reproducible evidence;
- the controlled case study reports uncertainty and limitations;
- offline artifacts and live sessions terminate reliably;
- the viewer cannot display stale asynchronous results as current;
- setup and analysis are documented from a clean checkout;
- no conclusion depends on unimplemented 3DGS training.

A well-measured negative result satisfies the thesis. Unrelated features must
not be added to avoid reporting a limitation.

---

## 15. Required Workflow per Work Package

For each work package, the implementer must:

1. read this ADR and inspect the current repository and worktree;
2. identify the next incomplete work package;
3. state the supported RQ or reproducibility requirement;
4. identify pre-existing user changes and preserve them;
5. add a failing regression test before each bug fix;
6. implement the smallest coherent solution;
7. run package-scoped checks during iteration;
8. run all applicable quality gates before completion;
9. report files changed, tests added, commands and outcomes, schema effects,
   reproducibility effects, and remaining risks;
10. stop at the work-package boundary for review.

The implementer must request confirmation before:

- changing an accepted artifact or wire schema;
- invalidating existing experiment results;
- adding a runtime dependency;
- deleting or migrating artifacts, datasets, footage, or weights;
- changing a research question or frozen protocol;
- performing work listed as a non-goal.

---

## 16. Consequences

### Positive

- The thesis is narrow enough to complete while retaining meaningful AI,
  computer-vision, geometry, backend, concurrency, and frontend work.
- Existing body-capture and viewer code becomes the primary contribution.
- Previous pose work remains useful as evidence for reformulating the problem.
- Negative findings about monocular depth remain valid research outcomes.
- Typed artifacts and pre-registered experiments create a defensible chain from
  source input to thesis claim.
- Local inference supports a clear privacy argument.

### Negative and accepted trade-offs

- The system does not reconstruct a photorealistic environment.
- One primary pretrained estimator limits model-comparison claims.
- Findings are specific to a small set of scenes and one hardware class.
- A controlled case study cannot establish clinical or population-level
  accuracy.
- Reliability and measurement work takes precedence over visible new features.

### Future work

Not part of this thesis:

- avatar retargeting and animation export;
- sports or ergonomics analysis after domain validation;
- multi-camera capture;
- telepresence;
- privacy-preserving occupancy analytics;
- dynamic human-scene reconstruction;
- broader estimator and hardware comparisons.

---

## 17. References and Evidence Policy

The thesis bibliography should include primary sources for:

- Multi-HMR 2;
- the Anny body model;
- Structure-from-Motion and COLMAP;
- the PyTorch MPS backend;
- any metric or statistical method used in final evaluation.

Exact bibliographic metadata and licensing claims must be verified against
primary project or publication sources before submission. Repository notes,
manifests, configurations, and raw measurements are evidence for implementation
and experimental claims, not substitutes for scholarly citations.
