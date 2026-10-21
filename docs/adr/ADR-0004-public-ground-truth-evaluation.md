# ADR-0004 — Public Ground-Truth Evaluation with MuPoTS-3D and Anny-One

**Status:** Accepted
**Date:** 2026-09-08
**Type:** Experimental-method and evaluation-architecture decision
**Audience:** Thesis author, supervisors, and contributors
**Extends:** ADR-0002 and ADR-0003
**Supersedes:** ADR-0002 §8.4, the manually measured position-error part of
ADR-0002 WP5 and WP6, and the controlled-scene collection plan in
`docs/experiments/protocol.md`
**Preserves:** The research questions, local Apple Silicon constraint, raw body
artifacts, reproducibility requirements, missing-detection policy, raw-versus-
smoothed separation, and the prohibition on re-identification

> **Note for contributors:** Read ADR-0002, ADR-0003 and this ADR
> completely before implementation. Implement the work packages in
> order. Stop at every review gate. Do not download a restricted dataset,
> accept a licence, change an evaluation split, or invent a joint mapping on the
> author's behalf. Dataset facts discovered during implementation must be written
> to a dated note and supported by the released files or an official source.

---

## 1. Decision Summary

The thesis will replace the unrecorded, manually measured RQ3 corridor study
with evaluation on two public datasets with 3D ground truth:

1. **MuPoTS-3D is the primary real-video benchmark.** It measures multi-person
   detection coverage, 3D joint accuracy, camera-space placement, depth error,
   and temporal stability on real indoor and outdoor sequences.
2. **Anny-One is the secondary synthetic compatibility benchmark.** It measures
   frame-level geometry, camera estimation, and Anny pose and shape outputs
   against annotations expressed in the same body-model family as Multi-HMR 2.

They answer different parts of RQ3:

```text
MuPoTS-3D                       Anny-One
real multi-person video        synthetic multi-person images
        |                               |
        v                               v
common 3D joints               native Anny annotations
        |                               |
        +-----------+   +---------------+
                    v   v
          evidence about accuracy,
          depth, stability and limits
```

MuPoTS-3D remains the primary accuracy result because it is real video and is
designed for monocular multi-person 3D pose evaluation. Anny-One is not treated
as independent evidence of real-world generalisation unless the final
Multi-HMR 2 paper and checkpoint documentation establish that the evaluated
images were excluded from training and model selection.

No new participant recording, tape measurement, camera surveying, or manually
placed floor marker is required for the thesis evaluation. Existing manual RQ3
code and templates remain as preliminary work but receive no further feature
development unless needed to preserve current tests or explain the project
history.

---

## 2. Why This Decision Is Necessary

### 2.1 Problem with the previous protocol

ADR-0002 proposed a fixed-camera corridor with floor positions measured by
hand. That protocol is defensible as a small case study, but it has avoidable
costs:

- the author must obtain a suitable space, camera, participant, and consent;
- camera and marker placement introduce manual uncertainty;
- a single corridor and participant provide weak external validity;
- measurements are hard for another researcher to reproduce;
- the resulting evidence cannot be compared directly with published methods;
- collection consumes time that is better spent validating conversions,
  metrics, and conclusions.

The recording had not been made when this ADR was accepted. Replacing it does
not discard collected thesis data.

### 2.2 Why two datasets are required

No single practical dataset covers all required evidence.

| Requirement | MuPoTS-3D | Anny-One |
|---|---:|---:|
| Real RGB observations | Yes | No |
| Consecutive video frames | Yes | Not the primary release purpose |
| Multiple people and occlusion | Yes | Yes |
| Camera-space 3D joints | Yes | Yes, subject to released schema verification |
| Native Anny pose and shape | No | Yes |
| Direct Anny mesh comparison | No | Potentially, subject to the implementation gate in §6.4 |
| Exact synthetic camera parameters | No | Yes |

MuPoTS-3D cannot validate the detailed Anny surface or Anny phenotype
parameters. Anny-One cannot by itself establish performance on real video or
temporal tracking. Reporting them separately prevents one dataset from being
used to support a claim it cannot test.

### 2.3 What “ground truth” means here

Ground truth is a reference annotation, not a claim of metaphysical certainty.

- MuPoTS-3D poses were obtained from a multi-view markerless performance-
  capture process. They are a standard reference with their own measurement
  and annotation limitations.
- Anny-One images are rendered from virtual scenes. The generator knows the
  camera and Anny body parameters used to create each image, so their numerical
  annotations are exact with respect to that synthetic scene.

The thesis must call the first **capture-derived ground truth** and the second
**synthetic ground truth** when the distinction matters.

---

## 3. Scope and Claims

### 3.1 Questions this work answers

This ADR primarily supports RQ3:

> How stable and geometrically credible are reconstructed bodies over time,
> especially along the monocular depth axis?

It contributes secondary observations to RQ1 and RQ2 because inference time,
memory, failures, and detected-person counts may be recorded while processing
the datasets. Those observations do not replace the pre-registered RQ1/RQ2
benchmark matrix.

### 3.2 Permitted claims

Subject to results, the thesis may claim:

- accuracy on the exact MuPoTS-3D release and protocol used;
- accuracy on the exact pre-registered Anny-One subset used;
- differences between root-relative pose quality and absolute camera-space
  placement;
- depth-axis bias and error as a function of annotated range;
- temporal joint-velocity error on valid consecutive matched observations;
- detection coverage under the declared confidence threshold;
- differences between real capture-derived and synthetic evidence;
- full local execution on the declared Apple Silicon machine.

### 3.3 Prohibited claims

The thesis must not claim:

- that MuPoTS-3D validates the complete Anny mesh or body shape;
- that Anny-One alone establishes real-world accuracy;
- dataset-wide Anny-One performance when only a subset or shard was evaluated;
- independent generalisation on any data used to train or select the checkpoint;
- identity accuracy from Multi-HMR 2 track IDs without verified identity labels
  and a separately declared tracking protocol;
- metric accuracy for unmatched or missing people;
- medical, biometric, ergonomic, or safety-grade measurement;
- that root-aligned metrics measure absolute depth or scene placement;
- that synthetic ground truth removes the synthetic-to-real domain gap.

### 3.4 Scope filter

Every implementation task must state which one it enables:

```text
M1  MuPoTS detection and joint evaluation
M2  MuPoTS camera-space depth evaluation
M3  MuPoTS temporal stability evaluation
A1  Anny-One native-Anny geometric evaluation
A2  Anny-One camera and shape diagnostic evaluation
R   Reproducibility, validity, or thesis-ready reporting
```

If a task enables none of these, it does not enter this work.

---

## 4. Dataset Policy

### 4.1 Human-controlled acquisition

Dataset licences may require registration or affirmative acceptance. Only the
author performs either action.

The author is responsible for:

1. reading and accepting each official licence;
2. obtaining the files from the official provider;
3. placing them under the documented local directory;
4. retaining any proof of the release version or download date required for
   reproducibility.

The project supplies verification and indexing commands, not a downloader that
bypasses an access gate.

### 4.2 Local layout

Use the following logical layout unless the released archive makes a different
read-only layout necessary:

```text
datasets/
  mupots-3d/             # ignored, original files left unchanged
  anny-one/              # ignored, original files left unchanged
  _indexes/              # generated local inventories, ignored if large

experiments/ground-truth/
  mupots-3d.toml
  anny-one.toml
  mappings/
    mupots-3d-to-anny-v1.toml
  subsets/
    anny-one-v1.txt
```

Raw dataset files are immutable inputs. Code must never rename, rewrite, or
silently repair them. Any derived frame, normalized annotation, cache, or
prediction belongs under `artifacts/`, not inside the dataset.

### 4.3 Version and integrity record

For each dataset record:

- official name and canonical URL;
- licence identifier or a hash of the supplied licence text;
- release/version label when one exists;
- acquisition date;
- official archive checksums when supplied;
- locally computed SHA-256 for annotation files and subset manifests;
- file count and byte count;
- excluded files and the reason for exclusion.

Do not commit restricted images, videos, annotations, extracted frames, or a
public remote from which they can be recovered. Commit only configuration,
small non-sensitive manifests, official file identifiers, hashes, mappings,
and instructions allowed by the licence. DVC may track local identity only if
its metadata and remote respect the same restriction; introducing DVC is not a
prerequisite for this ADR.

### 4.4 Dataset-specific selection

**MuPoTS-3D:** evaluate all 20 official test sequences if the released files
and licence permit local evaluation. Preserve per-sequence results. A pilot on
one short sequence may validate the pipeline but cannot appear as the final
MuPoTS result.

**Anny-One:** do not download or evaluate all 780,964 images by default. The
release is split across archives of approximately 12–17 GB each, so exhaustive
evaluation would violate the thesis time and storage priorities. Before any
model result is inspected:

1. inventory the metadata and official issue/exclusion list;
2. determine whether annotations can be obtained without every RGB archive;
3. define a deterministic subset from the locally available release;
4. stratify, where annotations allow it, by visible-person count and camera
   field-of-view or distance bins;
5. use a fixed seed and sorted stable identifiers;
6. commit the selected identifiers and selection algorithm;
7. name the result as a **subset** or **part-N subset**, never as full
   Anny-One performance.

The initial target is 300 images, with a minimum acceptable pilot of 60. Change
the target only before inference and document the storage/runtime reason.

### 4.5 Training-overlap gate

Before Anny-One results are classified, inspect the official Multi-HMR 2 paper,
checkpoint card, repository, or author documentation and record the known
training and model-selection datasets.

- If evaluated Anny-One images may have participated in training or selection,
  label the result **in-domain compatibility diagnostic**.
- If an official held-out split is documented and provably unused, label that
  exact split accordingly.
- If the evidence is ambiguous, use the conservative first label.

This gate does not block running Anny-One. It blocks presenting the result as
independent generalisation evidence.

---

## 5. Evaluation Architecture

### 5.1 Reuse the existing inference path

Dataset evaluation must reuse:

- `MultiHmrEstimator` and `EstimatorSession`;
- `BodiesConfig` and its pinned checkpoint identity;
- the stage [B] confidence and NMS behaviour;
- the existing `BodiesArtifact`, including `joints_3d`, `root_translation`,
  frame intrinsics, boxes, and track IDs;
- the existing camera-relative OpenCV convention in metres.

Do not copy Multi-HMR preprocessing or inference code into a dataset adapter.
Do not call upstream CLI scripts as an alternative implementation.

### 5.2 New package boundary

Add one focused evaluation package:

```text
packages/pipeline/src/unproject/ground_truth/
  artifacts.py          typed output contract
  config.py             strict experiment config
  dataset.py            small shared normalized types
  mupots.py             MuPoTS reader only
  anny_one.py           Anny-One reader only
  mapping.py            explicit joint mapping and validation
  matching.py           prediction-to-person association
  metrics.py            pure numerical definitions
  aggregate.py          deterministic sequence/dataset summaries
  figures.py            thesis-ready figures from saved results
  stage.py              orchestration; no metric definitions
```

This is not a plugin system and not a general dataset framework. A small
`GroundTruthDataset` protocol is justified by exactly two implementations; it
must expose only operations both adapters need.

### 5.3 Normalized ground-truth contract

Both adapters convert released annotations into an internal typed contract. At
minimum it contains:

```text
DatasetRecord
  dataset_name
  release_identity
  sample_id
  sequence_id | null
  frame_index | null
  image_path
  image_width, image_height
  camera convention and units
  camera intrinsics, when available
  people[]
    ground_truth_person_id | null
    joints_3d[J, 3]
    joint_names[J]
    joint_valid[J]
    joints_2d[J, 2] | null
    joint_visible[J] | null
    root_translation[3] | null
    native Anny pose/shape | null
```

Use Pydantic for the file boundary and NumPy arrays only behind validated shape,
dtype, unit, finiteness, and cardinality checks. Dataset-native objects do not
leak into metric functions.

### 5.4 Coordinate normalization

The canonical evaluation frame is:

```text
camera-relative OpenCV axes
+X right, +Y down, +Z forward
metres internally; millimetres only in reported distance metrics
```

Each adapter must document and test the native-to-canonical transform. A
transform is not accepted because a rendered skeleton “looks right”. It must
pass synthetic checks for:

- handedness;
- axis direction;
- units;
- homogeneous-transform order;
- camera-to-world versus world-to-camera interpretation;
- projection through released intrinsics;
- round-trip conversion where an inverse exists.

### 5.5 Joint mapping instead of body-parameter conversion

MuPoTS and Anny use different body representations. Do not compare their pose
vectors, shape coefficients, mesh vertices, or bone indices directly.

Map both to a declared common joint set:

```text
Anny joints_3d ---- explicit versioned mapping ----+
                                                   +--> K common joints
MuPoTS joints_3d -- released semantic names -------+
```

The mapping file records, for every common joint:

- canonical name;
- MuPoTS source name/index;
- Anny source name/index;
- whether it is direct or derived;
- derivation, such as midpoint of left and right hip;
- provenance for the anatomical interpretation;
- supported Anny package version, bone count, and checkpoint identity.

Never infer a mapping only from matching array lengths or similar-looking
indices. The evaluator fails closed on an unknown body-model version, wrong
joint count, duplicate mapping, missing required joint, or non-finite value.

The first implementation should use the largest unambiguous shared set among
pelvis/root, shoulders, elbows, wrists, hips, knees, ankles, neck, and head.
Spine or face landmarks enter only when both released definitions match.

Do not change `BodiesArtifact` solely to store names if a versioned mapping can
validate the current artifact unambiguously. A body-artifact schema change
requires a separate migration decision if implementation evidence shows it is
necessary.

### 5.6 Prediction-to-ground-truth matching

Match people per frame before computing person-level metrics.

Preferred association cost:

1. common visible 2D joint distance normalized by image diagonal;
2. released 2D bounding-box distance or IoU when common 2D joints are absent;
3. no use of ground-truth 3D distance to choose a match for a metric that later
   claims 3D accuracy.

Use deterministic minimum-cost bipartite assignment with an explicit gate.
Unmatched ground-truth people are false negatives. Unmatched predictions are
false positives. Never drop either silently. Ties are resolved by stable source
order and recorded.

Prefer the official MuPoTS association/evaluation code for the headline
benchmark if it is available, readable, licence-compatible, and its behaviour
can be fixed to a revision or checksum. Keep the project-native normalized
evaluation for diagnostics and test both paths on a tiny fixture. Any
difference between official and project-native results must be explained
before final collection.

### 5.7 Dependency policy

Dataset-only parsers and assignment utilities belong in a dedicated optional
extra, for example `ground-truth`, so an ordinary application install does not
pull them in.

Before adding a package, record:

- why the standard library, NumPy, or current dependencies are insufficient;
- macOS ARM64 wheel availability;
- licence;
- locked version;
- import boundary;
- whether the same released format can be parsed safely without executing
  dataset-provided code.

Do not deserialize an untrusted pickle merely because an official sample script
does. Inspect the format first. Prefer documented MAT, JSON, NPZ, CSV, or text
fields parsed with safe readers. Do not add CUDA, a second body-reconstruction
model, a database server, or a cloud service.

---

## 6. Metrics

### 6.1 Common reporting rules

Every metric records:

- dataset, release, split/subset, sequence, and sample counts;
- number of ground-truth people, predictions, matches, false negatives, and
  false positives;
- number of valid joints contributing to the value;
- units, alignment, threshold, and aggregation method;
- confidence threshold and person-association gate;
- raw per-sample observations or a lossless typed sidecar;
- macro and micro aggregation kept distinct.

Never compute an accuracy average over matched people without reporting recall.
A model that detects one easy person out of ten must not appear highly accurate.

### 6.2 MuPoTS headline metrics

Implement and report the official protocol first, including the official joint
set and threshold when confirmed from the release:

- **3DPCKrel:** percentage of joints within 150 mm after root alignment;
- **3DPCKabs:** percentage within 150 mm in camera coordinates without root
  alignment;
- official all-annotation and detected-subject variants when the evaluator
  defines both;
- per-sequence and overall results using the official aggregation.

Project-native diagnostics, clearly separated from the headline protocol:

- MPJPE in millimetres over matched valid common joints;
- root-relative MPJPE;
- root translation MAE and signed bias for X, Y, and Z;
- depth absolute error and signed bias by ground-truth distance bin;
- precision, recall, F1, false positives, and false negatives at the frozen
  detector threshold;
- per-joint and per-sequence breakdowns.

`3DPCKrel` answers pose articulation after removing global placement.
`3DPCKabs`, root error, and Z error answer scene placement and monocular depth.
Do not substitute one for the other.

### 6.3 MuPoTS temporal extension

For consecutive frames with a valid, stable ground-truth identity and valid
matches in both frames, report:

- MPJVE: mean per-joint velocity error in mm/s;
- root velocity error by X, Y, and Z;
- coverage: eligible pairs, evaluated pairs, and pairs lost to detection or
  association gaps;
- Multi-HMR track-ID switches only if released identity continuity is verified.

Do not bridge missing frames. Do not manually join Multi-HMR tracks. Do not
compute velocity across a gap larger than the frozen maximum interval. If
MuPoTS does not expose stable person identity in a form that can be verified,
omit ID-switch claims and state the limitation; frame-local accuracy still
proceeds.

Raw and viewer-smoothed predictions remain separate. Smoothing metrics are
secondary and use exactly the same eligible pairs as raw metrics.

### 6.4 Anny-One metrics

After the training-overlap classification in §4.5, report on the exact fixed
subset:

- person detection precision, recall, and F1;
- common Anny-joint MPJPE, root-relative MPJPE, and PCK at declared thresholds;
- root X/Y/Z error and depth bias;
- focal-length or field-of-view error when released and predicted camera fields
  are definitionally compatible;
- per-parameter shape error only for parameters whose names, units, ranges, and
  ordering are verified identical;
- results stratified by person count and declared camera/distance bins when the
  subset supports them.

Pose-matrix geodesic error and per-vertex error are optional. They enter only
after the WP5 gate proves that:

- prediction and ground truth use the same Anny skeleton and conventions;
- a posed mesh can be reconstructed locally without NVIDIA Warp or CUDA;
- correspondence is vertex-to-identical-vertex, not nearest-neighbour surface
  distance presented as correspondence;
- the implementation is validated on an identity/rest-pose fixture;
- runtime fits the thesis budget.

If any condition fails, joint-level evaluation is the complete Anny-One
deliverable. This is a successful scoped result, not unfinished work.

### 6.5 Aggregation

Preserve both:

- **micro average:** every valid joint/person contributes directly;
- **macro average:** compute per sequence or stratum first, then average groups.

The main table follows the dataset's official aggregation. Diagnostic figures
state their aggregation in title or caption metadata. Report median and p90 or
interquartile range for skewed continuous errors; retain means required by a
named standard metric such as MPJPE.

Do not pool MuPoTS and Anny-One into one score.

---

## 7. Artifact and Configuration Contracts

### 7.1 Configuration

Add a strict, versioned ground-truth evaluation configuration separate from
the existing RQ1/RQ2 benchmark schema. It must reject unknown fields and contain
at least:

- dataset kind and local root;
- release and split/subset identifier;
- subset-manifest path where applicable;
- annotation hashes or inventory identity;
- body experiment/defaults and pinned checkpoint identity;
- joint-mapping identifier and hash;
- confidence and association thresholds;
- metric thresholds and distance bins;
- maximum frame interval for temporal metrics;
- output directory and deterministic seed;
- official evaluator identity when used.

Paths are locations and do not replace content identity in the configuration
hash.

### 7.2 Outputs

Write evaluation artifacts beneath a distinct namespace:

```text
artifacts/ground-truth/<evaluation-id>/
  evaluation.json          descriptor, identity, summary and completion marker
  observations.npz         validated numerical observations, no object arrays
  per-sequence.csv
  per-joint.csv
  figures/
  logs/
```

`evaluation.json` is written last and is the atomic completion marker. It
contains schema version, config hash, Git SHA and dirty state, dataset identity,
subset-manifest hash, mapping hash, source body-artifact identities, model
identity, environment, failure/exclusion counts, and references with byte sizes
and SHA-256 hashes.

Do not use pickle or an NPZ object dtype. CSV files and figures are derived
views, never the source of truth. Regenerating them from `evaluation.json` and
`observations.npz` must be deterministic.

### 7.3 CLI

Expose the smallest sufficient command surface:

```text
uv run unproject gt-evaluate experiments/ground-truth/mupots-3d.toml
uv run unproject gt-evaluate experiments/ground-truth/anny-one.toml
uv run unproject gt-summary artifacts/ground-truth/<evaluation-id>
```

`gt-evaluate` validates the dataset before loading model weights. It resumes
complete per-sequence body artifacts and never silently overwrites a completed
result. `gt-summary` performs no inference and requires no model weights.

The existing `unproject rq3` command remains available for old uncontrolled or
manual-annotation analyses but is not the final ground-truth benchmark command.

---

## 8. Testing Strategy

Tests must not require dataset downloads, model weights, network access, or an
MPS device.

### 8.1 Required deterministic fixtures

Create tiny, hand-authored fixtures that cover:

- one empty frame;
- one perfectly predicted person;
- two people presented in opposite source orders;
- one false positive and one false negative;
- one invalid or invisible joint;
- metres-to-millimetres conversion;
- one axis inversion and one camera-transform round trip;
- direct and derived joint mappings;
- a root translation offset that changes absolute but not root-relative error;
- a 149 mm and a 151 mm joint for the 150 mm PCK boundary;
- a temporal gap that must not be bridged;
- deterministic subset selection and aggregation;
- rejection of NaN, infinity, wrong shapes, unknown schema versions, wrong
  mapping hashes, and unexpected dataset layouts.

### 8.2 Golden compatibility fixture

After lawful acquisition, create a minimal metadata-only compatibility fixture
from each release if its licence allows redistribution. Otherwise create an
equivalent synthetic fixture matching the documented schema and store hashes of
the real files used for local integration testing.

The fixture proves parser compatibility, not model accuracy.

### 8.3 Official-evaluator cross-check

If the official MuPoTS evaluator is used, construct one tiny valid prediction
with a known expected score and compare it to the project-native metric. Pin the
official code by source hash. Never edit it silently to make results agree.

### 8.4 Quality gates

Every implementation work package ends with the relevant subset of:

```shell
uv run ruff check --fix .
uv run ruff format .
uv run mypy --strict packages/pipeline apps/api
uv run pytest packages/pipeline apps/api
uv run unproject validate-config experiments/
```

Dataset integration tests are explicitly marked and skipped with an actionable
message when local licensed data is absent. Unit tests must still pass.

---

## 9. Work Packages and Implementation Order

### WP0 — Freeze evidence and inspect releases

**Supports:** R, prevents invalid results

Tasks:

1. Create `docs/experiments/ground-truth-protocol.md` from this ADR.
2. Record official URLs, papers, licences, download/release metadata, and known
   checkpoint training data in a dated note.
3. After the author supplies each dataset, inventory its actual layout and
   annotation schema without executing dataset-bundled code.
4. Confirm MuPoTS native joints, units, coordinate frame, visibility fields,
   person identity semantics, and official evaluator behaviour.
5. Confirm Anny-One pose, shape, camera, scene, person, view, segmentation, and
   issue-list fields.
6. Define final configurations and freeze all metric thresholds before model
   output is inspected.

Acceptance criteria:

- every factual assumption in the adapters has an official or file-level
  source;
- licence-dependent manual steps are explicit;
- no raw dataset file is modified or committed;
- unknown schema details remain named gates, not guesses;
- the protocol says exactly which claims each dataset may support.

**Review gate:** The author confirms lawful local availability. If either
dataset is unavailable, stop that adapter and report the blocker; do not replace
it with an unapproved dataset.

### WP1 — Add typed contracts and dataset validation

**Supports:** M1, M2, A1, A2, R

Tasks:

1. Add strict configuration models and CLI config detection.
2. Add normalized ground-truth and evaluation artifact models.
3. Implement dataset inventory and fail-fast validation.
4. Implement content identities and transactional result publication.
5. Add unit fixtures for malformed schemas and missing files.

Acceptance criteria:

- validation runs without model weights;
- error messages name the file, field, expected convention, and corrective
  action;
- dataset path changes do not change identity, but content changes do;
- no descriptor can point to a partial sidecar;
- all contracts pass strict type checking.

### WP2 — Implement and verify the MuPoTS adapter

**Supports:** M1, M2, M3

Tasks:

1. Parse the released MuPoTS annotations with a safe, documented reader.
2. Normalize images, frames, people, joint validity, and camera-space pose.
3. Create the explicit MuPoTS-to-Anny joint mapping.
4. Validate units and coordinate transforms with projections and synthetic
   fixtures.
5. Convert or expose every sequence through the existing stage [B] path without
   duplicating inference.
6. Preserve sequence boundaries so each sequence receives a fresh tracker.

Acceptance criteria:

- all 20 official sequence identifiers are inventoried;
- one metadata-only fixture parses deterministically;
- a wrong axis, scale, mapping version, or joint count causes a test failure;
- a one-sequence pilot produces reusable `BodiesArtifact` data and normalized
  ground truth;
- no final metric is reported yet.

**Review gate:** Inspect projected joints and numerical invariants for a small
pilot. Visual inspection is a diagnostic only; automated checks decide pass or
fail.

### WP3 — Implement MuPoTS matching and metrics

**Supports:** M1, M2, M3

Tasks:

1. Implement deterministic 2D person assignment and unmatched-person counts.
2. Implement official 3DPCK metrics and aggregation.
3. Implement MPJPE, root-relative MPJPE, root-axis error, depth bins, and
   detection metrics.
4. Implement temporal velocity error only for verified eligible frame pairs.
5. Cross-check against the official evaluator where feasible.
6. Write raw observations, summary tables, and deterministic figures.

Acceptance criteria:

- root alignment changes relative metrics but never absolute metrics;
- missing detections reduce coverage and cannot disappear from reports;
- official and native results agree on the golden case or their semantic
  difference is documented;
- macro and micro values are distinguishable;
- tables and figures regenerate byte-stably from saved observations.

### WP4 — Execute the complete MuPoTS evaluation

**Supports:** M1, M2, M3, R

Tasks:

1. Freeze code revision and effective configuration.
2. Run all sequences locally with one tracker session per sequence.
3. Preserve failures, zero-detection frames, raw predictions, and timings.
4. Regenerate the official headline table and project diagnostics.
5. Write a dated note separating observations, interpretations, and anomalies.

Acceptance criteria:

- every reported score traces to a sequence, mapping, configuration, and body
  artifact;
- all official sequences are completed or visibly listed as failed;
- no surprising result is excluded post hoc;
- rerunning aggregation performs no inference and produces identical values;
- MuPoTS results stand alone before Anny-One work starts.

**Review gate:** Do not start Anny-One implementation until the primary real-
video result is reproducible. If time is constrained, stop here; this is the
minimum complete thesis benchmark.

### WP5 — Implement the Anny-One adapter and frozen subset

**Supports:** A1, A2, R

Tasks:

1. Complete the training-overlap classification.
2. Parse the released metadata, issue list, camera fields, and Anny annotations.
3. Implement and commit the deterministic subset selection before inference.
4. Verify that predicted and ground-truth Anny versions and parameter semantics
   are compatible.
5. Normalize the selected images through the same inference path.
6. Implement joint, root, camera, and verified shape diagnostics.
7. Evaluate optional pose-matrix or mesh metrics only if every §6.4 gate passes.

Acceptance criteria:

- the subset manifest is stable, hashed, and generated without model output;
- excluded official issue-list scenes cannot enter the subset;
- the result names the exact subset/shard and training-overlap classification;
- incompatible Anny parameters fail closed instead of being compared;
- joint-level evaluation is complete even when mesh evaluation is rejected.

### WP6 — Execute Anny-One and compare evidence, not scores

**Supports:** A1, A2, R

Tasks:

1. Run the frozen subset locally.
2. Generate per-stratum results and failure counts.
3. Compare patterns with MuPoTS, especially relative versus absolute error and
   depth behaviour.
4. Discuss synthetic-to-real differences without combining scores.
5. Record runtime and storage cost to justify the subset size.

Acceptance criteria:

- no image was selected after inspecting its prediction error;
- no MuPoTS and Anny-One score is pooled into one number;
- claims respect the training-overlap classification;
- raw observations and derived figures are reproducible;
- anomalies and parser limitations are retained in the lab note.

### WP7 — Documentation and delivery integration

**Supports:** R and thesis communication

Tasks:

1. Update `docs/experiments/protocol.md` to mark the manual corridor as
   superseded and link to the ground-truth protocol.
2. Update the beginner thesis guide with a plain-language explanation of
   synthetic versus capture-derived ground truth, joint mapping, matching,
   alignment, and metrics.
3. Update `docs/reproduce.md` with licence-aware setup and commands.
4. Add architecture and data-flow diagrams generated from versioned sources.
5. Add CI for unit tests that require neither datasets nor model weights.
6. Check the repository for accidentally tracked dataset material before
   delivery.

Acceptance criteria:

- a clean checkout can validate configs and run all fixture tests;
- a licensed local checkout can reproduce both evaluations from documented
  commands;
- documentation never instructs the reader to obtain data from an unofficial
  mirror;
- every thesis table and figure names dataset, subset, alignment, units, and
  sample coverage;
- the old manual protocol cannot be mistaken for the final experiment.

---

## 10. Stop Conditions and Fallbacks

Implementation must stop and wait for a decision when:

- the released licence conflicts with repository publication or intended use;
- official annotations differ materially from the schema assumed here;
- camera-space MuPoTS ground truth cannot be obtained or interpreted;
- Anny joint or shape semantics cannot be matched unambiguously;
- an official evaluator requires an incompatible runtime or executes unpinned
  code;
- a dependency introduces CUDA, cloud inference, or an incompatible licence;
- final metrics would require changing a frozen split or threshold after
  results are known;
- implementing posed-mesh evaluation would materially delay the primary
  MuPoTS result.

Allowed fallbacks, in order:

1. keep MuPoTS headline metrics and omit unsupported diagnostics;
2. keep Anny-One joint/root evaluation and omit mesh or shape metrics;
3. reduce the pre-inference Anny-One subset for documented resource reasons;
4. omit Anny-One entirely if access or compatibility fails, while retaining a
   complete MuPoTS thesis result and documenting the attempted extension.

Do not silently replace MuPoTS or Anny-One with 3DPW, AGORA, BEDLAM, or another
dataset. That is a new experimental decision and requires author approval.

---

## 11. Consequences

### 11.1 Benefits

- no physical camera survey or manual distance measurement;
- reproducible comparison with a recognised real multi-person benchmark;
- explicit separation of pose articulation from absolute depth placement;
- native-Anny diagnostic evidence without converting body parameter spaces;
- stronger portfolio evidence through typed dataset adapters, reproducible
  evaluation, and validity-aware reporting;
- a smaller and more defensible undergraduate-thesis scope.

### 11.2 Costs

- licence-gated datasets still require manual acquisition;
- dataset parsing, coordinate conversion, joint mapping, and person association
  become critical correctness work;
- Anny-One storage is large even when inference uses only a subset;
- the two datasets produce complementary rather than directly comparable
  headline numbers;
- training overlap may limit the scientific strength of Anny-One results.

### 11.3 Risks and mitigations

| Risk | Mitigation |
|---|---|
| Wrong coordinate convention | Typed conventions, synthetic transforms, projection checks |
| Wrong joint correspondence | Versioned semantic mapping and fail-closed validation |
| High matched-only score with poor detection | Always report false negatives and coverage |
| Training leakage on Anny-One | Conservative in-domain classification gate |
| Post-hoc subset bias | Commit deterministic manifest before inference |
| Restricted data leaks into Git | Ignore raw data, verify tracked files, publish hashes only |
| Optional mesh work expands scope | MuPoTS first; hard gate and explicit joint-only success condition |
| Official and native metrics diverge | Golden cross-check and documented semantic differences |
| Tracking claims exceed annotations | Omit ID metrics unless identity continuity is verified |

---

## 12. Rejected Alternatives

### 12.1 Keep the manually measured corridor as the primary benchmark

Rejected because it provides weaker reproducibility and coverage at higher
collection cost. A personal video may remain a qualitative demonstration.

### 12.2 Convert MuPoTS bodies into Anny parameters

Rejected because MuPoTS is a joint-pose benchmark, not native Anny ground truth.
Fitting Anny would introduce another optimisation model and turn reference data
into pseudo-ground truth. Common semantic joints are the defensible comparison.

### 12.3 Evaluate only Anny-One

Rejected because synthetic images do not establish real-video temporal or
domain performance and may overlap the checkpoint's training distribution.

### 12.4 Evaluate only MuPoTS

Acceptable as the minimum fallback, but not the chosen full plan because it
cannot examine native Anny shape or surface-level compatibility.

### 12.5 Add several more public benchmarks

Rejected for thesis scope. 3DPW, AGORA, and BEDLAM remain future-work options,
not incomplete deliverables.

### 12.6 Build a generic benchmark platform

Rejected as over-engineering. Two explicit, well-tested adapters and one small
normalized contract are sufficient.

---

## 13. Source References

- Multi-HMR 2 official repository and output description:
  <https://github.com/naver/multi-hmr2>
- MuPoTS-3D official project and download page:
  <https://vcai.mpi-inf.mpg.de/projects/SingleShotMultiPerson/>
- MuPoTS-3D paper: Mehta et al., *Single-Shot Multi-Person 3D Pose Estimation
  From Monocular RGB*, 3DV 2018.
- Anny-One official dataset page and release files:
  <https://europe.naverlabs.com/research/vision-foundation-models-for-human-understanding/anny-one/>
- Anny official project: <https://github.com/naver/anny>

Official release files, licences, and evaluator documentation take precedence
over secondary descriptions. A discrepancy discovered during implementation is
recorded before code or metrics are changed.

---

## 14. Definition of Done

This ADR is fully implemented when:

1. the exact licensed dataset releases and configurations are recorded;
2. all 20 MuPoTS sequences have a reproducible result or visible failure;
3. official MuPoTS headline metrics and project-native depth diagnostics are
   generated from preserved observations;
4. the frozen Anny-One subset has a reproducible native-Anny joint/root result;
5. detection coverage accompanies every matched-person accuracy result;
6. coordinate conversions, joint mappings, matching, and aggregation are
   deterministic and unit-tested without external data;
7. final figures regenerate from typed artifacts without inference;
8. no raw licensed dataset or private footage is committed;
9. documentation distinguishes real capture-derived from synthetic evidence;
10. conclusions answer RQ3 without claiming mesh, identity, generalisation, or
    absolute accuracy beyond the measurements actually performed.
