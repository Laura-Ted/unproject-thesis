# ADR-0003 — Per-Person Inspection, and Not Re-Identification

**Status:** Accepted
**Date:** 2026-09-06
**Accepted:** 2026-09-08
**Type:** Feature and scope decision
**Audience:** Thesis author, supervisors, and contributors
**Extends:** ADR-0002. Nothing in ADR-0002 is superseded; this adds one viewer
capability and closes one direction with evidence.

> **Note for contributors:** this ADR authorises a viewer feature and
> explicitly forbids an adjacent one. The forbidden part is the one that looks
> more impressive. Read §4 before proposing anything that links two tracks
> together.

---

## 1. Decision Summary

Add a **per-person inspection layer**: every track in a finished run gets a
stable identifier, the viewer lists them, and the operator can show, hide and
isolate individuals, seeing one person's path and statistics on their own.

Do **not** add re-identification across occlusions, and do not interpolate a
body through a gap. Both are deferred, with the conditions for reopening them
stated in §5.

The raw artifact does not change. This is a reading and presentation layer over
what stage [B] already wrote.

---

## 2. Context

### 2.1 What prompted it

Watching a finished run, the operator observed that bodies are lost when
somebody passes behind another person or an object, and that a body does not
follow one person across the clip. That observation is correct, and it was
already measured under ADR-0002 WP5:

| Run | Clip | Most people at once | Distinct tracks | Median track life |
|---|---|---|---|---|
| `crowd` | 19.1 s | 20 | 62 | 4.17 s |
| `pedestrians` | 15.0 s | 17 | 73 | 2.12 s |

An identity survives roughly a fifth of the clip. One person becomes three or
four tracks. The reconstruction is placing bodies correctly — reprojection is
1.0 over 2753 bodies fully in frame on `pedestrians` — and then losing track of
whose body it is.

### 2.2 Why the obvious fix was measured before it was built

The natural response is: we know what a person looks like, so match them when
they come back. Two signals are available without adding a model, and both were
measured on the 73 tracks of `pedestrians` before any matcher was designed.

| Signal | Spread within one track | Spread between tracks | Ratio |
|---|---|---|---|
| Body shape, the 11 estimated parameters | median 0.159, p90 0.326 | median 0.208, p10 0.118 | **1.31×** |
| Torso colour, HSV histogram of the box | median 0.536, p90 0.838 | median 0.739, p10 0.534 | **1.38×** |

Neither separates people. The within-track p90 exceeds the between-track median
in both cases: the same person, seen twice, frequently looks *less* like
themselves than two different people look like each other.

A matcher built on either would work in a demonstration and link the wrong
people in the data. It would not fail loudly; it would produce a confident,
wrong answer, which is the failure mode ADR-0002 §3.3 exists to prevent.

Motion continuity does carry structure — 61 pairs where one track ends and
another begins within 3 s and 3 m, across 31 of the 73 tracks — so there is
something to join. What is missing is any way to know whether a join is correct.

### 2.3 The blocking problem

**No ground truth for identity exists.** Nobody has annotated who is who in any
clip in this repository. Without that, a re-identification algorithm cannot be
evaluated, only admired. A thesis whose contribution is measurement cannot ship
a component whose accuracy is unmeasured, and ADR-0002 §8.5 already says track
fragments are reported rather than joined "without a record".

---

## 3. Decision

### 3.1 Identity

A **person id** is a track id from the artifact, scoped to one run. It is
metadata for inspection and nothing else.

- It does not identify a real person. ADR-0002 §3.3 forbids that claim and this
  ADR does not weaken it.
- It is not stable across runs, across clips, or across re-runs of the same
  clip with a different configuration.
- No signature, template or embedding of a person is computed, stored, or
  carried between runs.

### 3.2 What the viewer gains

- A list of the people in the current run, each with its id, its colour, when it
  first and last appears, and how many frames it holds.
- Per-person visibility: show, hide, isolate. Hiding is a viewer state; it
  changes nothing on disk.
- For a selected person: their path over time, and the statistics RQ3 already
  computes for them — apparent velocity by axis, depth jitter as a fraction of
  range, shape variation.
- The list makes fragmentation visible. When one person is three tracks, the
  operator sees three entries whose lifetimes abut, and that is the finding
  rather than a defect to hide.

### 3.3 What the pipeline gains

Per-person statistics are computed by the existing RQ3 analysis and exposed
through the API. No new inference, no new model, no new runtime dependency.

### 3.4 Constraints carried over from ADR-0002

- The raw artifact is not modified, and no derived identity is written into it
  (§7.6, raw observations are immutable).
- Only the latest user action owns visible state (§WP3).
- The render loop stays independent of inference cadence (§10.2).
- Runtime JSON is parsed, not asserted (§10.2).
- Selection state belongs to the viewer and is never a wire concept the server
  has to remember.

---

## 4. Explicitly Out of Scope

These are forbidden under this ADR. Implementing any of them requires a new ADR
and the evidence named in §5.

1. **Re-identification across occlusions.** Linking two tracks as the same
   person, by any signal, automatic or heuristic.
2. **Interpolating a body through a gap.** A frame where nobody was found stays
   empty. ADR-0002 §8.5 permits interpolation for visual playback only, and this
   ADR does not exercise that permission: a drawn body that was never estimated
   is indistinguishable from one that was, and the viewer is where the thesis
   figures come from.
3. **Appearance sampling onto the mesh.** Colouring a body from the person's own
   pixels was requested and is deliberately excluded here. It measured 1.38× as
   a matching signal, so it does not help the machine; it helps a human
   recognise individuals, which moves the project toward the biometric framing
   ADR-0002 §3.2 rules out. If it is wanted for a demonstration, it needs its
   own decision, not this one.
4. **Any persisted per-person descriptor.** Nothing that could be carried from
   one recording to another.

---

## 5. Conditions For Reopening Re-Identification

Re-identification becomes admissible when all of the following hold:

1. **Ground truth exists.** A typed annotation file records, for at least one
   clip, which track ids are the same person — enough cases to compute a rate,
   not a handful of illustrations. This extends the RQ3 annotation format and is
   hand work.
2. **The metric is declared before the matcher is written.** Correct links,
   wrong links, and missed links, reported separately. A single accuracy figure
   hides the one that matters: a wrong link is worse than a missed one, because
   it fabricates a trajectory.
3. **The signal is measured first.** As in §2.2. Shape and naive colour are
   already ruled out; anything else is measured before it is built on.
4. **The privacy position is restated.** A learned re-identification model is a
   soft-biometric component. Whether that belongs in a thesis whose argument is
   local, privacy-preserving processing is a question to answer in writing, not
   by merging code.

Until then, the honest presentation of a fragmented track is a fragmented track.

---

## 6. Consequences

### Positive

- The operator can finally inspect one person instead of a crowd, which is what
  makes the depth behaviour legible at all.
- Track fragmentation stops being a number in a JSON file and becomes something
  visible, which strengthens the RQ3 argument rather than papering over it.
- No new model, dependency, weights or inference cost.
- The per-person statistics already exist; this exposes them.

### Negative and accepted

- The viewer will show, plainly, that identities break every few seconds. That
  is the truth about the system and it will look worse than a version that
  silently joined tracks.
- The requested occlusion recovery is not delivered. The reason is recorded
  above and is evidence, not reluctance.
- Selection is per-run and does not survive a reload beyond whatever the viewer
  chooses to keep locally.

### Risks

- **Scope creep toward re-identification.** A per-person list invites "and now
  merge these two". §4 exists because that invitation will be felt.
- **Misreading an id as a person.** Mitigated by naming: the viewer says
  "track", never "person 12", and the documentation repeats that a track id is
  not a person.

---

## 7. References

- ADR-0002 §3.2, §3.3 — what the project is not, and what it must not claim.
- ADR-0002 §7.6 — raw observations are immutable.
- ADR-0002 §8.5 — missing data and post-processing.
- `docs/notes/2026-09-05-wp5-rq3-metrics.md` — where the fragmentation numbers
  come from.
- The measurements in §2.2 were taken on
  `artifacts/pedestrians/bodies/4f4f09628526`, whose footage is CC0 and
  reproducible from `experiments/pedestrians.toml`.
