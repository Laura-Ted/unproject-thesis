# Footage: where it comes from, where it is kept, how it is cited

ADR-0006 §5. The repository carries no footage and never will; it carries the
recipe, the digests and the terms for each input, in the `[provenance]` table
of the experiment that uses it. This page is the policy behind those tables.

## The rule

| Question | Answer |
|---|---|
| Is a video ever committed, with or without Git LFS? | No. A repository is not an archive a thesis can cite, LFS has quotas and bandwidth limits, and footage is input, not source. |
| Where is a video kept so a reader can get the same bytes in ten years? | In a **deposit with a DOI** (Zenodo, CERN; free, versioned, permanent) where the licence allows redistribution; otherwise it is **derived from the official source** by a recorded recipe, and the source's digest is pinned. |
| Where is the licence statement kept? | On the source page, **and** in a Wayback Machine snapshot of it (`archived_url`), because pages move. `benchmarks/archive-sources.py` takes the snapshots. |
| How does the thesis cite it? | The deposit's DOI where there is one; otherwise the source URL with the date of access and the archived URL. The live link stays as a courtesy. |
| How does a machine know it has the right bytes? | `expected_sha256` in the experiment, checked by `unproject benchmark-preflight` before a suite runs and recorded by every job as `source_sha256`. |

## The inputs

| Experiment | What | Terms | Kept where | Identity |
|---|---|---|---|---|
| `mot16-09`, `mot16-09-half`, `mot16-09-letterbox`, `mot16-09-letterbox-half` | the matrix suite: one MOT16 sequence at four geometries | as MOT16 below | the same JPEG frames; the three variants are made by the ingest (`[ingest.geometry]`: exact 2×2 box average, black bands to a square), so every machine derives identical bytes and nothing is transcoded or carried | `expected_sha256` of the source directory, the same for all four; the ingest hash tells the variants apart |
| `mot16-09`, `mot16-08` | two sequences of MOT16, MOTChallenge; the medium and high scenes of the people suite | publicly downloadable for research, citation of Milan et al. 2016 requested, no licence text published | **not redistributed**; every machine fetches `MOT16.zip` from motchallenge.net; the archive's digest is pinned | `original_sha256` of the archive; `expected_sha256` of the ordered content of `img1/` as the ingest hashes it |
| `mupots-ts10`, `mupots-ts11` | two outdoor sequences of MuPoTS-3D, the "low" scenes of the people suite (three and two people walking, fixed camera) | MPI for Informatics licence (`license.txt`): non-commercial research only, no redistribution, citation of Mehta et al. 3DV 2018 and 3DV 2017 | **not redistributed**; every machine fetches `MultiPersonTestSet.zip` from the project page once the operator has accepted the terms (`fetch-datasets.sh --accept-mupots-terms`); the archive's digest is pinned; the sequences are extracted into `datasets/mupots-3d/`, the same copy the ground-truth evaluation reads | `original_sha256` of the archive; `expected_sha256` of the ordered content of the sequence directory as the ingest hashes it |
| MuPoTS-3D, all twenty sequences | ADR-0004, the accuracy evaluation | as above | obtained by the author from the project page | annotation inventory hash |
| `pedestrians` (+ `-half`, `-letterbox`, `-letterbox-half`) | the Munich pedestrian street, Xiph.org *derf* via Wikimedia Commons | CC0 1.0; the publisher's readme: "No restrictions / No Copyright" | **not a campaign input** since 2026-09-15 evening: the clip of the RQ3 temporal analysis and of the v2 pilot of that morning; the original `.webm` is on Commons, the four transcodes stay on machine B | `original_sha256` of the `.webm` in `experiments/pedestrians.toml` |
| `crowd` | the v1 pilot's clip | unknown | nowhere; not reproducible, not used by the campaign | - |

Frames of `pedestrians` may be printed in the thesis; frames of
MOT16 and MuPoTS are not, and the figures drawn from them are the model's
reconstructions, not the images. The camera of every MuPoTS sequence was
checked still before TS10 and TS11 were chosen (optical-flow translation of
the background, median 0.01 px per frame pair, drift under 2 px over a whole
sequence, 2026-09-15); the two were preferred among the 1920×1080 sequences
with two or three people for showing people walking, close to the camera
(median box height 34-45 % of the frame) and with nothing else moving.

## The deposit

No campaign input needs a deposit any more: every input is derived from an
official archive by the ingest, identically on every machine. A Zenodo
record, "Unproject thesis: benchmark footage", stays an option for the
pedestrians clip of the RQ3 temporal analysis (`pedestrian_area.webm`, the
transcode the analysis read, `SHA256SUMS`, a `README` naming the licence and
the experiment); its DOI would go into `deposit_doi` of
`experiments/pedestrians.toml` and into the thesis's bibliography.

## Consent

No recording of the author's own is part of the campaign (the `walkers`
clip planned earlier on 2026-09-15 was dropped the same day). Should one
ever be added, it is published only with the written consent of every
person shown - to the recording, to its deposit under CC BY 4.0 and to the
printing of frames in the thesis; the thesis repository holds a template.
Public footage from a source that has already settled the question (a
codec-test collection under CC0; benchmarks that have published their
sequences for years) is used on the source's terms.

## What the digest of a directory is

For an image directory the ingest's `content_sha256` is the SHA-256 over the
files in natural filename order (`unproject.ingest.image_dir`), so a directory
with one frame replaced, added or renamed hashes differently. It is what
`expected_sha256` pins for MOT16 and what `source_sha256` records in every
job: the same identity on every machine, with no transcode in between.
