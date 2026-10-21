# 2026-09-15 - WP7, the evidence a reader can hold without the data

WP7 of ADR-0005. Every number in the thesis comes from an artifact, and most
artifacts cannot be committed: the footage is not ours, MuPoTS-3D forbids
redistribution, the weights are 1.6 GB, `observations.npz` is derived from
restricted frames. What a reader can be given is small - summaries, tables,
figures, descriptors, the state of the gates, and a hash for everything that
stayed behind - and this commit is the machinery that produces exactly that
and nothing else, plus the README and the release checklist the ADR asks
for.

## The bundle

`experiments/evidence/thesis.toml` names each final result by kind, path and
the exact command that produced it. `unproject evidence-bundle` reads it and
writes `evidence/thesis/`:

- per result, only the files its kind allows - `summary.json`, the CSVs, the
  SVGs, the descriptor, the hypotheses, the suite run record and every job
  file - none over 2 MB, with a refusal rather than a skip past that;
- for a bodies run, not the run but two derived views the temporal figures
  need: the longest track's pelvis depth raw and filtered by the analysis's
  own smoother, and the middle frame's skeletons with their camera;
- `release.json`: commit, dirty state, whether the gates were green, and per
  result its identity (evaluation id, protocol, matcher, camera source,
  frame transform, mapping and inventory hashes, checkpoint digest, coverage;
  or suite hash, commits, machines, source digests), every carried file with
  its SHA-256, and every local-only file with its SHA-256 and the reason it
  stayed local;
- `gates.json`, when passed or run: exit code, duration and output tail per
  gate.

`require_final` is what turns a result into a thesis result, and the build
refuses instead of noting: an evaluation without `final: true` or scored from
a dirty tree or over bodies from one; a suite with a job from a dirty tree or
from a second commit; a bodies run whose manifest does not record a clean
tree. `optional` lets a bundle be assembled before every experiment has run
and say which are absent; at release it is deleted from each result that ran.
`unproject evidence-verify` re-hashes a bundle on disk against its manifest
and names every file that moved, appeared or vanished.

## One gate list

`unproject.gates.GATES` is the fifteen commands CI runs, in CI's order, and a
test parses `.github/workflows/ci.yml` and fails if the two ever differ.
`unproject gates` runs them from the repository root without stopping at the
first failure, prints one line per gate and `green` or `RED`, and writes the
record the bundle carries.

## Figures from the bundle only

`docs/thesis-figures/generate.py` no longer opens `artifacts/`: it reads
`--bundle`, defaulting to `evidence/thesis`, draws a figure only when every
input it needs is there, names the ones it skips, and writes
`provenienza.json` with the files each figure was drawn from, their SHA-256
taken from `release.json`, the commit and the state of the gates. The
committed figures are still the author's v1 pilot and say so in the
directory's README; they are regenerated from the final bundle at release.

## README and release

The README now opens with the thesis - bodies, viewer, measurement,
evaluation - and keeps stages [1]-[5] as *Scene reconstruction, the
preliminary work* at the end. It states what is downloaded when (wheels at
sync, the checkpoint and the DINOv3 source at first use, footage and the
dataset only by hand) and that nothing is sent during inference; the
hand-written "twenty seconds, then four" model-load time is gone, replaced
by the name of the field that measures it. A table under *Status* puts every
result in one of four states - implemented, pilot, final, optional - and
says what would move it to final. *Checks* ends with the gate as one
command, the bundle commands, and the minimal path from a fresh clone.
`docs/release.md` is the ordered checklist from the final runs to the tag,
with ADR-0005 §7's twelve criteria each mapped to how it is checked.

## Not done here, and why

No bundle is committed: `evidence/thesis` is built from the final runs,
which need the M3 Pro, the dataset and a clean commit (`docs/release.md`).
No tag: ADR-0005 makes it conditional on WP0-WP4 and WP7 complete *and* the
final runs, which are the author's. The v2 pilot on this machine is written
up in `2026-09-15-adr0005-v2-pilot-on-m4.md` and is not carried anywhere.

## Verified

18 tests in `test_evidence.py`: the gate list against the workflow, the
shipped spec, a gate run recording its exits; allowed files carried and the
rest indexed, a non-final evaluation refused, a dirty or two-commit suite
refused, a dirty or unrecorded run refused, a missing optional result
recorded and a missing required one naming its command, the size limit, the
inputs indexed and never copied, the gates record carried, a rebuild
replacing the bundle whole, the hypotheses carried, verification naming a
moved hash and a missing manifest, the release reading back, the rq3-run
derivations; two in `test_cli.py` for the evidence spec under
`validate-config`. `unproject gates` was run from a fresh `git clone` of the
commit (README, *The minimal path*) and came back RED the first time: the
`validate-config experiments/` gate walks every TOML under `experiments/`
and read the new `experiments/evidence/thesis.toml` as a broken experiment.
That is the runner doing its job from a checkout that had never seen the
spec; `validate-config` now knows the seventh kind, and the second run from a
second fresh clone was green in 1 min 26 s. Full gate from the clean
environment as well.
