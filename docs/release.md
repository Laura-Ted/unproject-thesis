# Releasing the thesis

The checklist from the final runs to the immutable tag, in order. ADR-0005
§4 WP7 says when a tag may exist: WP0-WP4 and WP7 complete, the remote
candidate commit green in CI, and, for any conclusion that stays in the
thesis, WP5 and WP6 as well. §7 lists what has to be true; each line below
names how it is checked, not only that it is.

Every step is the author's to run on the protocol machine (M3 Pro, 18 GB),
with the dataset and the footage in place; the other machines of the campaign
(ADR-0006) run the performance suites only, and their archives are unpacked
under `artifacts/campaign/` before the bundle is built. Nothing below edits a number: a
result that comes out different from the pilot is written down with its raw
data, per ADR-0005 §6, and the thesis text follows it.

## 0. The candidate commit

```shell
git switch adr-0005 && git pull --ff-only
git status --porcelain            # must print nothing
gh pr checks 1                    # or the Actions page: every job green on HEAD
uv sync --extra multihmr --extra ground-truth
pnpm install --frozen-lockfile
```

The pilot numbers that are still in `README.md` are labelled as such and stay
until the final ones replace them (step 5).

## 1. Inputs

| Input | Where | How it is pinned |
| --- | --- | --- |
| Multi-HMR 2 checkpoint | `models/multihmr2.pt`, fetched on first use | sha256 in every `[bodies]` config |
| DINOv3 source | `torch.hub` cache | `backbone_ref` commit in the config |
| Pedestrians clip and its three transformations | `datasets/pedestrians*/clip.mp4` | `docs/reproduce.md` §2 and §4; the source digest is in `experiments/pedestrians.toml` and in every artifact manifest |
| MuPoTS-3D | `datasets/mupots-3d/` | `docs/reproduce.md` §7; never fetched by the code, never committed |

`experiments/evidence/thesis.toml` lists the first five under `inputs`; the
bundle indexes them by hash and carries none of them.

## 2. The final runs

All on mains, low-power mode off, nothing else on the GPU, the lid open
(protocol v2 §2 and the live protocol record the conditions per job and say
so if they changed).

```shell
# RQ1/RQ2, protocol v2, as machine A of the campaign (ADR-0006, benchmarks/README.md):
# the matrix (24 processes; 33 minutes in the M4 pilot), the person-count suite
# and the CPU point, each preflighted, under artifacts/campaign/A/
make bench-datasets FROM=<dir with the campaign clips>   # digests pinned in experiments/
make bench-all MACHINE=A

# The live path, docs/experiments/live-protocol.md: the transport client, then the browser
uv run unproject-live-benchmark datasets/pedestrians/clip.mp4 --fps 1 --spawn
uv run unproject-live-benchmark datasets/pedestrians/clip.mp4 --fps 8 --spawn
uv run unproject-live-benchmark datasets/pedestrians/clip.mp4 --fps 1 --bodies --spawn
uv run unproject-api            # then, in the browser, ?measure=1&replay=datasets/pedestrians/clip.mp4&fps=1 and fps=8
uv run unproject live-summary

# MuPoTS-3D under the Unproject protocol: docs/notes/2026-09-15-adr0005-wp4-final-evidence.md, steps 1-3
uv run unproject gt-evaluate experiments/ground-truth/mupots-3d.toml --final --release-frames
uv run unproject gt-evaluate experiments/ground-truth/mupots-3d-annotation-derived-camera.toml --final --release-frames
uv run unproject gt-evaluate experiments/ground-truth/mupots-3d-letterbox-square.toml --final --release-frames
uv run unproject gt-hypotheses artifacts/ground-truth/<id-control>
uv run unproject gt-hypotheses artifacts/ground-truth/<id-treatment>
uv run unproject gt-compare artifacts/ground-truth/<id-control> artifacts/ground-truth/<id-treatment> \
    --label-a model_estimated --label-b annotation_derived
uv run unproject gt-compare artifacts/ground-truth/<id-treatment> artifacts/ground-truth/<id-letterbox> \
    --label-a wide --label-b letterboxed

# The filter, deterministic; and the run the temporal figures are drawn from
uv run unproject filter-characterise
uv run unproject bodies experiments/pedestrians.toml
uv run unproject rq3 artifacts/pedestrians/bodies/<id-run> experiments/rq3/pedestrians-uncontrolled.toml
```

What each must show before it counts:

- every `suite-run.json` and job records the candidate commit and
  `git_dirty: false`; a job whose thermal limit changed is reported with that
  fact, not dropped;
- every ground-truth `evaluation.json` records `final: true`; `gt-summary`
  prints no "exploratory";
- the two `gt-compare` of WP4 step 1 (the schema-1 evaluations against their
  rescoring) printed `numbers: identical`, or the difference is written down
  as a re-inference difference in a dated note;
- `gt-hypotheses` was read by the pre-registered rule in
  `docs/experiments/ground-truth-protocol.md` §4.6, and anything added
  afterwards is labelled post-hoc.

Then name the results: replace every `<id-...>` in
`experiments/evidence/thesis.toml` and delete `optional = true` from each
result that ran, so that the build refuses an absence instead of recording
it. Commit that alone: `release: name the final results`.

## 3. Gates and bundle

From the commit of step 2, tree clean:

```shell
uv run unproject gates                                            # ends with "green"
uv run unproject evidence-bundle experiments/evidence/thesis.toml evidence/thesis \
    --gates artifacts/gates.json
uv run unproject evidence-verify evidence/thesis
```

`evidence-bundle` must print every result as `collected`, no `(tree dirty)`
after the commit, and no `gates: not recorded`. Where the spec says
`require_final` it refuses an evaluation that is not final, a suite with a
job from a dirty tree or from a second commit, and a bodies run whose
manifest does not record a clean tree; everywhere it refuses a file over
2 MB. `observations.npz`, the bodies fitted to dataset frames, footage and
weights are indexed by SHA-256 and stay local. The thesis's tables and figures
are not drawn here: the scripts of the thesis repository regenerate them from
stored results, without inference, and record the hash of every input they read.

```shell
git add evidence/thesis
git commit -m "release: evidence bundle for <commit of step 2>"
git push
gh pr checks 1                                                    # green on the bundle commit
```

`release.json` names the commit that produced the results - the one of step
2, this commit's parent - and that is right: it identifies the code that ran.
The tag goes on the commit that carries the bundle.

## 4. The tag

```shell
git tag -a thesis-v1 -m "Unproject, thesis evidence release" <bundle commit>
git push origin thesis-v1
```

Immutable: a correction after this is `thesis-v1.1` on a new commit, never
the same tag moved. A GitHub release may point at it and attach nothing that
is not in the bundle.

## 5. The text

- `README.md`, *What is evidence and what is not*: every `pilot` row whose
  final run exists becomes `final`, naming the bundle label; the pilot tables
  under *Stage [B]* and *Evaluating against a public ground truth* are replaced
  by the final numbers or kept only as labelled history.
- The thesis takes its numbers and figures from the scripts of its own
  repository, which read the stored results and record the hash of each, and
  its conclusions on the camera, the scale, the aspect ratio and the smoothing
  follow ADR-0005 §3: hypotheses are called hypotheses, smoothing is
  smoothness, and nothing is "official".

## What is checked, and where

| ADR-0005 §7 | How |
| --- | --- |
| 1. CI green on the remote candidate | `gh pr checks 1`, on the bundle commit |
| 2. A fresh install runs every gate | `README.md` *The minimal path*: `git clone`, `uv run unproject gates` |
| 3. No "official" metric, no "released" camera | the grep below prints only the vocabulary table of the protocol (§7a) and the config comment that forbids the words |
| 4. Benchmark from tracked public input, fresh processes, raw samples | `evidence/thesis/rq1rq2-v2/*-suite-run.json` and the per-job files, every frame in them; `release.json` `identity` (`git_dirty: false`, one commit, the clip's digest) |
| 5. Inference and live measured apart | `evidence/thesis/live/` beside `rq1rq2-v2/`, definitions in the two protocols |
| 6. `git_dirty=false` on the final evaluations | `final: true` in each `evaluation.json`, refused otherwise by `--final` and by the bundle |
| 7. Every number traces to artifact, config, commit, hash | `release.json`; the provenance manifests the thesis scripts write beside their tables |
| 8. Figures without inference from bundle data | the thesis repository's scripts: no inference, stored results only (not the bundle), every input hashed |
| 9. §3 rules on camera, scale, aspect ratio, smoothing | step 5; the reading rule in `ground-truth-protocol.md` §4.6 |
| 10. README and reproduction guide describe the real state | the state table, `docs/reproduce.md` |
| 11. No restricted or local content in git | `git status` clean; `find evidence/thesis -size +2M` prints nothing; no `.npz`, `.mp4`, `.pt`, `.mat` under `evidence/` |
| 12. An immutable release | `git ls-remote --tags origin thesis-v1` names it; `git show thesis-v1` is the bundle commit |

```shell
git grep -n -i -E "official (3dpck|score)|released camera|true camera" -- ':!docs/notes' ':!docs/adr'
```

## If something is red

Stop and write it down (ADR-0005 §6): which gate, which result, which
condition. A threshold, a subset or an exclusion rule is not changed after a
result has been seen; an analysis added now is labelled post-hoc; a number
that moved against the pilot is reported with the raw data that says so.
