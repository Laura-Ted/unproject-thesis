# Handoff — running the performance campaign (ADR-0006)

*Written 2026-09-15, evening, for whoever runs the benchmarks. Read
it whole before touching anything. It is the context you do not have.*

## 0. What you are here to do, and to what standard

Produce the **final performance numbers of a bachelor thesis** (Laura
Tedeschini, Unimore: *ricostruzione 3D di più persone da video monoculare su
hardware consumer*). The numbers answer two research questions:

- **RQ1** — can the pipeline run locally on consumer hardware, at what
  latency per frame, throughput and memory? (verdicts, pre-declared:
  *eseguibile* / *streaming interattivo* ≥ 1 estimate/s with median ≤ 1 s /
  *tempo reale* 40 ms per frame);
- **RQ2** — what does the cost depend on: the tensor shape (16:9 vs square),
  the source resolution, the output mode (skeleton vs mesh), and the number
  of people in frame.

They are produced by **protocol v2** (`docs/experiments/protocol-v2.md`,
frozen; amendments dated) with three suites, run **from one commit** on a
set of **declared machines** (`benchmarks/machines.toml`): two Apple Silicon
Macs (A = the author's MacBook Pro M3 Pro, B = the Mac mini M4 you are
sitting on) and rented Linux hosts with consumer NVIDIA cards (C-*). **Linux
first**: the CUDA path is the one number the thesis does not have yet, and
it has never been executed (see §4).

Work as a team of senior engineers would on a measurement that will be
printed and defended:

- **No number that the protocol would refuse.** Red preflight, dirty tree,
  unregistered machine id, another process on the accelerator, a suite
  edited after a job ran: any of these makes the run a pilot, not a result.
  Stop and fix; never "note it and continue".
- **Never fabricate, estimate or fill in.** A missing or failed job is
  reported as such. Every figure the thesis prints traces to a JSON written
  by a job under a commit hash.
- **Never edit a suite, an experiment or the protocol after a job of the
  campaign has run**, on any machine. If a change is unavoidable, every
  machine reruns from the new commit and the old results are discarded, and
  you say so in the lab note.
- **One measurement at a time per machine.** Nothing else on the GPU or
  competing for the CPU while a suite runs. On B: close the VM
  (`com.apple.Virtualization.VirtualMachine`), NoMachine (`nxnode`) and
  Paseo before measuring; today's preflight flagged all three.
- **Money is the user's.** Rented instances cost by the hour; create them
  when you are ready to use them, destroy them as soon as `pull` succeeded,
  never leave one idle. Report spend; ask for a top-up *before* running out.
- **Report faithfully**, at each milestone, in Italian to the user: what
  ran, on what, the commit, the verdicts, what was excluded and why.

## 1. Where things are

| What | Where | State |
|---|---|---|
| Software repository | `~/thesis/Unproject` | branch `adr-0006-campaign`, HEAD = the commit that adds this note (its parent `30bc9a1` added the MuPoTS scenes), tree clean; 792 tests pass (`uv run --frozen pytest packages/pipeline`), `uv run --frozen unproject gates` green. Remote `github.com/Laura-Ted/Unproject` (private); the branch is not pushed and does not need to be — rented instances receive a `git bundle` of HEAD |
| Thesis repository | `~/thesis/TesiTriennale` | LaTeX; ~48 files modified by today's review, **uncommitted on purpose**. You only add results under `data/benchmark/<machine-id>/` (§8); you do not write thesis prose |
| Runbook | `benchmarks/README.md` | the commands, per machine |
| Design and rules | `docs/adr/ADR-0006-multi-machine-campaign.md`, `docs/experiments/protocol-v2.md` (§2, §4.2, §5–§8, §10, §11), `docs/datasets.md` | read before running |
| Rented-host driver | `benchmarks/vast/campaign.py`, `benchmarks/vast/setup-instance.sh` | stdlib Python, wraps the `vastai` CLI (`uvx vastai`) and ssh |
| Machine registry | `benchmarks/machines.toml` | A, B, C-rtx3060, C-rtx4070, C-rtx4090, C-rtx5090, C-rtx5070 |
| vast.ai API key | `~/.vast_api_key` (64 bytes) | the `vastai` CLI reads it from there; never copy it into the repository, a log or a commit |
| Weights | `models/multihmr2.pt` (1.6 GB, sha256 pinned in the experiment config); DINOv3 backbone via `torch.hub` at a pinned commit | fetched by the preflight (`--fetch-weights`) on each machine |

### The suites (all under `experiments/benchmark/v2/`)

| Make target | File | Suite name (= results directory) | Cells × reps | What it answers | ≈ time on B |
|---|---|---|---|---|---|
| `bench-matrix` | `matrix-suite.toml` | `rq1rq2-v2-matrix` | 8 × 3 = 24 jobs | RQ1 + RQ2 tensor/resolution/output on MOT16-09, four geometries made by the ingest | 40 min |
| `bench-people` | `people-suite.toml` | `rq2-v2-people` | 8 × 3 = 24 jobs | RQ2 person count at a fixed tensor: MuPoTS TS10/TS11, MOT16-09, MOT16-08 | 40 min |
| `bench-cpu` | `cpu-suite.toml` | `rq1-v2-cpu` | 2 × 3 = 6 jobs | RQ1 with the model held to the CPU, MOT16-09 | 20 min |

Every job is a fresh process: model load, 5 warm-up frames, 60 measured
frames spread along the clip, then a JSON (`JobResult` schema 2) with
per-frame `inference_seconds`, `rest_body_seconds`, `serialize_seconds`,
`people`, memory counters, the environment (device, card, CUDA/driver,
torch threads), the conditions before/after (power, load, `nvidia-smi`),
the commit and `git_dirty`, and the input digests. `benchmark-suite-summary`
derives `summary.json` per suite; `meets_protocol` and `one_machine` must be
true on every cell.

### The inputs (details and digests: `docs/datasets.md`, `experiments/*.toml`, `TesiTriennale/data/clips/README.md`)

| Scene | Role | Bytes the campaign uses | On this Mac | On a rented host |
|---|---|---|---|---|
| `mot16-09` (+ `-half`, `-letterbox`, `-letterbox-half`: the ingest's geometries of the same frames) | matrix and CPU suites; medium scene of the people suite (≈11 people) | the JPEG frames of MOT16-09, read as a directory; content digest pinned, the same for the four | `datasets/mot16/MOT16-09/img1` | `fetch-datasets.sh` downloads `MOT16.zip` (1.9 GB) from motchallenge.net and verifies it |
| `mot16-08` (≈23 people) | high | the JPEG frames of MOT16-08; content digest pinned | `datasets/mot16/MOT16-08/img1` | same archive |
| `mupots-ts10` (3 people), `mupots-ts11` (2) | low | the JPEG frames of MuPoTS-3D TS10/TS11; content digest pinned | `datasets/mupots-3d/MultiPersonTestSet/TS10`, `TS11` (all 20 sequences are there) | `fetch-datasets.sh --accept-mupots-terms` downloads `MultiPersonTestSet.zip` (5.6 GB) from MPI and verifies it; the flag exists because the licence (research only, no redistribution) is the operator's to accept — the user has accepted it |

Nothing is carried from machine to machine: every input is derived from its
official archive by the ingest, with exact integer arithmetic for the
matrix's half-resolution and letterboxed variants, so the same bytes come
out everywhere. The CC0 `pedestrians` clip is **not** a campaign input
(dropped 2026-09-15 evening, before any measurement); it remains the clip
of the RQ3 temporal analysis in the thesis.

Licences are settled and documented: MOT16 public for research with
citation (no licence text; frames never printed), MuPoTS-3D MPI research
licence (frames never printed). Nothing is redistributed; the repository
carries only recipes and digests.

## 2. The suites are final; nothing is added

Two earlier drafts of the same day had the author's own recording
(`walkers`) as an extra low scene and the CC0 `pedestrians` clip as the
matrix's sequence and a second medium scene; both left the campaign on
2026-09-15 before any measurement (protocol v2 §3 and §4.2, dated notes):
the inputs are the two research datasets only. The suites are the ones
above, on every machine, from one commit. Adding an experiment to a suite
changes its hash and would force every machine to rerun: do not add
anything, and refuse a request to do so once a campaign job has run — say
why.

## 3. Order of work and budget

Priority: **C-rtx4070** (the Linux machine the evidence bundle requires:
`experiments/evidence/thesis.toml` lists its matrix, people and cpu suites
as `require_final`), then **C-rtx3060** (entry tier), then C-rtx4090, then
C-rtx5090 / C-rtx5070 as budget allows; then **B** (this Mac; run it
yourself, machine quiet); then **A** (the author's MacBook: give her the
exact commands, she runs them, you validate the archive she sends).

Budget: the author loaded **$5** on vast.ai. Rough rates (check `search`):
RTX 3060 ≈ $0.10/h, 4070 ≈ $0.20/h, 4090 ≈ $0.35–0.45/h, 5090 ≈ $0.6–0.9/h.
One instance needs ≈ 1–1.5 h: setup 15–25 min (apt, `uv sync` with CUDA
torch ≈ 3–4 GB, weights 1.6 GB, DINOv3, MOT16 1.9 GB, MuPoTS 5.6 GB) plus
the three suites (fast cards do the matrix in ≈ 10 min, people ≈ 15 min;
the CPU suite depends on the host's CPU, 15–40 min). Expect ≈ $0.4 for the
4070, $0.2 for the 3060, $0.7 for the 4090, $1.3 for the 5090. Two things
eat budget silently: **bandwidth** (some hosts bill `inet_down_cost` per GB;
you download ≈ 12 GB — prefer offers with zero or near-zero bandwidth cost,
visible in `vastai search offers` raw JSON as `inet_down_cost`) and **idle
time** (destroy right after `pull`). Tell the user the running spend after
each instance; ask for a top-up when the balance cannot cover the next
instance (`uvx vastai show user --raw` shows `credit`).

## 4. The CUDA path has never run: shakedown first

`docs/notes/2026-09-15-adr0006-campaign.md`: *"The CUDA path has not run
yet: no NVIDIA machine was available today."* The device policy
(`unproject/bodies/device.py`: cuda → mps → cpu, `UNPROJECT_DEVICE`,
`synchronize`, CUDA memory peaks) is tested on Metal and CPU only.
Therefore the first rented instance is a **shakedown**, not a campaign
run, on the cheapest card that satisfies the requirements:

1. `setup` (which ends with the preflight): every check must be `ok`,
   `device` must read `cuda: NVIDIA GeForce RTX …`; `torch.cuda.is_available()`
   True in the `bench-env` line.
2. One real job on the fixture and on one real cell, e.g.
   `uv run --frozen unproject benchmark-suite experiments/benchmark/v2/fixture-suite.toml --machine <id> --artifacts-root artifacts/shakedown`
   then the matrix suite with `--cell mot16-09-skeleton` into
   `artifacts/shakedown`. Open the JSON: `environment.device == "cuda"`,
   `device_name`, `cuda_version`, `driver_version` filled; `conditions.raw`
   has `nvidia-smi` readings before and after; `memory` has CUDA peaks;
   per-frame `inference_seconds` plausible (a 4070 should be well under
   0.2 s per frame at 768×432; if it is ≈ 0.6 s or more, the model is not on
   the card); `git_dirty` false.
3. If anything is wrong, fix it **in the repository** (minimal change, with
   a test where the code is unit-testable), run the pipeline tests and the
   gates, commit, `push` again and repeat. The shakedown results are
   pilots: keep them under `artifacts/shakedown/`, never under
   `artifacts/campaign/`.

Only when the shakedown is clean does the campaign start, on the final
commit, and that commit is then fixed for every machine (write it in the lab
note).

Requirements of a host (the driver's `search` encodes most of them):
driver supporting CUDA 13 (`cuda_max_good ≥ 13.0`, driver ≥ 580 — torch
2.13 from the lockfile ships CUDA 13 libraries), one card, ≥ 60 GB disk
(use `--disk 80`), ≥ 200 Mbit/s down, `verified`, reliability ≥ 0.95. The
image is `nvidia/cuda:13.0.1-base-ubuntu24.04`. A host whose `uv sync` or
preflight fails for driver reasons is destroyed and replaced, not debugged.

## 5. Runbook, one rented machine (from this Mac)

Preconditions, once:

```shell
cd ~/thesis/Unproject
git status --porcelain                 # empty
uvx vastai show user --raw | head      # verified 2026-09-15: the key at ~/.vast_api_key works, credit 5.0
# The account's registered SSH key is ~/.ssh/vastai_lally.pub (ed25519, verified identical);
# ~/.ssh/config has no vast entry, so make ssh offer that private key to the hosts:
ssh-add ~/.ssh/vastai_lally            # or pass -i through an ssh config entry for the host
```

Then, per machine id (example C-rtx4070):

```shell
python3 benchmarks/vast/campaign.py search --gpu RTX_4070 --max-dph 0.35      # add --consumer-cpu to prefer Ryzen/Core hosts
python3 benchmarks/vast/campaign.py create <offer> --machine C-rtx4070 --disk 80
python3 benchmarks/vast/campaign.py wait   <instance>
python3 benchmarks/vast/campaign.py push   <instance>                          # bundle of HEAD + the four clips + setup script
python3 benchmarks/vast/campaign.py setup  <instance> --machine C-rtx4070 --accept-mupots-terms   # apt, uv, repo, env, datasets, preflight
python3 benchmarks/vast/campaign.py run    <instance> --machine C-rtx4070 [--suites matrix,people,cpu]
python3 benchmarks/vast/campaign.py ssh    <instance> -- tail -f /root/unproject-campaign/Unproject/campaign-C-rtx4070.log
python3 benchmarks/vast/campaign.py pull   <instance> --machine C-rtx4070      # archive → artifacts/campaign/C-rtx4070/…
python3 benchmarks/vast/campaign.py destroy <instance>
```

`run` launches `make bench-<suite>` under `nohup`; each target preflights,
runs the suite, writes the summary. The log tells you when it is done
(`benchmark-suite-summary` output for the last suite). Before `pull`, read
the tail of the log for `not ready`, `FAIL`, `Traceback`, `refused`.

After `pull`, validate on this Mac (§7) **before** destroying the instance,
so a broken archive can be re-collected.

Notes on the driver: every remote command is printed before it runs; a
failed step is rerun alone; `push` refuses a dirty tree; `create` labels the
instance `thesis-<id>` and sets `UNPROJECT_MACHINE_ID`; `pull` also fetches
the campaign log. If a host does not support `--direct` ssh, the CLI falls
back to the proxy (slower scp; acceptable).

## 6. Runbook, machine B (this Mac) and machine A

B, after the Linux runs, from the same commit:

```shell
cd ~/thesis/Unproject
git status --porcelain && git rev-parse --short HEAD
# quit the VM, NoMachine, Paseo; mains power; leave the machine alone
make bench-env                             # also after `unproject gates`, which re-syncs without the estimator (§9)
make bench-datasets MUPOTS=accept          # everything is already in place; this only verifies digests
make bench-preflight MACHINE=B             # must end "ready" three times
make bench-all MACHINE=B                   # ≈ 2 h; caffeinate keeps it awake
make bench-collect MACHINE=B
```

A (the author's MacBook Pro M3 Pro, 18 GiB): send her the clean commit
(bundle or push) and the same commands with `MACHINE=A`;
`make bench-datasets MUPOTS=accept` fetches MOT16 and, if her
`datasets/mupots-3d/` copy lacks TS10/TS11 frames, MuPoTS (she accepted the
terms for RQ3 already). Nothing else travels. She sends back the
`bench-collect` archive; you validate it exactly like a Linux one. A's
matrix suite is the release measurement of ADR-0005 (`require_final` in
`experiments/evidence/thesis.toml`); the machine must be on mains, lid open,
nothing else running.

## 7. What "perfect" means: validation of every machine

For each machine id, under `artifacts/campaign/<id>/benchmark-v2/<suite>/`:

- a `preflight-*.json` with `ok: true` and no check with `status: "fail"`
  (warnings are read and judged: load, thermal);
- N job JSONs = cells × 3, all `status: "ok"`, each with 60 `steady` frames
  and 5 `warmup`; `environment.git_commit` identical across every job of
  every machine; `git_dirty: false`; `identity.suite_hash` identical across
  machines for the same suite; `environment.machine_id` = the id;
  `environment.device` = the registry's device (`cuda`, `mps`; `cpu` for
  the CPU suite);
- `summary.json` with every cell `meets_protocol: true` and
  `one_machine: true`;
- `conditions`: `power_source` mains on Macs; Linux `nvidia-smi` readings
  before/after with no other process on the card; load averages recorded;
- the three suites present for C-rtx4070, B and A (cpu suite optional on
  the other Linux cards if budget is short — matrix and people first);
- `uv run --frozen unproject benchmark-campaign-summary artifacts/campaign`
  builds `campaign.json` / `campaign.csv` without refusing anything;
- sanity of the numbers, read, not assumed: the 16:9→square ratio ≈ 1.6–1.8
  on every backend (if a backend shows something else, that *is* a result
  — write it down, do not "fix" it); resolution effect ≈ 0; people suite
  cost rising with the count; CPU suite far slower than the accelerator;
  Apple numbers in the range of the pilots (B, Metal, 16:9 skeleton on a
  scene of ≈10 people: ≈ 0.63–0.72 s per frame; letterboxed ≈ 1.1 s).

Anything that fails a line above is not a thesis number. Say which line.

## 8. Deliverables

1. `artifacts/campaign/<id>/…` for every machine (raw jobs, run records,
   preflight reports, summaries, campaign log), plus the `bench-collect`
   archives with their sha256, kept as delivered;
2. `artifacts/campaign/campaign.json` and `campaign.csv`;
3. one lab note per machine, `docs/notes/2026-09-<dd>-campaign-<id>.md`,
   in the style of `docs/notes/2026-09-15-adr0006-campaign.md`: offer,
   host CPU, card, driver, CUDA, torch, commit, start/end times, spend,
   anything anomalous, anything excluded and why. Commit the notes (nothing
   under `artifacts/` is committed; it is gitignored);
4. in the thesis repository, `data/benchmark/<id>/`: the `summary.json` of
   each suite, the preflight report, the run record and the raw job JSONs
   (they are small), so the thesis can regenerate its tables — the existing
   `TesiTriennale/scripts/export_m4_results.py` reads schema-2 job JSONs
   from `data/benchmark-m4/` and is the model for a per-machine exporter;
   writing that exporter is welcome, writing thesis text is not your task;
5. a final report to the user: the table `campaign.csv` in prose, per
   machine, with the RQ1 verdicts and the RQ2 ratios, and the list of what
   is missing (e.g. machine A pending).

## 9. Known pitfalls, from today

- Background load on B makes numbers 10–15 % slower and the preflight
  warns; the campaign must not run with the VM/NoMachine/Paseo up.
- The preflight refuses a dirty tree: `git status --porcelain` must be
  empty, and `artifacts/`, `datasets/`, `models/` are gitignored, so a run
  leaves the tree clean.
- `ls ~/.ssh` holds many keys; the one registered on the vast account is
  `vastai_lally` (verified), and the driver calls plain `ssh`/`scp`, so the
  key must be in the agent (`ssh-add`) or `wait`/`push` fail at `ssh … true`.
- Wikimedia/archive.org rate limits are irrelevant here; motchallenge.net
  and gvv.mpi-inf.mpg.de serve the archives directly (no login).
- The MuPoTS archive is 5.6 GB per instance; `fetch-datasets.sh` keeps the
  zip under `datasets/_download/` and extracts only TS10/TS11.
- `pmset` does not exist on Linux and `nvidia-smi` does not exist on macOS;
  the job code handles both, the preflight warns on what it cannot read.
- `uv run --frozen unproject gates` runs `uv sync --dev --extra ground-truth`,
  which **removes the `multihmr` extra** from the environment: after the
  gates, `make bench-env` again, or the first job fails with "the bodies
  stage needs its extra". The preflight's `estimator` check catches it.
- The pedestrians-based suites (`pedestrians-suite.toml`,
  `pedestrians-cpu-suite.toml`) are retired records of the morning's pilot;
  the Makefile does not run them. Do not.
- `torch.hub` fetches the DINOv3 code from GitHub on first use; the
  preflight warns if it is not cached, and `setup-instance.sh` runs the
  preflight with `--fetch-weights`, so the first job should find it cached.

## 10. Checklist to hand back

- [ ] shakedown on CUDA clean; commit fixed and written down
- [ ] C-rtx4070: matrix, people, cpu — validated, instance destroyed, spend reported
- [ ] C-rtx3060 (and further cards as budget allows) — same
- [ ] B: three suites, machine quiet — validated
- [ ] A: commands sent to the author; archive validated when it arrives
- [ ] `campaign.csv` built; `data/benchmark/<id>/` in the thesis repository
- [ ] lab notes committed; final report to the user, in Italian
