# 2026-09-15 - Campaign, machine D (the operator's MacBook Pro 16", M4 Pro)

The third Apple Silicon machine, added to the registry the same evening
(`16b2231`) when the operator dropped the Mac mini (B) from the campaign
and put his own 16-inch laptop in its place, so that the Apple pair is two
laptops of the same Pro line one generation apart (A: M3 Pro, D: M4 Pro).
Run over ssh from the operator's Mac mini, in parallel with A (run by a
sibling session from the same Mac mini) and with the rented Linux hosts.
Numbers below are read from the job JSONs and `summary.json` under
`artifacts/campaign/D/`; nothing is estimated.

## The machine

| | |
|---|---|
| Hardware | Apple MacBook Pro 16-inch (2024), Apple M4 Pro, 24 GiB unified memory, 14 CPU cores (10 performance + 4 efficiency), 20 GPU cores, Metal 4; built-in display only |
| Software | **macOS 27.0 (26A428)**, Darwin 27.0.0 arm64 — one major release ahead of A (26.6.2) and of B; installed 2026-09-14 22:40 UTC with a reboot at 22:46 UTC, i.e. the run happened ~22 h after the upgrade; Python 3.12.14 (uv-managed), torch 2.13.0 from the lockfile, MPS available, CUDA not |
| torch threads | 10 (the default: the performance-core count; recorded, not tuned) |
| Commit | **`16b2231`** — `7e7ac5c` (the campaign commit of A and the C-* hosts) plus one additive commit that registers D in `benchmarks/machines.toml` (+7 lines, nothing else); clean tree on every job (`git_dirty: false`). The registry is part of the pre-registration, so the entry had to exist before the machine measured; the diff touches no code path of the runs |
| Where | a **separate checkout** `~/unproject-campaign/Unproject` on the laptop, cloned from the operator's git bundle of `7e7ac5c` and moved to `16b2231` with a second bundle; the operator's own checkout (`~/PycharmProjects/Unproject`, `main` at `3bc3a10`) was read for its layout and for its weights file and never modified |
| Power | mains from 20:25 UTC on (the first inventory at 20:23 UTC found the laptop on battery at 94 %; the operator plugged it in), energy mode `powermode 0` (automatic) on both AC and battery — macOS 27's `pmset -g` has no `lowpowermode` key any more, which is why the preflight reports the low-power state as "not reported"; lid open (`AppleClamshellState = No`), display **kept on** for the whole run by a `UserIsActive` + `PreventUserIdleDisplaySleep` assertion (`caffeinate -u -d`; see the first, aborted launch below for why the opposite choice, made on A, was wrong on this machine) |
| Awake | `caffeinate -u -d -i -s` around `make bench-all` (user active, display awake, no idle sleep, no system sleep; with a utility the assertions last for its lifetime), on top of the Makefile's own `caffeinate -i` per suite |
| Disk | 460 GiB SSD **95 % full**: 24.4 GB free at preflight time, just under the preflight's 25 GB threshold (one `WARN disk` per suite). This is why the inputs travelled as extracted frames and the weights as an APFS clone (below) |

## How the inputs got there

Nothing was fetched from the origin servers and, unlike A, the archives
did not travel either: with 24 GiB free on the laptop, 7.6 GB of archives
to be unzipped once would have pushed the disk to the edge. The fetch
script is idempotent over sequences that are already in place ("the
preflight checks the content digest"), so the four extracted sequences were
`rsync`ed from the Mac mini over the LAN (`-aR --partial` over ssh, 817 MB
in 35 s, 20:30:47–20:31:22 UTC) into the new checkout: `datasets/mot16/
MOT16-09/` (525 frames + `seqinfo.ini`), `MOT16-08/` (625), `datasets/
mupots-3d/MultiPersonTestSet/TS10/` (251) and `TS11/` (701). These are the
bytes the Mac mini's own `fetch-datasets.sh` had extracted from the
digest-verified archives (`MOT16.zip` `b944a7dd…`, `MultiPersonTestSet.zip`
`0f5efe13…`). `make bench-datasets MUPOTS=accept` then found every sequence
present and reported `ok` for each, and the preflight verified every
input's content digest — `6b1238af…` (MOT16-09), `1eca5681…` (MOT16-08),
`46fe005f…` (TS10), `351bddd5…` (TS11): `digest matches` on all, the same
values as on A. The MuPoTS terms were accepted by the operator, who had
accepted them on the Mac mini already.

The weights: the operator's own checkout on the laptop already held
`models/multihmr2.pt`; it was hashed in place (read-only) and found to be
the pinned `fd2d9ab7…`, then cloned into the campaign checkout with
`cp -c` (an APFS clone: an independent copy that costs no space). The
preflight's `--fetch-weights` re-verified the digest. The DINOv3 backbone
was already in the user's `torch.hub` cache; the Anny body-model cache
(`~/.cache/anny`, 1.7 GB) was already built, so no model data was
generated on this machine.

Setup timeline (UTC): bundle + clone 20:29, weights clone 20:29, seeding
20:30:47–20:31:22, `make bench-env` 20:30:49–20:31:19 (30 s: the uv cache
already held every wheel, torch included, from the operator's own checkout;
`torch 2.13.0 cuda False mps True`), registry commit `16b2231` on the Mac
mini 20:33 and checkout on the laptop 20:33, `make bench-datasets` +
`make bench-preflight MACHINE=D` 20:33:50–20:35: **three `ready`, 0
failures**, 4 warnings each — disk (24.4 GB < 25 GB), low-power state not
reported (macOS 27, above), thermal state unrecorded (as on every Apple
machine) and a 1-minute load of 2.5 (`WindowServer` at 25 %: the operator
was still working on the laptop; a `find` at 52 % then 35 % that was gone
before the cpu-suite preflight and was not ours). Forty seconds before the
launch the load read 2.43 → 2.12 with `WindowServer` 20.6 %, `fileproviderd`
7.3 %, Little Snitch 5.9 %, and nothing else above 5 %.

What else was on the machine: the operator's own desktop session (console
login since the post-upgrade boot), Spotlight re-indexing after the major
upgrade (`mds` 37 % and `mds_stores` 2.4 % averaged since boot; not in the
top eight at launch time), `fileproviderd` (iCloud/file-provider sync),
Little Snitch, NotificationCenter; no VM, no Docker, no remote desktop, no
other model. The operator agreed to leave the laptop untouched (mains, lid
open) for the length of the run.

## The first launch, aborted after four jobs

First launch 20:35:26 UTC, `nohup caffeinate -i -s bash -c "make bench-all
MACHINE=D"`, with the display put to sleep right after it (`pmset
displaysleepnow`, as on A). Its first job (`mot16-09-half-mesh`, repetition
2) came back `ok` with an inference median of 0.3541 s and 10 people — but
with a 1-minute load of **4.40 before and 6.44 after** the job, against
2.4 → 2.3 on A. A 5-second `top` at 20:38 UTC explained it: `mediaanalysisd`
at 133 % of a core (63 threads), `hybridsearchd` at 95 %,
`spotlightknowledged` at 13 % — the Photos/Visual-Intelligence analysis and
Spotlight's semantic indexer, discretionary background work that macOS
schedules when the machine is on mains and *unused*, and that a major
upgrade the night before had queued up in quantity. Putting the display to
sleep is precisely the "unused" signal that starts them: the right move on
A (no such backlog) was the wrong one on D. The test that settled it: a
45-second `caffeinate -u -d` (declare the user active, wake the display)
made all three daemons leave the top of the list within 25 s, leaving the
job's `python3.12` (22 %), `WindowServer` (21 %, the animated "Aerials"
wallpaper decoding video, `WallpaperAerials` 7 %) and nothing else above
5 % — the same picture as the pre-launch idle measurement.

So the first launch was stopped at 20:40 UTC (`make` exited 143, four jobs
written: `half-mesh` rep 2, `letterbox-skeleton` rep 0, `letterbox-mesh`
rep 0 and rep 1 — plus a fifth, see below), and its results directory and log were **moved aside,
not deleted**, to `~/unproject-campaign/aborted-D-20260915T2035/` and
`aborted-D-20260915T2035.log` on the laptop, so that the numbers of the
contaminated jobs stay readable next to the clean ones. The `mediaanalysisd`
work is deferred, not lost: it resumes when the assertion is released after
the run.

The same check on A a few minutes later found `mediaanalysisd` at 85 % of a
core with 40 CPU-minutes accumulated under its sleeping display, and 0.0 %
within seconds of `caffeinate -u -d`; A was therefore stopped and relaunched
under the same assertions at 20:45:59 UTC (see the A note). So the two Apple
laptops ran their final suites under identical conditions: mains, lid open,
display on, user declared active.

## The campaign run

Second launch 20:40:31 UTC: `nohup caffeinate -u -d -i -s bash -c "make
bench-all MACHINE=D"`, log at `~/unproject-campaign/campaign-D.log`
(outside the checkout, as the Linux driver does since `7e7ac5c`); `pmset -g
assertions` confirmed `UserIsActive 1`, `PreventUserIdleDisplaySleep 1`,
`PreventSystemSleep 1`. The launch-time preflight of the matrix suite:
`ready`, 0 failures, the same 4 warnings.

One job of the first launch outlived the abort: the `kill` reached
`caffeinate`, `bash` and `make`, but the running suite process
(`unproject benchmark-suite`, not `benchmark-run` as the `pkill` pattern
assumed) finished its job — `letterbox-half-skeleton`, repetition 2, job
004, started 20:39:51 UTC — and wrote its JSON at 20:40 UTC into the
*fresh* results directory that the second launch was about to use (the
launch-time preflight had found that directory "new" a few seconds
earlier). It ran through the 20:40:02–20:40:28 window in which the display
had gone off again (the 45-second test assertion had expired), with an
inference median of 0.5689 s against 0.5742/0.5741/0.5753 s for the three
clean repetitions — indistinguishable — but it is not a job of this run:
the two launches have the same deterministic job order (job 004 is that
cell and repetition in both), which is how it was identified. At 21:02 UTC
it was moved to the aborted launch's directory (which therefore holds five
jobs), and the matrix summary — first written with four jobs in that cell,
240 samples — is regenerated with `make bench-summary MACHINE=D` after the
run, so the summary shipped has three repetitions per cell everywhere.

## Matrix suite (rq1rq2-v2-matrix)

20:40:31–20:57 UTC, 24 clean jobs, all `ok`, `meets_protocol true` on the
eight cells, `git_dirty false` and `mps` on every job; per-job medians
within 1 % of each other in every cell (60 steady frames per job, 180 per
cell). Inference median, MOT16-09, ~10 people:

| cell | median (s) | per job |
|---|---|---|
| skeleton, 1920×1080 | 0.3642 | 0.3632 / 0.3642 / 0.3647 |
| mesh, 1920×1080 | 0.3654 | 0.3639 / 0.3661 / 0.3640 |
| half-skeleton, 960×540 | 0.3615 | 0.3623 / 0.3580 / 0.3639 |
| half-mesh, 960×540 | 0.3639 | 0.3646 / 0.3638 / 0.3630 |
| letterbox-skeleton, 1920×1920 | 0.5743 | 0.5744 / 0.5745 / 0.5740 |
| letterbox-mesh, 1920×1920 | 0.5763 | 0.5759 / 0.5772 / 0.5757 |
| letterbox-half-skeleton, 960×960 | 0.5742 | 0.5742 / 0.5741 / 0.5753 (clean three) |
| letterbox-half-mesh, 960×960 | 0.5749 | 0.5740 / 0.5749 / 0.5749 |

`pairs.csv` (same frames, 180 pairs each): mesh over skeleton +0.007 s at
16:9 and +0.010 s at 1:1 (ratios 1.03); 1:1 over 16:9 **+0.210 s at full
and +0.212 s at half resolution (ratio 1.58–1.59)**, the slower one in
100 % of the pairs; half against full resolution +0.002 s (ratio 1.008,
64 % of pairs) at 16:9 and −0.0003 s (ratio 1.000) at 1:1 — the input
resolution does not move the time, the aspect ratio does, as on the other
machines. Against A's first-launch job of the same cell (`half-mesh`,
0.4944 s) D reads 0.364 s: 1.36× faster.

Loads: 1.6–2.8 before and after every job except a burst at 20:54 UTC
(job 018 `half-skeleton` rep 1: 1.56 → 12.63; the next two jobs started at
11.86 and 6.27). It was not the display or the user (`pmset -g log` has no
event between the relaunch at 20:40:28 and 20:58) and not the analysis
daemons (`mediaanalysisd` and `hybridsearchd` gained 0 s of CPU over a
30-second sample at 21:01, and only the former had been busy earlier). The
burst itself was not sampled; the likeliest source, from a `ps` at 21:00
that showed a dozen `mdworker_shared` processes a few seconds old, is
Spotlight indexing the frames the benchmark itself writes under
`artifacts/` — many short-lived, I/O-bound processes that inflate the
run-queue length far more than they use the CPU. Consistently, the job
that carries the 12.63 was the *fastest* of its cell (0.3580 s against
0.3623 and 0.3639), so whatever it was did not take the GPU.

## People suite (rq2-v2-people)

20:57:58–21:12 UTC, 24 jobs, all `ok`, `meets_protocol true` on the eight
cells, loads between 1.87 and 2.73 before and after every job (no burst),
`mps`, clean tree. Inference median (three jobs of 60 steady frames each):

| cell | people (median) | median (s) | per job |
|---|---|---|---|
| mupots-ts11-skeleton | 2 | 0.3361 | 0.3364 / 0.3365 / 0.3357 |
| mupots-ts11-mesh | 2 | 0.3351 | 0.3340 / 0.3357 / 0.3354 |
| mupots-ts10-skeleton | 3 | 0.3383 | 0.3379 / 0.3391 / 0.3383 |
| mupots-ts10-mesh | 3 | 0.3386 | 0.3397 / 0.3384 / 0.3384 |
| mot16-09-skeleton | 10 | 0.3640 | 0.3645 / 0.3640 / 0.3637 |
| mot16-09-mesh | 10 | 0.3647 | 0.3643 / 0.3657 / 0.3644 |
| mot16-08-skeleton | 23 | 0.4068 | 0.4063 / 0.4061 / 0.4073 |
| mot16-08-mesh | 23 | 0.4069 | 0.4073 / 0.4064 / 0.4071 |

`pairs.csv` (180 pairs each): low → medium count +0.024 s (ratio 1.08),
medium → high +0.045 s (1.12), low → high +0.068 s (ratio 1.20, the
slower one in 99 % of pairs); the two low-count scenes differ by −0.002 s
(0.99, 26 % of pairs slower: no scene effect); mesh over skeleton +0.002 s
at the low count (1.007) and +0.049 s at the high count (1.117) — the
per-person cost of the mesh output shows only where there are people to
pay it for. The MOT16-09 cells repeat the matrix suite's full-resolution
cells to within 0.001 s (0.3640/0.3647 here, 0.3642/0.3654 there).

## CPU suite (rq1-v2-cpu)

21:12:34–21:16:33 UTC, 6 jobs, all `ok`, `meets_protocol true`, `device
cpu` with 10 torch threads. Inference median 0.4857 s (skeleton:
0.4855 / 0.4858 / 0.4859) and 0.4860 s (mesh: 0.4860 / 0.4856 / 0.4865);
mesh over skeleton +0.006 s (ratio 1.014, 99 % of pairs). Loads 5.0–6.8
before and after every job but the first (2.31 before): that is the ten
threads of the job itself, which the validator flags as `load_average_max`
warnings on both cells. **CPU against the accelerator, MOT16-09 skeleton:
1.33** (0.4857 / 0.3642) — the ten performance cores of the M4 Pro are
within a third of its 20-core GPU on this model.

The whole run: `CAMPAIGN-DONE exit=0` at 21:16:33 UTC, 36 minutes after
the relaunch (matrix 17 min, people 14 min, CPU 4 min), 54 jobs, 0 failed.

## Delivery

- `make bench-summary MACHINE=D` re-run at 21:17 UTC after the stray job
  was moved (matrix summary: 24 jobs, three per cell); the "unreadable
  result preflight-…json" notes it prints are the summary reader skipping
  the preflight reports that sit in the suite directory, as on every
  machine. *Addendum by the operator (thesis-c3), 21:45 UTC:* that reader
  was fixed in `a35f96c` and the three `summary.json`/`summary.csv`/
  `pairs.csv` of D were regenerated from the raw jobs on the Mac mini with
  it (`unreadable` empty; every median unchanged, the summary is a
  derivation), then validated again (`validate-campaign.py D --commit
  16b2231…`: valid) and copied to the thesis repository.
- `make bench-collect MACHINE=D` → `D-20260915T211706Z.tar.gz` (200 KB),
  sha256 `090fef36cd56521efb3e1d5a882f7816f407eaec672da6cf18a08cb8317a3fd7`,
  copied to the Mac mini, digest re-computed there (`shasum -c`: OK),
  extracted to `artifacts/campaign/D/benchmark-v2/` (69 files: 54 job
  JSONs, 3 preflights, 3 suite-run records, summary/pairs per suite); the
  campaign log beside it as `D-campaign.log`.
- `python3 benchmarks/validate-campaign.py D --commit 16b2231c0b43…`:
  **`verdict: valid`**, exit 0, every job at `16b2231c0b43`, three
  preflights `ok=True`; warnings only (disk < 25 GB, low-power state not
  reported on macOS 27, thermal unrecorded, the load bursts discussed
  above, the launch-time preflight's load 3.99 with `mediaanalysisd` still
  at 21.8 % as it wound down). Output saved as
  `artifacts/campaign/D-validate-20260915T2118Z.txt`.
- Thesis copy: `TesiTriennale/data/benchmark/D/` = `benchmark-v2/` (byte-
  identical, `diff -rq`), `campaign.log`, `archive.sha256` — the layout of
  the C-* machines. Not committed: `data/benchmark/` is untracked for every
  machine and belongs to the end-of-campaign batch.
- The first, aborted launch: its five job JSONs, three preflights, suite-run
  record and log are in `artifacts/campaign/pilot-D-firstlaunch-20260915T2035Z/`
  (no ingest caches); they are not thesis numbers.
- On the laptop everything is under `~/unproject-campaign/` (checkout,
  `.venv` 1.2 GB, `datasets/` 0.8 GB, `artifacts/campaign/D/` 5.8 GB of
  which almost all is ingest caches, the aborted launch 2 GB with its own
  caches, the logs) — about 10 GB, all of it reclaimable now that the
  archive is verified on the Mac mini. Nothing else on the laptop was
  written to, except the shared uv/torch caches read by `uv sync`. The
  `caffeinate` assertions ended with the run; the deferred Photos/Spotlight
  analysis resumes on the machine's own schedule.

