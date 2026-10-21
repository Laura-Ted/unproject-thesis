# 2026-09-15 - Campaign, machine A (the author's MacBook Pro 14", M3 Pro)

The protocol machine of ADR-0005, run over ssh from the operator's Mac mini
(machine B) on the same evening as the rented Linux hosts. Numbers below are
read from the job JSONs and `summary.json` under `artifacts/campaign/A/`;
nothing is estimated.

## The machine

| | |
|---|---|
| Hardware | Apple MacBook Pro 14-inch (2023), Apple M3 Pro, 18 GiB unified memory (`memory_total_bytes` 19327352832), `cpu_count` 11 |
| Software | macOS 26.6.2 (25G83), Darwin 25.6.0 arm64, Python 3.12.14 (uv-managed), torch 2.13.0 from the lockfile, MPS available, CUDA not |
| torch threads | 5 (the default; recorded, not tuned) |
| Commit | **`7e7ac5c`** — the campaign commit, clean tree on every job (`git_dirty: false`) |
| Where | a **separate checkout** `~/unproject-campaign/Unproject` on the machine, cloned from the operator's git bundle of `7e7ac5c`; the author's own checkout (`~/PycharmProjects/Unproject`, `main` at `fbc810f`) was read (for nothing but its layout) and never touched |
| Power | mains, battery 100 %, `lowpowermode 0`, lid open (`AppleClamshellState = No`) |
| Awake, and "in use" | the campaign run is wrapped in `caffeinate -u -d -i -s` (user-activity and display-on assertions, no idle or system sleep) on top of the Makefile's own `caffeinate -i` per suite — see "The first launch" below for why the display must stay on |

## How the inputs got there

Nothing was fetched from the origin servers: the two archives and the
weights travelled from machine B over the LAN (measured 106 MB/s with a
200 MB probe; `rsync --partial` over ssh, 100 s for 9.2 GB) into the new
checkout's `datasets/_download/` and `models/`. Then the runbook's own
`make bench-datasets MUPOTS=accept` verified both archives against the
pinned digests — `MOT16.zip` `b944a7dd…`, `MultiPersonTestSet.zip`
`0f5efe13…`, the same bytes as on every other machine — and extracted
MOT16-09 (525 frames), MOT16-08 (625), MuPoTS TS10 (251) and TS11 (701).
The preflight then verified every input's content digest (`6b1238af…`,
`1eca5681…`, `46fe005f…`, `351bddd5…`: `digest matches` on all), the
weights (`fd2d9ab7…`) and the DINOv3 backbone already in the user's
`torch.hub` cache. The MuPoTS terms were accepted by the operator on the
author's behalf: she had accepted them for RQ3 already (the whole test set
sits extracted in her checkout).

Setup timeline (UTC): bundle + clone 20:09, `make bench-env` 20:11–20:13
(uv sync from the lockfile, `torch 2.13.0 cuda False mps True`), seeding
20:11–20:13, `make bench-datasets` 20:13:27–20:13:49, `make bench-preflight
MACHINE=A` 20:13:57–20:14:20: **three `ready`, 0 failures**, 2 warnings each
— the thermal state `pmset` cannot report on Apple Silicon (as on B) and a
1-minute load of 4.1–4.4, which was the preflight's own hashing of 1.3 GB
of frames on the heels of the unzip and the uv sync; a minute later the
load read 1.66 with nothing above 17 % of a core (`WindowServer`, display
still on), and the launch-time preflight of the matrix suite reported one
warning only (thermal).

What else was on the machine: NoMachine's server (`nxserver`, `nxnode`,
`nxd`) with **no session connected** (0 established connections on port
4000, 0.0 % CPU), Little Snitch, the Docker privileged helper without
Docker Desktop, Safari and Teams background agents; no VM, no remote
desktop, no other model. The console user had been logged in since boot
(17:05 UTC) with nothing in the foreground.

## The first launch, kept as a pilot

The first launch (20:16:52 UTC, `nohup caffeinate -i -s make bench-all
MACHINE=A`) put the display to sleep with `pmset displaysleepnow` on the
reasoning that it would sleep by itself after ten minutes anyway. Two things
then happened that the second Apple machine (D, run in parallel by another
operator session) found first and A confirmed:

1. **a Mac with its display off runs Photos' analysis**: `mediaanalysisd`
   (`MediaAnalysis.framework`) rose to 85 % of a core — it also uses the
   Neural Engine — and had 40 min of CPU time by 20:44 UTC; with a
   user-activity assertion (`caffeinate -u -d`, `pmset -g assertions` →
   `UserIsActive 1`) it dropped to 0.0 % within seconds and stayed there.
   The laptop's own `hybridsearchd` (Spotlight) did the same on D;
2. `pmset -g log` shows `Display is turned on` at 20:18:30 UTC and off again
   at 20:34:24 — someone touched the laptop during the matrix suite.

The matrix suite of that launch completed (24 jobs, all `ok`, 20:16–20:40
UTC) and the people suite was stopped at its fourth job at 20:43:5x
(SIGTERM to the outer `make` and to the suite's `uv run`; the job in flight,
`mupots-ts11-skeleton` rep 0, finished on its own and wrote its record at
20:44:30, `ok`). Everything from that launch — 24 matrix jobs, 4 people jobs,
the campaign log and the preflight reports — is kept **outside the results
root**, in `~/unproject-campaign/pilot-A-firstlaunch-20260915T2016Z/` on the
machine (copied to the operator's Mac beside the archive), as a pilot: not a
campaign result, but evidence. Its matrix numbers, for the record (median
over 180 steady frames, s): 16:9 cells 0.492–0.501, 1:1 cells 0.836–0.843;
16:9 → 1:1 ratio 1.68 (full) and 1.71 (half), b slower on 100 % of frames;
resolution 1.01 / 1.00; mesh − skeleton 1.04 / 1.03; per-job dispersion
within 3 %.

## The campaign run

Relaunched at 20:45:59 UTC after a fresh results root: `nohup caffeinate -u
-d -i -s bash -c "make bench-all MACHINE=A"`, display on, user-activity
asserted for the whole run, `mediaanalysisd` at 0.0 % (checked every minute
by the operator's monitor), log at `~/unproject-campaign/campaign-A.log`
(outside the checkout, as the Linux driver does since `7e7ac5c`). These are
the conditions D runs under too. Launch preflight: `ready`, 0 failures, 2
warnings — thermal (unrecorded, as always on Apple Silicon) and
`WindowServer` at 46.6 % in the very second the display came back on (0.6 %
a minute later, load 0.84).

Timeline (UTC): matrix 20:46–21:08, people 21:08–21:26, CPU 21:26–21:33
(54 jobs, every one `status: ok`, 5 warm-up + 60 steady frames, `git_dirty:
false`, `git_commit: 7e7ac5c…`, `machine_id: A`, `device: mps` / `cpu`,
`device_name: Apple M3 Pro`, suite hashes as on the operator's Mac).
`BENCH-ALL-EXIT=0` at 21:33:24; `make bench-collect MACHINE=A` at 21:34:30.

Inference per frame, median over 180 steady frames (3 jobs × 60), seconds;
throughput is the median of the three per-job figures:

| Suite / cell | skeleton | mesh | fps (skeleton) | people (median) |
|---|---|---|---|---|
| matrix `mot16-09` 1920×1080 → 768×432 | 0.5039 | 0.5039 | 1.97 | 10 |
| matrix `mot16-09-half` 960×540 → 768×432 | 0.5019 | 0.5025 | 1.99 | 10 |
| matrix `mot16-09-letterbox` 1920×1920 → 768×768 | 0.8408 | 0.8416 | 1.18 | 9.5 |
| matrix `mot16-09-letterbox-half` 960×960 → 768×768 | 0.8375 | 0.8381 | 1.19 | 10 |
| people `mupots-ts11` | 0.4638 | 0.4644 | 2.14 | 2 |
| people `mupots-ts10` | 0.4678 | 0.4692 | 2.13 | 3 |
| people `mot16-09` | 0.5033 | 0.5042 | 1.98 | 10 |
| people `mot16-08` | 0.5641 | 0.5638 | 1.74 | 23 |
| CPU `mot16-09` (Apple M3 Pro, 5 threads) | 0.9689 | 1.0020 | 1.03 | 10 |

Repetitions agree to within 0.1–0.5 % in every accelerated cell (the pilot
launch: 0.6–3 %), 0.4–1.6 % on the CPU.

Paired readings (b / a on the same 180 frames): tensor 16:9 → 1:1 **1.668**
at full resolution and **1.668** at half, b slower on 100 % of frames — in
the 1.6–1.8 band of the Apple pilots and above the CUDA machines (1.49 on
the 3060, 1.34 on the 5070, 1.09 on the 4090), the backend property ADR-0006
§6 anticipated; resolution 1920 vs 960: 1.004 at either aspect (b slower on
71–72 % of frames: no effect); output mesh − skeleton 1.023 at 16:9 and at
1:1 (the rest body, paid per new track: p75 of the paired total +12 ms).
People suite (different clips, `same_frames: false`, the reading is the
paired total per frame): low → medium +34.3 ms (1.076, b slower on 100 %),
medium → high +62.0 ms (1.122, 99 %), low → high **+96.3 ms (1.207)**, the
two low scenes −3.4 ms apart (0.991: the count, not the scene); mesh −
skeleton +54.9 ms median at 23 people and +2.6 ms at 2. CPU against MPS on
the same cell: 0.969 / 0.504 = **1.92×** — the M3 Pro's five performance
cores are within a factor of two of its GPU on this model, where the rented
hosts' CPUs sit at 18–29× their cards.

RQ1 verdicts on this machine, pre-declared thresholds: *eseguibile* and
*streaming interattivo* (median ≤ 1 s, > 1 estimate/s) in every accelerated
cell — the 1:1 cells at 0.84 s and 1.18 fps are the closest to the line —
and, on the CPU, the skeleton cell 0.969 s / 1.03 fps just inside it while
the mesh cell at 1.002 s / 1.00 fps sits on the threshold (read with §7: the
number is reported, the verdict is the reader's); not *tempo reale* (40 ms)
anywhere.

Memory: RSS peak 3.39–3.97 GiB on MPS, 5.40 GiB on the CPU; Metal driver
peak 4.10–4.13 GiB in the matrix, 3.07–7.10 GiB in the people suite (the
23-person scene at the top). Model load 6.4–7.5 s per job (39 s once, the
first job of the pilot launch, cold cache). Load average before/after each
job 0.48–1.39 on the accelerated suites and 1.14–3.02 on the CPU suite,
where the job's own five threads are the load — the validator's only WARN
lines beside the thermal one.

## Pilot against campaign

The pilot launch's matrix (display off, `mediaanalysisd` running part of
the time, someone at the keyboard for 16 of its 24 minutes) against the
campaign's (display on and locked, user-activity asserted, daemon at 0 %):
cell medians differ by −0.5 % to +2.1 % (campaign / pilot 0.995–1.021; the
16:9 cells 1.6–2.1 % slower in the campaign, the 1:1 cells equal), the
pilot's repetitions 2–6× more dispersed. The four pilot people jobs read
0.4527–0.4924 s against 0.4636–0.5035 s in the campaign (−1.9 to −2.6 %).
So the display's compositing costs about 2 % at 16:9 and the daemon, on the
efficiency cores, cost nothing measurable — but the pilot's conditions
could not be stated, the campaign's can, and the two agree to the level of
the effect under study (the aspect-ratio factor is 67 %, the people factor
21 %). The pilot is kept for that comparison and nothing else.

## Validation (handoff §7)

`python3 benchmarks/validate-campaign.py A --commit 7e7ac5cf7645…`
(`artifacts/campaign/A-validate-20260915T2135Z.txt`): **valid**, every hard
line passes on the three suites; warnings: thermal (three suites), the two
launch-instant load lines (WindowServer 46.6 % as the display came back on;
WindowServer 37.9 % + loginwindow 23.4 % at the people preflight, the lock
screen's first paint — 0.9 % and absent ten seconds later), and
`load_average_max` 3.02 on the two CPU cells (the job's own threads).
Preflight reports `ok`, 0 failures, on all three suites. The summaries
written by the run (code at `7e7ac5c`) listed those reports under
`unreadable` — the cosmetic defect of `read_results` fixed in `a35f96c`;
the three `summary.json`/`summary.csv`/`pairs.csv` were regenerated from
the raw jobs with that code at 21:5x UTC by the campaign operator, the
validator re-run (`valid`), the cell medians unchanged to the fourth
decimal (the jobs are the record; the summaries are derived).
Archive `artifacts/campaign/A-20260915T213430Z.tar.gz`, sha256
`6f17dbfe9637882e905daa284a1b9ee18874e1058c77bffea71df99a55a15456`, verified
after the copy, unpacked to `artifacts/campaign/A/` and, byte-identical, to
the thesis repository as `data/benchmark/A/` (with `campaign.log` and
`archive.sha256`); the thesis exporter then produced its tables for five
machines. The pilot launch sits in `artifacts/pilots/pilot-A-firstlaunch-
20260915T2016Z/`, the setup logs in `artifacts/campaign/A-setup-logs/`.

## What is left on the machine

Only `~/unproject-campaign/` (the checkout with its `.venv`, the two
archives 7.6 GB, the extracted frames, the results, the pilot, the logs;
≈ 12 GB) and the uv cache it filled. Nothing outside it was written and
nothing was deleted; the author's checkout is as it was; the display was
turned on by the run's `caffeinate -u -d` and stays so until the machine is
used. The directory can be removed by its owner when the campaign is closed.
