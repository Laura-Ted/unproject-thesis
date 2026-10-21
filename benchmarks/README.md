# Running the campaign

The performance campaign of the thesis (ADR-0006): protocol v2's suites, from
one commit, on every machine in `machines.toml`. This is the runbook. Every
command below is the same on every machine; what differs is `MACHINE=<id>`.

Nothing here deletes, moves or edits anything outside `artifacts/campaign/`
(gitignored) and `models/` (the pinned weights, fetched once). The Macs this
runs on hold other people's data; the tooling was written with that in mind,
and `git status` after a run shows nothing changed.

## 0. Before anything: the commit

The campaign is one commit. Every result records it, and a suite with jobs
from two commits or from a dirty tree is refused by the evidence bundle.

```shell
git status --porcelain      # must print nothing
git rev-parse --short HEAD  # write it in the lab note
```

The suites are final: the matrix and the CPU point on MOT16-09, the people
suite with its low level (`mupots-ts10`, `mupots-ts11`), its medium scene
(`mot16-09`) and its high one (`mot16-08`). An experiment added to a suite changes the suite hash, so
nothing is added once the first job of the campaign has run anywhere.

## 1. The environment

```shell
make bench-env
```

`uv sync --frozen --extra multihmr`: the lockfile's versions and nothing else,
so torch, numpy and the estimator are the same on every machine. On Linux
x86-64 the lockfile resolves torch 2.13 with the CUDA 13 libraries; the host
driver has to be new enough for them (≥ 580), which the preflight checks by
trying. The last line prints what torch found: `cuda True` on a Linux box
with a working card, `mps True` on a Mac.

## 2. The footage

```shell
make bench-datasets MUPOTS=accept
```

Two inputs, both research datasets, both derived from their official archive
on every machine (`docs/datasets.md`); nothing is carried as a file:

- MOT16-08 and MOT16-09 are **fetched from MOTChallenge** as the official
  1.9 GB archive, verified by digest, and read as the JPEG frames they ship
  as. Nothing is transcoded; the matrix suite's half-resolution and
  letterboxed variants of MOT16-09 are made by the ingest from those frames
  with exact integer arithmetic (`[ingest.geometry]`), so they are the same
  bytes everywhere by construction;
- MuPoTS-3D TS10 and TS11 (the two "low" scenes) are **fetched from MPI for
  Informatics** the same way - the official 5.6 GB archive, its digest
  verified, two sequences extracted into `datasets/mupots-3d/` where the
  ground-truth evaluation already reads the dataset - but only with
  `MUPOTS=accept` (`--accept-mupots-terms`): the terms are research only, no
  redistribution, cite Mehta et al. 3DV 2018 and 3DV 2017, and they are yours
  to read at the project page before you say so. Without the flag the script
  reports the two sequences absent and fetches nothing.

A file already in place that hashes right is left alone; one that hashes
wrong is reported and left alone too - move it aside yourself.

## 3. The preflight

```shell
make bench-preflight MACHINE=B
```

Runs `unproject benchmark-preflight` for each of the three suites. Read-only
apart from one JSON report beside the results. It **fails** on: a dirty tree,
an id not in `machines.toml`, a device other than the one the suite or the
registry asks for, footage whose digest is not the campaign's, a clip too
short for the policy, less than 10 GB free, another process on the card. It
**warns** on what it cannot judge: a load average above 3, a process over
20 % of a core, the weights or the backbone not yet cached (the first job
fetches them), an unreported thermal state. Fix the failures; read the
warnings and close what they name.

The suites do not run while the preflight fails: `make bench-*` runs it
first and stops on red.

## 4. The conditions

Before `make bench-all`, on every machine:

- mains power, low-power mode off (a Mac's `pmset -g` shows `lowpowermode 0`);
- nothing else in the foreground: no video call, no virtual machine, no
  remote desktop session, no other model on the GPU. This afternoon's check
  ran 13 % slower with a VM and two remote desktops open;
- the lid open on a laptop, so it does not sleep and does not throttle on a
  closed lid (`caffeinate` keeps it awake for the run anyway);
- leave the machine alone until it prints the summary. The suite is
  resumable in the sense that every finished job is on disk, but a job
  interrupted halfway is recorded as failed and is not topped up.

The job records the load average, the power source and, on Linux, the
card's temperature and throttle reasons before and after; a summary with
`load_average_max` well above idle is a run to look at.

## 5. The suites

```shell
make bench-all MACHINE=B        # the three suites in order, each preflighted
make bench-matrix MACHINE=B     # or one at a time
make bench-people MACHINE=B
make bench-cpu MACHINE=B
```

| Suite | Cells × reps | What it measures | Time on the M4 |
|---|---|---|---|
| `matrix-suite` | 8 × 3 = 24 jobs | aspect ratio × resolution × output on MOT16-09, geometries by the ingest | ≈ 40 min |
| `people-suite` | 8 × 3 = 24 jobs | the person-count factor at a fixed tensor (MuPoTS TS10/TS11, MOT16-09, MOT16-08) | ≈ 40 min |
| `cpu-suite` | 2 × 3 = 6 jobs | the model held to the CPU, on MOT16-09 | ≈ 20 min |

Each job is a fresh process: the model loads every time (about 10 s), so
the wall clock is dominated by the frames. The results land under
`artifacts/campaign/<MACHINE>/benchmark-v2/<suite>/`, one JSON per job, the
run record, the preflight report, and after each suite `summary.json`,
`summary.csv`, `pairs.csv` derived from the jobs.

To regenerate the summaries without running anything:

```shell
make bench-summary MACHINE=B
```

## 6. Carrying the results back

```shell
make bench-collect MACHINE=B
```

One `artifacts/campaign/B-<stamp>.tar.gz` with the results, run records,
preflight reports and summaries - not the ingest frames, which regenerate.
Copy it to the operator's machine and unpack it under the same
`artifacts/campaign/` so the machines sit side by side:

```shell
tar -xzf B-<stamp>.tar.gz -C artifacts/campaign
make bench-campaign            # -> artifacts/campaign/campaign.{json,csv}, campaign-pairs.csv
```

`bench-campaign` reads only the `summary.json` files; a machine directory
whose results name another machine id is flagged `consistent: false`.

## 7. Rented Linux machines

`vast/campaign.py` drives one instance end to end from the operator's Mac;
`vast/setup-instance.sh` is what runs on it. The repository is private, so
the instance receives a `git bundle` of the commit over scp; no credential
of any kind reaches it.

```shell
python3 benchmarks/vast/campaign.py search --gpu RTX_4070 --consumer-cpu
python3 benchmarks/vast/campaign.py create <offer> --machine C-rtx4070
python3 benchmarks/vast/campaign.py wait <instance>
python3 benchmarks/vast/campaign.py probe <instance> --direct --ssh-key ~/.ssh/<key>   # before spending: read it
python3 benchmarks/vast/campaign.py push <instance> [--commit <sha>]
python3 benchmarks/vast/campaign.py seed <instance> --direct --ssh-key ~/.ssh/<key>    # optional: archives + weights from here
python3 benchmarks/vast/campaign.py setup <instance> --machine C-rtx4070 --accept-mupots-terms [--commit <sha>]
python3 benchmarks/vast/campaign.py run <instance> --machine C-rtx4070
python3 benchmarks/vast/campaign.py ssh <instance> -- tail -f /root/unproject-campaign/campaign-C-rtx4070.log
python3 benchmarks/vast/campaign.py pull <instance> --machine C-rtx4070
python3 benchmarks/vast/campaign.py destroy <instance>
```

**Read `probe` before anything else.** It prints the host's load (external,
since nothing of ours runs yet), whether the card is idle, and the speeds
the instance actually gets from motchallenge.net, MPI, naverlabs, PyPI and
GitHub, plus what your machine can push to it. On 2026-09-15 the advertised
bandwidth predicted none of this: the same origins gave 0.0 to 112 MB/s
depending on the host, one host had its card at 100 % for a tenant the
container cannot see, and one ran its other tenants' load at 12-19 on a
12-core quota - all visible in a minute, all ruinous after an hour. Destroy
a host that reads badly; the two minutes cost cents. Then:

- `push --commit <sha>` bundles that commit instead of HEAD, for the case
  where the branch has moved (a note, another machine's registry line)
  and the campaign's commit must not: the rented hosts of 2026-09-15 all
  ran `7e7ac5c` that way after D was registered as `16b2231`;
- `seed` copies `datasets/_download/MOT16.zip`, `MultiPersonTestSet.zip`
  and `models/multihmr2.pt` from this machine over the host's direct ssh
  port with `rsync --partial --inplace` (the vast proxy carried 1 MB/s,
  the direct port 10-40); `setup` then verifies the pinned digests exactly
  as after a download, so the bytes are the campaign's either way. Use it
  when `probe` shows the origins slow and your uplink fast; skip it when
  the origins are fast (the Italian host of the 5090 fetched everything in
  five minutes);
- `destroy` passes `--yes`: without a terminal the CLI would print
  "Aborted" and keep billing.

Needs an API key set with `uvx vastai set api-key <key>` (kept by the CLI
under `$HOME`, never here) and an SSH public key registered with the
account. `search` filters for one card, a driver that supports CUDA 13, 60
GB of disk and, with `--consumer-cpu`, a Ryzen or Core host CPU; the offer
list shows the card, the price per hour, the driver and the CPU so the
choice is deliberate. Each card gets its own id in `machines.toml` before it
runs.

What the instance records that the Macs cannot: the card's name, the
driver, CUDA and cuDNN versions, its temperature, power, clocks and throttle
reasons before and after every job. What it cannot give: a consumer CPU
figure - the host CPU is a server part, and the CPU suite's result there is
labelled by the registry's hardware class.

## 8. What not to do

- Do not run two suites at once, on one machine or on two machines sharing
  a GPU.
- Do not edit a suite file between machines; the hash changes and the
  results stop being one campaign.
- Do not delete a failed job's file; the protocol wants the failure visible.
- Do not point two machines at one artifacts root.
- Do not re-transcode the pedestrians clips; carry the files.
- Do not measure on a rented host whose `probe` shows an external load
  (the preflight warns at 3; a shared host at 12-19 gave 19-27 % spread
  between repetitions of one cell) or a card that is not idle (P8, 0 %):
  the result would be a pilot, and it was, once.
- Do not run long remote steps in a foreground ssh session through the
  vast proxy: it drops silent sessions. `run` already uses `nohup`; do the
  same for anything of your own.

## 9. The campaign of 2026-09-15/16: what exists

The suites ran on seven machines - A, D, C-rtx3060, C-rtx4070, C-rtx4090,
C-rtx5070, C-rtx5090 - from commit `7e7ac5c` (D from `16b2231`, its
registry line), 378 jobs, none failed, every machine validated. The
report and index is `docs/notes/2026-09-16-campaign-report.md`: what was
measured, where every file is (`artifacts/campaign/`, `artifacts/pilots/`,
`artifacts/campaign-logs/`, and the thesis repository), how to validate
again, and how to read the numbers. One lab note per machine sits beside
it in `docs/notes/`, plus `2026-09-15-campaign-rented-hosts.md` for the
hosts, the money and the network. `benchmarks/validate-campaign.py` is the
handoff's validation list as a script:

```shell
python3 benchmarks/validate-campaign.py C-rtx3060 C-rtx4070 --commit <full sha>
```
