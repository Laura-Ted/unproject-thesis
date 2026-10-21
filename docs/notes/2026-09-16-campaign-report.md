# 2026-09-16 - The performance campaign of ADR-0006: report and index

Start here if you come to these results later. This note says what was
measured, on what, from which commit, where every file is, how to check
it again, and what was decided along the way. The numbers themselves are
in the per-machine notes and in `artifacts/campaign/campaign.csv`; nothing
here is estimated, every figure traces to a job JSON.

## What was measured

Protocol v2 (`docs/experiments/protocol-v2.md`, frozen 2026-09-15) with its
three suites - `rq1rq2-v2-matrix` (8 cells), `rq2-v2-people` (8 cells),
`rq1-v2-cpu` (2 cells), 3 repetitions each, one fresh process per job, 5
warm-up + 60 measured frames - on the inputs of ADR-0006 §5: MOT16-09 and
its three ingest geometries, MOT16-08, MuPoTS-3D TS10 and TS11, every byte
verified against the pinned digests on every machine.

| Id | Machine | Backend | Commit | Jobs | Verdict | Note |
|---|---|---|---|---|---|---|
| A | MacBook Pro 14" M3 Pro, 18 GiB (the author's) | MPS | `7e7ac5c` | 54/54 ok | valid | `2026-09-15-campaign-A.md` |
| D | MacBook Pro 16" M4 Pro, 24 GiB | MPS | `16b2231` | 54/54 ok | valid | `2026-09-15-campaign-D.md` |
| C-rtx3060 | rented, RTX 3060 12 GiB, Ryzen 5 3600 | CUDA | `7e7ac5c` | 54/54 ok | valid | `2026-09-15-campaign-C-rtx3060.md` |
| C-rtx4070 | rented, RTX 4070 12 GiB, Ryzen 9 5900X | CUDA | `7e7ac5c` | 54/54 ok | valid (second run) | `2026-09-15-campaign-C-rtx4070.md`; first run a pilot, `-pilot.md` |
| C-rtx4090 | rented, RTX 4090 24 GiB, EPYC 7B13 (shared) | CUDA | `7e7ac5c` | 54/54 ok | valid, shared-host spread 2-7 % | `2026-09-15-campaign-C-rtx4090.md` |
| C-rtx5070 | rented, RTX 5070 12 GiB, i5-11600KF | CUDA | `7e7ac5c` | 54/54 ok | valid | `2026-09-15-campaign-C-rtx5070.md` |
| C-rtx5090 | rented, RTX 5090 32 GiB, EPYC 7C13 (128 threads) | CUDA | `7e7ac5c` | 54/54 ok | valid, CPU-bound host regime | `2026-09-16-campaign-C-rtx5090.md` |

`16b2231` differs from `7e7ac5c` by the seven lines that register D in
`benchmarks/machines.toml` (`git diff 7e7ac5c 16b2231 --stat`); no measured
code differs. Machine B (the Mac mini M4, the operator's station) was in the
handoff and was dropped by the author's decision at 20:35 UTC on the 15th;
the registry still lists it, nothing was measured on it. Summaries of every
machine were regenerated from the raw jobs under `a35f96c` (which only
stops the summary reader from listing the preflight reports as unreadable);
the cross-machine table `campaign.csv` is derived from those summaries and
nothing else: 126 cells, 91 pairs, nothing missing, no cell inconsistent.

The rented-host side - the provider, the money (5.00 → 0.37 $), the eleven
hosts created and the seven destroyed unused, what the network measurements
taught - is `2026-09-15-campaign-rented-hosts.md`.

## Where every file is

Under the software repository (all gitignored; carried by the archives and
by the thesis repository, below):

```
artifacts/campaign/<id>/benchmark-v2/<suite>/   preflight-*.json, <stamp>-suite-run.json,
                                                <cell>/<stamp>-rep<k>-job<i>.json (54 per machine),
                                                summary.json, summary.csv, pairs.csv
artifacts/campaign/<id>-<stamp>.tar.gz(.sha256) the bench-collect archive as delivered, and its digest
artifacts/campaign/<id>-campaign.log            make bench-all's log on that machine
artifacts/campaign/campaign.{json,csv}, campaign-pairs.csv   the cross-machine table
artifacts/pilots/                               runs that are not results: pilot-C-rtx4070-… (host under
                                                external load), pilot-A-firstlaunch-…, pilot-D-firstlaunch-…,
                                                B-preflights-… (the morning's preflights of retired suites)
artifacts/campaign-logs/                        setup logs of the rented hosts, validator outputs, note drafts
```

In the thesis repository, `TesiTriennale/data/benchmark/<id>/` holds the
same `benchmark-v2/` tree (raw jobs byte-identical, checked with sha256),
`campaign.log` and `archive.sha256`, plus `data/benchmark/campaign.{csv,json}`;
`scripts/export_campaign_results.py` regenerates the thesis tables from the
raw jobs there.

## How to check it again

```shell
# every hard line of the handoff's validation list, per machine:
python3 benchmarks/validate-campaign.py A C-rtx3060 C-rtx4070 C-rtx4090 C-rtx5070 C-rtx5090 \
    --commit 7e7ac5cf7645a64244f87b9c22d49be032bee448
python3 benchmarks/validate-campaign.py D --commit $(git rev-parse 16b2231)
# the summaries and the cross-machine table, from the raw jobs, no inference:
make bench-summary MACHINE=<id>       # per machine
make bench-campaign                   # -> artifacts/campaign/campaign.{json,csv}
# the archive of a machine against its recorded digest:
shasum -a 256 -c artifacts/campaign/<id>-<stamp>.tar.gz.sha256
```

What the validator checks: preflight `ok` with no failed check; cells ×
repetitions job files, all `status: ok`, 5 warm-up + 60 steady frames;
`git_dirty: false` and one commit across the jobs; `machine_id` and `device`
as the registry says; `nvidia-smi` readings before and after on Linux,
mains power on a Mac; `meets_protocol` and `one_machine` on every cell; one
suite hash per suite across machines. Warnings (load, thermal) are printed
for a person to read, and each note says how they were read.

## What to know when reading the numbers

- A number of a machine is a number of that machine (ADR-0006 §6); the
  comparable things across machines are the ratios, per machine.
- The tensor ratio 16:9 → 1:1 is 1.67 (A), 1.58 (D), 1.49 (3060), 1.36
  (4070), 1.34 (5070), 1.09 (4090), 1.08 (5090): it falls as the card gets
  faster because the CPU-side part of the frame does not scale with the
  tensor. The 5090 is the extreme: torch's default 128 threads on a
  256-thread host over-subscribe that part and the fastest card gives the
  slowest CUDA frame (0.111 s against 0.071 s on the 5070). The thread
  count is recorded and not tuned, as protocol v2 §10 requires; pinning it
  would be an amendment and a rerun of every machine.
- The resolution factor is ≈ 1.0 everywhere (1.00-1.05); the person count
  raises the cost monotonically from 2 to 23 people on every machine
  (×1.14-1.23); the CPU suite is 1.3-1.9× the accelerator on the Macs and
  15-29× on the rented hosts.
- RQ1 verdicts (thresholds pre-declared): every accelerated cell on every
  machine is *eseguibile* and *streaming interattivo*; none is *tempo reale*
  (40 ms); the fastest cell is 65 ms (C-rtx5070, two people).
- The C-rtx4090 run has 2-7 % spread between repetitions from other
  tenants' load on its host; the per-job medians are in the summary. The
  first C-rtx4070 run (19-27 % spread) is a pilot and is not in the table.

## Commits

```
49ff14c  vast: push stages the clips directory even when no clip is carried   (shakedown)
868cd90  benchmarks: validate-campaign.py                                     (shakedown)
2ac0c4b  preflight: describe the device the suite will use                    (shakedown)
7e7ac5c  vast: the campaign log lives beside the checkout; setup makes the fixture   <- campaign commit
16b2231  machines: register D                                                 <- D's commit
a35f96c  tooling: seed, probe, --commit, destroy --yes, busy card = stop, summary skips preflight reports
914f5cf  notes: one per machine + rented hosts
d63bfb7  notes: C-rtx5090
```

The shakedown that produced the first three fixes is in the C-rtx3060
note. The tooling commit changes no measured code; the runbook
(`benchmarks/README.md` §7, §9) describes the subcommands it added and the
rule the evening taught: measure a host's network and load before spending
on it.
