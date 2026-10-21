# 2026-09-15 - The campaign on rented Linux hosts: what happened, host by host

The operator's log of the ADR-0006 campaign's Linux side, run from the Mac
mini (machine B, the operator's station) between 14:56 UTC on the 15th
and 00:24 UTC on the 16th.
One note per machine holds the numbers; this one holds what is common:
the provider, the money, the hosts that were tried and refused, and the
two changes of plan of the evening. Every fact below is from the terminal
or from `vastai show user`; nothing is estimated.

## Commit

The campaign commit is **`7e7ac5c`** for A, C-rtx3060, C-rtx4070,
C-rtx4090 and C-rtx5070. Machine D was added to the registry at 20:33 UTC
as `16b2231` (`benchmarks/machines.toml`, seven lines, nothing else) and
its jobs record that commit; `git diff 7e7ac5c 16b2231 --stat` shows the
registry line only. The tooling commit `a35f96c` (driver, preflight,
summary; no measured code) came after every job of every machine, and the
summaries of all machines were regenerated from the raw jobs under it, so
`summary.json` no longer lists the `preflight-*.json` reports as
unreadable; no number changed, the summary is a derivation.

## Machines: what was decided

The handoff named A (M3 Pro), B (this Mac mini) and the C-* hosts. At
20:35 UTC the author decided the Apple machines are **A and D** (a MacBook
Pro 16" M4 Pro, 24 GiB) and **B is not measured**; B stayed the operator's
station. A and D were run by two other sessions over ssh, from separate
checkouts; their notes are `2026-09-15-campaign-A.md` and `-D.md`.

## vast.ai: the money

Credit 5.00 $ at 14:56; 0.37 $ at the end (4.63 $ spent). Instances and what they
cost, in order (prices per hour include the 80 GB disk; bandwidth is
billed per GB by some hosts):

| Id | Machine | Host | Created → destroyed (UTC) | Outcome | ≈ $ |
|---|---|---|---|---|---|
| 51127808 | C-rtx3060 | NJ, Ryzen 5 3600, 0.0985 $/h | 14:56 → 17:58 | shakedown + valid campaign | 0.33 |
| 51143079 | C-rtx4070 | NY, Ryzen 9 5900X, 0.264 $/h, 0.033 $/GB | 17:59 → 19:33 | **pilot**: host under external load 12–19 | 0.80 |
| 51145084 | C-rtx4090 | SE, EPYC 7B13, 0.50 $/h | 18:28 → 19:33 | valid (shared host, spread 2–7 %) | 0.55 |
| 51146329 | C-rtx5070 | FR, Ryzen 7 3700X | 18:42 → 18:54 | ssh key never accepted | 0.04 |
| 51147246 | C-rtx5070 | NV, i5-11600KF, 0.21 $/h | 18:54 → 20:10 | valid | 0.33 |
| 51149639 | C-rtx4070 | CA, Ryzen 9 3950X | 19:34 → 19:48 | ssh key never accepted | 0.11 |
| 51150389 | C-rtx4070 | CN, Xeon E5-2650 v4 | 19:48 → 20:08 | Docker image never finished pulling | 0.03 |
| 51151556 | C-rtx4070 | NY (same host as 51143079) | 20:05 → 20:10 | probe only: load 19 | 0.02 |
| 51151661 | C-rtx5090 | KR, Ryzen 7 7800X3D | 20:11 → 20:23 | card at 100 %, P0, 98 W with no visible process | 0.12 |
| 51152408 | C-rtx5090 | CA, EPYC 7742 | 20:23 → 20:45 | network 0.6 MB/s to PyPI, 0.0 to the origin servers | 0.20 |
| 51153762 | C-rtx5090 | US, EPYC 7B12 | 20:45 → 20:49 | probe only: 0.8 MB/s | 0.03 |
| 51154308 | C-rtx5090 | CA, EPYC 7K62 | 20:49 → 21:19 | 1 MB/s in every direction but MPI; abandoned | 0.30 |
| 51156171 | C-rtx4070 | NY (same host, now idle: load 0.05) | 21:25 → 22:18 | **valid campaign** | 0.53 |
| 51159467 | C-rtx5090 | US, EPYC 7B12 | 22:25 → 22:35 | ssh key never accepted, no direct port | 0.06 |
| 51159996 | C-rtx5090 | US, EPYC 7K62 | 22:35 → 22:45 | probe: PyPI 0.3 MB/s per connection, from the Mac 0.3 with 8 streams, motchallenge/naverlabs 0.0 | 0.08 |
| 51160639 | C-rtx5090 | KR, Ryzen 7 7800X3D | 22:45 → 22:51 | probe: everything 0.0–0.1 MB/s | 0.06 |
| 51162504 | C-rtx5090 | IT, EPYC 7C13, 0.688 $/h | 23:18 → 00:24 | **valid campaign** (probe: 38–112 MB/s from every origin, 39 from the Mac) | 0.85 |

The 5090 was measured on the **seventh** host tried, after midnight
(`2026-09-16-campaign-C-rtx5090.md`): the first six had either the card in
use by a tenant this container cannot see, a network at 0.1–1 MB/s in
every direction that mattered (parallel streams raised MPI to 40 MB/s but
not PyPI, motchallenge or the Mac), or never accepted the ssh key. An
Italian datacenter host with 38–112 MB/s from every origin did it in an
hour. The 4070's first run is a pilot
(`2026-09-15-campaign-C-rtx4070-pilot.md`) and the second, on the same
host once its other tenant had finished, is the result.

## What the rented hosts taught, and what changed in the tooling

- **A host's load is the first thing to read.** The preflight warned
  (load 12 on a 12-core quota) after 25 minutes of setup; the card was
  idle, the CPU-side work was not, and the three repetitions of a cell
  came out 19–27 % apart. `campaign.py probe` now prints the load, the
  card's idle state and the network speeds before anything is spent.
- **A card can be busy without a visible process.** `nvidia-smi
  --query-compute-apps` sees this container only; a 5090 read 100 %, P0,
  98 W with "no process". The preflight now fails above 20 % utilisation.
- **Advertised bandwidth means nothing.** Measured from the instances:
  motchallenge.net 0.0–19 MB/s, MPI 4.6–22, naverlabs 0.0–19, PyPI
  0.6–34, the operator's Mac 0.8–30 (over the direct port; the vast ssh
  proxy carried 1 MB/s). `campaign.py seed` copies the two archives and
  the weights from the operator's machine over the direct port with
  `rsync --partial --inplace`; `setup` then verifies the pinned digests
  exactly as it does after a download, so the bytes are the campaign's
  whichever way they came. Two archives were completed from two sources
  (MPI + the Mac, `rsync --append`) and verified whole.
- **Two hosts never accepted the registered ssh key** in 12–14 minutes;
  a third never finished pulling the image. Replaced, not debugged.
- **The proxy drops long silent sessions**; every long remote step ran
  under `nohup` with a log, and the campaign log itself moved out of the
  checkout (`7e7ac5c`) after the shakedown found it made the tree dirty.
- `vastai destroy` asks for confirmation and, without a terminal,
  "Aborted" and kept billing; the driver passes `--yes` (`a35f96c`).
- The operator's `~/.ssh/config` disables agents (`IdentityAgent none`);
  a `Host *.vast.ai` entry offers the vast key; the direct addresses need
  `--ssh-key`.

## Reading across the cards (ADR-0006 §11), from the summaries

Inference per frame at 768×432, 16:9 skeleton on MOT16-09 (≈ 10 people),
median of 180 steady frames, and the tensor ratio 16:9 → 1:1 on the same
frames:

| Machine | 16:9 skeleton | 1:1 skeleton | ratio | CPU suite | CPU / card |
|---|---|---|---|---|---|
| A (M3 Pro, MPS) | 0.5039 | 0.8408 | 1.67 | 0.9689 | 1.9 |
| D (M4 Pro, MPS) | 0.3642 | 0.5743 | 1.58 | 0.4857 | 1.3 |
| C-rtx3060 | 0.1385 | 0.2065 | 1.49 | 2.5872 | 18.7 |
| C-rtx4070 | 0.0830 | 0.1132 | 1.36 | 1.2939 | 15.6 |
| C-rtx4090 | 0.0950 | 0.1035 | 1.09 | 2.4849 | 26.2 |
| C-rtx5070 | 0.0711 | 0.0954 | 1.34 | 2.0532 | 28.9 |
| C-rtx5090 | 0.1107 | 0.1193 | 1.08 | 2.1055 | 19.0 |

The tensor ratio falls as the card gets faster (1.67 on Metal, 1.49 →
1.08 on CUDA): the 768×768 tensor costs 1.78× the pixels, and the share
of the frame that scales with them shrinks as the GPU part shrinks
against the fixed CPU-side work (preprocessing, tracker, conversion). The
RTX 5090 is the extreme case: on a 256-thread host with torch's default
128 threads that CPU-side work runs over-subscribed and the fastest card
gives the slowest CUDA frame (0.111 s against the 5070's 0.071); the
thread count is recorded, not tuned (protocol v2 §10), and the reading is
in that machine's note.
Per §11 that is a property of the backend and host, reported per
machine; no figure averages machines. The resolution factor is ≈ 1.0 on
every machine (1.00–1.07); the people suite rises monotonically from 2
to 23 people on every machine (A ×1.21, D ×1.20, 3060 ×1.16, 4090 ×1.20,
5070 ×1.19, 5090 ×1.14). The consumer CPU points of RQ1 are the Macs' (A 0.97 s, D
0.49 s per frame); the rented hosts' CPU cells are the host CPU's and are
labelled by the registry's class.

RQ1 verdicts, pre-declared thresholds: every accelerated cell on every
machine is *eseguibile* and *streaming interattivo* (median ≤ 1 s, above
one estimate per second); no cell reaches *tempo reale* (40 ms per frame;
the fastest measured is 65.8 ms, C-rtx5070 on two people).
