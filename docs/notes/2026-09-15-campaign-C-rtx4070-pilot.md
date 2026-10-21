# 2026-09-15 - C-rtx4070, first run: a pilot (host under external load)

Instance 51143079 (offer 50860550, New York, RTX 4070 12 GiB, AMD Ryzen 9
5900X 12C/24T with 12 cores allocated, driver 580.159.04, CUDA 13.0,
0.264 $/h with disk, bandwidth 0.0326 $/GB), created 17:59, suites
18:24–19:29, destroyed 19:33 UTC. Commit `7e7ac5c`, clean tree, preflights
`ready ×3` with a **load warning: 1-minute load 12.1–13.9 before each
suite**, external (nothing of ours ran; the host's 5- and 15-minute averages
were 12.3 and 13.5 before our setup), on a 12-core quota; 19 by the time a
probe instance looked again at 20:05.

All 54 jobs ran `ok` and the mechanical validation passes, but the three
repetitions of a cell disagree by **19–27 % in 12 of 18 cells** (e.g.
`letterbox-half-skeleton` 0.1254 / 0.1500 / 0.1482 s; `mot16-09-skeleton`
0.1284 / 0.1034 / 0.1049; CPU `mot16-09-mesh` 2.58 / 3.02 / 3.13 s), against
0.1–1.8 % on the idle 3060 host; the same cell `mot16-09-mesh` reads 1.20×
its skeleton twin in the matrix suite and 1.00× in the people suite, and the
high-count scene came out *cheaper* than the low-count ones (0.978). These
are measurements of the host's contention, not of the card. **Not a thesis
number**: kept as delivered under
`artifacts/campaign/pilot-C-rtx4070-20260915T1824-loaded-host/` (archive
`C-rtx4070-20260915T193135Z.tar.gz`, sha256 `301e353a…`), never under
`artifacts/campaign/C-rtx4070/`, not copied to the thesis repository.

The rerun needs an idle host. At 19:30–20:45 UTC the only RTX 4070 offers
with a driver ≥ 580 were: this host (loaded), one in California (offer
50971030, never accepted the ssh key in 14 minutes, destroyed unused) and
one in Fujian (offer 34049710, still pulling the Docker image after 20
minutes, destroyed unused). Spend on the pilot ≈ 0.80 $ (1.57 h; the
bandwidth alone ≈ 0.39 $), on the two failed attempts ≈ 0.15 $.

Lesson for the driver (end-of-campaign batch): read `/proc/loadavg` right
after creation and refuse a host whose external load is above a few units
before spending on the transfer; the preflight's warning came only after
the 25-minute setup.
