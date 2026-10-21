# 2026-09-15 - The live path run once on an M4, and the drops it was not counting

The transport client of WP3 (`unproject-live-benchmark`, protocol
`docs/experiments/live-protocol.md`) was run against a spawned server on
this machine, an Apple M4 with 16 GB, with the CC0 pedestrians clip: 60
frames at 1 fps, then 240 at 8 fps, skeleton output, nothing else on the
GPU. The reports are under `artifacts/live/`, gitignored. They are **not
thesis numbers**: wrong machine (the protocol names the M3 Pro), and the
second pair of sessions ran from a dirty tree, because the first session
found a bug and the fix was in the tree when they were repeated.

## The bug

First session, 1 fps: sent 60, answered 51, dropped by the server 0,
**unaccounted 9**. Frames 1-9 were sent while frame 0 was in the model -
the first inference includes the model load, 9.4 s here - and were never
answered and never reported dropped. The mailbox had counted them; the
session loop had not passed the count on. It read `mailbox.dropped` before
handing a frame to the model and again after the answer, and set its
"already reported" mark from the second reading, so every drop that
happened *during* an inference was marked reported without ever being put
in a message. At 1 fps on a fast machine that is only the load; on a camera
at 30 fps it is every frame that arrives while one is being estimated,
which is most of them. The line dates from the commit that added the API
(`a13cd4c`), so the viewer's "dropped" figure has under-reported since the
live path existed; the README's description of the policy was right and the
number beside it was not.

Fixed in `_serve_camera`: what is put in the message is what is added to
the mark. A test holds the first frame in the model until three more are in
the slot and asserts the answer to the third carries `dropped: 2`; on the
old code it carries 0.

## After the fix

| | 1 fps, 60 sent | 8 fps, 240 sent |
| --- | --- | --- |
| answered / dropped by the server / unaccounted | 52 / 8 / 0 | 33 / 207 / 0 |
| answered rate | 0.87 fps | 1.06 fps |
| round trip, median (p95) | 687 ms (890) | 737 ms (1210) |
| server inference, median | 671 ms | 654 ms |
| server decode, median | 8.2 ms | 3.5 ms |
| queue wait, median | 0.1 ms | 56 ms |
| serialisation, median | 0.2 ms | 0.2 ms |
| first answer | 9.4 s after the first send | 9.9 s |

The books balance now: sent = answered + dropped + unaccounted, in both.
What the socket adds on top of inference on this machine is 16 ms at the
median when the sender is slower than the model and 83 ms when it is
faster, 56 of which is the frame waiting in the slot for the previous
inference to finish - the latest-only policy doing what it is for.

## The browser, once

The viewer's measure mode (`?measure=1&replay=datasets/pedestrians/clip.mp4`
at `fps=1` and `fps=8`) was then driven from an automated Chromium tab
(Electron 44, Chrome 152) against the same server, from the clean tree of
the fix's commit. The clip plays at its natural rate, so a 15 s clip minus
the model load gives 11 frames at 1 fps and 107 at 8: answered 5 and 23,
dropped 6 and 84, unaccounted 0 and 0. Round trip median 722 ms and 729 ms;
capture-to-present 722 ms and 738 ms; receive-to-present - the wait for the
first animation frame after the bodies were applied, which only the browser
can measure - median 0.0 ms and 5.0 ms, p95 0.4 ms and 15.4 ms. The report
was posted to `/api/live-metrics` and kept under `artifacts/live/`, and
`unproject live-summary` tabulated the four sessions together. The path
works end to end; the tab was not in the foreground, and the protocol's
browser sessions are the author's, in one, on the M3 Pro (`docs/release.md`
§2).
