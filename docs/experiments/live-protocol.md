# Live End-to-End Measurement Protocol

**Status:** Frozen
**Created:** 2026-09-15
**Frozen:** 2026-09-15, with the commit that adds it, before any measurement
recorded under it
**Governs:** the end-to-end overhead of the live path in RQ1
(`docs/adr/ADR-0002-live-body-capture-from-monocular-video.md` §4), as required
by `docs/adr/ADR-0005-thesis-evidence-hardening.md` WP3
**Replaces:** the README's "2.45 fps against 2.60 for inference alone", which
predated the benchmark and could not be regenerated

This is the pre-registered protocol for the live path. It contains no results.
Raw observations are the report files under `artifacts/live/`, and the table is
derived from them by `unproject live-summary`.

---

## 1. What is measured, and on whose clock

Protocol v2 (`protocol-v2.md`) measures the estimator on frames already in
memory. Everything between the camera and the screen is outside it: JPEG
encoding in the browser, the socket, the server's decode, the wait in the
latest-only mailbox, the socket back, and the render loop. That is what this
protocol measures - **as two separate things, on two separate clocks, never
joined.**

Every frame the client sends carries a monotone `frame_id`; the server echoes
it in its answer. On its own clock (`performance.now()` in the browser,
`time.perf_counter()` in the Python client), the client records when it sent
the frame, when the answer arrived, and - in the browser - the first animation
frame after the bodies were applied. The server attaches to each answer its
own durations on its own monotonic clock: decode, queue wait, inference, and
the serialisation of the previous message.

**No interval subtracts a browser timestamp from a server timestamp.** The two
clocks have different origins, and a "latency" across them would measure the
origins. Every number below is one clock's.

## 2. Sources

The CC0 pedestrians clip of protocol v2 §3.1, replayed into the product's own
camera endpoint. Two clients do the replaying:

| Client | What it is | What it measures |
|---|---|---|
| **transport** | `unproject-live-benchmark`, a Python client on the real socket; frames JPEG-encoded up front at the viewer's size and quality; runs unattended | capture-to-response round trip, server phases, drops, bytes |
| **browser** | the viewer in measure mode, `?measure=1&replay=…`: the clip plays in a `<video>` element and is captured into a canvas at a controlled rate, as a camera is | the same, plus response-to-first-animation-frame and the browser's own encode cost inside the round trip |

The camera endpoint, not the file endpoint: the file path is ordered and
lossless and cannot show the latest-only behaviour that a camera session has.
No live camera is used: a camera is not a repeatable source.

## 3. Send rates

At least two, one below and one above the estimator's capacity, so the
protocol shows both latency at steady state and the effect of the latest-only
mailbox when the source outruns inference:

| Rate | Why |
|---|---|
| 1 fps | below capacity on the protocol machine: every frame answered, queue wait near zero, the round trip is the path's own cost |
| 8 fps | the viewer's own capture rate and above capacity: the mailbox replaces frames, `dropped` is non-zero, and the answered rate is the estimator's |

Both rates, both clients, skeleton mode; mesh mode at 1 fps in addition, so the
extra bytes of bone transforms and rest bodies are measured on the wire.

## 4. Environment

The machine of every other protocol here, the M3 Pro; the server and the
client on the same machine over loopback, which is the supported deployment
(ADR-0002 §6.7). Each report records the client platform, user agent and
viewport (browser), the server's chip and device, the commit and its dirty
state, the clip and its digest (transport), the send rate, frame size, JPEG
quality, mode, and the live clock's lag.

## 5. Metrics

Recorded per frame into `unproject.benchmark.live.LiveSample` and summarised by
`summarise_live`, which runs on the server for browser reports and in the
client for transport reports - one implementation. Milliseconds. Percentiles
nearest-rank.

| Metric | Clock | Definition |
|---|---|---|
| Round trip | client | receive − send, per answered frame |
| Capture to present | client, browser only | first animation frame after the answer was applied − send |
| Receive to present | client, browser only | the previous minus the round trip |
| Decode | server | JPEG decode of the received frame |
| Queue wait | server | from hand-over to the mailbox to the moment the model took the frame |
| Inference | server | `session.frame`, off the event loop |
| Serialise | server | encoding of the previous message on the socket (a message cannot carry its own) |
| Frames sent, answered | client | rows, and rows with an answer |
| Dropped by the server | server, via client | the sum of `dropped` over answers: frames the mailbox replaced |
| Unaccounted | derived | sent − answered − dropped: in flight at the end, lost, or dropped after the last answer |
| Answered fps | client | answered frames over the session's duration |
| Bytes sent, received | client | JPEG bytes (with the 4-byte id) out; message bytes in |
| People | server | detections per answered frame, a covariate |

**Presented is not drawn.** The live clock trails the newest frame by a lag
(0.4 s by default) so the render loop has something to interpolate towards;
the bodies of a given frame are drawn at least that much later than they were
presented in the sense above. The lag is in every browser report's config and
is added in the reading, not hidden in the number.

## 6. Commands

```shell
# transport, unattended; --spawn starts and stops the server itself
uv run unproject-live-benchmark datasets/pedestrians/clip.mp4 --fps 1 --spawn
uv run unproject-live-benchmark datasets/pedestrians/clip.mp4 --fps 8 --spawn
uv run unproject-live-benchmark datasets/pedestrians/clip.mp4 --fps 1 --bodies --spawn

# browser: start the server, open the URL, wait for "Report kept at …"
uv run unproject-api
open "http://localhost:8000/?measure=1&replay=datasets/pedestrians/clip.mp4&fps=1"
open "http://localhost:8000/?measure=1&replay=datasets/pedestrians/clip.mp4&fps=8"

uv run unproject live-summary
```

The browser posts its rows to `POST /api/live-metrics` when the clip ends;
the server computes the summary and writes the report. A second submission of
the same session is refused, so a report is never replaced.

## 7. Failure and exclusion rules

Written before collection:

- A session that ends with an `error` message from the server is a report
  with the message in `notes`; it is kept and listed, not averaged with the
  others.
- An answer for a frame the client did not send is recorded as a note and
  ignored; it would mean the ids were misread.
- A report whose stored summary disagrees with its own rows is refused on read.
- **No session is excluded for being surprising.** A dropped frame is the
  mailbox working; a long round trip at 8 fps is the finding.

## 8. Outputs

| Output | Contents |
|---|---|
| `artifacts/live/<stamp>-<kind>-<mode>-<fps>fps.json` | one session: every frame row, the summary, environment, config, notes |
| `artifacts/live/summary.csv` | one row per report, from `unproject live-summary` |

The transport and browser rows sit in the same table and are labelled by
`kind`. No number joins them: the browser's capture-to-present is reported as
the browser's, the transport client's round trip as the transport's, and the
difference between the two clients' round trips at the same rate is the
browser's own encode-and-dispatch cost, read from the table rather than
computed into a third number.
