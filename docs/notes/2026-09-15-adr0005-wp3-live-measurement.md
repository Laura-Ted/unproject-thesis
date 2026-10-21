# 2026-09-15 - WP3, measuring the live path on one clock at a time

WP3 of ADR-0005. `docs/experiments/live-protocol.md` is frozen with this
commit; the README's "2.45 fps against 2.60" is withdrawn, because nothing
could regenerate it. This is the design record; no measurement has been taken
under the protocol yet.

## The rule everything else follows

A browser's `performance.now()` and a server's `perf_counter()` have
different origins. Subtracting one from the other gives a number that is
mostly the origins, so the ADR forbids it and so does the code: the server
attaches **durations** to each answer - decode, queue wait, inference, and
the serialisation of the previous message - and the client measures its
intervals on its own clock, pairing an answer with the send it answers by a
`frame_id` the client chose and the server echoed. Every reported interval is
one clock's.

## Wire schema 2

Additive. `Frame` gains `frame_id` (the client's number, or a file session's
own count from zero) and `timing` (the four durations, in milliseconds).
`StartCamera` gains `frame_ids`; when true, each binary frame starts with a
4-byte little-endian id, and a message too short for its id is dropped with a
warning like a frame that will not decode. `Ready` gains `environment` - the
server's chip, device, commit and dirty state - so a client can write a report
that says where the estimator ran. The viewer's zod schemas read all of it.

Serialisation is reported for the *previous* message. A message cannot carry
the time it took to encode itself, and encoding it twice to find out would be
the cost being measured. For a distribution over a session the one-frame
shift is immaterial, and the field is named so nobody pairs it wrongly.

Queue wait is stamped by the mailbox, not the producer: the latest-only slot
records `perf_counter()` on `put`, and the wait a frame is charged with is
the time it sat there before the model took it. A file session stamps at
hand-over to the read-ahead queue for the same reason.

## Two clients, one report schema, one summary

`unproject.benchmark.live` defines the report: rows (`LiveSample`), context,
and a summary that the model refuses to store unless it equals what
`summarise_live` computes from the rows. Drops are the server's own count,
summed over the answers that reported them; a frame sent and neither answered
nor reported is *unaccounted* - in flight at the end, or dropped after the
last answer, which has no messenger. The test that found that edge is in
`test_live_measurement.py`.

**Transport**: `unproject-live-benchmark` (`unproject_api.measure`) plays a
clip into the camera endpoint over a real `websockets` connection, frames
JPEG-encoded up front at the viewer's own 1280-px width and 0.75 quality,
sends scheduled on absolute times so a slow send does not shift the schedule,
a receiver task pairing answers, a grace period for the last ones. `--spawn`
starts the API on the given port and stops it. It writes the report itself.

**Browser**: `?measure=1&replay=<clip>&fps=N[&bodies=1][&frames=N]` plays
the clip in the source pane and captures it into a canvas as a camera is
captured - same `captureFrames`, same size and quality - with the id prefix
and a `LiveMetrics` collector recording sends, answers, and the first
`requestAnimationFrame` after each answer was applied. When the clip ends or
the frame limit is reached, capture stops, the last answers get 2.5 s, and
the rows go to `POST /api/live-metrics`; the server computes the summary,
refuses a second submission of the same session, and keeps the file. The
status bar names it. The run list's auto-open is suppressed in measure mode
so it cannot tear the replay down.

"Presented" is the first animation frame after the bodies reached the
timeline. The live clock then trails the newest frame by its lag (0.4 s) by
design; the lag is in the report's config, and the protocol says to add it in
the reading rather than fold it into a number that would mean two things.

**Summary**: `unproject live-summary` tabulates every report under
`artifacts/live/`, transport and browser labelled and never joined.

## Verified

- API: a numbered camera frame is answered with its number and timings
  (`serialize_previous_ms` null on the first, set on the second); a bare
  JPEG stays valid when ids were not asked for; a too-short numbered frame is
  dropped and the session continues; a file session numbers frames from
  zero by what it sent; `POST /api/live-metrics` keeps a report with a
  server-computed summary, refuses a second copy (409), a draft carrying a
  summary (422) and an impossible row (422).
- Transport client: frames encoded at the viewer's size; a session against
  a small WebSocket server speaking the protocol pairs answers with sends and
  accounts for drops and the unaccounted tail.
- Pipeline: the row invariants (an answer cannot precede its send;
  presentation follows reception), the summary's arithmetic, the
  summary-matches-rows rule, unique ids, write/read.
- Viewer: 105 tests; the measured session asks for ids, prefixes each frame,
  pairs answers on the client clock, keeps server timings, ignores an answer
  it did not send, and sends bare frames when not measured; the URL parser.
- Full gate from the clean environment.

## Not done here

No measurement. The protocol names the M3 Pro; the six sessions - two rates
and two clients in skeleton mode, plus mesh at 1 fps - are the author's to
run from a clean commit, then `unproject live-summary`. A pilot of the
tooling on another machine proves the path and is not a thesis number.
