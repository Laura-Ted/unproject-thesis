# 2026-09-05 - WP6, running it, and what running it broke

WP6 of ADR-0002. The protocol was frozen first, then eight matrix cells were
measured, then the RQ3 analysis was pointed at real footage for the first time.
The third of those found three defects in metrics that had passed every
synthetic test in WP5.

## What was run

`docs/experiments/protocol.md` is **Frozen** as of today, before the first
measured run. Every cell uses 5 warm-up frames, 40 measured, 3 repetitions: 120
steady-state samples each, on an Apple M3 Pro through MPS.

Eight cells, three factors fully crossed - aspect ratio, resolution, output
mode - over `datasets/crowd/clip.mp4` and three controlled ffmpeg
transformations of it. Regenerate the table with `unproject benchmark-summary`.

| | 16:9 | square |
|---|---|---|
| full resolution | 0.603 s (16 people) | 0.870 s (8 people) |
| half resolution | 0.550 s (15 people) | 0.856 s (7.5 people) |

- **Resolution barely matters.** Halving the long edge moved the median 8.7% at
  16:9 and 1.6% square.
- **Shape matters a great deal.** Square costs 44% more at full resolution and
  55% more at half - while showing *half as many people*, because the square is
  a centre crop. Fewer people and still half again the cost is what rules the
  people out.
- **The mesh costs nothing resolvable.** Against the skeleton: -3.2%, +3.3%,
  +0.03%, +0.32%. The sign flips, so the effect is under the run-to-run spread.
- Cold start is 7.4-10.3 s per process with the weights in the page cache, 33.4 s
  on the first run of the day. Opening a *second* session inside one process
  costs microseconds, because the model is kept and only the tracker is new.

## The person-count factor was not measured

RQ2 asks for low, medium and high person counts. That needs three scenes and one
exists. No footage was recorded, nothing is claimed, and §3.1 of the protocol
says so in place rather than the factor being quietly dropped.

Two README claims went with it: that nine against fifteen people cost seven per
cent more, and that patch count correlates with time at +0.990. Neither traces
to a reproducible run. They are withdrawn rather than restated.

## The RQ3 case study was not recorded

It needs floor marks measured with a tape and a consenting participant. That is
physical work, not code, and it has not been done. **No position error is
reported for anything.**

What could be measured without ground truth was:
`experiments/rq3/crowd-uncontrolled.toml` declares the crowd clip as an
uncontrolled scene with no markers, and the analysis produces the half of RQ3
that needs none. Over 2665 measured steps: lateral 0.16 m/s median, depth
**4.66 m/s** - twenty-nine times faster along the axis the camera cannot see -
jitter at 3.1% of range, filtered depth 1.28 m/s, 62 tracks with a median life of
4.2 s.

## Three metrics that only real footage could break

**Reprojection consistency read 58% and should have read 100%.** The failures had
the projected pelvis horizontally inside the box and vertically below the frame:
people walking out of the bottom of the shot, whose legs are cut off, so the box
stops at the image edge while the pelvis is genuinely below it. Split by whether
the box touches the frame edge, bodies fully in frame are **112 of 112**. A
single figure there was not a hard number, it was a wrong one.

**Every motion metric was unreachable without annotations.** They were only
computed over annotated intervals, so a run with no marks produced nothing - not
even the jitter, which needs no ground truth at all. There is a whole-run figure
now.

**And when it was reachable it returned zero.** `axis_motion` walked the
detections in time order and skipped pairs whose track had changed. With a dozen
people in a frame, consecutive entries *are* different people, so every pair was
skipped: 2832 detections, 0 steps. It groups by track now. The synthetic tests
never caught it because they all had one person in them.

**The response delay was a confident 0.0 s over 62 interleaved tracks.** Two
tracks are not one signal. It is reported only for a single continuous track now.

All four are covered by regression tests written from the real failure, and the
RQ3 summary schema went to 2.

## What this changes about the claims

Reviewed against ADR-0002 §3.3, as WP6 task 7 asks. The README now separates
numbers that regenerate from a command from numbers that do not, and the ones
that do not are either withdrawn or labelled. Nothing in the repository now
quotes a performance figure without a way to reproduce it, except the live
transport pair, which is labelled as such because the protocol deliberately
keeps the browser out of the benchmark.

## Still open

- **Consent for the crowd clip is unresolved.** It shows identifiable people in a
  public place and nothing here records where it came from or on what terms. It
  is gitignored and never leaves the machine, which satisfies storage and
  processing, and does not settle whether a frame of it may appear in a
  submitted thesis. Protocol §8 says so and it needs answering before any figure
  drawn from that clip is published.
- The RQ3 controlled recording.
- The person-count scenes.
