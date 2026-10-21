# 2026-09-05 - WP5, measuring the thing the project is about

WP5 of ADR-0002. RQ3 is the research question with an actual finding behind it:
bodies that look right from the camera are strewn across ten metres of ground
from a quarter turn away. Until now that was an observation. This makes it a
measurement.

## Positions are measured from the camera, not from the room

The obvious annotation format puts floor markers in room coordinates. That needs
the camera's pose in that room, which nothing in this project estimates - stage
[3] is preliminary work and does not run on this footage. So markers are
measured with a tape from directly under the camera and recorded in the
artifact's own frame: camera-relative OpenCV metres.

The cost is real and worth writing down: the method only describes scenes shot
from one fixed camera. It buys not having to introduce a coordinate frame that
would itself need estimating, and every marker carries its own uncertainty so an
error smaller than the tape is visibly not a result.

## Two implementations of one filter

The RQ3 numbers are computed in Python; the filter being scored lives in
TypeScript, in the viewer. Doing the analysis in TypeScript would have put it
away from every other analysis in the project; porting the filter duplicates it.

Ported, and pinned. `packages/pipeline/tests/fixtures/one_euro.json` holds 24
samples and the outputs both implementations must produce, and both suites read
it. Verified in both directions: nudging the viewer's depth cutoff from 0.6 to
0.65 fails two viewer tests, and the same change on the Python side fails two
pipeline tests. Without that fixture the two would drift and one thesis number
would quietly move.

## Reprojection consistency is not accuracy

ADR-0002 §4 asks for "2D reprojection consistency" and there is no 2D ground
truth to compare against. What is checkable with what the artifact holds: does
the 3D root, projected through that frame's own recorded intrinsics, land inside
that person's own recorded 2D box.

That catches a coordinate convention read backwards or intrinsics recorded at
the wrong resolution. It does not catch a body in the wrong place, and the
metric's docstring says so, because a number named "consistency" is exactly the
sort of thing that gets quoted as accuracy later.

## What "no interpolation" cost

Velocity between consecutive detections is only computed when they are less than
0.5 s apart. A wider spacing is a gap - the person was not found - and dividing
the distance by the elapsed time there would describe the gap rather than the
person. Skipped pairs are counted into `gaps_skipped` and reported, so a metric
computed over a fragmented track cannot pass as one computed over a whole one.

## Figures, and why they are byte-stable

Three, drawn from the summary rather than from anything held aside, so a figure
cannot disagree with the table beside it: position error per marker with the
marker's own uncertainty drawn across each pair of bars, apparent velocity per
axis raw against smoothed, and depth jitter as a percentage of range.

matplotlib was added as a dependency of the pipeline rather than as an extra -
approved first, per ADR-0002 §15. ADR-0002 §9.1 lists reproducible tables and
figures as a deliverable, so it is on the required path; the alternative was
about two hundred lines of hand-rolled SVG, which is worse than one small
dependency. It is imported lazily and pinned to the Agg backend, so the API
process never loads it and nothing looks for a display.

SVG carries a timestamp and a random id salt by default. Both are pinned, and a
test writes the same summary twice and compares the bytes; removing either pin
fails it. A figure whose bytes move on every run cannot be checked into a thesis
repository and trusted to be the one the numbers came from.

Checked by rendering a synthetic corridor: at 3 m the depth error sits under the
marker uncertainty, at 10 m it is four times it and eleven times the lateral
error, and someone standing still at 10 m shows 1.1 m/s of apparent depth
motion. That is the argument, and the figure makes it at a glance.

## Not done here

**Any actual measurement.** No controlled recording exists. The scene shape and
the annotation format are fixed in `docs/experiments/protocol.md` and
`experiments/rq3/corridor.example.toml`, and the template holds placeholders
rather than numbers. Recording it is WP6.
