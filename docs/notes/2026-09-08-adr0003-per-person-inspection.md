# 2026-09-08 - ADR-0003, one track at a time

Implemented ADR-0003, and accepted it in the same change: the status line said
`Proposed` and asking for the implementation is what settled it.

## What it does

The viewer lists the tracks in a finished run. Each row has the colour the body
is drawn in, when the track starts and ends, and how many frames it holds. A
checkbox hides it, **only** hides everything else, and picking one draws its
path through the scene as a plain white polyline.

Where `unproject rq3` has been run beside the artifact, the selected track also
shows its own numbers: distance, apparent velocity per axis, depth jitter as a
fraction of range, longest gap, shape drift.

## The path is the figure

Isolating one track and looking at it from 45 degrees off is the clearest
picture of the finding this project has produced so far. A person walking in a
straight line across a pedestrian street comes out as a jagged streak running
tens of metres into the distance. The 3% jitter figure and that line are the
same fact; one of them is legible without a table.

## Decisions worth recording

**No API route.** `artifacts/` is already mounted as static files, so the viewer
fetches `rq3.json` the same way it fetches `bodies.json`. That keeps selection
entirely inside the viewer, which ADR-0003 §3.4 asks for, and adds nothing the
server has to remember.

**The list is derived, not fetched.** It comes from the timeline the viewer
already holds, so it works for a live session too, where no analysis exists.
The statistics are the only part that needs `rq3.json`, and a missing file is
normal rather than an error - the list just carries fewer columns.

**The palette was duplicated.** `TRACK_COLORS` existed identically in
`BodyFigures.tsx` and `BodyMeshes.tsx`. The swatch beside a row has to be the
colour of the body in the scene, or the list actively misleads, so it moved to
`palette.ts` and all three read it. Three copies of a constant that must agree
is worse than the refactor.

**Hidden tracks are filtered in one place**, the effect that derives the frames
the render loop reads. A frame emptied by hiding stays in the timeline: dropping
it would shift the times, and the video is the clock, so the two panes would
drift apart.

## What was not built, again

Nothing here joins two tracks and nothing fills a gap. The line breaks where the
data breaks. ADR-0003 §4 forbids both, §5 says what would have to exist first,
and the measurements in §2.2 are why: shape separates people at 1.31x and torso
colour at 1.38x, which is not enough to build on.

## Tests

Eleven new ones, in `apps/web/src/tracks.test.ts` and `test_rq3.py`. Checked by
breaking the two invariants that matter - the visibility filter, and skipping
frames a track is absent from - and confirming five fail. Notably one asserts
that an emptied frame survives, and one that the path does not interpolate
across the frames a track was missing from.
