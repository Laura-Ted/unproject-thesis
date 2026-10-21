# 2026-09-08 - Draggable panes, and a delete button

Two operator requests: remove runs that are no longer wanted, and make the panes
resizable because the track statistics did not fit in a 17rem rail.

## Deleting is the one destructive thing the server does

ADR-0002 §10.5 forbids deleting artifacts without the user asking. The button is
the asking; what stops it meaning more than it says is three checks in
`resolve_run`, all of which have to hold:

- it resolves inside `artifacts/`, so `../` buys nothing;
- it is a directory;
- it holds a `bodies.json`.

The third is the one that matters. Without it the endpoint would remove any
directory in the tree, including the ingest artifact holding every decoded
frame. Verified against a running server: `crowd/ingest/0140aa2c6860` came back
`400 does not hold a bodies.json, so it is not a run`.

The viewer confirms before calling, naming the run and what goes with it.

**A test found a real defect in the first version.** The success handler was
guarded by the ownership turn, like every other network completion in the app.
That is wrong here: the run is gone from disk whatever the operator did next, so
dropping the list update would leave a row pointing at a file that no longer
exists. Both updates name the run they mean, so they are safe out of turn; only
`setBusy(null)` still checks.

## calc() cannot multiply an fr

The pane splitter was written as `grid-template-columns: calc(var(--source) * 1fr) 3px 1fr`.
Browsers reject `fr` inside `calc()`, and a rejected `grid-template-columns`
takes the whole declaration with it - so the splitter came out **zero pixels
wide**, invisible and undraggable, and the panes fell back to auto sizing.

Nothing caught this: it type-checked, it linted, the tests passed, and the page
rendered. It was only visible in `getComputedStyle`, which said `805px 0px 300px`
where the middle column should have been 3.

Now a percentage, `var(--source, 50%) 3px minmax(0, 1fr)`, and a test asserts
both halves of the lesson through `CSS.supports` so the next person does not try
the same thing.

The rail splitter worked all along, because a plain `var(--rail, 17rem)` needs no
`calc`. That is why it looked half-working rather than broken.

## Widths are remembered

`localStorage`, wrapped in try/catch at both ends: storage throws rather than
being absent where a browser blocks site data, and a viewer that will not start
because a preference is malformed is worse than one that opens at its default
width. Tested against broken JSON, a stored non-object, and storage that throws.

## Verifying the delete needed a second server

The API process on port 8000 had been running since 3 September and does not
reload, so it answered the new route with `405`. Started a second instance on
another port rather than killing the operator's, ran the four cases against it,
then restored the run it removed from a backup - deleting somebody's artifact to
test a button is not a decision to make on their behalf.


## Two smaller things, later the same day

**The clip no longer loops.** It is the clock for the 3D pane, so restarting it
jumped the bodies back to the beginning under whoever was studying the end. It
now stops on the last frame, and the bodies stop with it.

**Dragging the camera now says so.** The two placed viewpoints are positions,
not locks - orbiting away from Source view left the button still claiming you
were there, and left you with the source lens, about eleven degrees on the crowd
clip, which is a miserable thing to orbit in. A drag switches the mode to Free,
which also restores the ordinary fifty-degree lens.

Wired through drei's `onStart`, which is a pointer press: the programmatic
`update()` in `Viewpoints` does not trip it, so clicking a viewpoint button does
not immediately undo itself.

Tested at the App level, where `Stage3D` is already mocked, by exposing
`onUserMove` as a button. What drei does with the callback was checked by
reading drei's source rather than asserted: a synthetic pointer event does not
reach three's `OrbitControls`, because it calls `setPointerCapture` with an id
that never belonged to a real pointer. Worth knowing before someone spends an
afternoon on it.


## The clip restarted whenever anything changed

Reported as "the video restarts every time I change viewpoint", and the
viewpoint was a red herring: it restarted on any re-render of `App`, which a
slider or a checkbox also causes.

Two things together:

- `toVideoSource` built a fresh object on every render, and `onElement` was an
  inline arrow, so `SourceVideo`'s effect - which depends on both - re-ran every
  time;
- that effect assigns `video.srcObject`, and **assigning `srcObject` runs the
  media element load algorithm even when the value has not changed**, which
  rewinds a playing clip to zero.

Fixed at both ends. The assignment is guarded on both branches, so the element
is never reloaded without cause whatever the caller does; and the source object
and the callback are memoised, so the effect stops re-running for no reason.

The guard alone would have fixed the symptom. Both are correct on their own -
one protects the media element, the other stops the churn that also drove a
`setVideoEl` round trip per render - so both are in.

Verified in the browser: the clip holds at 9 s across three viewpoint changes,
two sliders and a track checkbox. It used to go back to 0 on each.
