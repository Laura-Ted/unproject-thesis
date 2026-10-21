# 2026-09-05 - WP3, who owns the screen

WP3 of ADR-0002, in `apps/web`. Every one of these is the same bug wearing a
different hat: network work resolves in whatever order it likes, and the code
assumed the order it was asked in.

## Turns

`App.tsx` now keeps a counter in a ref. Every user action takes the next number
and holds a closure that asks whether it still has it; every completion checks
before touching `session`, `bodies`, `busy` or `error`. `teardown` bumps the
counter as well, so unmounting invalidates whatever was in the air.

What that fixes, concretely:

- opening run A then run B showed **A** if A's 45 MB happened to land second;
- a mesh fetched for run A was drawn on run B's skeleton, because only
  `setSession` was guarded by a run check and `setBodies` was not;
- an upload finishing after the operator went back to a finished run started a
  live session over the top of it;
- `getUserMedia` resolving after teardown started a session with a stream
  nothing would ever stop.

That last one needed more than ignoring the result: the camera light stays on
until the tracks are stopped, so a superseded permission now stops them.

## A closed socket was not a quiet socket

`LiveSession.close()` closed the socket and dropped the reference, but the
listeners stayed attached to the object. A message already in flight, and the
close event itself, still ran `update()` and pushed state into a session the
operator had left. There is a `closed` flag now, checked in `update` as the
last gate rather than the only one.

## Testing this needed a DOM

`vitest` ran with `environment: 'node'` and `include: ['src/**/*.test.ts']`, so
nothing could render a component and four of the five required cases were not
reachable. Added `jsdom` and `@testing-library/react` as **devDependencies** -
approved first, per ADR-0002 §15 - and extended the include to `.test.tsx`. The
bundle is unchanged; these never leave the development machine.

## The tests lied first

Written straightforwardly they passed against the broken code too. Two reasons,
both worth remembering:

- `await Promise.resolve()` a couple of times is not enough to flush the chain
  under test. `openRun` awaits twice and then React has to re-render, so the
  assertion ran *before* the stale completion could do damage. There is a
  `settle()` helper now that runs a real macrotask inside `act`.
- the first fixture was built with `as BodiesArtifact` and had the *live wire*
  person shape rather than the artifact's. A cast will agree to anything;
  `bodiesArtifactSchema.parse` said what was wrong.

Every test here was checked by disabling the guard it covers and confirming it
fails: six in `App.test.tsx`, five in `live.test.ts`. A test that passes either
way is not evidence, and the first version of these was not evidence.
