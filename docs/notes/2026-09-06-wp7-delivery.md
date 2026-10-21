# 2026-09-06 - WP7, hardening the handover

WP7 of ADR-0002. Six tasks; five done, one that needs a screen recorder and a
person.

## CI runs on Linux, not on the target platform

`.github/workflows/ci.yml`: two jobs, lint and types and tests and configs and
the lockfile on one, the whole frontend on the other. Neither needs the weights
or any footage - the tests synthesise the video they use.

Linux rather than macOS even though the project targets Apple Silicon, because
**this repository is private** and macOS runners bill at ten times the rate. What
runs there never touches Metal, so the platform costs nothing. Performance is
deliberately not measured in CI: a shared runner cannot produce a number worth
reporting, and the benchmark is run by hand on the machine the protocol names.

All four action versions I first wrote were out of date - checkout@v5 against a
current v7, setup-uv@v7 against v10. Checked against the registry rather than
guessed, which is the only way that check ever works.

## A clean checkout found a real defect

Cloned to a fresh directory with no datasets, no models, no artifacts, and ran
the documented commands. Confirmed: 3.7 MB of repository, no footage or weights
committed, `uv sync --dev` then every check passes, and `uv run unproject-api`
builds the viewer itself and serves it.

Then, without the `multihmr` extra:

```
GET /api/body/topology  ->  500 Internal Server Error
```

That fails WP7's own acceptance criterion outright. A reader who has run plain
`uv sync` meets that route before anything else, and it tells them nothing.

Worse, the socket was the same, **and that one was my doing**: WP2 replaced
`str(error)` with a fixed message so a stack trace or a local path could not
reach a browser. Correct in general, and it also swallowed the one message that
was safe and useful - the name of the extra to install.

Both fixed. `BodyEstimationError` is now caught separately from everything else:
it names an install command and says nothing about this machine, so it goes on
the wire. The topology routes answer 503 with the same hint. Three tests cover
it, including that finished runs are still served with no model at all.

The lesson is narrower than "be careful with error messages": a stable
user-facing error is right, and it needs an exception type that is explicitly
allowed through it. Blanket suppression takes the good ones with the bad.

## Diagrams that were checked, not just written

`docs/architecture.md` has four: what talks to what, the pipeline with the
deferred and preliminary stages marked, how an artifact hash is chained, and the
concurrency as a sequence. Mermaid, so GitHub renders them and there is no
dependency.

Rendered all four through mermaid 11 before committing - 4 rendered, 0 syntax
errors. A diagram that only exists as text nobody has parsed is a diagram that is
probably broken.

## Reproduction guide

`docs/reproduce.md`, written after running each step from the fresh clone rather
than from memory. It ends with what **cannot** be reproduced from the
repository: the crowd clip, the RQ3 case study, the person-count factor, and the
live transport figure. Naming those is the point of the document.

The pedestrians clip is fully reproducible - URL, sha256 of the original, and the
transcode command - so a reader has one scene they can actually fetch.

## Not done

**The recorded demo.** ADR-0002 §12 WP7 task 5 asks for a short recording showing
source view, off-axis, raw against smoothed, skeletons and meshes. That needs a
screen recorder and someone driving the viewer. Everything it would show works
and was verified interactively today; the recording is yours to make.

## Left alone deliberately

`package-lock.json`, 706 bytes, untracked, dated before any of this work. It is
the non-authoritative lockfile ADR-0002 §11 names, and it is debris from a
one-off `npm install`: never committed, and prettier resolves through
`pnpm-lock.yaml` like everything else. WP7 task 6 allows removing it "after
confirming they are not user work" and the confirmation is above, but it is a
file on someone's disk that costs nothing to keep, so it is reported rather than
deleted. `rm package-lock.json` is the whole of it.

`docs/guides/thesis-project-guide.it.md` is untracked too, and is the user's own
writing. Not mine to move or commit.
