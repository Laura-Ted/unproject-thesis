# 2026-09-04 - WP0, consolidating the source of truth

WP0 of ADR-0002. No code changed; this is documentation moving into the
repository and two factual corrections.

## docs/ was never in git

`docs/` was untracked and ADR-0001, the notes and the pose-backend brief were
all under `tmp/`, which is gitignored. So the two things ADR-0002 calls the
source of truth - the accepted decisions and the dated notebook - existed only
on this disk. `git log -- docs` was empty. They are now under `docs/adr` and
`docs/notes`, moved verbatim.

The brief got a preamble saying the question it asks was answered by
LingBot-Map. Its own header still reads `Status: open question, blocking`,
which was true on 2026-08-26 and is the point of keeping it.

## The M4 Pro was wrong

`experiments/fixture-lingbot.toml` attributed the kv-cache memory measurement to
an M4 Pro. ADR-0001 and the brief both say M3 Pro, 18 GB, and the machine
reports `Apple M3 Pro` with 19327352832 bytes. Corrected to M3 Pro.

Only the attribution was wrong, and the number it attributes - 64 peaking near
16 GB against 16 holding flat around 10 GB - was not re-measured. It was taken
on this machine, so it stands, but it is a comment rather than a recorded
observation and it is not evidence for anything in the thesis. WP4 is where
numbers like that start being collected properly.

## Two weeks of work with no notes

The notebook stops at 2026-08-25, stage [3]. Everything after it - the LingBot
backend, stage [5], stage [B], the API, the viewer, the mesh path, the root
filter, the socket - is 19 commits on 27 and 28 August with nothing written
down. The findings survive as README prose and config comments, which is where
they were read back from today; the reasoning, the alternatives tried and the
dead ends did not survive at all.

They are not reconstructable now, and inventing dated notes for them would be
worse than the gap. Recording the gap is the honest option. Notes resume from
this one.

## References removed rather than repaired

The README pointed at four files that do not exist and never did:
`docs/figures/bodies_overlay_source.png`, `docs/figures/bodies_mesh_source.png`,
`docs/figures/bodies_45_off.png` and
`docs/notes/2026-08-28-what-it-costs-and-what-it-cannot-say.md`. The claims
around them stand on their own, so the pointers were dropped. `docs/figures/`
stays uncreated until something reproducible generates a figure into it, which
is WP5 and WP6.

`experiments/fixture-lingbot.toml` pointed at `docs/notes/2026-08-27`, one of
the notes never written. It now points at the README section that holds the
same reasoning.

## Left where it was

`tmp/docs/guides/stage-2-preprocess.md` is still in `tmp/`. It is an
implementation guide rather than a note, ADR-0002 §6.1 has no `docs/guides`, and
it is written in Italian where the repository is English throughout. Moving it
would import both problems. Deciding its fate is not WP0's job.
