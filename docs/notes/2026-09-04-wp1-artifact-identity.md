# 2026-09-04 - WP1, making artifact identity and publication trustworthy

WP1 of ADR-0002. Two defects, both of the kind that produce a plausible number
from the wrong inputs rather than an error.

## A path was being used as an input identity

`ingest_hash` was `config_hash(config.ingest)` and nothing else. The content
digest already existed - `SourceMetadata.content_sha256`, computed by both
sources - but it was only *recorded in* the artifact, never part of *addressing*
it. So a second video dropped at `datasets/crowd/clip.mp4` would have been
served the first one's frames, and every downstream hash chains off the ingest
directory name, so poses and bodies would have inherited the same lie.

Now `ingest_hash = sha256(config_hash : content_sha256)`. Because preprocess,
pose and bodies already chain off `ingested.artifact_dir.name`, the propagation
came for free; the only place that needed real work was
`pose.resolve_artifact_dir`, which stage [5] uses to find two runs without
executing either. It could be a pure function of the config text before and
cannot be now - it takes a `config_dir` and reads the source. That cost is a
read of the file: 5 ms for the 8.3 MB reference clip, about 1.7 GB/s, paid on
every invocation including the ones that end in reuse.

Approved before implementing, per ADR-0002 §15: the identity change relocates
every artifact directory. The 520 MB already in `artifacts/` is not deleted and
not migrated, it is simply no longer addressed. `artifacts/fixture/ingest` now
holds both `7e3dff555abb` (the old identity) and `74307396d90a` (the new one).

## A stage wrote into the directory a reader was reusing

Ingest `rmtree`'d the published `frames/` and rebuilt it in place, so an
overwrite that failed halfway had already destroyed the good artifact. The
descriptor was correctly written last, which meant a half-built directory did
not read as complete - but only because nothing had checked whether the frames a
complete descriptor names were actually there.

Both are now handled by `run/publish.py`: every stage builds in a sibling
`.<name>.staging-<id>` directory and moves it into place at the end. A rename
cannot replace a non-empty directory, so the previous artifact steps aside first
and is deleted only once the new one has landed; if that second rename fails the
first is undone. A run that raises publishes anyway when there is nothing to
lose, so its manifest lands where the user will look - unless a complete
artifact is sitting there, which is kept untouched instead.

`run/reuse.py` is the other half: before trusting a descriptor, check that every
file it references exists at the size it declares.

## What the sizes cost

Frames and keyframes did not record their own size, so ingest went to schema 3
and preprocess to schema 3, both adding `byte_length` to their per-file record,
and pose to schema 2 adding `points_byte_length`. Bodies needed no schema change:
its sidecar layout already declares `byte_length` per array, so the expected
total was already derivable.

This was the half of the work I was least sure was worth it, and the truncation
tests settled it: a descriptor validates fine against a PNG that a full disk cut
in half, and existence alone does not catch that.

## Not done here

Stage [B]'s reuse path is checked by the same code as the other three, but there
is no end-to-end test of it, because no test can run the estimator without the
weights - a constraint the suite already had. It is covered at the level of
`BodiesArtifact.expected_files()` and `unusable_reason()` instead.
