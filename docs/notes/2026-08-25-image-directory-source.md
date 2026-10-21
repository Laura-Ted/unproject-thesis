# 2026-08-25 - ImageDirectorySource

Second `FrameSource`. Motivation is not the feature itself: with one
implementation, the protocol was an untested guess. Writing a second one while
the first is still fresh is cheaper than discovering the interface is wrong in
six months with half a pipeline hanging off it.

The concrete use is public datasets. Mip-NeRF 360, Tanks and Temples and the
rest ship as numbered image folders, not video. Without this, any comparison
against published numbers is impossible.

## The abstraction did leak, roughly where expected

`SourceMetadata` was written against a video file. Two fields did not survive
contact with a directory of images:

- **`fps`.** Images carry no frame rate. It now has to be declared in the config
  (`image_fps`) and is recorded as declared, not measured. For a genuinely
  unordered photo collection it is a pure convention and the timestamps mean
  nothing. That is worth knowing before stage [3] treats them as a motion cue.
- **`content_sha256`.** There is no single file to hash. It is now a digest over
  every filename and its contents, in order. A rename changes it, which is
  intended: the order is part of what the pipeline consumes.

`codec` survived unchanged, which was a surprise. JPEG and PNG are codecs, so
`codec: "jpeg"` on an image directory is literally true. A directory holding
both reports `"mixed"`.

Added one field, `kind: "video" | "image_directory"`. It earns its place because
stage [3] will need it: video frames are temporally ordered and admit sequential
matching in COLMAP, an unordered photo collection needs exhaustive matching. A
reader cannot recover that from the other fields.

## Natural sort, not lexicographic

`frame2.png` sorts after `frame10.png` lexicographically. A dataset that does not
zero-pad would be silently reordered, and a scrambled frame order is invisible to
every later stage - the same shape of failure as the timestamp off-by-one from
yesterday. Sorting splits digit runs and compares them numerically, which agrees
with lexicographic order on zero-padded names and fixes the rest.

Verified end to end on a 10-image folder named `frame1..frame10`: with
`frame_stride = 2` the artifact holds source indices 0, 2, 4, 6, 8, which are
`frame1, frame3, frame5, frame7, frame9`.

## Schema version 2 and what happens to old artifacts

`kind` is required, so an artifact written before today cannot be parsed.
`INGEST_SCHEMA_VERSION` went to 2 and `IngestArtifact` now rejects any other
version.

`run_ingest` treats an unreadable artifact as an absent one and runs the stage
again, rather than failing. An artifact is always reproducible from its config,
so a stale schema is not something the user should have to clear by hand. The
same path covers a truncated or corrupt file.

## Renamed `video_path` to `source_path`

It accepts a directory now, so the old name was a lie. This changes the config
hash of every experiment, which moves every artifact directory. Done now, while
the only experiment is the fixture and nothing is published.

The source is chosen from what the path is on disk: a directory gets
`ImageDirectorySource`, a file gets `VideoFileSource`. The config does not repeat
the distinction, so it cannot contradict the filesystem.

## Decisions taken

- Images are re-encoded to PNG on the way into `frames/`, exactly as video
  frames are. Copying a JPEG dataset through unchanged would be faster and
  smaller, but stage [2] would then have to handle two formats. Uniformity is
  worth the disk.
- All frames in a directory must share a size; a mismatch is an error. The
  schema carries one width and height, so anything else would be a lie. The
  check happens during iteration rather than in `metadata()`, since validating
  up front means decoding the whole directory twice.
- Non-image files are ignored, not an error. Dataset folders routinely carry a
  `poses.txt` or a stray `.DS_Store`.
- `file_sha256` moved to `ingest/hashing.py`. Two sources computing input
  identity differently would make their artifacts incomparable.

## Open

- `RTSPCameraSource` is still unbuilt, and is the one that will stress the schema
  hardest: a live stream has no file to hash and no known frame count. Expect
  `content_sha256` and `reported_frame_count` to need widening when it lands.
- No DVC yet.
- `datasets/fixture/clip.mp4` is still `testsrc2` and not a reconstructable scene.
