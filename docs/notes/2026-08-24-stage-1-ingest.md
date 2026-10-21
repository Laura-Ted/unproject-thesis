# 2026-08-24 - Stage [1] Ingest

First code in the repository. Scope: toolchain, repository layout, and a working
`VideoFileSource` behind the `FrameSource` protocol.

## Dead end: Python 3.14

The repository arrived with a PyCharm virtualenv on Python 3.14.6 (Homebrew).
Dropped before writing any code: neither torch nor pycolmap publish arm64 macOS
wheels for 3.13+, and building them from source is not a path I want to be on.
Pinned to 3.12 in `.python-version` and bounded in every `requires-python`
(`>=3.12,<3.13`), so the ceiling shows up where it bites rather than only in a
README.

The requirement was only "Python 3.11+", which is a floor. It is the ceiling that
turned out to matter.

## Finding: OpenCV frame timestamps are off by one

My first implementation read `CAP_PROP_POS_MSEC` before `read()`, assuming the
property points at the frame about to be decoded. It does not. Measured on
OpenCV 5.0.0, MJPG/AVI, 8 frames at 10 fps:

| index | POS_MSEC before `read()` | POS_MSEC after `read()` | true PTS |
|---|---|---|---|
| 0 | 0 | 0 | 0 |
| 1 | 0 | 100 | 100 |
| 2 | 100 | 200 | 200 |
| 7 | 600 | 700 | 700 |

Read after `read()`, the property is the timestamp of the frame just returned.
Read before, the whole timeline shifts by one frame and the last frame falls off
the end.

Worth noting how quiet this failure is. Nothing downstream breaks: the timestamps
stay monotonic and evenly spaced, so ingest, preprocessing and training all
succeed. It would only surface as a systematic bias in the stage [3] pose
estimates, which is the quantity the thesis measures. The test that caught it
asserts the value of the last timestamp, not just its ordering.

Two tests now pin the behaviour: exact timestamps on a known clip, and a fake
capture reporting a flat 0 ms to exercise the nominal-fps fallback.

## Decisions taken

- Frames are written as lossless PNG. JPEG at ingest would make the held-out
  evaluation of stage [5] partly a measurement of compression artifacts. Cost is
  ~1.1 MB for 30 frames at 320x240; a real capture will be much larger, and
  `artifacts/` is gitignored.
- Artifacts are keyed by the hash of the stage's config, not the whole
  experiment's: `artifacts/<experiment>/ingest/<hash12>/`. Changing a training
  parameter should not invalidate frames that are still good, which is what makes
  the stages independently resumable.
- The config hash is computed over the validated model, not the file bytes.
  Comments and key order do not change a run's identity; an applied default does.
- `ingest.json` is written last. While it is missing the directory is visibly
  incomplete, so an interrupted run cannot be mistaken for a finished one.
- The `uri` in `SourceMetadata` is the path as the config declared it, not the
  resolved absolute path, so artifacts stay comparable across machines.
- The manifest is written on the failure path too, with `status: "failed"` and
  the exception on one line. A stage that dies nine minutes in still says how
  long it ran and what stopped it, which is the run worth knowing about. The
  write is wrapped so an I/O error there cannot mask the original failure. The
  artifact file is still absent, so a failed directory does not read as finished.

## Fixture data

`datasets/fixture/clip.mp4` is a synthetic clip, generated locally:

```shell
ffmpeg -f lavfi -i "testsrc2=size=320x240:rate=10:duration=3" \
       -c:v libx264 -pix_fmt yuv420p datasets/fixture/clip.mp4
```

It exercises the plumbing and nothing else. `testsrc2` has no consistent 3D
geometry, so it cannot be reconstructed and must not be used as the regression
scene. Replacing it with real footage, and tracking it with DVC, is open work.

## Open

- No DVC yet; `datasets/` is only gitignored.
- `ImageDirectorySource` and `RTSPCameraSource` are unbuilt. The `FrameSource`
  protocol exists for them, but only one implementation is real.
