# 2026-08-25 - Stage [2] Preprocess

Keyframe selection, resizing, and the chained artifact hash. Undistortion is
specified but deliberately unimplemented, for reasons below.

Verified end to end on a 4 s synthetic pan at 30 fps: 120 frames in, 14
keyframes out, measured disparities clustered at 10.7-10.8 px against a 10.0 px
threshold.

## Undistortion cannot happen here yet

The stage list assigns undistortion to stage [2], but correcting lens distortion
requires the distortion coefficients, and those are produced by stage [3] - both
COLMAP and DA3 estimate them alongside the poses. At stage [2] they do not exist.

The only cases where they would is a separately calibrated camera
(`cv2.calibrateCamera` against a checkerboard) or a fixed, known camera, which is
the university setup later on. For handheld phone video, the current development
path, there is nothing to apply. COLMAP handles distortion internally anyway.

`calibration_path` therefore exists in the config and is rejected with an
explanation if set. Writing undistortion code with no calibration file to test it
against would be worse than not having it.

This is an ordering limitation of the pipeline, not an oversight, and belongs in
the thesis as such.

## The blur threshold is a percentile, not a number

Variance of the Laplacian is the standard blur metric (Pech-Pacheco et al.,
2000), but its scale is entirely scene-dependent. On the fixture the blur floor
resolved to 639; a flat-textured scene would sit two orders of magnitude lower.
Any absolute threshold works on one scene and destroys another.

`blur_percentile` is resolved against the sequence being processed, and the
absolute value it landed on is written into the artifact as `blur_floor` - the
percentile alone does not tell you what actually applied.

Known limitation: the percentile is global, so a take whose lighting changes
significantly part-way through gets a floor that is not representative of either
half. A rolling median over a window would be more robust. Not doing it yet.

## What tracked_ratio actually measures

Measured, rather than assumed:

| pair | tracked_ratio |
|---|---|
| translated 20 px | 0.84 |
| translated 160 px | 0.38 |
| translated 260 px | 0.10 |
| **entirely different scene** | **0.87** |
| **uniform gray frame** | **0.85** |

It falls as content leaves the frame, which is what `min_tracked_ratio` uses it
for and that part works. It does **not** detect wrong tracking: Lucas-Kanade's
status flag means "converged", not "converged on the right point". An unrelated
frame still scores 0.87, and a blank frame yields a confident, meaningless 16 px.

What saves the algorithm is the blur floor, which rejects a blank frame long
before tracking would. Worth knowing, because it means `tracked_ratio` must not
later be reused as a confidence measure. Pinned by a test.

Separately: a frame with no corners at all is now reported as `measurable=False`
rather than as `tracked_ratio=0.0`. Those are different statements, and the
selector was relying on the coincidence that both compare below the threshold.

## The chained hash

`preprocess_hash = sha256(ingest_hash + config_hash(preprocess))`.

Not the stage config alone. Changing `frame_stride` in stage [1] changes the
frames going in, so it must change the keyframe identity even when stage [2] is
untouched. Verified: with stride 1 the run landed in `dfb7d6247275`; with stride
3 the identical preprocess config landed in `4d26ffb72e38`.

## Two test lessons

**Thresholds on step boundaries test float noise, not logic.** Optical flow
measures a 4 px/frame pan to about 1e-5 px. With `min_disparity_px = 8.0` on that
sequence, some pairs measured 8.000006 and others 7.999992, so the selection
flipped on rounding. The fix is in the tests, not the code: thresholds now sit
between two achievable displacements (10 px against a 4 px step).

**Type annotations do not pick pytest fixtures, names do.** Renaming
`make_video: MakeVideo` to `make_video: MakePanningVideo` type-checked cleanly
and kept injecting the old flat fixture. mypy cannot catch this because fixture
resolution is by parameter name. Cost about twenty minutes of debugging a stage
that was working.

The underlying cause was worth having: the stage [1] fixture writes almost
untextured frames, which was fine when the only assertion was a frame count.
Corner tracking needs texture, so stage [2] has its own panning fixture.

## Decisions taken

- `min_disparity_px` is stated in pixels of the **output** image, while the
  measurement happens at `metric_long_edge`. The stage converts between them, so
  tuning the working resolution for speed does not silently move the threshold.
- Metrics are measured at a fixed working resolution rather than natively, so
  values are comparable across scenes shot at different resolutions.
- The safety net counts **blur** rejections only. A stretch skipped for low
  disparity is the algorithm working - a static camera has no parallax, and
  forcing a keyframe there would add a degenerate pair.
- The first keyframe is the sharpest of `first_frame_window`, not frame zero. On
  the fixture it picked source index 6.
- A run ending with fewer than two keyframes is an error. Two views is the
  mathematical floor for triangulation, so one is a configuration mistake rather
  than a thin result. This replaced a check for an empty list that could never
  fire, since selection always returns the anchor.
- `INTER_AREA` for downscaling; `INTER_LINEAR` aliases beyond a factor of two.
- Every keyframe carries its `sharpness`, `displacement_px`, `tracked_ratio` and
  `forced` flag into the artifact. When a scene reconstructs badly, these are the
  difference between reporting a low score and explaining it.

## Open

- **The held-out split is still undecided.** It has to exist before stage [4] and
  cannot be changed after results are produced without invalidating them. The
  natural home is a `held_out: bool` on `KeyframeRecord`, derived from the seed.
  Not added, so adding it later is a schema bump.
- `min_disparity_px = 12.0` is a starting point, not a justified value. Finding
  one is an experiment: run the stage at three settings on a real scene and record
  the keyframe counts and their spacing.
- Pose-aware reselection (select roughly, estimate poses, reselect) would be
  better, but each pose backend would then produce a different selection and the
  DA3-vs-COLMAP comparison would no longer be measuring one variable. Not doing it.
- The global blur percentile, as above.

---

## Addendum, same day: the held-out split

Decided and implemented, so it is fixed before any result exists.

**Where.** Marked at stage [2] as `held_out: bool` on each `KeyframeRecord`,
rather than left to stage [4]. It is then visible by opening the artifact,
inspectable by hand, and reproducible without re-running training.

**Which.** Every 8th keyframe by position. Not an invented number: it is the
convention of Mip-NeRF 360 and of Kerbl et al., so the resulting scores are
directly comparable with published ones and the choice needs no defence of its
own beyond the citation.

Deterministic by construction, so it needs no seed and cannot drift between runs
of the same config. The seed stays reserved for things that are actually random.

**Guard.** A split that leaves fewer than two keyframes to train on is an error,
not a thin run. `held_out_every` is also bounded at 2, since 1 would withhold
everything.

**On by default** (`held_out_every = 8`). Off by default would make it easy to
produce a run that cannot be honestly scored and only notice at stage [5].
`null` disables it and the CLI says so out loud:

```
kept 14 of 120 frame(s) in artifacts/pan/preprocess/e84336c97921
      12 for training, 2 held out (every 8)
```

**Schema version 2.** `held_out` is required, so artifacts written before today
cannot be read. The existing "unreadable artifact is re-run rather than an error"
path covers it, the same as when `kind` was added to stage [1].

**Limitation to carry into the write-up.** In densely sampled footage a withheld
view can sit almost on top of a training view, which makes the test easier than
it looks and inflates the scores. This is a property of the every-Nth protocol
that the whole field shares, not something the split can fix. Better to state it
than to have it asked.

Verified on the 4 s pan: 14 keyframes, positions 0 and 8 withheld, 12 left for
training. Changing `held_out_every` moves the artifact to a different directory,
so results under two splits cannot silently reuse each other.
