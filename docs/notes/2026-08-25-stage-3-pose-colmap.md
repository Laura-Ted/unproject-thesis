# 2026-08-25 - Stage [3] Pose, COLMAP backend

The experiment's variable. `poses.json` is the contract both backends write, so
this note is mostly about the contract and the conversions into it.

Ran end to end: 180 frames, 58 keyframes, COLMAP registered 3 of them on a
deliberately degenerate synthetic scene (see below).

## The convention, and why this one

`colmap_world_to_camera_opencv`, recorded as a literal string in every artifact.

- extrinsics map world into camera: `x_cam = R @ x_world + t`
- quaternion order `(w, x, y, z)`
- camera axes `+X` right, `+Y` down, `+Z` forward

Chosen because COLMAP is one of the two backends, so the reference path needs no
conversion and has one fewer place to be wrong, and because every 3DGS trainer
including Brush consumes this layout. DA3 will convert into it.

The string lives in the file rather than only in the source. A pose file whose
convention has to be inferred is a pose file that will eventually be read wrong.

## pycolmap orders quaternions the other way

`Rotation3d.quat` is Eigen's coefficient order, `(x, y, z, w)`. This project
stores `(w, x, y, z)`. Verified rather than assumed:

```
Rz(90 degrees) -> quat [0, 0, 0.7071, 0.7071]
(w,x,y,z) would be     [0.7071, 0, 0, 0.7071]
```

Both orderings produce a valid unit quaternion and a valid rotation matrix, so a
copy instead of a reorder gives poses that pass every schema check and place
every camera wrong. Two tests cover it: one asserts the reordered result matches
the known rotation, and one asserts that reading the array as-is does **not**.

## Two API facts that cost time

`Image.cam_from_world` is a **method**. An early probe tried to set it in the
`Image` constructor, got "attribute is read-only", and I concluded it was a
property. Reading it without calling returns the bound method, and the failure
surfaces three frames later as `'builtin_function_or_method' object has no
attribute 'rotation'`. `has_pose` next to it *is* a property.

Building a posed `Reconstruction` by hand needs COLMAP 4.x's rig and frame
objects, and a minimal text model fails to load without `rigs.txt`/`frames.txt`.
So the shape of the API is pinned off the classes instead:

```python
callable(pycolmap.Image.cam_from_world)
isinstance(pycolmap.Image.has_pose, property)
callable(pycolmap.Camera.extra_params_idxs)
```

Unusual as a test, but it catches the upstream change that would otherwise reach
the code as a runtime error on a real scene.

## No `Any` for an unstubbed extension

pycolmap is pybind11 with no stubs, so `mypy --strict` sees `Any` everywhere. The
project forbids that. Six narrow Protocols describe exactly what the backend
reads, with a single `cast` at the boundary. They double as documentation of the
dependency surface, and `test_pose_colmap.py` asserts the real classes still
satisfy them.

## TOML has no null

Found by running the pipeline, not by reading the code. The config documented
`held_out_every = null` and `max_long_edge = null`, which is not valid TOML - the
spec omits null deliberately. Omitting the key cannot substitute, since both
options have non-null defaults.

`0` is now the off sentinel, converted to `None` by a `before` validator.
Documented in the README and covered by a test.

## Decisions taken

- Partial registration is a result, not an error. The artifact records
  `keyframes_registered` against `keyframes_total`, and that ratio is a headline
  comparison number: a backend that solves half the views is not comparable on
  reconstruction quality alone.
- COLMAP returns disconnected pieces as separate reconstructions. The largest is
  kept and the rest counted as unregistered. Merging them would need a scale
  relation we do not have.
- `mean_reprojection_error_px` is optional. COLMAP minimises it; a learned
  backend has no equivalent, and a placeholder would put a meaningless number in
  a results table.
- The matcher defaults to `auto`: sequential for video, exhaustive for an image
  directory. This is the first real use of `SourceMetadata.kind`, added at stage
  [1] on the guess that stage [3] would need it.
- `single_camera = true` by default. A handheld take is one physical camera, and
  sharing intrinsics across keyframes conditions the solve far better than
  solving them per image.
- The sparse cloud is a binary PLY beside `poses.json`, not inline. Stage [4]
  initialises from it, and a hundred thousand points do not belong in JSON.
- The COLMAP database is deleted before each run. Left in place it would be
  matched against incrementally, and the result would depend on run history.

## The fixture scene does not reconstruct, on purpose

`testsrc2` rotated in plane is a flat pattern under pure rotation: no parallax,
no 3D structure. COLMAP registering 3 of 58 keyframes is the correct answer, and
it exercises the partial-registration path honestly. It is **not** evidence that
the backend works well, only that the plumbing does.

Real footage is now the blocking item for this stage, not code.

## Open

- **DA3 is unimplemented.** `backend = "da3"` is refused rather than silently
  substituted. It needs torch (~2 GB of wheels) plus model weights downloaded
  from HuggingFace, which is a machine-state change worth deciding on its own.
  The conversion side is ready: `invert_pose` and `opengl_to_opencv` exist and
  are tested, because camera-to-world and OpenGL axes are what learned models
  usually emit.
- Pose accuracy between backends (rotation and translation error) belongs to
  stage [5], not here. The geometry primitives it needs are in `pose/geometry.py`.
- COLMAP feature extraction and matching run on CPU. On a real scene this is
  minutes, and the per-stage timing is already in the manifest.
