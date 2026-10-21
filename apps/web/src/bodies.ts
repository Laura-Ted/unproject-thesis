/**
 * The bodies timeline, and how playback samples it.
 *
 * A finished artifact and a live socket produce the same thing here: a list of
 * frames, each holding people with their joints. Everything downstream reads
 * that one shape, so the renderer has no idea whether the bodies came off disk
 * or out of a camera a moment ago.
 *
 * The sampling is pure and lives outside any component. It is what the render
 * loop calls every displayed frame, and it must never depend on new data having
 * arrived.
 */

import { z } from 'zod';

const cameraSchema = z.object({
  width: z.number().int().positive(),
  height: z.number().int().positive(),
  fx: z.number().positive(),
  fy: z.number().positive(),
  cx: z.number(),
  cy: z.number(),
  source: z.enum(['estimated', 'assumed', 'known']),
});

const personSchema = z.object({
  parameter_index: z.number().int().nonnegative(),
  track_id: z.number().int().nonnegative(),
  confidence: z.number().min(0).max(1),
  box_xyxy: z.tuple([z.number(), z.number(), z.number(), z.number()]),
});

const parameterArraySchema = z.object({
  name: z.string(),
  dtype: z.literal('<f4'),
  shape: z.array(z.number().int().nonnegative()),
  byte_offset: z.number().int().nonnegative(),
  byte_length: z.number().int().nonnegative(),
});

export const bodyModelSchema = z.object({
  name: z.literal('anny'),
  version: z.string(),
  bone_count: z.number().int().positive(),
  shape_count: z.number().int().positive(),
  bone_parents: z.array(z.number().int()),
});

const meshArraySchema = z.object({
  name: z.string(),
  dtype: z.enum(['<f4', '<u2']),
  shape: z.array(z.number().int().nonnegative()),
  byte_offset: z.number().int().nonnegative(),
  byte_length: z.number().int().nonnegative(),
});

const meshAssetsSchema = z.object({
  path: z.string(),
  vertex_count: z.number().int().positive(),
  face_count: z.number().int().positive(),
  influences: z.number().int().positive(),
  // Which track each row of rest_vertices and rest_bone_poses belongs to.
  tracks: z.array(z.number().int().nonnegative()),
  arrays: z.array(meshArraySchema),
});

export const bodiesArtifactSchema = z.object({
  schema_version: z.literal(2),
  // Refused rather than guessed at: camera-relative, OpenCV axes, metres.
  convention: z.literal('camera_relative_opencv_metres'),
  ingest_hash: z.string(),
  body_model: bodyModelSchema,
  estimator: z.object({
    name: z.literal('multihmr2'),
    checkpoint_sha256: z.string(),
    backbone_repo: z.string(),
    backbone_ref: z.string(),
    config_hash: z.string(),
  }),
  parameters: z.object({
    path: z.string(),
    person_count: z.number().int().nonnegative(),
    arrays: z.array(parameterArraySchema),
  }),
  // Absent when the estimator could not describe its own body: the skeleton
  // still draws, the mesh does not.
  mesh: meshAssetsSchema.nullable().default(null),
  frames: z.array(
    z.object({
      frame_index: z.number().int().nonnegative(),
      source_index: z.number().int().nonnegative(),
      timestamp_seconds: z.number().nonnegative(),
      camera: cameraSchema,
      people: z.array(personSchema),
    }),
  ),
  summary: z.object({
    frames_total: z.number().int().nonnegative(),
    frames_with_people: z.number().int().nonnegative(),
    people_total: z.number().int().nonnegative(),
    max_people_in_frame: z.number().int().nonnegative(),
    confidence_threshold: z.number().min(0).max(1),
    track_count: z.number().int().nonnegative(),
  }),
});

export type BodiesArtifact = z.infer<typeof bodiesArtifactSchema>;
export type BodyModel = z.infer<typeof bodyModelSchema>;

/** One person at one instant. */
export interface TimelinePerson {
  readonly trackId: number;
  readonly confidence: number;
  /** Camera-space metres, three values per bone. */
  readonly joints: Float32Array;
  /**
   * Posed bone transforms, sixteen floats each, root-relative.
   *
   * A finished run carries them once its mesh is loaded; a live frame carries
   * them only when bodies were asked for.
   */
  readonly bones?: Float32Array;
}

export interface TimelineFrame {
  readonly time: number;
  readonly people: readonly TimelinePerson[];
}

/**
 * Turn a finished artifact into the timeline the renderer reads.
 *
 * The joint arrays are views into the sidecar rather than copies, so this costs
 * a few thousand small objects and no pixels of memory.
 */
export function toTimeline(
  artifact: BodiesArtifact,
  joints: Float32Array,
  bones?: Float32Array,
): TimelineFrame[] {
  const stride = artifact.body_model.bone_count * 3;
  const boneStride = artifact.body_model.bone_count * 16;
  return artifact.frames.map((frame) => ({
    time: frame.timestamp_seconds,
    people: frame.people.map((person) => ({
      trackId: person.track_id,
      confidence: person.confidence,
      joints: joints.subarray(
        person.parameter_index * stride,
        (person.parameter_index + 1) * stride,
      ),
      ...(bones === undefined
        ? {}
        : {
            bones: bones.subarray(
              person.parameter_index * boneStride,
              (person.parameter_index + 1) * boneStride,
            ),
          }),
    })),
  }));
}

/**
 * Index of the last frame at or before `time`.
 *
 * Binary search: this runs once per displayed frame over hundreds of entries.
 */
export function frameIndexAt(frames: readonly TimelineFrame[], time: number): number {
  let low = 0;
  let high = frames.length - 1;
  while (low < high) {
    const middle = Math.ceil((low + high) / 2);
    if ((frames[middle]?.time ?? 0) <= time) low = middle;
    else high = middle - 1;
  }
  return low;
}

/**
 * Everybody visible at `time`, interpolated between the frames either side.
 *
 * Interpolation is **by track id, never by position in the list**. Person 0 in
 * one frame is not person 0 in the next, so lerping index-wise slides every
 * figure towards whoever happened to be listed in the same slot - which looks
 * smoother than the truth, and is wrong.
 *
 * People come from whichever bracketing frame is nearer, so a figure appears
 * when the timeline reaches them rather than fading in beforehand.
 */
export function sampleAt(frames: readonly TimelineFrame[], time: number): TimelinePerson[] {
  if (frames.length === 0) return [];

  const index = frameIndexAt(frames, time);
  const current = frames[index];
  if (current === undefined) return [];
  const next = frames[index + 1];

  const span = next === undefined ? 0 : next.time - current.time;
  const alpha = span > 0 ? Math.min(1, Math.max(0, (time - current.time) / span)) : 0;

  const nearer = alpha < 0.5 || next === undefined ? current : next;
  const other = nearer === current ? next : current;
  // Measured away from whichever frame we are standing on, so it is 0 there.
  const blend = nearer === current ? alpha : 1 - alpha;

  const partner = new Map(other?.people.map((person) => [person.trackId, person]) ?? []);

  return nearer.people.map((person) => {
    const match = partner.get(person.trackId);
    if (match === undefined || blend === 0) return person;

    const joints = new Float32Array(person.joints.length);
    for (let i = 0; i < joints.length; i += 1) {
      const a = person.joints[i] ?? 0;
      const b = match.joints[i] ?? 0;
      joints[i] = a + (b - a) * blend;
    }
    const bones =
      person.bones && match.bones ? blendBones(person.bones, match.bones, blend) : person.bones;
    return {
      trackId: person.trackId,
      confidence: person.confidence,
      joints,
      ...(bones === undefined ? {} : { bones }),
    };
  });
}

/**
 * The playhead for a stream, which has no end and no seeking.
 *
 * It runs a little behind the newest frame received. That lag is what buys
 * something to interpolate towards: without it every figure would sit on the
 * last frame and jump when the next arrived.
 *
 * Time advances by the display's own delta, so motion stays smooth while frames
 * arrive irregularly - which, at a few frames a second, is always. Two clamps
 * keep it honest: it never runs past data that has not arrived, and if it falls
 * far behind it is pulled forward rather than crawling.
 *
 * The default lag is set above the interval between frames the estimator
 * actually manages. Set it shorter and the playhead spends its time pinned
 * against the newest frame, which is the stutter it exists to avoid.
 */
export class LiveClock {
  private time = 0;
  private newest = 0;
  private started = false;

  /** Public because a measured session records it: presentation is this much later by design. */
  constructor(readonly lag = 0.4) {}

  now(): number {
    return this.time;
  }

  /** Told about each frame as it arrives; never called from the render loop. */
  observe(time: number): void {
    this.newest = Math.max(this.newest, time);
    if (!this.started) {
      this.started = true;
      this.time = Math.max(0, time - this.lag);
    }
  }

  tick(delta: number): number {
    if (!this.started) return 0;
    const advanced = this.time + delta;
    // Falling more than two lags behind means the stream jumped; catch up
    // rather than walking through everything that was missed.
    const floor = this.newest - 2 * this.lag;
    // And never past the newest frame: there is nothing there to draw.
    this.time = Math.min(Math.max(advanced, floor), this.newest);
    return this.time;
  }
}

/**
 * One Euro filter: a low-pass whose cutoff rises with the speed of the signal.
 *
 * A fixed low-pass has to choose between jitter and lag. This one adapts - it
 * smooths hard while a value is roughly still, and gets out of the way when it
 * is genuinely moving - which is what makes it usual for interactive tracking.
 *
 * Casiez, Roussel and Vogel, CHI 2012.
 */
class OneEuro {
  private value: number | null = null;
  private rate = 0;

  constructor(
    private readonly minCutoff: number,
    private readonly beta: number,
    private readonly rateCutoff = 1,
  ) {}

  /** `scale` weakens the cutoff for this sample; see the range note below. */
  filter(sample: number, dt: number, scale = 1): number {
    if (this.value === null || dt <= 0) {
      this.value = sample;
      return sample;
    }
    const rawRate = (sample - this.value) / dt;
    this.rate += alphaFor(this.rateCutoff, dt) * (rawRate - this.rate);
    // The faster it is really moving, the less it is smoothed.
    const cutoff = (this.minCutoff + this.beta * Math.abs(this.rate)) * scale;
    this.value += alphaFor(cutoff, dt) * (sample - this.value);
    return this.value;
  }
}

function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value));
}

function alphaFor(cutoff: number, dt: number): number {
  const tau = 1 / (2 * Math.PI * cutoff);
  return 1 / (1 + tau / dt);
}

export interface AxisSmoothing {
  /** Low-pass cutoff while the value is still. Lower is smoother. */
  readonly cutoffHz: number;
  /** How far real movement opens the filter up. Higher is less lag. */
  readonly speedCoupling: number;
  /** How quickly the speed estimate itself reacts. */
  readonly rateCutoffHz: number;
}

export interface Smoothing {
  /** Sideways, where the estimate is already good. */
  readonly lateral: AxisSmoothing;
  /** Along the view, where it is not. */
  readonly depth: AxisSmoothing;
  /**
   * Range at which the depth cutoff is used as written, in metres.
   *
   * The filter tracks range in proportion, because the error does: the measured
   * jitter is about 2.5% of range at every distance, so someone at 60 m is
   * displaced five times as far per frame as someone at 12 m, and someone at
   * arm's length is hardly displaced at all. One fixed cutoff would either
   * leave the far ones flying or freeze the near ones.
   */
  readonly referenceRangeMetres: number;
}

/**
 * Settled by measuring the reference clip, not guessed.
 *
 * The two axes are filtered very differently because they are not equally
 * wrong: frame to frame the bodies move 0.25 m/s sideways and 7.8 m/s in depth.
 * Smoothing sideways as hard as in depth would blur motion that was never bad.
 *
 * Depth also gets a much weaker `speedCoupling`. A One Euro filter opens up when
 * the signal appears to be moving fast, which is exactly the wrong response when
 * the apparent speed is mostly noise - it lets the jitter straight through. On
 * an axis where the error is five times the real motion, the filter has to stay
 * shut and let the slow rate estimate carry the genuine walking.
 */
export const DEFAULT_SMOOTHING: Smoothing = {
  lateral: { cutoffHz: 2.5, speedCoupling: 0.4, rateCutoffHz: 1 },
  depth: { cutoffHz: 0.6, speedCoupling: 0, rateCutoffHz: 0.25 },
  referenceRangeMetres: 8,
};

/**
 * Holds each person still enough to look like a person.
 *
 * Only the root moves. The pose the model returns is good - the stature is
 * right to within a few centimetres - and it is *where* the body is put that
 * jumps, so the whole skeleton is shifted by the correction rather than
 * filtered joint by joint. Filtering joints individually would smooth away the
 * walking as well.
 *
 * This does not make the depth more accurate and is not meant to. The error is
 * inherent to one camera and is the thing the project is about; this only stops
 * a metre of per-frame noise being drawn as a metre of movement. Turn it off to
 * see what the model actually said.
 */
export class TrackSmoother {
  private readonly filters = new Map<number, [OneEuro, OneEuro, OneEuro]>();
  private lastTime: number | null = null;

  constructor(private readonly options: Smoothing = DEFAULT_SMOOTHING) {}

  frame(time: number, people: readonly TimelinePerson[]): TimelinePerson[] {
    const dt = this.lastTime === null ? 0 : time - this.lastTime;
    this.lastTime = time;

    const seen = new Set<number>();
    const smoothed = people.map((person) => {
      seen.add(person.trackId);
      return this.one(person, dt);
    });
    // A track that has gone should not resume from where it left off: the
    // person may be somewhere else entirely by the time the id is reused.
    for (const id of [...this.filters.keys()]) {
      if (!seen.has(id)) this.filters.delete(id);
    }
    return smoothed;
  }

  private one(person: TimelinePerson, dt: number): TimelinePerson {
    const { lateral, depth } = this.options;
    let axes = this.filters.get(person.trackId);
    if (axes === undefined) {
      const build = (a: AxisSmoothing): OneEuro =>
        new OneEuro(a.cutoffHz, a.speedCoupling, a.rateCutoffHz);
      axes = [build(lateral), build(lateral), build(depth)];
      this.filters.set(person.trackId, axes);
    }

    // Joint 0 is the pelvis, and the root the model reports.
    const [rawX, rawY, rawZ] = [
      person.joints[0] ?? 0,
      person.joints[1] ?? 0,
      person.joints[2] ?? 0,
    ];
    // Proportional both ways: someone at two metres is barely noisy and should
    // barely be filtered. Clamped at both ends so one bad depth cannot make the
    // filter useless or immovable.
    const tighten = clamp(
      this.options.referenceRangeMetres / Math.max(Math.abs(rawZ), 0.5),
      0.1,
      4,
    );
    const [alongX, alongY, alongZ] = axes;
    // Only depth follows the range; sideways is accurate at any distance.
    const shift = [
      alongX.filter(rawX, dt) - rawX,
      alongY.filter(rawY, dt) - rawY,
      alongZ.filter(rawZ, dt, tighten) - rawZ,
    ];
    if (shift.every((value) => value === 0)) return person;

    const joints = new Float32Array(person.joints.length);
    for (let i = 0; i < joints.length; i += 1) {
      joints[i] = (person.joints[i] ?? 0) + (shift[i % 3] ?? 0);
    }
    // The bones travel through untouched. They are root-relative, so moving the
    // root is the whole correction, and a person rebuilt without them is not
    // drawn as a mesh.
    return { ...person, joints };
  }
}

/** Run a whole timeline through a smoother, in order. */
export function smoothTimeline(
  frames: readonly TimelineFrame[],
  options: Smoothing = DEFAULT_SMOOTHING,
): TimelineFrame[] {
  const smoother = new TrackSmoother(options);
  return frames.map((frame) => ({
    time: frame.time,
    people: smoother.frame(frame.time, frame.people),
  }));
}

/**
 * Blend two sets of bone transforms.
 *
 * Straight interpolation of a rotation matrix does not give a rotation - the
 * result shrinks and shears - so the upper 3x3 is re-orthonormalised
 * afterwards. Between frames a fortieth of a second apart the angles are small
 * enough that this and a proper slerp are indistinguishable, and this costs a
 * fraction of decomposing a couple of thousand matrices every displayed frame.
 */
export function blendBones(from: Float32Array, to: Float32Array, blend: number): Float32Array {
  const out = new Float32Array(from.length);
  for (let i = 0; i < out.length; i += 1) {
    const a = from[i] ?? 0;
    out[i] = a + ((to[i] ?? 0) - a) * blend;
  }
  for (let base = 0; base < out.length; base += 16) orthonormalise(out, base);
  return out;
}

/**
 * Make the rotation part of a column-major 4x4 a rotation again.
 *
 * Gram-Schmidt over the three basis columns. A matrix that is nearly a rotation
 * comes back as one; a degenerate one is left alone rather than producing NaNs
 * that would take the whole body off screen.
 */
function orthonormalise(m: Float32Array, at: number): void {
  const col = (c: number): [number, number, number] => [
    m[at + c * 4] ?? 0,
    m[at + c * 4 + 1] ?? 0,
    m[at + c * 4 + 2] ?? 0,
  ];
  const put = (c: number, v: readonly [number, number, number]): void => {
    m[at + c * 4] = v[0];
    m[at + c * 4 + 1] = v[1];
    m[at + c * 4 + 2] = v[2];
  };
  const dot = (a: readonly number[], b: readonly number[]): number =>
    (a[0] ?? 0) * (b[0] ?? 0) + (a[1] ?? 0) * (b[1] ?? 0) + (a[2] ?? 0) * (b[2] ?? 0);

  const x = col(0);
  let length = Math.hypot(...x);
  if (length < 1e-8) return;
  const ex: [number, number, number] = [x[0] / length, x[1] / length, x[2] / length];

  const y = col(1);
  const projection = dot(y, ex);
  const ry: [number, number, number] = [
    y[0] - projection * ex[0],
    y[1] - projection * ex[1],
    y[2] - projection * ex[2],
  ];
  length = Math.hypot(...ry);
  if (length < 1e-8) return;
  const ey: [number, number, number] = [ry[0] / length, ry[1] / length, ry[2] / length];

  put(0, ex);
  put(1, ey);
  // The third is fixed by the other two, up to the handedness they imply.
  put(2, [
    ex[1] * ey[2] - ex[2] * ey[1],
    ex[2] * ey[0] - ex[0] * ey[2],
    ex[0] * ey[1] - ex[1] * ey[0],
  ]);
}

/**
 * Transpose every 4x4 in a flat buffer, in place.
 *
 * numpy writes a matrix row by row and three.js reads one column by column, so
 * a transform crossing that boundary arrives transposed. Two transposed
 * matrices do not compose into the transpose of their product, so the error
 * does not cancel: bodies land in the right places and are crumpled.
 */
export function transposeMatrices(flat: Float32Array): Float32Array {
  for (let base = 0; base < flat.length; base += 16) {
    for (let row = 0; row < 4; row += 1) {
      for (let column = row + 1; column < 4; column += 1) {
        const a = base + row * 4 + column;
        const b = base + column * 4 + row;
        const swap = flat[a] ?? 0;
        flat[a] = flat[b] ?? 0;
        flat[b] = swap;
      }
    }
  }
  return flat;
}

/** The constant half of a body: what it is made of and what pulls on it. */
export interface BodyTopology {
  readonly faces: Uint16Array;
  readonly skinIndices: Uint16Array;
  readonly skinWeights: Float32Array;
  readonly vertexCount: number;
  readonly boneCount: number;
}

/** One person's unposed body. Column-major bind bones. */
export interface RestBody {
  readonly vertices: Float32Array;
  readonly bones: Float32Array;
}

/**
 * Where the renderer gets bodies from.
 *
 * A finished run has every track's body up front; a stream learns them as
 * people appear. Both answer the same question, so the renderer builds a body
 * the first time one is available and never asks about the source.
 */
export interface BodySource {
  readonly topology: BodyTopology;
  rest(track: number): RestBody | null;
}
