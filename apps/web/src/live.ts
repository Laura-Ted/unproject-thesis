/**
 * The live socket: bodies arriving while the operator watches.
 *
 * Frames are appended to a timeline the render loop reads. Nothing here touches
 * the loop and the loop never waits here, which is the whole point - inference
 * runs at a few frames a second and the view has to stay at sixty.
 *
 * The timeline is trimmed rather than kept: a camera left running would
 * otherwise grow without bound. Playback only reads the newest frames, but the
 * smoother and the track list run over all that is kept.
 */

import { z } from 'zod';

import type { BodyModel, BodySource, BodyTopology, RestBody, TimelineFrame } from './bodies';
import { bodyModelSchema, transposeMatrices } from './bodies';
import type { LiveMetrics } from './livemetrics';

/** Enough history to interpolate through, and no more. */
const KEEP_SECONDS = 4;

const serverEnvironmentSchema = z.object({
  machine: z.string(),
  device: z.string(),
  git_commit: z.string().nullable(),
  git_dirty: z.boolean().nullable(),
});

export type ServerEnvironment = z.infer<typeof serverEnvironmentSchema>;

const readySchema = z.object({
  kind: z.literal('ready'),
  // 2 adds `frame_id` and `timing` to the frame, both optional.
  schema_version: z.literal(2),
  convention: z.literal('camera_relative_opencv_metres'),
  body_model: bodyModelSchema,
  source: z.string(),
  duration_seconds: z.number().nullable(),
  environment: serverEnvironmentSchema.nullable().default(null),
});

/**
 * What the server spent on a frame, on its own clock. Durations only: a
 * timestamp from another machine cannot be compared with `performance.now()`.
 */
const serverTimingSchema = z.object({
  decode_ms: z.number().nonnegative(),
  queue_wait_ms: z.number().nonnegative(),
  inference_ms: z.number().nonnegative(),
  serialize_previous_ms: z.number().nonnegative().nullable(),
});

export type ServerTiming = z.infer<typeof serverTimingSchema>;

const frameSchema = z.object({
  kind: z.literal('frame'),
  // Our own number for the frame this answers, echoed back; null for a file.
  frame_id: z.number().int().nonnegative().nullable().default(null),
  timing: serverTimingSchema.nullable().default(null),
  time_seconds: z.number().nonnegative(),
  camera: z.object({
    width: z.number().int().positive(),
    height: z.number().int().positive(),
    fx: z.number().positive(),
    fy: z.number().positive(),
    cx: z.number(),
    cy: z.number(),
    source: z.enum(['estimated', 'assumed', 'known']),
  }),
  people: z.array(
    z.object({
      track_id: z.number().int().nonnegative(),
      confidence: z.number().min(0).max(1),
      joints: z.array(z.number()),
      // Empty unless bodies were asked for.
      bones: z.array(z.number()).default([]),
    }),
  ),
  dropped: z.number().int().nonnegative(),
});

const messageSchema = z.discriminatedUnion('kind', [
  readySchema,
  frameSchema,
  z.object({ kind: z.literal('done'), frames: z.number().int().nonnegative() }),
  z.object({ kind: z.literal('error'), message: z.string() }),
]);

export type LiveStatus = 'connecting' | 'starting' | 'running' | 'done' | 'error';

export interface LiveState {
  readonly status: LiveStatus;
  readonly message: string | null;
  readonly bodyModel: BodyModel | null;
  readonly frames: readonly TimelineFrame[];
  /** Source frames inference could not keep up with. Camera only. */
  readonly dropped: number;
  readonly processed: number;
  readonly durationSeconds: number | null;
  /** From the most recent frame, so the backdrop knows the source camera. */
  readonly camera: SourceCamera | null;
  /** Where the estimator runs, as the server said when it was ready. */
  readonly environment: ServerEnvironment | null;
}

/** The camera a frame was estimated against, at source resolution. */
export interface SourceCamera {
  readonly width: number;
  readonly height: number;
  readonly fx: number;
  readonly fy: number;
  readonly cx: number;
  readonly cy: number;
}

export type LiveRequest =
  | {
      readonly kind: 'video';
      readonly path: string;
      readonly stride: number;
      readonly bodies: boolean;
    }
  | {
      readonly kind: 'camera';
      readonly bodies: boolean;
      /** Prefix every frame with a 4-byte id the server echoes back. */
      readonly frame_ids?: boolean;
    };

const EMPTY: LiveState = {
  status: 'connecting',
  message: null,
  bodyModel: null,
  frames: [],
  dropped: 0,
  processed: 0,
  durationSeconds: null,
  camera: null,
  environment: null,
};

/**
 * One run of the estimator, watched.
 *
 * The frame list is replaced rather than mutated on each arrival so React sees
 * a change, but the render loop reads `timeline()` directly and never re-renders
 * for it.
 */
export class LiveSession {
  private socket: WebSocket | null = null;
  /**
   * Set by `close`, and the only thing that stops a superseded session talking.
   *
   * Closing a socket does not silence it: a message already in flight, and the
   * close event itself, still reach the listeners, and the owner has moved on
   * by then.
   */
  private closed = false;
  private state: LiveState = EMPTY;
  private frames: TimelineFrame[] = [];
  // Filled by the binary messages the server sends once per person.
  private readonly rest = new Map<number, RestBody>();

  constructor(
    private readonly request: LiveRequest,
    /** Needed before a body message can be read; null draws skeletons. */
    private readonly topology: BodyTopology | null,
    private readonly onChange: (state: LiveState) => void,
    /** Told about each frame's timestamp, so a clock can follow the stream. */
    private readonly onFrame: (time: number) => void,
    /** Records sends, answers and presentation when a session is measured. */
    private readonly metrics: LiveMetrics | null = null,
  ) {}

  /** Raw frames, oldest first. Smoothing is the caller's business. */
  timeline(): readonly TimelineFrame[] {
    return this.frames;
  }

  open(): void {
    if (this.closed) return;
    const url = new URL('/api/live', globalThis.location.href);
    url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
    const socket = new WebSocket(url);
    this.socket = socket;

    socket.addEventListener('open', () => {
      socket.send(JSON.stringify(this.request));
      this.update({ status: 'starting' });
    });
    socket.addEventListener('message', (event: MessageEvent<string | ArrayBuffer>) => {
      if (this.closed) return;
      if (typeof event.data === 'string') this.receive(event.data);
      else this.receiveBody(event.data);
    });
    // Bodies arrive as bytes; a fifth of a megabyte printed as JSON numbers
    // would cost five times as much for nothing.
    socket.binaryType = 'arraybuffer';
    socket.addEventListener('error', () => {
      this.update({ status: 'error', message: 'the live connection failed' });
    });
    socket.addEventListener('close', () => {
      if (this.state.status === 'running' || this.state.status === 'starting') {
        this.update({ status: 'done' });
      }
    });
  }

  /**
   * Push one camera frame. Dropped on the floor if the socket is not ready.
   *
   * When the session is measured the frame gets the next id, as a 4-byte
   * little-endian prefix the server echoes back, and the send is recorded.
   */
  send(payload: Blob): void {
    if (this.closed) return;
    if (this.socket?.readyState !== WebSocket.OPEN) return;
    if (this.metrics === null) {
      this.socket.send(payload);
      return;
    }
    const header = new Uint32Array([0]);
    const frameId = this.metrics.sent(payload.size + header.byteLength);
    header[0] = frameId;
    this.socket.send(new Blob([header, payload]));
  }

  /** How many frames were sent and answered, for a measured session's status. */
  measured(): { sent: number; answered: number } | null {
    return this.metrics?.count() ?? null;
  }

  /** Close the socket and stop reporting. Safe to call more than once. */
  close(): void {
    this.closed = true;
    this.socket?.close();
    this.socket = null;
  }

  /** Whether this session has been closed by its owner. */
  isClosed(): boolean {
    return this.closed;
  }

  private receive(raw: string): void {
    const parsed = messageSchema.safeParse(JSON.parse(raw) as unknown);
    if (!parsed.success) {
      this.update({ status: 'error', message: `unreadable message: ${parsed.error.message}` });
      return;
    }
    const message = parsed.data;

    if (message.kind === 'ready') {
      this.update({
        status: 'running',
        bodyModel: message.body_model,
        durationSeconds: message.duration_seconds,
        environment: message.environment,
      });
      return;
    }
    if (message.kind === 'error') {
      this.update({ status: 'error', message: message.message });
      return;
    }
    if (message.kind === 'done') {
      this.update({ status: 'done' });
      return;
    }

    this.frames.push({
      time: message.time_seconds,
      people: message.people.map((person) => ({
        trackId: person.track_id,
        confidence: person.confidence,
        joints: Float32Array.from(person.joints),
        ...(person.bones.length === 0
          ? {}
          : { bones: transposeMatrices(Float32Array.from(person.bones)) }),
      })),
    });
    this.trim(message.time_seconds);
    this.onFrame(message.time_seconds);
    if (this.metrics !== null && message.frame_id !== null) {
      const frameId = message.frame_id;
      this.metrics.received(
        frameId,
        raw.length,
        message.people.length,
        message.dropped,
        message.timing,
      );
      // The bodies are on the timeline now; the next animation frame is the
      // first one that could draw them.
      requestAnimationFrame((at) => {
        this.metrics?.presented(frameId, at);
      });
    }
    this.update({
      status: 'running',
      dropped: this.state.dropped + message.dropped,
      processed: this.state.processed + 1,
      camera: message.camera,
    });
  }

  private receiveBody(payload: ArrayBuffer): void {
    const shape = this.topology;
    if (shape === null) return;
    try {
      const [track, rest] = readRestBody(payload, shape);
      this.rest.set(track, rest);
    } catch (cause: unknown) {
      this.update({
        status: 'error',
        message: cause instanceof Error ? cause.message : String(cause),
      });
    }
  }

  /** Bodies as the renderer wants them, once the topology is known. */
  bodySource(): BodySource | null {
    const shape = this.topology;
    if (shape === null) return null;
    const rest = this.rest;
    return { topology: shape, rest: (track) => rest.get(track) ?? null };
  }

  private trim(now: number): void {
    const cutoff = now - KEEP_SECONDS;
    let keep = 0;
    while (keep < this.frames.length && (this.frames[keep]?.time ?? 0) < cutoff) keep += 1;
    // One frame before the cutoff stays, so there is always something to
    // interpolate from at the oldest end.
    if (keep > 1) this.frames.splice(0, keep - 1);
  }

  private update(patch: Partial<LiveState>): void {
    // The last gate, rather than the only one: `open` and `close` can race
    // through the browser's own event queue, and a closed session must not
    // move the state of whatever replaced it.
    if (this.closed) return;
    this.state = { ...this.state, ...patch, frames: this.frames.slice() };
    this.onChange(this.state);
  }
}

/**
 * The send rate is deliberately slower than the display and faster than
 * inference: the server keeps only the newest, so sending more would just be
 * encoding work thrown away, and sending less would add latency the model is
 * not responsible for.
 */
export const CAPTURE_DEFAULTS = { fps: 8, maxWidth: 1280, jpegQuality: 0.75 } as const;

/** Send frames from a video element down a socket at a steady cadence. */
export function captureFrames(
  video: HTMLVideoElement,
  session: LiveSession,
  {
    fps = CAPTURE_DEFAULTS.fps,
    maxWidth = CAPTURE_DEFAULTS.maxWidth,
    limit = null,
    onLimit,
  }: {
    fps?: number;
    maxWidth?: number;
    /** Stop after this many frames were sent; null sends until stopped. */
    limit?: number | null;
    onLimit?: () => void;
  } = {},
): () => void {
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  let stopped = false;
  let sent = 0;

  const tick = (): void => {
    if (stopped || context === null) return;
    // A paused or ended clip has nothing new to send; the camera never pauses.
    if (video.paused || video.ended) return;
    const width = Math.min(video.videoWidth, maxWidth);
    if (width > 0) {
      canvas.width = width;
      canvas.height = Math.round((video.videoHeight * width) / video.videoWidth);
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob(
        (blob) => {
          if (blob === null || stopped) return;
          session.send(blob);
          sent += 1;
          if (limit !== null && sent >= limit) {
            stopped = true;
            clearInterval(handle);
            onLimit?.();
          }
        },
        'image/jpeg',
        CAPTURE_DEFAULTS.jpegQuality,
      );
    }
  };

  const handle = setInterval(tick, Math.round(1000 / fps));
  return () => {
    stopped = true;
    clearInterval(handle);
  };
}

/**
 * One person's unposed body, as the socket sends it.
 *
 * A track id, then the rest vertices, then the bind bones. The counts come from
 * the topology the viewer already fetched, so the message carries no header
 * beyond the id it belongs to - and a length that disagrees is refused rather
 * than reshaped into a body that is not one.
 */
export function readRestBody(
  payload: ArrayBuffer,
  topology: BodyTopology,
): [track: number, rest: RestBody] {
  if (payload.byteLength < 4) throw new Error('a body arrived with no track id');
  const track = new Uint32Array(payload.slice(0, 4))[0];
  if (track === undefined) throw new Error('a body arrived with no track id');

  const vertices = topology.vertexCount * 3;
  const bones = topology.boneCount * 16;
  const floats = new Float32Array(payload.slice(4));
  if (floats.length !== vertices + bones) {
    throw new Error(
      `a body arrived with ${String(floats.length)} floats, expected ${String(vertices + bones)}`,
    );
  }
  return [
    track,
    {
      vertices: floats.subarray(0, vertices),
      // Column-major from here on, which is what three.js reads.
      bones: transposeMatrices(floats.slice(vertices)),
    },
  ];
}
