import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { BodyTopology } from './bodies';
import type { LiveState } from './live';
import { LiveSession, readRestBody } from './live';
import { LiveMetrics, measureRequestFrom } from './livemetrics';

const topology: BodyTopology = {
  faces: new Uint16Array(),
  skinIndices: new Uint16Array(),
  skinWeights: new Float32Array(),
  vertexCount: 2,
  boneCount: 1,
};

/** A track id, then 2x3 vertices, then one 4x4 bind bone, as the server packs it. */
function packed(track: number, floats: number[]): ArrayBuffer {
  const buffer = new ArrayBuffer(4 + floats.length * 4);
  new Uint32Array(buffer, 0, 1)[0] = track;
  new Float32Array(buffer, 4).set(floats);
  return buffer;
}

describe('readRestBody', () => {
  const rows = [1, 2, 3, 4, 5, 6];
  const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 7, 8, 9, 1];

  it('splits a message into its track, vertices and bones', () => {
    const [track, rest] = readRestBody(packed(12, [...rows, ...identity]), topology);
    expect(track).toBe(12);
    expect(Array.from(rest.vertices)).toEqual(rows);
    expect(rest.bones).toHaveLength(16);
  });

  it('transposes the bind bones into what three.js reads', () => {
    // The server packs row by row; the translation is the last row there and
    // the last column here.
    const [, rest] = readRestBody(packed(0, [...rows, ...identity]), topology);
    expect(Array.from(rest.bones.slice(12))).toEqual([0, 0, 0, 1]);
    expect(Array.from(rest.bones.slice(3, 4))).toEqual([7]);
  });

  it('refuses a message of the wrong length', () => {
    // Reshaping it anyway would build a body out of somebody else's numbers.
    expect(() => readRestBody(packed(1, [1, 2, 3]), topology)).toThrow(/expected 22/);
  });

  it('refuses a message with no track id', () => {
    expect(() => readRestBody(new ArrayBuffer(2), topology)).toThrow(/no track id/);
  });
});

/** A socket that never connects, so listeners can be fired by hand. */
class FakeSocket {
  static last: FakeSocket | null = null;
  /** The real constant, so `readyState = 1` reads as open. */
  static readonly OPEN = 1;
  readonly listeners = new Map<string, ((event: unknown) => void)[]>();
  binaryType = '';
  readyState = 0;
  closes = 0;
  readonly sent: unknown[] = [];

  constructor(readonly url: URL | string) {
    FakeSocket.last = this;
  }

  addEventListener(kind: string, handler: (event: unknown) => void): void {
    const existing = this.listeners.get(kind) ?? [];
    this.listeners.set(kind, [...existing, handler]);
  }

  send(payload: unknown): void {
    this.sent.push(payload);
  }

  close(): void {
    this.closes += 1;
  }

  fire(kind: string, event: unknown = {}): void {
    (this.listeners.get(kind) ?? []).forEach((handler) => {
      handler(event);
    });
  }
}

function openSession(metrics: LiveMetrics | null = null): {
  session: LiveSession;
  states: LiveState[];
  socket: FakeSocket;
} {
  const states: LiveState[] = [];
  const session = new LiveSession(
    { kind: 'camera', bodies: false, frame_ids: metrics !== null },
    null,
    (state) => states.push(state),
    () => undefined,
    metrics,
  );
  session.open();
  const socket = FakeSocket.last;
  if (socket === null) throw new Error('no socket was opened');
  return { session, states, socket };
}

const READY = JSON.stringify({
  kind: 'ready',
  schema_version: 2,
  convention: 'camera_relative_opencv_metres',
  body_model: {
    name: 'anny',
    version: '0.6.0',
    bone_count: 1,
    shape_count: 11,
    bone_parents: [-1],
  },
  source: 'camera',
  duration_seconds: null,
  environment: { machine: 'Apple M0', device: 'mps', git_commit: '0'.repeat(40), git_dirty: false },
});

/** One answered frame, numbered, with the server's own durations on it. */
function frameMessage(frameId: number | null, dropped = 0): string {
  return JSON.stringify({
    kind: 'frame',
    frame_id: frameId,
    timing: { decode_ms: 1.5, queue_wait_ms: 3, inference_ms: 300, serialize_previous_ms: null },
    time_seconds: 0.5,
    camera: { width: 64, height: 48, fx: 100, fy: 100, cx: 32, cy: 24, source: 'estimated' },
    people: [{ track_id: 0, confidence: 0.9, joints: [0, 0, 3] }],
    dropped,
  });
}

describe('a closed LiveSession', () => {
  beforeEach(() => {
    FakeSocket.last = null;
    vi.stubGlobal('WebSocket', FakeSocket);
    vi.stubGlobal('location', { href: 'http://localhost:8000/' });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('reports state while it is open', () => {
    const { states, socket } = openSession();
    socket.fire('message', { data: READY });
    expect(states.at(-1)?.status).toBe('running');
  });

  it('says nothing after a message that arrives late', () => {
    // Closing a socket does not silence it; the browser still delivers what
    // was already in flight, and by then another session owns the screen.
    const { session, states, socket } = openSession();
    session.close();
    const seen = states.length;

    socket.fire('message', { data: READY });

    expect(states).toHaveLength(seen);
  });

  it('says nothing on its own close event', () => {
    const { session, states, socket } = openSession();
    socket.fire('message', { data: READY });
    session.close();
    const seen = states.length;

    socket.fire('close');

    expect(states).toHaveLength(seen);
  });

  it('says nothing when the connection then fails', () => {
    const { session, states, socket } = openSession();
    session.close();
    const seen = states.length;

    socket.fire('error');

    expect(states).toHaveLength(seen);
  });

  it('sends nothing more', () => {
    const { session, socket } = openSession();
    socket.readyState = 1;
    session.close();

    session.send(new Blob([]));

    expect(socket.sent).toHaveLength(0);
  });

  it('cannot be reopened by a late caller', () => {
    const { session } = openSession();
    session.close();
    const before = FakeSocket.last;

    session.open();

    expect(FakeSocket.last).toBe(before);
    expect(session.isClosed()).toBe(true);
  });

  it('can be closed twice without complaint', () => {
    const { session, socket } = openSession();
    session.close();
    session.close();
    expect(socket.closes).toBe(1);
  });
});

describe('a measured LiveSession', () => {
  beforeEach(() => {
    FakeSocket.last = null;
    vi.stubGlobal('WebSocket', FakeSocket);
    vi.stubGlobal('location', { href: 'http://localhost:8000/' });
    vi.stubGlobal('requestAnimationFrame', (callback: (at: number) => void) => {
      callback(1016);
      return 1;
    });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('asks the server for frame ids', () => {
    const { socket } = openSession(new LiveMetrics(0, () => 0));
    socket.fire('open');
    expect(JSON.parse(String(socket.sent[0]))).toEqual({
      kind: 'camera',
      bodies: false,
      frame_ids: true,
    });
  });

  it('numbers each frame in a 4-byte prefix and records the send', async () => {
    let clock = 1000;
    const metrics = new LiveMetrics(1000, () => clock);
    const { session, socket } = openSession(metrics);
    socket.readyState = 1;

    session.send(new Blob([new Uint8Array(10)]));
    clock = 1125;
    session.send(new Blob([new Uint8Array(20)]));

    expect(socket.sent).toHaveLength(2);
    const second = socket.sent[1] as Blob;
    expect(second.size).toBe(24);
    const header = new Uint32Array(await second.slice(0, 4).arrayBuffer());
    expect(header[0]).toBe(1);
    const draft = metrics.draft(context());
    expect(draft.samples.map((row) => [row.frame_id, row.sent_at_ms, row.sent_bytes])).toEqual([
      [0, 0, 14],
      [1, 125, 24],
    ]);
  });

  it('pairs an answer with its send on the client clock and keeps the server timings', () => {
    let clock = 1000;
    const metrics = new LiveMetrics(1000, () => clock);
    const { session, socket } = openSession(metrics);
    socket.readyState = 1;
    socket.fire('message', { data: READY });
    session.send(new Blob([new Uint8Array(10)]));
    session.send(new Blob([new Uint8Array(10)]));

    clock = 1400;
    socket.fire('message', { data: frameMessage(1, 1) });

    const [first, second] = metrics.draft(context()).samples;
    expect(first?.received_at_ms).toBeNull();
    expect(second?.received_at_ms).toBe(400);
    expect(second?.dropped_before).toBe(1);
    expect(second?.people).toBe(1);
    expect(second?.server?.inference_ms).toBe(300);
    // The animation frame stamped 1016 is before the reception at 1400 on the
    // origin-relative scale; presentation is never earlier than reception.
    expect(second?.presented_at_ms).toBe(400);
    expect(session.measured()).toEqual({ sent: 2, answered: 1 });
  });

  it('ignores an answer it did not send', () => {
    const metrics = new LiveMetrics(0, () => 0);
    const { socket } = openSession(metrics);
    socket.fire('message', { data: READY });

    socket.fire('message', { data: frameMessage(7) });

    expect(metrics.count()).toEqual({ sent: 0, answered: 0 });
  });

  it('sends bare frames when not measured', () => {
    const { session, socket } = openSession(null);
    socket.readyState = 1;
    session.send(new Blob([new Uint8Array(10)]));
    expect((socket.sent[0] as Blob).size).toBe(10);
    expect(session.measured()).toBeNull();
  });
});

function context(): Parameters<LiveMetrics['draft']>[0] {
  return {
    source: 'datasets/pedestrians/clip.mp4',
    mode: 'skeleton',
    frameWidth: 1280,
    frameHeight: 720,
    sendFps: 2,
    maxWidth: 1280,
    jpegQuality: 0.75,
    liveClockLagSeconds: 0.4,
    frameLimit: null,
    server: null,
    notes: [],
  };
}

describe('measureRequestFrom', () => {
  it('reads a replay request off the query string', () => {
    expect(
      measureRequestFrom(
        '?measure=1&replay=datasets/pedestrians/clip.mp4&fps=8&bodies=1&frames=60',
      ),
    ).toEqual({ replay: 'datasets/pedestrians/clip.mp4', fps: 8, bodies: true, frameLimit: 60 });
  });

  it('defaults the rate and the limit', () => {
    expect(measureRequestFrom('?measure=1&replay=datasets/x.mp4')).toEqual({
      replay: 'datasets/x.mp4',
      fps: 2,
      bodies: false,
      frameLimit: null,
    });
  });

  it('is null without both the flag and a clip', () => {
    expect(measureRequestFrom('?replay=datasets/x.mp4')).toBeNull();
    expect(measureRequestFrom('?measure=1')).toBeNull();
    expect(measureRequestFrom('')).toBeNull();
  });
});
