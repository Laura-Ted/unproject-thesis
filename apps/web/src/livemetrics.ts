/**
 * What the browser records about a live session, and how it hands it in.
 *
 * ADR-0005 WP3. Every frame the viewer sends gets a monotone id; every answer
 * carries it back, so a send and its answer can be paired on one clock -
 * `performance.now()`, relative to the moment the session opened. The server's
 * own durations arrive inside the answer and are kept as they came: they are on
 * the server's clock and are never subtracted from ours.
 *
 * "Presented" is the first animation frame after the bodies were applied to
 * the timeline. It is not the moment those bodies were drawn: the live clock
 * trails the newest frame by a deliberate lag so there is something to
 * interpolate towards, and that lag is recorded in the config rather than
 * folded into a number that would then mean two things.
 *
 * The rows go to the server as they are. The summary - percentiles, drop
 * accounting - is computed there, by the one implementation in the pipeline,
 * so this client and the Python one cannot disagree about a median.
 */

import type { ServerTiming } from './live';

export interface SentRecord {
  readonly frameId: number;
  readonly sentAtMs: number;
  readonly sentBytes: number;
  receivedAtMs: number | null;
  presentedAtMs: number | null;
  receivedBytes: number | null;
  people: number | null;
  droppedBefore: number | null;
  server: ServerTiming | null;
}

/** What the report needs beyond the rows. */
export interface ReportContext {
  readonly source: string;
  readonly mode: 'skeleton' | 'mesh';
  readonly frameWidth: number;
  readonly frameHeight: number;
  readonly sendFps: number;
  readonly maxWidth: number;
  readonly jpegQuality: number;
  readonly liveClockLagSeconds: number;
  readonly frameLimit: number | null;
  readonly server: {
    readonly machine: string;
    readonly device: string;
    readonly git_commit: string | null;
    readonly git_dirty: boolean | null;
  } | null;
  readonly notes: readonly string[];
}

/** The body of `POST /api/live-metrics`, as the server's `LiveReportDraft` reads it. */
export interface ReportDraft {
  readonly kind: 'browser';
  readonly started_at: string;
  readonly source: string;
  readonly source_sha256: null;
  readonly mode: 'skeleton' | 'mesh';
  readonly frame_width: number;
  readonly frame_height: number;
  readonly environment: {
    readonly user_agent: string;
    readonly viewport: string;
    readonly client_platform: string;
    readonly server_machine: string | null;
    readonly server_device: string | null;
    readonly git_commit: string | null;
    readonly git_dirty: boolean | null;
  };
  readonly config: {
    readonly send_fps: number;
    readonly max_width: number;
    readonly jpeg_quality: number;
    readonly bodies: boolean;
    readonly live_clock_lag_seconds: number;
    readonly frame_limit: number | null;
  };
  readonly samples: readonly {
    readonly frame_id: number;
    readonly sent_at_ms: number;
    readonly sent_bytes: number;
    readonly received_at_ms: number | null;
    readonly presented_at_ms: number | null;
    readonly received_bytes: number | null;
    readonly people: number | null;
    readonly dropped_before: number | null;
    readonly server: ServerTiming | null;
  }[];
  readonly notes: readonly string[];
}

export class LiveMetrics {
  private readonly rows = new Map<number, SentRecord>();
  private readonly startedAt = new Date();
  private next = 0;

  constructor(
    /** `performance.now()` when the session opened; every time here is relative to it. */
    private readonly origin: number,
    private readonly now: () => number = () => performance.now(),
  ) {}

  /** Hand out the next frame id and record the send. */
  sent(bytes: number): number {
    const frameId = this.next;
    this.next += 1;
    this.rows.set(frameId, {
      frameId,
      sentAtMs: this.elapsed(),
      sentBytes: bytes,
      receivedAtMs: null,
      presentedAtMs: null,
      receivedBytes: null,
      people: null,
      droppedBefore: null,
      server: null,
    });
    return frameId;
  }

  received(
    frameId: number,
    bytes: number,
    people: number,
    dropped: number,
    server: ServerTiming | null,
  ): void {
    const row = this.rows.get(frameId);
    if (row === undefined) return;
    row.receivedAtMs = this.elapsed();
    row.receivedBytes = bytes;
    row.people = people;
    row.droppedBefore = dropped;
    row.server = server;
  }

  /** The first animation frame after the answer was applied; `at` is its timestamp. */
  presented(frameId: number, at: number): void {
    const row = this.rows.get(frameId);
    if (row?.receivedAtMs === null || row === undefined) return;
    // Never earlier than the reception it follows: an animation frame's
    // timestamp is taken at the start of the frame, which can precede the
    // message handler that ran within it.
    row.presentedAtMs = Math.max(row.receivedAtMs, at - this.origin);
  }

  count(): { sent: number; answered: number } {
    let answered = 0;
    for (const row of this.rows.values()) if (row.receivedAtMs !== null) answered += 1;
    return { sent: this.rows.size, answered };
  }

  draft(context: ReportContext): ReportDraft {
    return {
      kind: 'browser',
      started_at: this.startedAt.toISOString(),
      source: context.source,
      source_sha256: null,
      mode: context.mode,
      frame_width: context.frameWidth,
      frame_height: context.frameHeight,
      environment: {
        user_agent: navigator.userAgent,
        viewport: `${String(window.innerWidth)}x${String(window.innerHeight)}`,
        client_platform: 'browser',
        server_machine: context.server?.machine ?? null,
        server_device: context.server?.device ?? null,
        git_commit: context.server?.git_commit ?? null,
        git_dirty: context.server?.git_dirty ?? null,
      },
      config: {
        send_fps: context.sendFps,
        max_width: context.maxWidth,
        jpeg_quality: context.jpegQuality,
        bodies: context.mode === 'mesh',
        live_clock_lag_seconds: context.liveClockLagSeconds,
        frame_limit: context.frameLimit,
      },
      samples: [...this.rows.values()]
        .sort((a, b) => a.frameId - b.frameId)
        .map((row) => ({
          frame_id: row.frameId,
          sent_at_ms: row.sentAtMs,
          sent_bytes: row.sentBytes,
          received_at_ms: row.receivedAtMs,
          presented_at_ms: row.presentedAtMs,
          received_bytes: row.receivedBytes,
          people: row.people,
          dropped_before: row.droppedBefore,
          server: row.server,
        })),
      notes: [...context.notes],
    };
  }

  private elapsed(): number {
    return Math.max(0, this.now() - this.origin);
  }
}

/** Hand a draft to the server, which computes the summary and keeps the file. */
export async function submitReport(draft: ReportDraft): Promise<string> {
  const response = await fetch('/api/live-metrics', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(draft),
  });
  if (!response.ok) {
    const detail = await response.text().catch(() => '');
    throw new Error(`the server refused the report (${String(response.status)}): ${detail}`);
  }
  const body = (await response.json()) as { path?: unknown };
  return typeof body.path === 'string' ? body.path : 'artifacts/live';
}

/**
 * Measurement settings named by the query string, or null when not measuring.
 *
 * `?measure=1&replay=datasets/pedestrians/clip.mp4&fps=2&bodies=0&frames=120`
 * plays a clip the server already has into the camera endpoint at two frames
 * a second, records every frame, and posts the report when the clip ends. A
 * URL rather than a button, so an automated run is a command line.
 */
export interface MeasureRequest {
  readonly replay: string;
  readonly fps: number;
  readonly bodies: boolean;
  readonly frameLimit: number | null;
}

export function measureRequestFrom(search: string): MeasureRequest | null {
  const params = new URLSearchParams(search);
  if (params.get('measure') !== '1') return null;
  const replay = params.get('replay');
  if (replay === null || replay === '') return null;
  const fps = Number(params.get('fps') ?? '2');
  const frames = Number(params.get('frames') ?? '');
  return {
    replay,
    fps: Number.isFinite(fps) && fps > 0 ? fps : 2,
    bodies: params.get('bodies') === '1',
    frameLimit: Number.isInteger(frames) && frames > 0 ? frames : null,
  };
}
