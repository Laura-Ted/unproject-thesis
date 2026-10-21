/**
 * Tracks, one at a time.
 *
 * **A track is not a person.** It is one unbroken run of detections the tracker
 * managed to hold together, and the id means nothing outside this run. When one
 * person becomes four tracks, four entries appear here with abutting lifetimes.
 * That is the finding rather than a defect to hide: see ADR-0003.
 *
 * Nothing here links two tracks, and nothing here fills a gap. ADR-0003 §4.
 */

import { z } from 'zod';

import type { TimelineFrame } from './bodies';

/** What can be derived from the timeline alone, so it is always available. */
export interface TrackListing {
  readonly trackId: number;
  readonly detections: number;
  readonly firstSeconds: number;
  readonly lastSeconds: number;
}

/**
 * Every track in a timeline, in id order.
 *
 * Derived from what the viewer already loaded rather than fetched, so the list
 * appears for a live session too, where no analysis has been run.
 */
export function listTracks(frames: readonly TimelineFrame[]): TrackListing[] {
  const seen = new Map<number, { detections: number; first: number; last: number }>();

  for (const frame of frames) {
    for (const person of frame.people) {
      const entry = seen.get(person.trackId);
      if (entry === undefined) {
        seen.set(person.trackId, { detections: 1, first: frame.time, last: frame.time });
      } else {
        entry.detections += 1;
        entry.last = Math.max(entry.last, frame.time);
        entry.first = Math.min(entry.first, frame.time);
      }
    }
  }

  return [...seen.entries()]
    .map(([trackId, entry]) => ({
      trackId,
      detections: entry.detections,
      firstSeconds: entry.first,
      lastSeconds: entry.last,
    }))
    .sort((a, b) => a.trackId - b.trackId);
}

/**
 * The subset of `rq3.json`'s per-track block the viewer shows.
 *
 * Parsed rather than asserted, and deliberately partial: the analysis owns the
 * full schema, and a viewer that demanded every field would break whenever a
 * metric was added to it.
 */
const trackStatsSchema = z.object({
  track_id: z.number().int().nonnegative(),
  longest_gap_seconds: z.number().nullable(),
  median_range_metres: z.number().nullable(),
  motion: z.object({
    steps: z.number().int().nonnegative(),
    lateral_median_metres_per_second: z.number().nullable(),
    depth_median_metres_per_second: z.number().nullable(),
    depth_jitter_fraction_of_range: z.number().nullable(),
  }),
  shape: z.object({ max_parameter_std: z.number().nullable() }),
});

const summarySchema = z.object({ per_track: z.array(trackStatsSchema) });

export type TrackStats = z.infer<typeof trackStatsSchema>;

/**
 * Per-track statistics for a finished run, or null when none were computed.
 *
 * The analysis is a separate command, so most runs will not have an `rq3.json`
 * beside them. That is not an error: the list still works, it just carries
 * fewer columns.
 */
export async function loadTrackStats(path: string): Promise<Map<number, TrackStats> | null> {
  let response: Response;
  try {
    response = await fetch(`/artifacts/${path}/rq3.json`);
  } catch {
    return null;
  }
  if (!response.ok) return null;

  const parsed = summarySchema.safeParse(await response.json());
  if (!parsed.success) return null;
  return new Map(parsed.data.per_track.map((stats) => [stats.track_id, stats]));
}

/**
 * The path one track took, in three.js space, oldest point first.
 *
 * Joint 0 is the pelvis, which is the root the model actually places, so this
 * is the trajectory of the estimate rather than of any particular limb. Frames
 * where the track was not found contribute nothing: the line joins the points
 * that exist and does not invent the ones that do not.
 */
export function trackPath(
  frames: readonly TimelineFrame[],
  trackId: number,
  signs: readonly [number, number, number],
): [number, number, number][] {
  const [signX, signY, signZ] = signs;
  const points: [number, number, number][] = [];

  for (const frame of frames) {
    for (const person of frame.people) {
      if (person.trackId !== trackId) continue;
      points.push([
        (person.joints[0] ?? 0) * signX,
        (person.joints[1] ?? 0) * signY,
        (person.joints[2] ?? 0) * signZ,
      ]);
    }
  }
  return points;
}

/** The timeline with hidden tracks removed. Frames with nobody left stay, empty. */
export function withoutHidden(
  frames: readonly TimelineFrame[],
  hidden: ReadonlySet<number>,
): readonly TimelineFrame[] {
  if (hidden.size === 0) return frames;
  return frames.map((frame) => ({
    time: frame.time,
    people: frame.people.filter((person) => !hidden.has(person.trackId)),
  }));
}
