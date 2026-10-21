import { describe, expect, it } from 'vitest';

import type { TimelineFrame } from './bodies';
import { listTracks, trackPath, withoutHidden } from './tracks';

/** Frames a tenth of a second apart; each person given as [track, x, y, z]. */
function timeline(frames: [number, number, number, number][][]): TimelineFrame[] {
  return frames.map((people, index) => ({
    time: index * 0.1,
    people: people.map(([trackId, x, y, z]) => ({
      trackId,
      confidence: 1,
      joints: Float32Array.from([x, y, z]),
    })),
  }));
}

describe('listTracks', () => {
  it('finds every track, in id order', () => {
    const frames = timeline([
      [
        [2, 0, 0, 8],
        [0, 1, 0, 9],
      ],
      [[1, 2, 0, 7]],
    ]);

    expect(listTracks(frames).map((t) => t.trackId)).toEqual([0, 1, 2]);
  });

  it('counts how many frames each track holds', () => {
    const frames = timeline([[[0, 0, 0, 8]], [[0, 0, 0, 8]], [[1, 0, 0, 8]]]);

    const [first, second] = listTracks(frames);

    expect(first?.detections).toBe(2);
    expect(second?.detections).toBe(1);
  });

  it('records when a track first and last appears', () => {
    // Somebody who walks in late and leaves early. The lifetimes are what make
    // fragmentation visible when one person becomes several tracks.
    const frames = timeline([[], [[3, 0, 0, 8]], [[3, 0, 0, 8]], []]);

    const [only] = listTracks(frames);

    expect(only?.firstSeconds).toBeCloseTo(0.1, 5);
    expect(only?.lastSeconds).toBeCloseTo(0.2, 5);
  });

  it('is empty for a timeline with nobody in it', () => {
    expect(listTracks(timeline([[], []]))).toEqual([]);
  });
});

describe('withoutHidden', () => {
  it('returns the same frames when nothing is hidden', () => {
    const frames = timeline([[[0, 0, 0, 8]]]);

    expect(withoutHidden(frames, new Set())).toBe(frames);
  });

  it('drops only the tracks that were switched off', () => {
    const frames = timeline([
      [
        [0, 0, 0, 8],
        [1, 1, 0, 9],
        [2, 2, 0, 10],
      ],
    ]);

    const shown = withoutHidden(frames, new Set([1]));

    expect(shown[0]?.people.map((p) => p.trackId)).toEqual([0, 2]);
  });

  it('keeps a frame that has been emptied', () => {
    // Dropping the frame instead would shift the timeline, and the video is the
    // clock: the panes would drift apart.
    const frames = timeline([[[0, 0, 0, 8]], [[1, 0, 0, 8]]]);

    const shown = withoutHidden(frames, new Set([0]));

    expect(shown).toHaveLength(2);
    expect(shown[0]?.people).toEqual([]);
  });

  it('leaves the original untouched', () => {
    const frames = timeline([
      [
        [0, 0, 0, 8],
        [1, 0, 0, 8],
      ],
    ]);

    withoutHidden(frames, new Set([0]));

    expect(frames[0]?.people).toHaveLength(2);
  });
});

describe('trackPath', () => {
  const signs = [1, -1, -1] as const;

  it('follows one track and converts to three.js space', () => {
    const frames = timeline([
      [
        [0, 1, 2, 8],
        [1, 9, 9, 9],
      ],
      [[0, 2, 3, 9]],
    ]);

    expect(trackPath(frames, 0, signs)).toEqual([
      [1, -2, -8],
      [2, -3, -9],
    ]);
  });

  it('skips the frames the track was not found in rather than filling them', () => {
    // A break in the line is a break in the data. ADR-0003 §4 forbids inventing
    // the missing points.
    const frames = timeline([[[0, 1, 0, 8]], [[1, 5, 0, 8]], [[0, 3, 0, 8]]]);

    expect(trackPath(frames, 0, signs)).toEqual([
      [1, -0, -8],
      [3, -0, -8],
    ]);
  });

  it('is empty for a track that is not there', () => {
    expect(trackPath(timeline([[[0, 1, 0, 8]]]), 7, signs)).toEqual([]);
  });
});
