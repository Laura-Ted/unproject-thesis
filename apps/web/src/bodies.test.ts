import { describe, expect, it } from 'vitest';

// Imported rather than read off disk: the path is then resolved by the
// bundler, so a fixture that moves breaks the build instead of a test at
// runtime.
import sharedFilterVector from '../../../packages/pipeline/tests/fixtures/one_euro.json';
import sharedSignalVector from '../../../packages/pipeline/tests/fixtures/one_euro_signals.json';

import type { Smoothing, TimelineFrame, TimelinePerson } from './bodies';
import {
  blendBones,
  DEFAULT_SMOOTHING,
  frameIndexAt,
  LiveClock,
  sampleAt,
  smoothTimeline,
  TrackSmoother,
  transposeMatrices,
} from './bodies';

/** Frames a tenth of a second apart, each person given as [track, x]. */
function timeline(frames: [number, number][][]): TimelineFrame[] {
  return frames.map((people, index) => ({
    time: index * 0.1,
    people: people.map(([trackId, x]) => ({
      trackId,
      confidence: 0.9,
      joints: Float32Array.from([x, 0, 0]),
    })),
  }));
}

const xs = (people: { trackId: number; joints: Float32Array }[]): Map<number, number> =>
  new Map(people.map((person) => [person.trackId, person.joints[0] ?? NaN]));

describe('frameIndexAt', () => {
  it('finds the last frame at or before the time', () => {
    const frames = timeline([[], [], [], []]);
    expect(frameIndexAt(frames, 0)).toBe(0);
    expect(frameIndexAt(frames, 0.15)).toBe(1);
    expect(frameIndexAt(frames, 0.2)).toBe(2);
  });

  it('clamps outside the timeline rather than running off the end', () => {
    const frames = timeline([[], []]);
    expect(frameIndexAt(frames, -5)).toBe(0);
    expect(frameIndexAt(frames, 99)).toBe(1);
  });
});

describe('sampleAt', () => {
  it('interpolates by track id, not by position in the list', () => {
    // The same two people, listed in the opposite order in the second frame.
    // Index-wise interpolation would slide each figure towards the other.
    const frames = timeline([
      [
        [7, 0],
        [3, 10],
      ],
      [
        [3, 12],
        [7, 4],
      ],
    ]);

    const at = xs(sampleAt(frames, 0.025));
    // A quarter of the way: track 7 goes 0 -> 4, track 3 goes 10 -> 12.
    expect(at.get(7)).toBeCloseTo(1, 5);
    expect(at.get(3)).toBeCloseTo(10.5, 5);
  });

  it('returns the stored values exactly at a frame time', () => {
    const frames = timeline([[[1, 2]], [[1, 20]]]);
    expect(xs(sampleAt(frames, 0)).get(1)).toBe(2);
    expect(xs(sampleAt(frames, 0.1)).get(1)).toBe(20);
  });

  it('does not show a person before the frame they appear in', () => {
    const frames = timeline([
      [[1, 0]],
      [
        [1, 1],
        [2, 5],
      ],
    ]);
    // Still nearer the first frame, where track 2 does not exist yet.
    expect(sampleAt(frames, 0.02).map((p) => p.trackId)).toEqual([1]);
    // Past halfway the second frame is nearer, and track 2 is in it.
    expect(
      sampleAt(frames, 0.08)
        .map((p) => p.trackId)
        .sort(),
    ).toEqual([1, 2]);
  });

  it('holds a person still when their track is missing from the other frame', () => {
    const frames = timeline([[[1, 7]], [[9, 99]]]);
    // No partner to move towards, so they stay put rather than lerping towards
    // an unrelated person.
    expect(xs(sampleAt(frames, 0.03)).get(1)).toBe(7);
  });

  it('has nothing to draw for an empty timeline', () => {
    expect(sampleAt([], 0)).toEqual([]);
  });
});

describe('LiveClock', () => {
  it('shows nothing before the first frame arrives', () => {
    expect(new LiveClock().tick(0.016)).toBe(0);
  });

  it('starts a lag behind the first frame', () => {
    const clock = new LiveClock(0.4);
    clock.observe(10);
    expect(clock.now()).toBeCloseTo(9.6, 6);
  });

  it('advances with the display while there is data ahead', () => {
    const clock = new LiveClock(0.4);
    clock.observe(10);
    const before = clock.now();
    clock.tick(0.1);
    expect(clock.now()).toBeCloseTo(before + 0.1, 6);
  });

  it('never runs past what has arrived', () => {
    // Interpolating towards a frame that does not exist is how a viewer ends up
    // confidently drawing the future.
    const clock = new LiveClock(0.4);
    clock.observe(10);
    for (let i = 0; i < 100; i += 1) clock.tick(0.1);
    expect(clock.now()).toBeCloseTo(10, 6);
  });

  it('catches up instead of crawling when the stream jumps', () => {
    const clock = new LiveClock(0.4);
    clock.observe(10);
    clock.observe(30);
    clock.tick(0.016);
    expect(clock.now()).toBeGreaterThan(29);
  });

  it('keeps following as frames keep arriving', () => {
    const clock = new LiveClock(0.4);
    clock.observe(10);
    for (let i = 1; i <= 5; i += 1) {
      clock.observe(10 + i * 0.4);
      clock.tick(0.4);
    }
    // Behind the newest, but tracking it rather than stuck at the start.
    expect(clock.now()).toBeGreaterThan(11);
    expect(clock.now()).toBeLessThanOrEqual(12);
  });
});

describe('TrackSmoother', () => {
  /** Frames 0.08 s apart, one person, x/y/z given per frame. */
  const walk = (path: [number, number, number][]): TimelineFrame[] =>
    path.map(([x, y, z], i) => ({
      time: i * 0.08,
      people: [
        { trackId: 1, confidence: 0.9, joints: Float32Array.from([x, y, z, x, y, z + 0.4]) },
      ],
    }));

  const depthOf = (frames: TimelineFrame[]): number[] =>
    frames.map((f) => f.people[0]?.joints[2] ?? 0);

  const wobble = (steps: number, base: number, amplitude: number): [number, number, number][] =>
    Array.from({ length: steps }, (_, i) => [0, 0, base + (i % 2 ? amplitude : -amplitude)]);

  it('quietens a standing person the estimator keeps moving', () => {
    // The measured failure: a metre of per-frame depth noise drawn as a metre
    // of movement.
    const jumpy = walk(wobble(40, 20, 0.5));
    const calm = smoothTimeline(jumpy);

    const spread = (frames: TimelineFrame[]): number => {
      const z = depthOf(frames).slice(10);
      return Math.max(...z) - Math.min(...z);
    };
    expect(spread(jumpy)).toBeCloseTo(1, 1);
    expect(spread(calm)).toBeLessThan(0.25);
  });

  it('still lets someone walk', () => {
    // Smoothing that froze real motion would be worse than the jitter.
    const approaching = walk(
      Array.from({ length: 60 }, (_, i) => [0, 0, 20 - i * 0.11] as [number, number, number]),
    );
    const calm = smoothTimeline(approaching);
    const z = depthOf(calm);
    expect((z[0] ?? 0) - (z.at(-1) ?? 0)).toBeGreaterThan(4);
  });

  it('leaves the pose alone and only moves the root', () => {
    // The stature the model returns is right; it is the placement that jumps.
    // Filtering joints one by one would smooth the walking away too.
    const calm = smoothTimeline(walk(wobble(20, 25, 0.6)));
    for (const frame of calm) {
      const j = frame.people[0]?.joints;
      if (!j) continue;
      // The second joint was placed 0.4 m behind the first.
      expect((j[5] ?? 0) - (j[2] ?? 0)).toBeCloseTo(0.4, 5);
    }
  });

  /**
   * Ripple left once the filter has settled.
   *
   * Measured at the end of a long run on purpose: a heavily smoothed filter
   * takes many frames to reach the mean, and measuring across that ramp reports
   * the transient rather than the jitter.
   */
  const rippleAt = (base: number): { input: number; left: number } => {
    const amplitude = base * 0.025;
    const z = depthOf(smoothTimeline(walk(wobble(400, base, amplitude)))).slice(-40);
    return { input: 2 * amplitude, left: Math.max(...z) - Math.min(...z) };
  };

  it('lets far less noise through from someone far away', () => {
    // The measured error is a fraction of range, so the filter follows it.
    const near = rippleAt(4);
    const far = rippleAt(40);
    expect(far.left / far.input).toBeLessThan(near.left / near.input / 4);
  });

  it('leaves about the same jitter in metres at any range', () => {
    // The point of tracking range: what is left over should be a few
    // centimetres whether the person is close or across a park, even though the
    // input at 40 m is ten times noisier.
    const near = rippleAt(4);
    const far = rippleAt(40);
    expect(far.left).toBeLessThan(near.left * 2);
  });

  it('keeps tracks apart', () => {
    const frames: TimelineFrame[] = [0, 1, 2, 3].map((i) => ({
      time: i * 0.08,
      people: [
        { trackId: 1, confidence: 0.9, joints: Float32Array.from([0, 0, 10]) },
        { trackId: 2, confidence: 0.9, joints: Float32Array.from([0, 0, 40]) },
      ],
    }));
    const calm = smoothTimeline(frames);
    expect(calm.at(-1)?.people[0]?.joints[2]).toBeCloseTo(10, 3);
    expect(calm.at(-1)?.people[1]?.joints[2]).toBeCloseTo(40, 3);
  });

  it('does not resume a track from where it vanished', () => {
    // The id may be reused for someone standing somewhere else entirely, and
    // dragging them across the scene would be worse than the jitter.
    const smoother = new TrackSmoother();
    const at = (z: number): TimelinePerson[] => [
      { trackId: 5, confidence: 0.9, joints: Float32Array.from([0, 0, z]) },
    ];
    for (let i = 0; i < 20; i += 1) smoother.frame(i * 0.08, at(10));
    smoother.frame(1.7, []);
    const back = smoother.frame(1.78, at(40));
    expect(back[0]?.joints[2]).toBeCloseTo(40, 3);
  });

  it('passes the first sighting through untouched', () => {
    const calm = smoothTimeline(walk([[1, 2, 30]]));
    expect(Array.from(calm[0]?.people[0]?.joints.slice(0, 3) ?? [])).toEqual([1, 2, 30]);
  });
});

describe('smoothing and the mesh', () => {
  it('carries the bone transforms through', () => {
    // They are root-relative, so smoothing the root is the whole correction.
    // Dropping them leaves the renderer with nothing to skin and it draws
    // nobody at all.
    const bones = Float32Array.from({ length: 16 }, (_, i) => i);
    const frames: TimelineFrame[] = [0, 1, 2, 3].map((i) => ({
      time: i * 0.08,
      people: [{ trackId: 1, confidence: 0.9, joints: Float32Array.from([0, 0, 20 + i]), bones }],
    }));
    for (const frame of smoothTimeline(frames)) {
      expect(frame.people[0]?.bones).toBeDefined();
    }
  });

  it('blends them when both frames have them', () => {
    const at = (v: number): Float32Array =>
      Float32Array.from([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, v, 0, 0, 1]);
    const frames: TimelineFrame[] = [
      {
        time: 0,
        people: [
          { trackId: 1, confidence: 0.9, joints: Float32Array.from([0, 0, 10]), bones: at(0) },
        ],
      },
      {
        time: 0.1,
        people: [
          { trackId: 1, confidence: 0.9, joints: Float32Array.from([0, 0, 10]), bones: at(4) },
        ],
      },
    ];
    // A quarter of the way between them.
    expect(sampleAt(frames, 0.025)[0]?.bones?.[12]).toBeCloseTo(1, 5);
  });

  it('keeps a blended rotation a rotation', () => {
    // Interpolating two rotation matrices gives something that is not one; the
    // columns are re-orthonormalised so a body does not shear as it moves.
    const spin = (a: number): Float32Array =>
      Float32Array.from([
        Math.cos(a),
        Math.sin(a),
        0,
        0,
        -Math.sin(a),
        Math.cos(a),
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
      ]);
    const mixed = blendBones(spin(0), spin(Math.PI / 4), 0.5);
    const column = [mixed[0] ?? 0, mixed[1] ?? 0, mixed[2] ?? 0];
    expect(Math.hypot(...column)).toBeCloseTo(1, 6);
  });
});

describe('transposeMatrices', () => {
  it('turns a row-major matrix into a column-major one', () => {
    const m = Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    expect(Array.from(transposeMatrices(m))).toEqual([
      1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16,
    ]);
  });

  it('does every matrix in the buffer', () => {
    const two = Float32Array.from([...Array(16).keys(), ...Array(16).keys()]);
    const done = transposeMatrices(two);
    expect(Array.from(done.slice(0, 16))).toEqual(Array.from(done.slice(16)));
    expect(done[1]).toBe(4);
  });

  it('is its own inverse', () => {
    const original = Float32Array.from({ length: 32 }, (_, i) => i * 1.5);
    const there = transposeMatrices(Float32Array.from(original));
    expect(Array.from(transposeMatrices(there))).toEqual(Array.from(original));
  });
});

describe('the shared filter vector', () => {
  /**
   * The analysis in `packages/pipeline/src/unproject/rq3/smoothing.py` is a
   * second implementation of this filter, because the RQ3 numbers are computed
   * in Python. Both read this file. If either drifts, both suites fail, which
   * is the point: the alternative is one thesis number quietly moving.
   */
  // No cast: TypeScript reads the JSON's own shape, which is stricter than
  // anything written here and catches a fixture that changed shape.
  const fixture = sharedFilterVector;

  it('is the viewer defaults it claims to be', () => {
    expect(fixture.options).toEqual<Smoothing>(DEFAULT_SMOOTHING);
  });

  it('is reproduced by the viewer filter', () => {
    const smoother = new TrackSmoother(DEFAULT_SMOOTHING);

    const produced = fixture.samples.map((sample) => {
      const [person] = smoother.frame(sample.time, [
        {
          trackId: 0,
          confidence: 1,
          joints: Float32Array.from([sample.x, sample.y, sample.z]),
        },
      ]);
      return Array.from(person?.joints ?? []);
    });

    produced.forEach((row, index) => {
      const want = fixture.expected[index];
      expect(want).toBeDefined();
      row.forEach((value, axis) => {
        expect(value).toBeCloseTo(want?.[axis] ?? Number.NaN, 5);
      });
    });
  });
});

describe('the shared signal vector', () => {
  /**
   * The second shared vector (ADR-0005 WP6): a noisy step in depth and a noisy
   * sine sideways at the slowest and fastest cadence the viewer sees, at the
   * nearest and farthest distance the range term is clamped over. Generated by
   * `unproject.rq3.characterise.parity_fixture`; the Python suite checks the
   * committed file is what that code produces, and this checks the viewer's
   * filter produces the same numbers.
   */
  const fixture = sharedSignalVector;

  it('is the viewer defaults it claims to be', () => {
    expect(fixture.options).toEqual<Smoothing>(DEFAULT_SMOOTHING);
  });

  it('spans both cadences and both distances', () => {
    const seen = fixture.cases.map((c) => `${String(c.cadence_hz)}@${String(c.distance_m)}`);
    expect(seen.sort()).toEqual(['2.5@2', '2.5@40', '25@2', '25@40']);
  });

  it('is reproduced by the viewer filter at every cadence and distance', () => {
    for (const testCase of fixture.cases) {
      const smoother = new TrackSmoother(DEFAULT_SMOOTHING);
      testCase.samples.forEach((sample, index) => {
        const [person] = smoother.frame(sample.time, [
          {
            trackId: 0,
            confidence: 1,
            joints: Float32Array.from([sample.x, sample.y, sample.z]),
          },
        ]);
        const want = testCase.expected[index];
        expect(want).toBeDefined();
        Array.from(person?.joints ?? []).forEach((value, axis) => {
          expect(value).toBeCloseTo(want?.[axis] ?? Number.NaN, 5);
        });
      });
    }
  });
});
