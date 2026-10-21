/**
 * Every body on screen, drawn as a skeleton, updated from the render loop.
 *
 * The loop runs at whatever rate the display gives it and draws the most recent
 * data it has. It never waits: with inference at a few frames a second, a loop
 * that waited would stutter every time the operator moved the camera.
 *
 * One `LineSegments` for the whole crowd rather than an object per person. The
 * vertex buffers are sized once and refilled in place, so a frame costs no
 * allocation and the draw range hides whoever is absent.
 */

import { useFrame } from '@react-three/fiber';
import { useMemo, useRef } from 'react';
import { BufferGeometry, Color } from 'three';

import type { TimelineFrame } from '../bodies';
import { sampleAt } from '../bodies';
import { TRACK_COLORS } from '../palette';
import { CAMERA_SPACE_SIGNS } from '../convention';

interface BodyFiguresProps {
  /** A getter, not an array: a live timeline grows under the loop's feet. */
  readonly frames: () => readonly TimelineFrame[];
  readonly boneParents: readonly number[];
  /** Most people to draw at once. Buffers are sized for this. */
  readonly capacity: number;
  /** Advances the playhead and returns the time to draw. */
  readonly tick: (delta: number) => number;
}

/** Child-to-parent pairs, which is what a skeleton is to draw. */
function boneSegments(parents: readonly number[]): [number, number][] {
  const segments: [number, number][] = [];
  parents.forEach((parent, bone) => {
    if (parent >= 0) segments.push([bone, parent]);
  });
  return segments;
}

export function BodyFigures({
  frames,
  boneParents,
  capacity,
  tick,
}: BodyFiguresProps): React.JSX.Element {
  const geometryRef = useRef<BufferGeometry>(null);

  const segments = useMemo(() => boneSegments(boneParents), [boneParents]);
  const palette = useMemo(() => TRACK_COLORS.map((hex) => new Color(hex)), []);

  const vertices = Math.max(1, capacity) * segments.length * 2;
  const initial = useMemo(
    () => ({ positions: new Float32Array(vertices * 3), colors: new Float32Array(vertices * 3) }),
    [vertices],
  );

  useFrame((_state, delta) => {
    const geometry = geometryRef.current;
    if (geometry === null) return;

    const people = sampleAt(frames(), tick(delta));
    const position = geometry.getAttribute('position');
    const color = geometry.getAttribute('color');
    const positions = position.array as Float32Array;
    const colors = color.array as Float32Array;
    const [signX, signY, signZ] = CAMERA_SPACE_SIGNS;

    let vertex = 0;
    for (const person of people) {
      const tint = palette[person.trackId % palette.length];
      for (const [child, parent] of segments) {
        if (vertex + 2 > vertices) break;
        for (const bone of [child, parent]) {
          const at = bone * 3;
          const offset = vertex * 3;
          positions[offset] = (person.joints[at] ?? 0) * signX;
          positions[offset + 1] = (person.joints[at + 1] ?? 0) * signY;
          positions[offset + 2] = (person.joints[at + 2] ?? 0) * signZ;
          colors[offset] = tint?.r ?? 1;
          colors[offset + 1] = tint?.g ?? 1;
          colors[offset + 2] = tint?.b ?? 1;
          vertex += 1;
        }
      }
    }

    geometry.setDrawRange(0, vertex);
    position.needsUpdate = true;
    color.needsUpdate = true;
  });

  return (
    // Culling is off: the bounding sphere would have to be recomputed from
    // thousands of vertices every frame to stay honest.
    <lineSegments frustumCulled={false}>
      <bufferGeometry ref={geometryRef}>
        <bufferAttribute attach="attributes-position" args={[initial.positions, 3]} />
        <bufferAttribute attach="attributes-color" args={[initial.colors, 3]} />
      </bufferGeometry>
      <lineBasicMaterial vertexColors />
    </lineSegments>
  );
}
