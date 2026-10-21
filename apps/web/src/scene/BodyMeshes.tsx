/**
 * The bodies as bodies, skinned on the GPU.
 *
 * The artifact stores what a body is made of and what its bones are doing, and
 * this is the arithmetic that turns one into the other: each vertex is carried
 * by the four bones that pull on it hardest, from the rest pose its track was
 * measured in to wherever the bones are now.
 *
 * three.js does the skinning in a shader, so the loop's work per frame is
 * writing bone matrices - a couple of thousand of them - not moving vertices.
 *
 * Everything here is root-relative and in the camera's own axes. The parent
 * group turns the whole scene the right way up; see `convention.ts`.
 */

import { useFrame } from '@react-three/fiber';
import { useEffect, useRef } from 'react';
import {
  Bone,
  BufferAttribute,
  BufferGeometry,
  Color,
  DetachedBindMode,
  type Group,
  Matrix4,
  MeshStandardMaterial,
  Skeleton,
  SkinnedMesh,
  Uint16BufferAttribute,
} from 'three';

import type { BodySource, TimelineFrame } from '../bodies';
import { sampleAt } from '../bodies';
import { TRACK_COLORS } from '../palette';

interface BodyMeshesProps {
  readonly frames: () => readonly TimelineFrame[];
  readonly bodies: BodySource;
  /** Advances the playhead and returns the time to draw. */
  readonly tick: (delta: number) => number;
}

/**
 * One skinned body, built the first time its track has one to build from.
 *
 * A finished run knows every body up front and a stream learns them as people
 * appear, so this is lazy either way rather than two paths.
 *
 * Topology and skinning are shared: those attributes are the same object in
 * every geometry rather than a copy each. Only the rest pose is the track's
 * own, because only shape is.
 */
function buildBody(
  source: BodySource,
  track: number,
  shared: SharedAttributes,
): SkinnedMesh | null {
  const rest = source.rest(track);
  if (rest === null) return null;

  const geometry = new BufferGeometry();
  geometry.setIndex(shared.index);
  geometry.setAttribute('position', new BufferAttribute(rest.vertices.slice(), 3));
  geometry.setAttribute('skinIndex', shared.skinIndex);
  geometry.setAttribute('skinWeight', shared.skinWeight);
  // Anny ships no normals, and a body without them is a silhouette.
  geometry.computeVertexNormals();

  // Flat bones: the transforms are already absolute, so there is no hierarchy
  // to walk and nothing should recompute them from parents.
  const bones = Array.from({ length: source.topology.boneCount }, () => {
    const bone = new Bone();
    bone.matrixAutoUpdate = false;
    return bone;
  });
  const inverses = bones.map((_, bone) => new Matrix4().fromArray(rest.bones, bone * 16).invert());

  const body = new SkinnedMesh(
    geometry,
    new MeshStandardMaterial({
      color: new Color(TRACK_COLORS[track % TRACK_COLORS.length] ?? '#ffffff'),
      roughness: 0.85,
      metalness: 0,
    }),
  );
  // Detached, bound to the identity. Left attached, three.js keeps
  // `bindMatrixInverse` equal to the inverse of the mesh's world matrix, which
  // cancels the position we are about to give it and piles every body up at
  // the origin.
  body.bindMode = DetachedBindMode;
  body.bind(new Skeleton(bones, inverses), new Matrix4());
  body.frustumCulled = false;
  body.visible = false;
  return body;
}

interface SharedAttributes {
  readonly index: Uint16BufferAttribute;
  readonly skinIndex: Uint16BufferAttribute;
  readonly skinWeight: BufferAttribute;
}

export function BodyMeshes({ frames, bodies, tick }: BodyMeshesProps): React.JSX.Element {
  const group = useRef<Group>(null);
  // Built in an effect and reached through a ref: the loop mutates these every
  // frame, and a value that came out of render may not be mutated.
  const built = useRef<Map<number, SkinnedMesh> | null>(null);
  const shared = useRef<SharedAttributes | null>(null);

  useEffect(() => {
    const made = new Map<number, SkinnedMesh>();
    built.current = made;
    shared.current = {
      index: new Uint16BufferAttribute(bodies.topology.faces, 1),
      skinIndex: new Uint16BufferAttribute(bodies.topology.skinIndices, 4),
      skinWeight: new BufferAttribute(bodies.topology.skinWeights, 4),
    };
    return () => {
      built.current = null;
      shared.current = null;
      for (const body of made.values()) {
        body.removeFromParent();
        body.geometry.dispose();
        // The type allows an array; this one is always the single material
        // built above.
        for (const material of [body.material].flat()) material.dispose();
        body.skeleton.dispose();
      }
    };
  }, [bodies]);

  useFrame((_state, delta) => {
    const parent = group.current;
    const made = built.current;
    const attributes = shared.current;
    if (parent === null || made === null || attributes === null) return;

    const people = sampleAt(frames(), tick(delta));
    const shown = new Set<number>();

    for (const person of people) {
      if (person.bones === undefined) continue;
      let body = made.get(person.trackId);
      if (body === undefined) {
        // A stream may not have sent this person's body yet; they are simply
        // not drawn until it arrives.
        body = buildBody(bodies, person.trackId, attributes) ?? undefined;
        if (body === undefined) continue;
        made.set(person.trackId, body);
      }
      shown.add(person.trackId);

      const bones = body.skeleton.bones;
      for (let index = 0; index < bones.length; index += 1) {
        const bone = bones[index];
        if (bone === undefined) continue;
        // Set the world matrix directly. The transforms already say where each
        // bone is rather than where it sits relative to a parent, and no bone
        // is in the scene graph, so nothing recomputes these from a hierarchy.
        bone.matrix.fromArray(person.bones, index * 16);
        bone.matrixWorld.copy(bone.matrix);
      }
      body.skeleton.update();

      // The pose is root-relative; the root itself is the person's place in the
      // camera's frame.
      body.position.set(person.joints[0] ?? 0, person.joints[1] ?? 0, person.joints[2] ?? 0);
      body.updateMatrix();
      body.visible = true;
      if (body.parent !== parent) parent.add(body);
    }

    for (const track of made.keys()) {
      const idle = made.get(track);
      if (idle !== undefined && !shown.has(track)) idle.visible = false;
    }
  });

  // Rotating half a turn about X is the axis flip of `convention.ts` written as
  // a rotation. A negative scale would do it too and would invert every face.
  return <group ref={group} rotation={[Math.PI, 0, 0]} />;
}
