/**
 * The right-hand pane: the reconstruction, with the viewpoint under the
 * operator's control.
 *
 * The render loop asks `tick` for the time and `frames` for the data on every
 * displayed frame, without going through React: driving component state at
 * frame rate would put the framework between the operator's mouse and the
 * screen.
 */

import { GizmoHelper, GizmoViewport, Grid, OrbitControls } from '@react-three/drei';
import { Canvas, useThree } from '@react-three/fiber';
import { useEffect, useLayoutEffect, useMemo } from 'react';
import {
  BufferGeometry,
  Float32BufferAttribute,
  PerspectiveCamera,
  SRGBColorSpace,
  VideoTexture,
} from 'three';

import type { BodySource, TimelineFrame } from '../bodies';
import { CAMERA_SPACE_SIGNS } from '../convention';
import type { SourceCamera } from '../live';
import { BodyFigures } from './BodyFigures';
import { BodyMeshes } from './BodyMeshes';

/**
 * Where to stand.
 *
 * `source` is the camera that shot the footage, the one viewpoint the depths
 * were fitted to look right from. `off45` is a quarter turn away, where what a
 * single camera cannot know becomes visible. The difference between them is the
 * measurement, so both are one click apart.
 */
export type Viewpoint = 'source' | 'off45' | 'free';

export interface Framing {
  readonly centre: [number, number, number];
  readonly ground: number;
  readonly spread: number;
  /** Just past the furthest body, where the source image plane is hung. */
  readonly far: number;
}

const DEFAULT_FRAMING: Framing = { centre: [0, 0, -12], ground: -1.6, spread: 20, far: 30 };

/** Comfortable for looking around a scene, and nothing to do with any lens. */
const DEFAULT_FOV = 50;

/**
 * Where the bodies are, so the scene can be framed around them.
 *
 * Taken from the data: one clip puts people 12 metres out and the next puts
 * them 70, and a fixed frame would lose one of them.
 */
export function frameScene(frames: readonly TimelineFrame[]): Framing {
  const [signX, signY, signZ] = CAMERA_SPACE_SIGNS;
  let count = 0;
  let sumX = 0;
  let sumZ = 0;
  let lowest = 0;
  let nearest = Infinity;
  let furthest = 0;

  // A handful of moments spread over the timeline is plenty to frame with.
  const step = Math.max(1, Math.floor(frames.length / 8));
  for (let i = 0; i < frames.length; i += step) {
    for (const person of frames[i]?.people ?? []) {
      for (let bone = 0; bone < person.joints.length / 3; bone += 1) {
        const y = (person.joints[bone * 3 + 1] ?? 0) * signY;
        const z = (person.joints[bone * 3 + 2] ?? 0) * signZ;
        sumX += (person.joints[bone * 3] ?? 0) * signX;
        sumZ += z;
        count += 1;
        lowest = Math.min(lowest, y);
        nearest = Math.min(nearest, -z);
        furthest = Math.max(furthest, -z);
      }
    }
  }

  if (count === 0) return DEFAULT_FRAMING;
  return {
    centre: [sumX / count, lowest, sumZ / count],
    ground: lowest,
    spread: Math.max(furthest - nearest, 5),
    far: furthest * 1.1,
  };
}

/**
 * Vertical field of view, in degrees, that shows the whole source image.
 *
 * From the source viewpoint the view should be the camera's own, or the footage
 * and the bodies are drawn through different lenses and nothing lines up. The
 * reference clip is a long lens, about eleven degrees across, so a default of
 * fifty would show the image plane as a small rectangle mid-screen.
 *
 * Widened when the pane is narrower than the image, so the sides are not cut
 * off; the fit is the same one `object-fit: contain` does.
 */
export function sourceFieldOfView(camera: SourceCamera, paneAspect: number): number {
  const tanHalfWidth = camera.width / 2 / camera.fx;
  const tanHalfHeight = camera.height / 2 / camera.fy;
  const imageAspect = tanHalfWidth / tanHalfHeight;
  const tanHalf = paneAspect >= imageAspect ? tanHalfHeight : tanHalfWidth / paneAspect;
  return (2 * Math.atan(tanHalf) * 180) / Math.PI;
}

/** Moves the camera when a viewpoint is picked, then hands control back. */
function Viewpoints({
  view,
  framing,
  source,
}: {
  readonly view: Viewpoint;
  readonly framing: Framing;
  readonly source: SourceCamera | null;
}): null {
  const camera = useThree((state) => state.camera);
  const controls = useThree((state) => state.controls);
  const size = useThree((state) => state.size);
  // Fetched inside the effects below rather than closed over: the field of view
  // is set by assignment, and a value that came out of render may not be.
  const get = useThree((state) => state.get);

  useLayoutEffect(() => {
    const active = get().camera;
    if (!(active instanceof PerspectiveCamera)) return;
    // The source viewpoint borrows the real lens; everything else is a viewing
    // angle of our own choosing and keeps a comfortable one.
    const wanted =
      view === 'source' && source !== null
        ? sourceFieldOfView(source, size.width / size.height)
        : DEFAULT_FOV;
    if (active.fov !== wanted) {
      active.fov = wanted;
      active.updateProjectionMatrix();
    }
  }, [view, source, size, get]);

  useLayoutEffect(() => {
    if (view === 'free') return;
    const [cx, , cz] = framing.centre;
    const back = framing.spread * 0.55;
    const orbit = controls as {
      target?: { set: (x: number, y: number, z: number) => void };
      update?: () => void;
    } | null;

    if (view === 'source') {
      // Exactly where the camera stood, pointing exactly where it pointed. The
      // convention says camera space has its origin at the camera looking down
      // -Z, so anything else here - even a few centimetres of lift to make the
      // ground read better - shows the bodies through a lens that never existed
      // and pulls the footage out of register with them.
      camera.position.set(0, 0, 0);
      orbit?.target?.set(0, 0, -Math.max(framing.far, 1));
    } else {
      camera.position.set(cx + back * 0.75, framing.ground + back * 0.35, cz + back * 0.75);
      orbit?.target?.set(cx, framing.ground, cz);
    }
    camera.lookAt(
      view === 'source' ? 0 : cx,
      view === 'source' ? 0 : framing.ground,
      view === 'source' ? -Math.max(framing.far, 1) : cz,
    );
    orbit?.update?.();
  }, [view, framing, camera, controls]);

  return null;
}

/**
 * The footage, drawn where it was actually taken from.
 *
 * Not a backdrop behind the canvas but the source camera's image plane, put in
 * the scene at the size and place its own intrinsics say. From the source
 * viewpoint that makes it line up with the bodies exactly, which is worth more
 * than the picture: if the skeletons sit on the people, the recorded intrinsics
 * and the coordinate convention are both right. From anywhere else you see the
 * plane edge-on, which is a fair description of what one camera saw.
 *
 * It is placed beyond the furthest body so the figures draw in front of it.
 */
function SourceBackdrop({
  video,
  camera,
  opacity,
  distance,
}: {
  readonly video: HTMLVideoElement;
  readonly camera: SourceCamera;
  readonly opacity: number;
  readonly distance: number;
}): React.JSX.Element {
  const texture = useMemo(() => {
    const made = new VideoTexture(video);
    made.colorSpace = SRGBColorSpace;
    return made;
  }, [video]);

  useEffect(
    () => () => {
      texture.dispose();
    },
    [texture],
  );

  // A pixel at (u, v) and depth d sits at ((u - cx) d / fx, (v - cy) d / fy, d)
  // in the camera's own axes, and three.js flips the last two.
  const width = (distance * camera.width) / camera.fx;
  const height = (distance * camera.height) / camera.fy;
  const offsetX = (distance * (camera.width / 2 - camera.cx)) / camera.fx;
  const offsetY = (-distance * (camera.height / 2 - camera.cy)) / camera.fy;

  return (
    <mesh position={[offsetX, offsetY, -distance]} renderOrder={-1}>
      <planeGeometry args={[width, height]} />
      <meshBasicMaterial
        map={texture}
        transparent
        opacity={opacity}
        // Behind everything, and never occluding a body in front of it.
        depthWrite={false}
        toneMapped={false}
      />
    </mesh>
  );
}

/**
 * The trajectory of one track, drawn as a plain polyline.
 *
 * White and thin on purpose: it is a reference line over the scene, not another
 * body. Depth-tested like everything else, so it goes behind the people it
 * belongs to rather than floating over them.
 */
function TrackPath({ points }: { readonly points: readonly [number, number, number][] }) {
  const geometry = useMemo(() => {
    const buffer = new BufferGeometry();
    buffer.setAttribute(
      'position',
      new Float32BufferAttribute(
        points.flatMap(([x, y, z]) => [x, y, z]),
        3,
      ),
    );
    return buffer;
  }, [points]);

  useEffect(
    () => () => {
      geometry.dispose();
    },
    [geometry],
  );

  return (
    <line>
      <primitive object={geometry} attach="geometry" />
      <lineBasicMaterial color="#ffffff" transparent opacity={0.7} />
    </line>
  );
}

interface Stage3DProps {
  readonly frames: () => readonly TimelineFrame[];
  readonly boneParents: readonly number[];
  readonly capacity: number;
  readonly tick: (delta: number) => number;
  readonly framing: Framing;
  readonly view: Viewpoint;
  readonly backdrop: { readonly video: HTMLVideoElement; readonly camera: SourceCamera } | null;
  readonly backdropOpacity: number;
  /** Bodies rather than skeletons, once there is something to build them from. */
  readonly bodies: BodySource | null;
  /**
   * Told when the operator moves the camera by hand.
   *
   * The two placed viewpoints are positions, not locks: dragging away from one
   * leaves it, and the button would otherwise keep claiming you are somewhere
   * you are not.
   */
  readonly onUserMove: () => void;
  /**
   * Where one track went, oldest point first, or null when none is selected.
   *
   * Drawn as it is: the line joins the frames the track was found in and skips
   * the ones it was not, so a break in the line is a break in the data.
   */
  readonly path: readonly [number, number, number][] | null;
}

export function Stage3D({
  frames,
  boneParents,
  capacity,
  tick,
  framing,
  view,
  backdrop,
  backdropOpacity,
  bodies,
  path,
  onUserMove,
}: Stage3DProps): React.JSX.Element {
  const distance = framing.spread * 0.8;
  const initial = useMemo<[number, number, number]>(
    () => [framing.centre[0], framing.ground + distance * 0.35, distance * 0.4],
    [framing, distance],
  );

  return (
    <Canvas camera={{ position: initial, fov: DEFAULT_FOV, near: 0.1, far: framing.far * 4 }}>
      <color attach="background" args={['#0f1419']} />
      {/* Flat light is enough for lines; a body needs a direction to read as a
          shape rather than a silhouette. */}
      <ambientLight intensity={bodies === null ? 1 : 0.55} />
      {bodies !== null && <directionalLight position={[2, 4, 3]} intensity={1.6} />}

      {bodies === null ? (
        <BodyFigures frames={frames} boneParents={boneParents} capacity={capacity} tick={tick} />
      ) : (
        <BodyMeshes frames={frames} bodies={bodies} tick={tick} />
      )}

      {path !== null && path.length > 1 && <TrackPath points={path} />}

      {backdrop !== null && backdropOpacity > 0 && (
        <SourceBackdrop
          video={backdrop.video}
          camera={backdrop.camera}
          opacity={backdropOpacity}
          distance={framing.far}
        />
      )}

      {/* The camera that shot the footage. Everything is placed relative to it,
          so the further the operator orbits from here, the less the depths can
          be trusted. */}
      <mesh position={[0, 0, 0]}>
        <sphereGeometry args={[framing.spread * 0.012, 12, 12]} />
        <meshBasicMaterial color="#ffb454" />
      </mesh>

      <Grid
        position={[framing.centre[0], framing.ground, framing.centre[2]]}
        args={[framing.spread * 3, framing.spread * 3]}
        cellSize={1}
        sectionSize={5}
        cellColor="#243b4a"
        sectionColor="#31576b"
        fadeDistance={framing.spread * 4}
        infiniteGrid
      />
      <OrbitControls
        makeDefault
        enableDamping
        dampingFactor={0.1}
        target={framing.centre}
        // `onStart` is a pointer press, so the programmatic `update()` in
        // Viewpoints does not trip it.
        onStart={onUserMove}
      />
      <Viewpoints view={view} framing={framing} source={backdrop?.camera ?? null} />
      <GizmoHelper alignment="bottom-right" margin={[64, 64]}>
        <GizmoViewport axisColors={['#f07178', '#c3e88d', '#82aaff']} labelColor="#0f1419" />
      </GizmoHelper>
    </Canvas>
  );
}
