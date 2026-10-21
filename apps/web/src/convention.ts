/**
 * The one coordinate conversion the viewer makes.
 *
 * Stage [B] stores `camera_relative_opencv_metres`: the camera that shot the
 * footage sits at the origin and everything is placed relative to it. OpenCV has
 * +Y down and +Z forward along the view; three.js has +Y up and a camera looking
 * down -Z, so both axes flip.
 *
 * Getting it wrong renders a scene that looks fine until you orbit it, at which
 * point the crowd is upside down or behind you.
 */

import { Vector3 } from 'three';

/**
 * Exported so the hot path that fills a vertex buffer and the helper below
 * cannot drift apart: one source of truth, one test.
 */
export const CAMERA_SPACE_SIGNS = [1, -1, -1] as const;

/** Readable form of the flip, for anything outside a hot loop. */
export function cameraSpaceToWorld(x: number, y: number, z: number): Vector3 {
  const [sx, sy, sz] = CAMERA_SPACE_SIGNS;
  return new Vector3(x * sx, y * sy, z * sz);
}
