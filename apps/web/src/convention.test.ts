import { describe, expect, it } from 'vitest';

import { CAMERA_SPACE_SIGNS, cameraSpaceToWorld } from './convention';
import { sourceFieldOfView } from './scene/Stage3D';

describe('cameraSpaceToWorld', () => {
  it('puts a point in front of the camera in front of the viewer', () => {
    // +Z is forward in OpenCV; a three.js camera at the origin looks down -Z.
    expect(cameraSpaceToWorld(0, 0, 10).z).toBeLessThan(0);
  });

  it('keeps right on the right and turns down into down', () => {
    expect(cameraSpaceToWorld(2, 0, 10).x).toBeGreaterThan(0);
    // +Y is down in OpenCV, and three.js has +Y up.
    expect(cameraSpaceToWorld(0, 1.5, 10).y).toBeLessThan(0);
  });

  it('leaves the camera itself at the origin', () => {
    expect(cameraSpaceToWorld(0, 0, 0).lengthSq()).toBe(0);
  });

  it('preserves distances, since it is only a pair of axis flips', () => {
    const a = cameraSpaceToWorld(1, 2, 3);
    const b = cameraSpaceToWorld(4, 6, 3);
    expect(a.distanceTo(b)).toBeCloseTo(5, 9);
  });

  it('agrees with the signs the vertex buffers are filled with', () => {
    // BodyFigures writes straight into a Float32Array for speed. This is what
    // stops that copy and the helper above from drifting apart.
    const [sx, sy, sz] = CAMERA_SPACE_SIGNS;
    const point = cameraSpaceToWorld(3, 5, 7);
    expect([point.x, point.y, point.z]).toEqual([3 * sx, 5 * sy, 7 * sz]);
  });
});

describe('sourceFieldOfView', () => {
  // What the model returned for the reference clip: a long lens, about eleven
  // degrees across, not the fifty a viewer would otherwise assume.
  const crowd = { width: 1280, height: 720, fx: 6510.3, fy: 6510.3, cx: 640, cy: 360 };

  it('matches the lens the footage was shot with', () => {
    const vertical = (2 * Math.atan(360 / 6510.3) * 180) / Math.PI;
    expect(sourceFieldOfView(crowd, 1280 / 720)).toBeCloseTo(vertical, 6);
  });

  it('opens up rather than cropping when the pane is narrow', () => {
    // Cutting the sides off would hide people who are in the footage.
    const wide = sourceFieldOfView(crowd, 16 / 9);
    expect(sourceFieldOfView(crowd, 4 / 3)).toBeGreaterThan(wide);
  });

  it('does not shrink when the pane is wider than the image', () => {
    // Past that the height is the limit, and the extra width is just empty
    // scene either side.
    expect(sourceFieldOfView(crowd, 21 / 9)).toBeCloseTo(sourceFieldOfView(crowd, 16 / 9), 6);
  });

  it('is a wide angle for a wide lens', () => {
    const phone = { width: 1280, height: 720, fx: 1100, fy: 1100, cx: 640, cy: 360 };
    expect(sourceFieldOfView(phone, 16 / 9)).toBeGreaterThan(30);
    expect(sourceFieldOfView(crowd, 16 / 9)).toBeLessThan(15);
  });
});
