/**
 * Tests for who owns the screen when network work resolves out of order.
 *
 * Every promise here is resolved by hand, so "run A comes back after run B" is
 * a fact of the test rather than a race it hopes to provoke. The 3D scene and
 * the video element are stubbed: neither has anything to do with ownership, and
 * WebGL does not exist in jsdom.
 */

import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { RunSummary } from './api';
import type { BodiesArtifact } from './bodies';
import { bodiesArtifactSchema } from './bodies';

vi.mock('./scene/Stage3D', () => ({
  // Exposes onUserMove as a button, so the wiring can be driven without WebGL.
  // What drei does with it is library behaviour, not this app's.
  Stage3D: ({ onUserMove }: { onUserMove: () => void }) => (
    <button type="button" data-testid="orbit" onClick={onUserMove} />
  ),
  frameScene: () => ({ centre: [0, 0, -12], ground: -1.6, spread: 20, far: 30 }),
}));
const sourcesSeen: unknown[] = [];
vi.mock('./scene/SourceVideo', () => ({
  SourceVideo: ({ source }: { source: unknown }) => {
    sourcesSeen.push(source);
    return null;
  },
}));
vi.mock('./api', () => ({
  deleteRun: vi.fn(),
  listRuns: vi.fn(),
  loadArtifact: vi.fn(),
  loadArray: vi.fn(),
  loadMesh: vi.fn(),
  loadTopology: vi.fn(),
  uploadVideo: vi.fn(),
  bodySourceFromMesh: vi.fn(() => ({ topology: null, rest: () => null })),
}));
vi.mock('./live', async (original) => ({
  ...(await original<typeof import('./live')>()),
  captureFrames: () => () => undefined,
  LiveSession: class {
    open = vi.fn();
    close = vi.fn();
    timeline = () => [];
    bodySource = () => null;
  },
}));

const api = await import('./api');
const { App } = await import('./App');

/** A promise whose settling this test decides. */
function deferred<T>(): { promise: Promise<T>; resolve: (value: T) => void } {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((settle) => {
    resolve = settle;
  });
  return { promise, resolve };
}

function run(name: string): RunSummary {
  return {
    path: `artifacts/${name}/bodies/abc`,
    experiment: name,
    hash: 'abc',
    frames: 2,
    people: 1,
    max_people: 1,
    tracks: 1,
    duration_seconds: 1,
    video: null,
  };
}

/** Parsed rather than cast, so the fixture is a real artifact or the test says so. */
function artifact(withMesh: boolean): BodiesArtifact {
  const model = {
    name: 'anny' as const,
    version: '0.6.0',
    bone_count: 1,
    shape_count: 11,
    bone_parents: [-1],
  };
  return bodiesArtifactSchema.parse({
    schema_version: 2,
    convention: 'camera_relative_opencv_metres',
    ingest_hash: '0'.repeat(12),
    body_model: model,
    estimator: {
      name: 'multihmr2',
      checkpoint_sha256: 'f'.repeat(64),
      backbone_repo: 'facebookresearch/dinov3',
      backbone_ref: '0'.repeat(40),
      config_hash: 'deadbeef',
    },
    parameters: { path: 'bodies.bin', person_count: 1, arrays: [] },
    mesh: withMesh
      ? {
          path: 'mesh.bin',
          vertex_count: 1,
          face_count: 1,
          influences: 4,
          tracks: [0],
          arrays: [],
        }
      : null,
    frames: [
      {
        frame_index: 0,
        source_index: 0,
        timestamp_seconds: 0,
        camera: {
          width: 64,
          height: 48,
          fx: 100,
          fy: 100,
          cx: 32,
          cy: 24,
          source: 'estimated' as const,
        },
        people: [
          {
            parameter_index: 0,
            track_id: 0,
            confidence: 0.9,
            box_xyxy: [0, 0, 10, 20],
          },
        ],
      },
    ],
    summary: {
      frames_total: 1,
      frames_with_people: 1,
      people_total: 1,
      max_people_in_frame: 1,
      confidence_threshold: 0.4,
      track_count: 1,
    },
  });
}

/** The mocked module, typed enough to set return values on. */
const mocked = api as unknown as {
  listRuns: ReturnType<typeof vi.fn>;
  loadArtifact: ReturnType<typeof vi.fn>;
  loadArray: ReturnType<typeof vi.fn>;
  loadMesh: ReturnType<typeof vi.fn>;
  loadTopology: ReturnType<typeof vi.fn>;
  uploadVideo: ReturnType<typeof vi.fn>;
  deleteRun: ReturnType<typeof vi.fn>;
};

const A = run('alpha');
const B = run('bravo');

async function openApp(runs: RunSummary[] = []): Promise<void> {
  mocked.listRuns.mockResolvedValue(runs);
  render(<App />);
  await waitFor(() => {
    expect(screen.getByText('Finished runs')).toBeDefined();
  });
}

function clickRun(name: string): void {
  const button = screen
    .getAllByRole('button')
    .find((element) => element.textContent.includes(name));
  if (button === undefined) throw new Error(`no button for ${name}`);
  fireEvent.click(button);
}

function selected(): string | null {
  return document.querySelector('.run.selected')?.querySelector('.name')?.textContent ?? null;
}

beforeEach(() => {
  vi.clearAllMocks();
  mocked.loadArray.mockResolvedValue(new Float32Array([0, 0, 1]));
  // The default settles at once, so the list's own auto-open is out of the way
  // before a test starts queueing promises it controls.
  mocked.loadArtifact.mockResolvedValue(artifact(false));
});

/** What the footer is currently saying. The text is split across nodes. */
function status(): string {
  return document.querySelector('.status')?.textContent ?? '';
}

/**
 * Run every pending continuation to the end, then let React paint.
 *
 * A handful of `await Promise.resolve()` is not enough and fails open: the
 * chain being tested has more awaits than that, so the assertion would run
 * before the stale completion had a chance to do damage.
 */
async function settle(): Promise<void> {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 0));
  });
}

afterEach(() => {
  cleanup();
});

describe('opening one run while another is still loading', () => {
  it('shows the run that was asked for last, whichever arrives first', async () => {
    // The list opens the newest by itself, so start from a settled state.
    await openApp([A, B]);
    await waitFor(() => {
      expect(selected()).toBe('alpha');
    });

    const slow = deferred<BodiesArtifact>();
    const quick = deferred<BodiesArtifact>();
    mocked.loadArtifact.mockReturnValueOnce(slow.promise).mockReturnValueOnce(quick.promise);
    clickRun('alpha');
    clickRun('bravo');

    quick.resolve(artifact(false));
    await waitFor(() => {
      expect(selected()).toBe('bravo');
    });
    // Alpha lands second, having been asked for first.
    slow.resolve(artifact(false));
    await settle();

    expect(selected()).toBe('bravo');
  });

  it('does not let the superseded run clear the status of the current one', async () => {
    await openApp([A, B]);
    await waitFor(() => {
      expect(selected()).toBe('alpha');
    });
    const slow = deferred<BodiesArtifact>();
    const pending = deferred<BodiesArtifact>();
    mocked.loadArtifact.mockReturnValueOnce(slow.promise).mockReturnValueOnce(pending.promise);

    clickRun('alpha');
    clickRun('bravo');
    await waitFor(() => {
      expect(status()).toContain('Loading bravo');
    });
    slow.resolve(artifact(false));
    await settle();

    // Bravo is still loading; alpha finishing must not say otherwise.
    expect(status()).toContain('Loading bravo');
  });

  it('keeps the error of a superseded run off the screen', async () => {
    await openApp([A, B]);
    await waitFor(() => {
      expect(selected()).toBe('alpha');
    });
    const failing = deferred<BodiesArtifact>();
    const pending = deferred<BodiesArtifact>();
    mocked.loadArtifact.mockReturnValueOnce(failing.promise).mockReturnValueOnce(pending.promise);

    clickRun('alpha');
    clickRun('bravo');
    failing.resolve(Promise.reject(new Error('alpha is gone')) as unknown as BodiesArtifact);
    await settle();

    expect(document.querySelector('.error')).toBeNull();
  });
});

describe('a mesh that arrives after the run changed', () => {
  it('is not put on the run that is showing now', async () => {
    mocked.loadArtifact.mockResolvedValue(artifact(true));
    await openApp([A, B]);
    await waitFor(() => {
      expect(selected()).toBe('alpha');
    });

    // Alpha's body and bravo's are told apart by what they contain, so the
    // assertion is about which one reached the renderer rather than about how
    // many fetches happened.
    const alphaMesh = deferred<unknown>();
    const bravoMesh = deferred<unknown>();
    mocked.loadMesh.mockReturnValueOnce(alphaMesh.promise).mockReturnValueOnce(bravoMesh.promise);
    mocked.loadArray.mockResolvedValue(new Float32Array([0, 0, 1]));
    fireEvent.click(screen.getByText('Mesh'));

    // Switch run while alpha's 45 MB is still in the air.
    clickRun('bravo');
    await waitFor(() => {
      expect(selected()).toBe('bravo');
    });

    alphaMesh.resolve({ from: 'alpha' });
    await settle();

    expect(api.bodySourceFromMesh).not.toHaveBeenCalledWith({ from: 'alpha' });
  });

  it('is fetched again for the run that replaced it, without touching the control', async () => {
    // The choice survives the change of run but the body does not, so the
    // new run has to fetch its own. Before this, Mesh stayed picked over a
    // screen of skeletons until it was toggled off and on again.
    mocked.loadArtifact.mockResolvedValue(artifact(true));
    mocked.loadMesh.mockResolvedValue({ from: 'whichever' });
    mocked.loadArray.mockResolvedValue(new Float32Array([0, 0, 1]));
    await openApp([A, B]);
    await waitFor(() => {
      expect(selected()).toBe('alpha');
    });

    fireEvent.click(screen.getByText('Mesh'));
    await settle();
    const mesh = screen.getByText('Mesh');
    expect(mesh.className).toBe('selected');
    const before = mocked.loadMesh.mock.calls.length;

    clickRun('bravo');

    // waitFor, not a single settle: the fetch is deferred by a tick to avoid a
    // cascading render, so it lands a macrotask after the run does.
    await waitFor(() => {
      expect(mocked.loadMesh.mock.calls.length).toBeGreaterThan(before);
    });
    expect(mesh.className).toBe('selected');
  });
});

describe('an upload that finishes after the operator moved on', () => {
  it('does not start a session nobody asked for', async () => {
    await openApp([A]);
    await waitFor(() => {
      expect(selected()).toBe('alpha');
    });

    const upload = deferred<{ path: string; name: string }>();
    mocked.uploadVideo.mockReturnValue(upload.promise);
    const input = document.querySelector('input[type="file"]');
    if (input === null) throw new Error('no file input');
    fireEvent.change(input, { target: { files: [new File(['x'], 'clip.mp4')] } });

    // The operator goes back to a finished run mid-upload.
    clickRun('alpha');
    await waitFor(() => {
      expect(selected()).toBe('alpha');
    });

    upload.resolve({ path: 'uploads/1.mp4', name: 'clip.mp4' });
    await settle();

    expect(selected()).toBe('alpha');
    expect(screen.queryByText(/clip\.mp4/)).toBeNull();
  });
});

describe('camera permission granted after the operator moved on', () => {
  it('stops the tracks instead of leaving the light on', async () => {
    await openApp([A]);
    await waitFor(() => {
      expect(selected()).toBe('alpha');
    });

    const permission = deferred<MediaStream>();
    const stop = vi.fn();
    const stream = { getTracks: () => [{ stop }] } as unknown as MediaStream;
    // Only the one method is needed, and spreading the real navigator would
    // flatten its prototype.
    vi.stubGlobal('navigator', { mediaDevices: { getUserMedia: () => permission.promise } });

    fireEvent.click(screen.getByText('Use the camera'));
    // Answered only after the operator has gone back to a run.
    clickRun('alpha');
    await waitFor(() => {
      expect(selected()).toBe('alpha');
    });

    permission.resolve(stream);
    await settle();

    expect(stop).toHaveBeenCalledTimes(1);
    vi.unstubAllGlobals();
  });
});

describe('removing a run', () => {
  function removeButton(name: string): HTMLElement {
    const row = [...document.querySelectorAll('.runs li')].find((li) =>
      li.textContent.includes(name),
    );
    const button = row?.querySelector('.remove');
    if (!(button instanceof HTMLElement)) throw new Error(`no remove button for ${name}`);
    return button;
  }

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('asks before it deletes anything', () => {
    // Irreversible, and the artifact is the evidence behind any number drawn
    // from it. ADR-0002 §10.5.
    const confirm = vi.fn(() => false);
    vi.stubGlobal('confirm', confirm);

    return openApp([A, B]).then(() => {
      fireEvent.click(removeButton('alpha'));

      expect(confirm).toHaveBeenCalledOnce();
      expect(mocked.deleteRun).not.toHaveBeenCalled();
    });
  });

  it('removes it once the operator says yes', async () => {
    vi.stubGlobal('confirm', () => true);
    mocked.deleteRun.mockResolvedValue(undefined);
    await openApp([A, B]);
    await waitFor(() => {
      expect(selected()).toBe('alpha');
    });

    fireEvent.click(removeButton('bravo'));
    await settle();

    expect(mocked.deleteRun).toHaveBeenCalledWith(B.path);
    expect(screen.queryByText('bravo')).toBeNull();
  });

  it('clears the screen when the run being shown is the one removed', async () => {
    vi.stubGlobal('confirm', () => true);
    mocked.deleteRun.mockResolvedValue(undefined);
    await openApp([A, B]);
    await waitFor(() => {
      expect(selected()).toBe('alpha');
    });

    fireEvent.click(removeButton('alpha'));
    await settle();

    // Leaving it drawn would show bodies read from a file that is gone.
    expect(selected()).toBeNull();
  });

  it('says so when the server refuses', async () => {
    vi.stubGlobal('confirm', () => true);
    mocked.deleteRun.mockRejectedValue(new Error('could not remove the run (400)'));
    await openApp([A, B]);
    await waitFor(() => {
      expect(selected()).toBe('alpha');
    });

    fireEvent.click(removeButton('bravo'));
    await settle();

    expect(document.querySelector('.error')?.textContent).toContain('could not remove');
    expect(screen.queryByText('bravo')).not.toBeNull();
  });
});

describe('moving the camera by hand', () => {
  function currentView(): string | undefined {
    return [...document.querySelectorAll('.views button')].find((b) =>
      b.className.includes('selected'),
    )?.textContent;
  }

  it('leaves the viewpoint that was clicked', async () => {
    // The two placed viewpoints are positions, not locks. A button still
    // claiming "Source view" after the operator has orbited away from it is
    // telling them something untrue about what they are looking at.
    await openApp([A]);
    await waitFor(() => {
      expect(selected()).toBe('alpha');
    });
    fireEvent.click(screen.getByText('Source view'));
    expect(currentView()).toBe('Source view');

    fireEvent.click(screen.getByTestId('orbit'));

    expect(currentView()).toBe('Free');
  });

  it('does the same from the off-axis view', async () => {
    await openApp([A]);
    await waitFor(() => {
      expect(selected()).toBe('alpha');
    });
    fireEvent.click(screen.getByText('45° off'));

    fireEvent.click(screen.getByTestId('orbit'));

    expect(currentView()).toBe('Free');
  });
});

describe('the clip while the operator fiddles with the view', () => {
  // A run with footage on disk, so App renders the video rather than the
  // placeholder it shows when there is nothing to play.
  const WITH_VIDEO: RunSummary = { ...A, video: '/datasets/alpha/clip.mp4' };

  it('is not handed a new source when the viewpoint changes', async () => {
    // SourceVideo's effect depends on the source object, and that effect
    // touches the media element. Assigning srcObject runs the media load
    // algorithm even when the value is unchanged, which rewinds a playing clip
    // to zero - so a fresh object per render made the video restart whenever
    // anything at all changed.
    await openApp([WITH_VIDEO]);
    await waitFor(() => {
      expect(selected()).toBe('alpha');
    });
    sourcesSeen.length = 0;

    fireEvent.click(screen.getByText('45° off'));
    fireEvent.click(screen.getByText('Source view'));
    fireEvent.click(screen.getByText('Free'));

    // Rendered several times over, always with the same object.
    expect(sourcesSeen.length).toBeGreaterThan(1);
    expect(new Set(sourcesSeen).size).toBe(1);
  });

  it('is not handed a new source when a slider moves', async () => {
    await openApp([WITH_VIDEO]);
    await waitFor(() => {
      expect(selected()).toBe('alpha');
    });
    sourcesSeen.length = 0;

    const [, depth] = document.querySelectorAll('input[type=range]');
    if (depth === undefined) throw new Error('no depth slider');
    fireEvent.input(depth, { target: { value: '1.2' } });
    await settle();

    expect(new Set(sourcesSeen).size).toBeLessThanOrEqual(1);
  });
});
