/**
 * Footage on the left, what was recovered from it on the right.
 *
 * Three ways in, all ending in the same pair of panes: a run already on disk, a
 * clip dropped in and estimated now, or a camera. The right-hand pane is always
 * live in the sense that matters - the operator can move while it updates.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import type { RunSummary } from './api';
import {
  bodySourceFromMesh,
  deleteRun,
  listRuns,
  loadArray,
  loadArtifact,
  loadMesh,
  loadTopology,
  uploadVideo,
} from './api';
import type {
  BodiesArtifact,
  BodyModel,
  BodySource,
  BodyTopology,
  Smoothing,
  TimelineFrame,
} from './bodies';
import {
  DEFAULT_SMOOTHING,
  LiveClock,
  smoothTimeline,
  toTimeline,
  transposeMatrices,
} from './bodies';
import type { LiveRequest, LiveState, SourceCamera } from './live';
import { CAPTURE_DEFAULTS, captureFrames, LiveSession } from './live';
import type { MeasureRequest } from './livemetrics';
import { LiveMetrics, measureRequestFrom, submitReport } from './livemetrics';
import { CAMERA_SPACE_SIGNS } from './convention';
import type { Layout } from './layout';
import { clampLayout, readLayout, writeLayout } from './layout';
import { trackColour } from './palette';
import type { TrackListing, TrackStats } from './tracks';
import { listTracks, loadTrackStats, trackPath, withoutHidden } from './tracks';
import type { Framing, Viewpoint } from './scene/Stage3D';
import { frameScene, Stage3D } from './scene/Stage3D';
import type { VideoSource } from './scene/SourceVideo';
import { SourceVideo } from './scene/SourceVideo';

/** Nobody has drawn more than sixteen at once; this leaves room and bounds it. */
const LIVE_CAPACITY = 24;

/**
 * Opening view, overlay and mesh, from the query string.
 *
 * `?view=source&overlay=0.6` names a whole picture, so a figure for the thesis
 * is a URL rather than a lucky screenshot.
 */
function opening(): {
  view: Viewpoint;
  overlay: number;
  mesh: boolean;
  measure: MeasureRequest | null;
} {
  const params = new URLSearchParams(globalThis.location.search);
  const asked = params.get('view');
  const overlay = Number(params.get('overlay'));
  return {
    view: asked === 'source' || asked === 'free' ? asked : 'off45',
    overlay: Number.isFinite(overlay) ? Math.min(1, Math.max(0, overlay)) : 0,
    mesh: params.get('mesh') === '1',
    // `?measure=1&replay=...` plays a clip into the camera endpoint and posts
    // what it measured when the clip ends. ADR-0005 WP3.
    measure: measureRequestFrom(globalThis.location.search),
  };
}

/** Answers still arriving after the last frame was sent get this long. */
const MEASURE_GRACE_MS = 2500;

type Session =
  | { readonly kind: 'idle' }
  | {
      readonly kind: 'run';
      readonly run: RunSummary;
      readonly artifact: BodiesArtifact;
      readonly joints: Float32Array;
      readonly frames: TimelineFrame[];
      readonly model: BodyModel;
      readonly camera: SourceCamera | null;
    }
  | {
      readonly kind: 'live';
      readonly session: LiveSession;
      readonly label: string;
      readonly videoUrl: string | null;
      readonly stream: MediaStream | null;
      /** Set when a clip is being replayed as a camera and measured. */
      readonly replay: { readonly request: MeasureRequest; readonly metrics: LiveMetrics } | null;
    };

function describe(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/** Release the camera. Nothing else turns the light off. */
function stopTracks(stream: MediaStream): void {
  stream.getTracks().forEach((track) => {
    track.stop();
  });
}

export function App(): React.JSX.Element {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [session, setSession] = useState<Session>({ kind: 'idle' });
  const [live, setLive] = useState<LiveState | null>(null);
  const [framing, setFraming] = useState<Framing | null>(null);
  const [start] = useState(opening);
  const [view, setView] = useState<Viewpoint>(start.view);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Replaced whenever a stream starts, so it lives in state rather than a ref:
  // the render pass reads it to keep the two panes on the same moment.
  const [liveClock, setLiveClock] = useState(() => new LiveClock());
  // null is off. The jitter this damps is the project's own result, so it is a
  // control and not a silent correction; turn it off to see what the model said.
  const [smoothing, setSmoothing] = useState<Smoothing | null>(DEFAULT_SMOOTHING);
  const [backdropOpacity, setBackdropOpacity] = useState(start.overlay);
  const [wantMesh, setWantMesh] = useState(false);
  const [bodies, setBodies] = useState<BodySource | null>(null);
  // Which tracks the operator has switched off, and which one they are
  // inspecting. Viewer state only: nothing here reaches the artifact or the
  // server. ADR-0003 §3.4.
  const [hidden, setHidden] = useState<ReadonlySet<number>>(() => new Set());
  const [selected, setSelected] = useState<number | null>(null);
  const [trackStats, setTrackStats] = useState<Map<number, TrackStats> | null>(null);
  const [videoEl, setVideoEl] = useState<HTMLVideoElement | null>(null);
  // Where the last measured session's report landed, for the status bar.
  const [reported, setReported] = useState<string | null>(null);
  // Pane widths, remembered between visits; layout.ts says why they are not fixed.
  const [layout, setLayout] = useState<Layout>(readLayout);

  // Mutable things the render loop touches. None of them belong in state: the
  // loop reads them sixty times a second and React must not re-render for it.
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const framesRef = useRef<readonly TimelineFrame[]>([]);
  const stopCapture = useRef<(() => void) | null>(null);
  // Which run's body has already been asked for. The click handler and the
  // effect below can both want it, and 45 MB is not worth fetching twice.
  const meshRequested = useRef<string | null>(null);
  // Every user action takes a turn. Network work resolves in whatever order it
  // likes, so a completion has to ask whether the operator is still waiting for
  // it before it touches anything visible.
  const turn = useRef(0);

  const claim = useCallback(() => {
    const mine = ++turn.current;
    return () => turn.current === mine;
  }, []);

  const teardown = useCallback(() => {
    // Nothing outstanding owns the screen any more, including on unmount.
    turn.current += 1;
    stopCapture.current?.();
    stopCapture.current = null;
    setSession((previous) => {
      if (previous.kind === 'live') {
        previous.session.close();
        if (previous.stream !== null) stopTracks(previous.stream);
      }
      return { kind: 'idle' };
    });
    framesRef.current = [];
    meshRequested.current = null;
    setHidden(new Set());
    setSelected(null);
    setTrackStats(null);
    setLive(null);
    setFraming(null);
    setBodies(null);
  }, []);

  useEffect(() => teardown, [teardown]);

  useEffect(() => {
    writeLayout(layout);
  }, [layout]);

  const openRun = useCallback(
    (run: RunSummary) => {
      teardown();
      const mine = claim();
      setError(null);
      setBusy(`Loading ${run.experiment}`);
      (async () => {
        const artifact = await loadArtifact(run.path);
        // Only the joints: the bone transforms are ten times the size and
        // nothing draws them until there is a mesh to skin.
        const joints = await loadArray(run.path, artifact, 'joints_3d');
        // Checked here rather than only at the end: another run opened while
        // these two fetches were in the air owns the screen now.
        if (!mine()) return;
        const frames = toTimeline(artifact, joints);
        setFraming(frameScene(frames));
        setSession({
          kind: 'run',
          run,
          artifact,
          joints,
          frames,
          model: artifact.body_model,
          camera: artifact.frames[0]?.camera ?? null,
        });
      })().then(
        () => {
          if (mine()) setBusy(null);
        },
        (cause: unknown) => {
          if (!mine()) return;
          setBusy(null);
          setError(describe(cause));
        },
      );
    },
    [claim, teardown],
  );

  const startLive = useCallback(
    (
      request: LiveRequest,
      label: string,
      videoUrl: string | null,
      stream: MediaStream | null,
      topology: BodyTopology | null,
      measure: MeasureRequest | null = null,
    ) => {
      teardown();
      setError(null);
      const clock = new LiveClock();
      setLiveClock(clock);
      const metrics = measure === null ? null : new LiveMetrics(performance.now());

      const socket = new LiveSession(
        request,
        topology,
        (state) => {
          setLive(state);
          // Frame the scene once there is something to frame, then leave it be:
          // re-framing mid-stream would move the ground under the operator.
          setFraming((current) =>
            current === null && state.frames.length > 4 ? frameScene(state.frames) : current,
          );
          if (state.status === 'error') setError(state.message);
        },
        (time) => {
          clock.observe(time);
        },
        metrics,
      );
      socket.open();
      setSession({
        kind: 'live',
        session: socket,
        label,
        videoUrl,
        stream,
        replay: measure === null || metrics === null ? null : { request: measure, metrics },
      });
    },
    [teardown],
  );

  // Listed after `openRun` exists, because it opens the newest as it arrives:
  // an empty split screen says nothing about what this is for.
  useEffect(() => {
    const mine = claim();
    listRuns().then(
      (found) => {
        if (!mine()) return;
        setRuns(found);
        // Opening the newest is a convenience, not an instruction: if the
        // operator picked one while the list was loading, theirs wins - and a
        // measured replay asked for by URL is not to be replaced by it.
        if (found[0] && start.measure === null) openRun(found[0]);
      },
      (cause: unknown) => {
        if (mine()) setError(describe(cause));
      },
    );
  }, [claim, openRun, start.measure]);

  // A measured replay: the clip named in the URL is played into the camera
  // endpoint, every frame numbered, and the report posted when it ends.
  useEffect(() => {
    const measure = start.measure;
    if (measure === null) return;
    const mine = claim();
    // Deferred by a tick, like the other URL-driven starts: setting state in
    // the pass that mounted the app is a cascade.
    const handle = setTimeout(() => {
      setBusy(`Measuring ${measure.replay}`);
    }, 0);
    (measure.bodies ? loadTopology() : Promise.resolve(null)).then(
      (topology) => {
        if (!mine()) return;
        setBusy(null);
        startLive(
          { kind: 'camera', bodies: topology !== null, frame_ids: true },
          `Replay ${measure.replay}`,
          `/${measure.replay}`,
          null,
          topology,
          measure,
        );
      },
      (cause: unknown) => {
        if (!mine()) return;
        setBusy(null);
        setError(describe(cause));
      },
    );
    return () => {
      clearTimeout(handle);
    };
  }, [claim, start.measure, startLive]);

  // Fetched when it is asked for rather than with every run: the mesh and a
  // transform per bone per frame come to about 45 MB.
  const onMesh = useCallback(
    (wanted: boolean) => {
      setWantMesh(wanted);
      if (!wanted || session.kind !== 'run' || session.artifact.mesh === null) return;

      const { run, artifact, joints } = session;
      meshRequested.current = run.path;
      const mine = claim();
      setBusy('Loading the body');
      Promise.all([
        loadMesh(run.path, artifact, artifact.body_model.bone_count),
        loadArray(run.path, artifact, 'bone_poses'),
      ]).then(
        ([body, rowMajor]) => {
          // 45 MB arriving after the operator changed run would otherwise put
          // one run's body on another run's skeleton.
          if (!mine()) return;
          const bones = transposeMatrices(rowMajor);
          setBusy(null);
          setBodies(bodySourceFromMesh(body));
          // The timeline has to carry the bone transforms now, which it did not
          // when only the skeleton was drawn.
          setSession((current) =>
            current.kind === 'run' && current.run.path === run.path
              ? { ...current, frames: toTimeline(artifact, joints, bones) }
              : current,
          );
        },
        (cause: unknown) => {
          if (!mine()) return;
          meshRequested.current = null;
          setBusy(null);
          setWantMesh(false);
          setError(describe(cause));
        },
      );
    },
    [claim, session],
  );

  // The choice survives a change of run; the 45 MB body does not, because
  // teardown drops it and fetching it for a run the operator may not linger on
  // would be wasteful. So a run opening while Mesh is picked has to fetch it
  // now - otherwise the control says "Mesh" over a screen full of skeletons
  // until it is toggled off and on again.
  useEffect(() => {
    if (!wantMesh || bodies !== null) return;
    if (session.kind !== 'run' || session.artifact.mesh === null) return;
    // Already in flight for this run: the operator's click got there first.
    if (meshRequested.current === session.run.path) return;
    // Deferred by a tick for the same reason as the effect below: asking for it
    // in the pass that opened the run is a cascade.
    const handle = setTimeout(() => {
      onMesh(true);
    }, 0);
    return () => {
      clearTimeout(handle);
    };
  }, [wantMesh, bodies, session, onMesh]);

  useEffect(() => {
    if (!start.mesh || session.kind !== 'run' || wantMesh || session.artifact.mesh === null) return;
    // Deferred by a tick: this only exists so a figure can be named by URL, and
    // asking for it in the same pass that opened the run is a cascade.
    const handle = setTimeout(() => {
      onMesh(true);
    }, 0);
    return () => {
      clearTimeout(handle);
    };
  }, [start.mesh, session, wantMesh, onMesh]);

  const onUpload = useCallback(
    (file: File) => {
      const mine = claim();
      setBusy(`Uploading ${file.name}`);
      Promise.all([uploadVideo(file), wantMesh ? loadTopology() : null]).then(
        ([upload, topology]) => {
          // The file stays on the server either way; it is the operator's, and
          // deleting it because they navigated away is not ours to decide.
          if (!mine()) return;
          setBusy(null);
          startLive(
            { kind: 'video', path: upload.path, stride: 2, bodies: topology !== null },
            upload.name,
            `/${upload.path}`,
            null,
            topology,
          );
        },
        (cause: unknown) => {
          if (!mine()) return;
          setBusy(null);
          setError(describe(cause));
        },
      );
    },
    [claim, startLive, wantMesh],
  );

  const onRemove = useCallback(
    (run: RunSummary) => {
      // Irreversible, and the artifact is the evidence behind any number drawn
      // from it, so it is worth one question. ADR-0002 §10.5.
      const sure = globalThis.confirm(
        `Remove the ${run.experiment} run ${run.hash} from disk?\n\n` +
          `${String(run.frames)} frames and ${String(run.people)} bodies, ` +
          'including any analysis and figures beside them. This cannot be undone.',
      );
      if (!sure) return;

      const mine = claim();
      setBusy(`Removing ${run.experiment}`);
      deleteRun(run.path).then(
        () => {
          if (mine()) setBusy(null);
          // Not guarded by the turn, unlike everything else here: the run is
          // gone from disk whatever the operator did next, so leaving the row
          // would leave a button that opens a file which no longer exists.
          // Both updates say which run they mean, so they are safe out of turn.
          setRuns((previous) => previous.filter((other) => other.path !== run.path));
          setSession((current) =>
            current.kind === 'run' && current.run.path === run.path ? { kind: 'idle' } : current,
          );
        },
        (cause: unknown) => {
          if (!mine()) return;
          setBusy(null);
          setError(describe(cause));
        },
      );
    },
    [claim],
  );

  const onCamera = useCallback(() => {
    const mine = claim();
    setBusy('Waiting for the camera');
    Promise.all([
      navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false }),
      wantMesh ? loadTopology() : null,
    ]).then(
      ([stream, topology]) => {
        if (!mine()) {
          // Permission granted to a session nobody is waiting for any more.
          // Ignoring it is not enough: the camera light stays on until the
          // tracks are stopped.
          stopTracks(stream);
          return;
        }
        setBusy(null);
        startLive({ kind: 'camera', bodies: topology !== null }, 'Camera', null, stream, topology);
      },
      (cause: unknown) => {
        if (!mine()) return;
        setBusy(null);
        setError(describe(cause));
      },
    );
  }, [claim, startLive, wantMesh]);

  // Null unless `unproject rq3` was run on this run; see `loadTrackStats`.
  useEffect(() => {
    if (session.kind !== 'run') return;
    const mine = claim();
    void loadTrackStats(session.run.path).then((stats) => {
      if (mine()) {
        setTrackStats(stats);
      }
    });
  }, [claim, session]);

  // The end of a measured replay: stop sending, give the last answers a
  // moment to arrive, hand the rows to the server, and close.
  const finishMeasurement = useCallback(
    (current: Session) => {
      if (current.kind !== 'live' || current.replay === null) return;
      stopCapture.current?.();
      stopCapture.current = null;
      const { request, metrics } = current.replay;
      const video = videoRef.current;
      const width = Math.min(
        video?.videoWidth ?? CAPTURE_DEFAULTS.maxWidth,
        CAPTURE_DEFAULTS.maxWidth,
      );
      const height =
        video !== null && video.videoWidth > 0
          ? Math.round((video.videoHeight * width) / video.videoWidth)
          : width;
      setBusy('Waiting for the last answers');
      const mine = claim();
      setTimeout(() => {
        if (!mine()) return;
        const draft = metrics.draft({
          source: request.replay,
          mode: request.bodies ? 'mesh' : 'skeleton',
          frameWidth: width,
          frameHeight: height,
          sendFps: request.fps,
          maxWidth: CAPTURE_DEFAULTS.maxWidth,
          jpegQuality: CAPTURE_DEFAULTS.jpegQuality,
          liveClockLagSeconds: liveClock.lag,
          frameLimit: request.frameLimit,
          server: live?.environment ?? null,
          notes: [
            'frames were captured from a <video> element replaying the clip into a canvas at ' +
              "the viewer's own size and quality; the clip played at its natural rate",
          ],
        });
        submitReport(draft).then(
          (path) => {
            if (!mine()) return;
            setBusy(null);
            setReported(path);
            teardown();
          },
          (cause: unknown) => {
            if (!mine()) return;
            setBusy(null);
            setError(describe(cause));
          },
        );
      }, MEASURE_GRACE_MS);
    },
    [claim, live, liveClock, teardown],
  );

  // Camera frames go out on a timer once both the element and socket exist. A
  // measured replay captures from the clip at the rate the URL asked for and
  // stops at its frame limit or at the clip's end, whichever comes first.
  useEffect(() => {
    if (session.kind !== 'live' || (session.stream === null && session.replay === null)) return;
    const video = videoRef.current;
    if (video === null) return;
    const replay = session.replay;
    stopCapture.current = captureFrames(video, session.session, {
      fps: replay?.request.fps ?? CAPTURE_DEFAULTS.fps,
      limit: replay?.request.frameLimit ?? null,
      onLimit: () => {
        finishMeasurement(session);
      },
    });
    const onEnded = (): void => {
      finishMeasurement(session);
    };
    if (replay !== null) video.addEventListener('ended', onEnded);
    return () => {
      if (replay !== null) video.removeEventListener('ended', onEnded);
      stopCapture.current?.();
      stopCapture.current = null;
    };
  }, [session, finishMeasurement]);

  // The one place the loop's frames are decided. Re-derived whenever the
  // parameters move or another frame lands; a live window is a handful of
  // frames and a finished clip is a few milliseconds, so neither is worth
  // caching against the confusion of two copies drifting apart.
  useEffect(() => {
    const raw =
      session.kind === 'run'
        ? session.frames
        : session.kind === 'live'
          ? session.session.timeline()
          : [];
    const shown = withoutHidden(raw, hidden);
    framesRef.current = smoothing === null ? shown : smoothTimeline(shown, smoothing);
  }, [session, smoothing, live, hidden]);

  // Derived from the timeline the viewer already holds, so the list is there
  // for a live session too, where no analysis has been run.
  const timeline =
    session.kind === 'run'
      ? session.frames
      : session.kind === 'live'
        ? session.session.timeline()
        : [];
  const tracks = listTracks(timeline);
  const path = selected === null ? null : trackPath(timeline, selected, CAMERA_SPACE_SIGNS);

  const isLive = session.kind === 'live';
  const tick = useCallback(
    (delta: number): number =>
      // A finished run has a real video, and that video is the clock: reading
      // it here is what keeps the two panes on the same frame.
      isLive ? liveClock.tick(delta) : (videoRef.current?.currentTime ?? 0),
    [isLive, liveClock],
  );
  const frames = useCallback(() => framesRef.current, []);
  // Stable, for the same reason as `videoSource`: an inline arrow here is a new
  // dependency on every render, and SourceVideo's effect depends on it.
  const onVideoElement = useCallback((element: HTMLVideoElement | null) => {
    videoRef.current = element;
    setVideoEl(element);
  }, []);

  const model = session.kind === 'run' ? session.model : live?.bodyModel;
  const capacity = session.kind === 'run' ? Math.max(1, session.run.max_people) : LIVE_CAPACITY;
  const followLive = useCallback(() => liveClock.now(), [liveClock]);
  // Memoised: a fresh object each render makes SourceVideo's effect re-run, and
  // it is the effect that touches the media element.
  const videoSource = useMemo(() => toVideoSource(session, followLive), [session, followLive]);
  const sourceCamera = session.kind === 'run' ? session.camera : (live?.camera ?? null);
  const backdrop =
    videoEl !== null && sourceCamera !== null ? { video: videoEl, camera: sourceCamera } : null;

  return (
    <div
      className="app"
      style={{ '--rail': `${String(layout.railRem)}rem` } as React.CSSProperties}
    >
      <aside className="rail">
        <header>
          <h1>Unproject</h1>
          <p>Bodies from a single camera</p>
        </header>

        <div className="inputs">
          <label className="button">
            Open a video
            <input
              type="file"
              accept="video/*"
              onChange={(event) => {
                const file = event.target.files?.[0];
                event.target.value = '';
                if (file) onUpload(file);
              }}
            />
          </label>
          <button type="button" className="button" onClick={onCamera}>
            Use the camera
          </button>
        </div>

        <h2>Finished runs</h2>
        {runs.length === 0 && (
          <p className="empty">
            None yet. <code>uv run unproject bodies experiments/crowd.toml</code>
          </p>
        )}
        <ul className="runs">
          {runs.map((run) => (
            <li key={run.path}>
              <button
                type="button"
                className={
                  session.kind === 'run' && session.run.path === run.path ? 'run selected' : 'run'
                }
                onClick={() => {
                  openRun(run);
                }}
              >
                <span className="name">{run.experiment}</span>
                <span className="hash">{run.hash}</span>
                <span className="stat">
                  {/* The average, not the total: a detection count summed over
                      every frame reads as a number of people and is not one. */}
                  {run.frames} frames ·{' '}
                  {run.frames === 0 ? '0' : (run.people / run.frames).toFixed(1)} people per frame ·{' '}
                  {run.tracks} tracks
                </span>
              </button>
              <button
                type="button"
                className="remove"
                title={`Remove ${run.experiment} ${run.hash} from disk`}
                aria-label={`Remove run ${run.hash}`}
                onClick={() => {
                  onRemove(run);
                }}
              >
                ×
              </button>
            </li>
          ))}
        </ul>

        {error !== null && <p className="error">{error}</p>}

        <Controls
          smoothing={smoothing}
          onSmoothing={setSmoothing}
          opacity={backdropOpacity}
          onOpacity={setBackdropOpacity}
          canOverlay={backdrop !== null}
          showMesh={wantMesh}
          onMesh={onMesh}
          canMesh={
            session.kind !== 'live' && (session.kind !== 'run' || session.artifact.mesh !== null)
          }
        />

        {tracks.length > 0 && (
          <TrackList
            tracks={tracks}
            stats={trackStats}
            hidden={hidden}
            selected={selected}
            onToggle={(id) => {
              setHidden((previous) => {
                const next = new Set(previous);
                if (!next.delete(id)) next.add(id);
                return next;
              });
            }}
            onIsolate={(id) => {
              setHidden((previous) =>
                previous.size === tracks.length - 1 && !previous.has(id)
                  ? new Set()
                  : new Set(tracks.map((t) => t.trackId).filter((other) => other !== id)),
              );
              setSelected(id);
            }}
            onSelect={(id) => {
              setSelected((previous) => (previous === id ? null : id));
            }}
            onShowAll={() => {
              setHidden(new Set());
            }}
          />
        )}

        {model && <ModelDetail model={model} live={live} />}
      </aside>

      <Splitter
        area="rail-split"
        label="Resize the sidebar"
        onDrag={(dx) => {
          setLayout((previous) =>
            clampLayout({ ...previous, railRem: previous.railRem + dx / 16 }),
          );
        }}
      />

      <main
        className="panes"
        style={{ '--source': `${String(layout.sourcePercent)}%` } as React.CSSProperties}
      >
        <section className="pane video">
          <h3>Source</h3>
          {videoSource === null ? (
            <p className="placeholder">
              {busy ?? 'Open a video, use the camera, or pick a finished run.'}
            </p>
          ) : (
            <SourceVideo source={videoSource} onElement={onVideoElement} />
          )}
        </section>

        <Splitter
          area="pane-split"
          label="Resize the source pane"
          onDrag={(dx, width) => {
            setLayout((previous) =>
              clampLayout({
                ...previous,
                sourcePercent: previous.sourcePercent + (dx / width) * 100,
              }),
            );
          }}
        />

        <section className="pane stage">
          <h3>Recovered</h3>
          {framing === null || model === null || model === undefined ? (
            <p className="placeholder">
              {isLive ? (live?.status === 'error' ? 'Failed' : 'Waiting for the first bodies') : ''}
            </p>
          ) : (
            <Stage3D
              frames={frames}
              boneParents={model.bone_parents}
              capacity={capacity}
              tick={tick}
              framing={framing}
              view={view}
              backdrop={backdrop}
              backdropOpacity={backdropOpacity}
              bodies={wantMesh ? bodies : null}
              path={path}
              onUserMove={() => {
                setView('free');
              }}
            />
          )}
          <div className="views">
            {(['source', 'off45', 'free'] as const).map((option) => (
              <button
                key={option}
                type="button"
                className={view === option ? 'selected' : undefined}
                onClick={() => {
                  setView(option);
                }}
              >
                {option === 'source' ? 'Source view' : option === 'off45' ? '45° off' : 'Free'}
              </button>
            ))}
          </div>
        </section>
      </main>

      <footer className="status">
        <Status session={session} live={live} busy={busy} reported={reported} />
      </footer>
    </div>
  );
}

/**
 * A drag handle between two panes.
 *
 * Pointer events rather than mouse: the capture keeps the drag alive when the
 * cursor leaves the three-pixel strip, which it does immediately. `onDrag` is
 * given the movement since the last event and the width of the whole row, so
 * the caller can turn pixels into whatever unit its layout is in.
 */
function Splitter({
  area,
  label,
  onDrag,
}: {
  readonly area: string;
  readonly label: string;
  readonly onDrag: (deltaX: number, containerWidth: number) => void;
}): React.JSX.Element {
  return (
    <div
      className="splitter"
      style={{ gridArea: area }}
      role="separator"
      aria-orientation="vertical"
      aria-label={label}
      onPointerDown={(event) => {
        event.currentTarget.setPointerCapture(event.pointerId);
      }}
      onPointerMove={(event) => {
        if (!event.currentTarget.hasPointerCapture(event.pointerId)) return;
        const width = event.currentTarget.parentElement?.clientWidth ?? window.innerWidth;
        onDrag(event.movementX, width);
      }}
      onPointerUp={(event) => {
        event.currentTarget.releasePointerCapture(event.pointerId);
      }}
    />
  );
}

/**
 * The tracks in the current run, one row each.
 *
 * Says "track", never "person 12". A track is one unbroken run of detections,
 * and when somebody walks behind a pillar and comes back they arrive as a new
 * one. Seeing four rows with abutting lifetimes where there was one person is
 * the point of this list, not a bug in it. ADR-0003 §3.1.
 */
function TrackList({
  tracks,
  stats,
  hidden,
  selected,
  onToggle,
  onIsolate,
  onSelect,
  onShowAll,
}: {
  readonly tracks: readonly TrackListing[];
  readonly stats: Map<number, TrackStats> | null;
  readonly hidden: ReadonlySet<number>;
  readonly selected: number | null;
  readonly onToggle: (id: number) => void;
  readonly onIsolate: (id: number) => void;
  readonly onSelect: (id: number) => void;
  readonly onShowAll: () => void;
}): React.JSX.Element {
  const detail = selected === null ? undefined : stats?.get(selected);

  return (
    <section className="tracks">
      <h2>
        Tracks <span className="count">{tracks.length}</span>
      </h2>
      {hidden.size > 0 && (
        <button type="button" className="link" onClick={onShowAll}>
          Show all {tracks.length}
        </button>
      )}

      <ul>
        {tracks.map((track) => (
          <li key={track.trackId} className={selected === track.trackId ? 'selected' : undefined}>
            <input
              type="checkbox"
              checked={!hidden.has(track.trackId)}
              aria-label={`Show track ${track.trackId}`}
              onChange={() => {
                onToggle(track.trackId);
              }}
            />
            <button
              type="button"
              className="pick"
              onClick={() => {
                onSelect(track.trackId);
              }}
            >
              <span className="swatch" style={{ background: trackColour(track.trackId) }} />
              <span className="id">track {track.trackId}</span>
              <span className="span">
                {track.firstSeconds.toFixed(1)}&ndash;{track.lastSeconds.toFixed(1)} s ·{' '}
                {track.detections} frames
              </span>
            </button>
            <button
              type="button"
              className="only"
              title="Hide every other track"
              onClick={() => {
                onIsolate(track.trackId);
              }}
            >
              only
            </button>
          </li>
        ))}
      </ul>

      {selected !== null && (
        <dl className="stats">
          {detail === undefined ? (
            <p className="empty">
              Run <code>unproject rq3</code> on this run for its per-track numbers.
            </p>
          ) : (
            <>
              <dt>Distance</dt>
              <dd>{fixed(detail.median_range_metres, 1, ' m')}</dd>
              <dt>Sideways</dt>
              <dd>{fixed(detail.motion.lateral_median_metres_per_second, 2, ' m/s')}</dd>
              <dt>Depth</dt>
              <dd>{fixed(detail.motion.depth_median_metres_per_second, 2, ' m/s')}</dd>
              <dt>Depth jitter</dt>
              <dd>
                {detail.motion.depth_jitter_fraction_of_range === null
                  ? '—'
                  : `${(100 * detail.motion.depth_jitter_fraction_of_range).toFixed(1)}% of range`}
              </dd>
              <dt>Longest gap</dt>
              <dd>{fixed(detail.longest_gap_seconds, 2, ' s')}</dd>
              <dt>Shape drift</dt>
              <dd>{fixed(detail.shape.max_parameter_std, 3, '')}</dd>
            </>
          )}
        </dl>
      )}
    </section>
  );
}

/** An em dash for a measurement that was not taken. Never a zero. */
function fixed(value: number | null, places: number, unit: string): string {
  return value === null ? '—' : `${value.toFixed(places)}${unit}`;
}

function toVideoSource(session: Session, followLive: () => number): VideoSource | null {
  if (session.kind === 'run') {
    return session.run.video === null
      ? null
      : { kind: 'file', url: session.run.video, follow: null };
  }
  if (session.kind === 'live') {
    if (session.stream !== null) return { kind: 'stream', stream: session.stream };
    if (session.replay !== null && session.videoUrl !== null) {
      // A replay stands in for a camera: the clip plays at its own rate and
      // frames are captured from it as they pass, answered or not.
      return { kind: 'file', url: session.videoUrl, follow: null };
    }
    if (session.videoUrl !== null) {
      // Slaved: estimation runs far slower than the clip, so the pane shows the
      // frame the bodies have actually reached rather than running ahead.
      return { kind: 'file', url: session.videoUrl, follow: followLive };
    }
  }
  return null;
}

function Slider({
  label,
  value,
  min,
  max,
  step,
  format,
  onChange,
  disabled,
}: {
  readonly label: string;
  readonly value: number;
  readonly min: number;
  readonly max: number;
  readonly step: number;
  readonly format: (value: number) => string;
  readonly onChange: (value: number) => void;
  readonly disabled?: boolean;
}): React.JSX.Element {
  return (
    <label className={disabled === true ? 'slider disabled' : 'slider'}>
      <span>
        {label}
        <b>{format(value)}</b>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(event) => {
          onChange(Number(event.target.value));
        }}
      />
    </label>
  );
}

/**
 * What is worth turning while looking at a scene.
 *
 * Each changes what is drawn and none changes what is stored, which is why
 * they are here rather than in the config: the artifact is always the raw
 * estimate, and these are ways of reading it.
 */
function Controls({
  smoothing,
  onSmoothing,
  opacity,
  onOpacity,
  canOverlay,
  showMesh,
  onMesh,
  canMesh,
}: {
  readonly smoothing: Smoothing | null;
  readonly onSmoothing: (value: Smoothing | null) => void;
  readonly opacity: number;
  readonly onOpacity: (value: number) => void;
  readonly canOverlay: boolean;
  readonly showMesh: boolean;
  readonly onMesh: (value: boolean) => void;
  readonly canMesh: boolean;
}): React.JSX.Element {
  const on = smoothing !== null;
  const current = smoothing ?? DEFAULT_SMOOTHING;

  return (
    <section className="controls">
      <h2>View</h2>

      <div className="figure">
        <span>Figure</span>
        <div>
          <button
            type="button"
            className={showMesh ? undefined : 'selected'}
            onClick={() => {
              onMesh(false);
            }}
          >
            Skeleton
          </button>
          <button
            type="button"
            className={showMesh ? 'selected' : undefined}
            disabled={!canMesh}
            title="The fitted body, about 45 MB for a run"
            onClick={() => {
              onMesh(true);
            }}
          >
            Mesh
          </button>
        </div>
      </div>

      <Slider
        label="Footage behind"
        value={opacity}
        min={0}
        max={1}
        step={0.05}
        disabled={!canOverlay}
        format={(v) => (v === 0 ? 'off' : `${String(Math.round(v * 100))}%`)}
        onChange={onOpacity}
      />

      <label className="switch">
        <input
          type="checkbox"
          checked={on}
          onChange={(event) => {
            onSmoothing(event.target.checked ? DEFAULT_SMOOTHING : null);
          }}
        />
        Low-pass filter
      </label>

      <Slider
        label="Depth"
        value={current.depth.cutoffHz}
        min={0.15}
        max={2}
        step={0.05}
        disabled={!on}
        format={(v) => `${v.toFixed(2)} Hz`}
        onChange={(v) => {
          onSmoothing({ ...current, depth: { ...current.depth, cutoffHz: v } });
        }}
      />
      <Slider
        label="Sideways"
        value={current.lateral.cutoffHz}
        min={0.5}
        max={6}
        step={0.1}
        disabled={!on}
        format={(v) => `${v.toFixed(1)} Hz`}
        onChange={(v) => {
          onSmoothing({ ...current, lateral: { ...current.lateral, cutoffHz: v } });
        }}
      />
    </section>
  );
}

function ModelDetail({
  model,
  live,
}: {
  readonly model: BodyModel;
  readonly live: LiveState | null;
}): React.JSX.Element {
  return (
    <dl className="detail">
      <dt>body model</dt>
      <dd>
        {model.name} {model.version}
      </dd>
      {live && (
        <>
          <dt>processed</dt>
          <dd>{live.processed}</dd>
          <dt>dropped</dt>
          <dd>
            {live.dropped}
            {/* Non-zero is honest rather than alarming: it is what running
                behind a camera looks like. */}
            {live.dropped > 0 && <div className="muted">inference is behind the source</div>}
          </dd>
        </>
      )}
    </dl>
  );
}

function Status({
  session,
  live,
  busy,
  reported,
}: {
  readonly session: Session;
  readonly live: LiveState | null;
  readonly busy: string | null;
  readonly reported: string | null;
}): React.JSX.Element | null {
  if (busy !== null) return <span>{busy}…</span>;
  if (session.kind === 'live') {
    const measured = session.session.measured();
    return (
      <span>
        {session.label} · {live?.status ?? 'connecting'}
        {live && live.processed > 0 && ` · ${String(live.processed)} frames estimated`}
        {measured !== null &&
          ` · measuring: sent ${String(measured.sent)}, answered ${String(measured.answered)}`}
      </span>
    );
  }
  if (session.kind === 'run') return null;
  if (reported !== null) return <span>Report kept at {reported}</span>;
  return <span>Idle</span>;
}
