/**
 * The left-hand pane: the footage the bodies were fitted to.
 *
 * Three arrangements, and which one is in use decides what keeps time.
 *
 * - A finished run plays its clip normally and **is** the clock. The 3D loop
 *   reads `currentTime` each frame, so the two panes cannot drift apart and
 *   scrubbing one scrubs both.
 * - A clip being estimated right now runs far slower than real time, so the
 *   video is paused and nudged to whatever frame has come back. Letting it play
 *   would show the operator a moment the bodies have not reached.
 * - A camera is just itself, with no seeking to do.
 */

import { useEffect, useRef } from 'react';

export type VideoSource =
  | { readonly kind: 'file'; readonly url: string; readonly follow: null }
  | { readonly kind: 'file'; readonly url: string; readonly follow: () => number }
  | { readonly kind: 'stream'; readonly stream: MediaStream };

interface SourceVideoProps {
  readonly source: VideoSource;
  /** Handed the element so the render loop can read its clock. */
  readonly onElement?: (element: HTMLVideoElement | null) => void;
}

/** Below this a seek is not worth the decode; above it the panes visibly differ. */
const SEEK_TOLERANCE_SECONDS = 0.05;

export function SourceVideo({ source, onElement }: SourceVideoProps): React.JSX.Element {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const video = videoRef.current;
    if (video === null) return;
    // Guarded, both branches: assigning `srcObject` runs the media element load
    // algorithm even when the value is unchanged, and that rewinds a playing
    // clip to zero. Unguarded, every re-run of this effect restarts the video.
    if (source.kind === 'stream') {
      if (video.srcObject !== source.stream) {
        video.srcObject = source.stream;
        void video.play().catch(() => undefined);
      }
    } else if (video.srcObject !== null) {
      video.srcObject = null;
    }
    onElement?.(video);
    return () => {
      onElement?.(null);
    };
  }, [source, onElement]);

  useEffect(() => {
    if (source.kind !== 'file' || source.follow === null) return;
    const video = videoRef.current;
    if (video === null) return;

    const follow = source.follow;
    let running = true;
    const step = (): void => {
      if (!running) return;
      const target = follow();
      if (
        Number.isFinite(target) &&
        Math.abs(video.currentTime - target) > SEEK_TOLERANCE_SECONDS
      ) {
        video.currentTime = target;
      }
      requestAnimationFrame(step);
    };
    video.pause();
    requestAnimationFrame(step);
    return () => {
      running = false;
    };
  }, [source]);

  // No `loop`: the clip is the clock, so restarting it would jump the bodies
  // back to the beginning under whoever was studying the end of it.
  const controls = source.kind === 'file' && source.follow === null;

  return (
    <video
      ref={videoRef}
      className="source-video"
      src={source.kind === 'file' ? source.url : undefined}
      controls={controls}
      // The clip is the clock the 3D pane reads, so its rate is part of the
      // run rather than a viewing preference. Chromium only; the other
      // browsers keep their speed menu whatever this says.
      controlsList="noplaybackrate nofullscreen"
      // Neither casting, nor a floating window, nor a full screen: this pane
      // only means anything beside the one next to it.
      disableRemotePlayback
      disablePictureInPicture
      autoPlay={controls}
      muted
      playsInline
    />
  );
}
