/**
 * The colours a track is drawn in.
 *
 * One copy, shared by the skeletons, the meshes and the swatch beside each
 * entry in the track list, so the swatch always matches the body it names.
 */

/** Distinct enough to tell neighbouring people apart, muted enough to look at. */
export const TRACK_COLORS = [
  '#5ccfe6',
  '#ffb454',
  '#c3e88d',
  '#f07178',
  '#c792ea',
  '#ffd580',
  '#82aaff',
  '#7fdbca',
] as const;

/**
 * The colour this track is drawn in, as CSS.
 *
 * Ids are assigned by the tracker and run past the end of the palette, so it
 * wraps. Two tracks can therefore share a colour, which is a limitation of
 * eight colours rather than a claim that they are the same person.
 */
export function trackColour(trackId: number): string {
  return TRACK_COLORS[trackId % TRACK_COLORS.length] ?? TRACK_COLORS[0];
}
