/**
 * How wide the three panes are, and remembering it.
 *
 * The rail carries the track list and its numbers, and at a fixed 17rem they do
 * not fit. Rather than pick a wider fixed number and be wrong for somebody
 * else's screen, the operator drags.
 */

export interface Layout {
  /** Width of the left rail, in rem. */
  readonly railRem: number;
  /**
   * How much of the remaining width the source pane takes, as a percentage.
   *
   * A percentage rather than a grid fraction because `calc()` cannot multiply
   * an `fr`: `calc(var(--x) * 1fr)` is rejected outright, and a rejected
   * `grid-template-columns` takes the splitter's own width down with it.
   */
  readonly sourcePercent: number;
}

export const DEFAULT_LAYOUT: Layout = { railRem: 17, sourcePercent: 50 };

// Narrow enough to get out of the way, wide enough to read the statistics.
const RAIL_MIN_REM = 12;
const RAIL_MAX_REM = 44;
// Neither pane may be squeezed shut: one with no width left cannot be dragged
// back open.
const SOURCE_MIN = 15;
const SOURCE_MAX = 85;

const STORAGE_KEY = 'unproject.layout';

function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value));
}

/** Bring a layout inside its bounds. Non-finite values fall back to the default. */
export function clampLayout(layout: Layout): Layout {
  return {
    railRem: Number.isFinite(layout.railRem)
      ? clamp(layout.railRem, RAIL_MIN_REM, RAIL_MAX_REM)
      : DEFAULT_LAYOUT.railRem,
    sourcePercent: Number.isFinite(layout.sourcePercent)
      ? clamp(layout.sourcePercent, SOURCE_MIN, SOURCE_MAX)
      : DEFAULT_LAYOUT.sourcePercent,
  };
}

/**
 * The layout from last time, or the default.
 *
 * Storage can be unavailable or hold anything at all; either way the viewer
 * opens at its default width instead of failing.
 */
export function readLayout(): Layout {
  try {
    // Not optional-chained: where storage is blocked the access throws rather
    // than being absent, which is what the catch is for.
    const stored = globalThis.localStorage.getItem(STORAGE_KEY);
    if (stored === null) return DEFAULT_LAYOUT;
    const parsed: unknown = JSON.parse(stored);
    if (typeof parsed !== 'object' || parsed === null) return DEFAULT_LAYOUT;
    const { railRem, sourcePercent } = parsed as Partial<Layout>;
    return clampLayout({
      railRem: typeof railRem === 'number' ? railRem : DEFAULT_LAYOUT.railRem,
      sourcePercent:
        typeof sourcePercent === 'number' ? sourcePercent : DEFAULT_LAYOUT.sourcePercent,
    });
  } catch {
    return DEFAULT_LAYOUT;
  }
}

/** Remember a layout. Silently does nothing where storage is unavailable. */
export function writeLayout(layout: Layout): void {
  try {
    globalThis.localStorage.setItem(STORAGE_KEY, JSON.stringify(layout));
  } catch {
    // An unsaved preference is not worth an error.
  }
}
