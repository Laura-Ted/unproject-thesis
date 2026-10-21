import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { clampLayout, DEFAULT_LAYOUT, readLayout, writeLayout } from './layout';

describe('clampLayout', () => {
  it('leaves a sensible layout alone', () => {
    const layout = { railRem: 24, sourcePercent: 65 };

    expect(clampLayout(layout)).toEqual(layout);
  });

  it('will not let the rail be dragged shut', () => {
    // A pane with no width left cannot be dragged back open.
    expect(clampLayout({ railRem: 0, sourcePercent: 50 }).railRem).toBeGreaterThan(0);
    expect(clampLayout({ railRem: -50, sourcePercent: 50 }).railRem).toBeGreaterThan(0);
  });

  it('will not let the rail eat the whole window', () => {
    expect(clampLayout({ railRem: 500, sourcePercent: 50 }).railRem).toBeLessThan(100);
  });

  it('keeps both panes on screen', () => {
    expect(clampLayout({ railRem: 17, sourcePercent: 0 }).sourcePercent).toBeGreaterThan(0);
    expect(clampLayout({ railRem: 17, sourcePercent: 100 }).sourcePercent).toBeLessThan(100);
  });

  it('falls back to the default for a value that is not a number', () => {
    const broken = { railRem: Number.NaN, sourcePercent: Number.POSITIVE_INFINITY };

    expect(clampLayout(broken)).toEqual(DEFAULT_LAYOUT);
  });
});

describe('readLayout', () => {
  beforeEach(() => {
    globalThis.localStorage.clear();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('returns the default when nothing was stored', () => {
    expect(readLayout()).toEqual(DEFAULT_LAYOUT);
  });

  it('returns what was written', () => {
    writeLayout({ railRem: 26, sourcePercent: 65 });

    expect(readLayout()).toEqual({ railRem: 26, sourcePercent: 65 });
  });

  it('clamps a stored value that is out of bounds', () => {
    // Someone edits storage by hand, or a future version writes a wider range.
    globalThis.localStorage.setItem(
      'unproject.layout',
      JSON.stringify({ railRem: 9999, sourcePercent: 50 }),
    );

    expect(readLayout().railRem).toBeLessThan(100);
  });

  it('survives storage holding something that is not a layout', () => {
    globalThis.localStorage.setItem('unproject.layout', '"a string"');

    expect(readLayout()).toEqual(DEFAULT_LAYOUT);
  });

  it('survives storage holding broken JSON', () => {
    globalThis.localStorage.setItem('unproject.layout', '{ not json');

    expect(readLayout()).toEqual(DEFAULT_LAYOUT);
  });

  it('survives storage throwing on access', () => {
    // A private window, or a browser set to block site data. Opening at the
    // default width beats refusing to start.
    vi.stubGlobal('localStorage', {
      getItem: () => {
        throw new Error('blocked');
      },
    });

    expect(readLayout()).toEqual(DEFAULT_LAYOUT);
  });

  it('does not throw when storage refuses a write', () => {
    vi.stubGlobal('localStorage', {
      setItem: () => {
        throw new Error('quota');
      },
    });

    expect(() => {
      writeLayout({ railRem: 20, sourcePercent: 50 });
    }).not.toThrow();
  });
});

describe('the CSS the layout drives', () => {
  it('is a grid template a browser will accept', () => {
    // calc() cannot multiply an fr, and a rejected declaration leaves the
    // splitter zero pixels wide - invisible and undraggable.
    expect(CSS.supports('grid-template-columns', 'calc(1 * 1fr) 3px 1fr')).toBe(false);
    expect(CSS.supports('grid-template-columns', '50% 3px minmax(0, 1fr)')).toBe(true);
  });
});
