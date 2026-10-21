"""Tests for deciding whether the served bundle is out of date.

Nothing here runs a build. What matters is the decision: a bundle quietly older
than its sources looks like broken code rather than a stale build.
"""

from __future__ import annotations

import os
from pathlib import Path

from unproject_api.viewer import ensure_built, is_stale


def _web(root: Path, *, built: bool) -> Path:
    web = root / "apps" / "web"
    (web / "src").mkdir(parents=True)
    (web / "package.json").write_text("{}", encoding="utf-8")
    (web / "src" / "App.tsx").write_text("//", encoding="utf-8")
    if built:
        (web / "dist").mkdir()
        (web / "dist" / "index.html").write_text("<html></html>", encoding="utf-8")
        _touch(web / "dist" / "index.html", 10_000)
        _touch(web / "src" / "App.tsx", 5_000)
        _touch(web / "package.json", 5_000)
    return web


def _touch(path: Path, when: float) -> None:
    os.utime(path, (when, when))


def test_a_missing_bundle_is_stale(tmp_path: Path) -> None:
    assert is_stale(_web(tmp_path, built=False))


def test_a_bundle_newer_than_its_sources_is_fresh(tmp_path: Path) -> None:
    assert not is_stale(_web(tmp_path, built=True))


def test_a_changed_source_makes_it_stale(tmp_path: Path) -> None:
    web = _web(tmp_path, built=True)
    _touch(web / "src" / "App.tsx", 20_000)
    assert is_stale(web)


def test_a_new_file_deep_in_the_tree_counts(tmp_path: Path) -> None:
    # The whole src tree matters, not just its top level.
    web = _web(tmp_path, built=True)
    nested = web / "src" / "scene"
    nested.mkdir()
    (nested / "Thing.tsx").write_text("//", encoding="utf-8")
    _touch(nested / "Thing.tsx", 20_000)
    assert is_stale(web)


def test_config_changes_count_too(tmp_path: Path) -> None:
    # A dependency or vite change alters the bundle without touching src.
    web = _web(tmp_path, built=True)
    _touch(web / "package.json", 20_000)
    assert is_stale(web)


def test_a_directory_that_is_not_a_viewer_is_left_alone(tmp_path: Path) -> None:
    # Serving from somewhere without a package.json is not an error, just not a
    # viewer, and must not start shelling out.
    assert not ensure_built(tmp_path / "nowhere")


def test_a_fresh_bundle_is_not_rebuilt(tmp_path: Path) -> None:
    # No pnpm is invoked, so this passing at all proves it short-circuited.
    assert ensure_built(_web(tmp_path, built=True))
