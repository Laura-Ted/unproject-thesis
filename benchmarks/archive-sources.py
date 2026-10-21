#!/usr/bin/env python3
"""Snapshot every source page the campaign cites in the Wayback Machine.

ADR-0006 §5: a licence statement is evidence, and a page is not an archive.
Each `source_url` and `download_url` page of the experiments, plus the pages
listed in `EXTRA`, is submitted to the Internet Archive's Save Page Now
(v2) API, and the snapshot URL is written to `benchmarks/archived-sources.json`
so it can be copied into the experiments' `archived_url` and into the
thesis's bibliography.

Needs an archive.org account's S3-style keys in the environment, never on
the command line or in this repository:

    IA_ACCESS_KEY=... IA_SECRET_KEY=... python3 benchmarks/archive-sources.py

(https://archive.org/account/s3.php). Anonymous Save Page Now is refused
for automated clients, which is why the keys are needed. Stdlib only.
"""

from __future__ import annotations

import json
import os
import sys
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "benchmarks" / "archived-sources.json"
SPN = "https://web.archive.org/save"
STATUS = "https://web.archive.org/save/status/"

# Pages that are not an experiment's but that the thesis leans on.
EXTRA = (
    "https://media.xiph.org/video/derf/y4m/ReadMe_1080p.txt",
    "https://github.com/naver/multi-hmr2",
    "https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/gluUnProject.xml",
)


def experiment_urls() -> list[str]:
    urls: list[str] = []
    for path in sorted((REPO / "experiments").glob("*.toml")):
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        provenance = data.get("provenance") or {}
        for key in ("source_url", "download_url"):
            value = provenance.get(key)
            if value and value not in urls:
                urls.append(str(value))
    return urls


def request(url: str, data: bytes | None, headers: dict[str, str]) -> dict[str, object]:
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=120) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body if isinstance(body, dict) else {}


def save(url: str, auth: str) -> str | None:
    headers = {"Accept": "application/json", "Authorization": auth}
    data = urllib.parse.urlencode({"url": url, "capture_all": "1"}).encode("utf-8")
    try:
        started = request(SPN, data, headers)
    except urllib.error.HTTPError as error:
        print(f"  {url}: HTTP {error.code} {error.reason}", file=sys.stderr)
        return None
    job = started.get("job_id")
    if not job:
        print(f"  {url}: no job id in {started}", file=sys.stderr)
        return None
    for _ in range(40):
        time.sleep(6)
        status = request(f"{STATUS}{job}", None, headers)
        state = status.get("status")
        if state == "success":
            stamp = status.get("timestamp")
            original = status.get("original_url") or url
            return f"https://web.archive.org/web/{stamp}/{original}"
        if state == "error":
            print(f"  {url}: {status.get('message') or status}", file=sys.stderr)
            return None
    print(f"  {url}: still pending after four minutes", file=sys.stderr)
    return None


def main() -> int:
    access = os.environ.get("IA_ACCESS_KEY")
    secret = os.environ.get("IA_SECRET_KEY")
    if not access or not secret:
        print(
            "set IA_ACCESS_KEY and IA_SECRET_KEY (https://archive.org/account/s3.php)",
            file=sys.stderr,
        )
        return 2
    auth = f"LOW {access}:{secret}"
    existing: dict[str, str] = {}
    if OUTPUT.is_file():
        existing = json.loads(OUTPUT.read_text(encoding="utf-8"))
    urls = [*experiment_urls(), *EXTRA]
    for url in urls:
        if url in existing:
            print(f"kept  {url} -> {existing[url]}")
            continue
        print(f"save  {url}")
        snapshot = save(url, auth)
        if snapshot is not None:
            existing[url] = snapshot
            print(f"  -> {snapshot}")
            OUTPUT.write_text(
                json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        time.sleep(10)
    missing = [url for url in urls if url not in existing]
    print(f"{len(existing)} archived, {len(missing)} not; record is {OUTPUT}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
