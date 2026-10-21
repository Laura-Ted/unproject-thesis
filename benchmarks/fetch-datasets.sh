#!/usr/bin/env bash
# The campaign's footage, put in place and verified (ADR-0006 §5).
#
# Two kinds of input, handled differently because their terms differ:
#
# - clips carried as files (--from-dir PATH holding datasets/<name>/clip.mp4,
#   or --from-deposit URL): none since 2026-09-15, the list is empty; kept
#   for a future input whose bytes cannot be derived on every machine;
# - MOT16-08 and MOT16-09 (MOTChallenge, research use, not redistributed):
#   fetched from the benchmark's own server as the official archive, verified
#   by digest, and read as the JPEG frames they ship as - nothing is
#   transcoded, so every machine reads the same bytes by construction;
# - MuPoTS-3D TS10 and TS11 (MPI for Informatics, research only, not
#   redistributed): the same treatment, but only once the operator has read
#   the dataset's terms and says so with --accept-mupots-terms, because the
#   licence is the operator's to accept, not this script's. Without the flag
#   the two sequences are reported as absent and nothing is fetched.
#
# Idempotent: a file that is present and hashes right is left alone; one that
# hashes wrong is reported and left alone too - this script never deletes or
# overwrites, it tells you what is wrong. The digests are the ones pinned in
# experiments/*.toml; change them there, not here.
set -euo pipefail

cd "$(dirname "$0")/.."

FROM_DIR=""
DEPOSIT=""
MUPOTS_TERMS="${UNPROJECT_ACCEPT_MUPOTS_TERMS:-}"
while [ $# -gt 0 ]; do
    case "$1" in
        --from-dir) FROM_DIR="$2"; shift 2 ;;
        --from-deposit) DEPOSIT="$2"; shift 2 ;;
        --accept-mupots-terms) MUPOTS_TERMS=1; shift ;;
        -h|--help) sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

# name -> pinned digest of datasets/<name>/clip.mp4, read from the experiment.
pinned() {
    sed -n 's/^expected_sha256 = "\([0-9a-f]\{64\}\)"/\1/p' "experiments/$1.toml"
}

digest() {
    if command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | cut -c1-64
    else sha256sum "$1" | cut -c1-64; fi
}

status=0
note() { printf '%s\n' "$*"; }
fail() { printf 'FAIL  %s\n' "$*" >&2; status=1; }

# --------------------------------------------------------------------------- #
# Clips carried as files
# --------------------------------------------------------------------------- #
# Empty since 2026-09-15: every campaign input is derived from an official
# archive by the ingest (MOT16, MuPoTS-3D below), so nothing is carried. The
# mechanism stays for a future clip whose bytes have to travel as a file.
CLIPS=""

for name in $CLIPS; do
    want="$(pinned "$name")"
    if [ -z "$want" ]; then
        note "skip  $name: no expected_sha256 pinned in experiments/$name.toml yet"
        continue
    fi
    target="datasets/$name/clip.mp4"
    if [ -f "$target" ]; then
        have="$(digest "$target")"
        if [ "$have" = "$want" ]; then note "ok    $target ($have)"; else
            fail "$target hashes to $have, the campaign expects $want; not the same bytes. Move it aside by hand and rerun."
        fi
        continue
    fi
    mkdir -p "datasets/$name"
    if [ -n "$FROM_DIR" ] && [ -f "$FROM_DIR/$name/clip.mp4" ]; then
        cp "$FROM_DIR/$name/clip.mp4" "$target.part" && mv "$target.part" "$target"
    elif [ -n "$DEPOSIT" ]; then
        curl -fL --retry 3 -o "$target.part" "$DEPOSIT/$name.clip.mp4" && mv "$target.part" "$target"
    else
        fail "$target is absent; pass --from-dir <dir holding $name/clip.mp4> or --from-deposit <url>"
        continue
    fi
    have="$(digest "$target")"
    if [ "$have" = "$want" ]; then note "ok    $target fetched ($have)"; else
        fail "$target fetched but hashes to $have, expected $want"
    fi
done

# --------------------------------------------------------------------------- #
# MOT16, from MOTChallenge
# --------------------------------------------------------------------------- #
MOT_URL="https://motchallenge.net/data/MOT16.zip"
MOT_SHA="$(sed -n 's/^original_sha256 = "\([0-9a-f]\{64\}\)"/\1/p' experiments/mot16-09.toml)"
MOT_ZIP="datasets/_download/MOT16.zip"
need_mot=0
for seq in MOT16-09:train MOT16-08:test; do
    name="${seq%%:*}"
    if [ ! -d "datasets/mot16/$name/img1" ]; then need_mot=1; fi
done
if [ "$need_mot" = 1 ]; then
    mkdir -p datasets/_download datasets/mot16
    if [ ! -f "$MOT_ZIP" ]; then
        note "fetch $MOT_URL (1.9 GB)"
        curl -fL --retry 3 -o "$MOT_ZIP.part" "$MOT_URL" && mv "$MOT_ZIP.part" "$MOT_ZIP"
    fi
    have="$(digest "$MOT_ZIP")"
    if [ "$have" != "$MOT_SHA" ]; then
        fail "$MOT_ZIP hashes to $have, expected $MOT_SHA; the archive is not the one the campaign used"
    else
        note "ok    $MOT_ZIP ($have)"
        for seq in MOT16-09:train MOT16-08:test; do
            name="${seq%%:*}"; split="${seq##*:}"
            if [ -d "datasets/mot16/$name/img1" ]; then continue; fi
            note "unzip $split/$name"
            unzip -q -o "$MOT_ZIP" "$split/$name/*" -d datasets/_download/mot16-extract
            mkdir -p "datasets/mot16/$name"
            cp -R "datasets/_download/mot16-extract/$split/$name/img1" "datasets/mot16/$name/"
            cp "datasets/_download/mot16-extract/$split/$name/seqinfo.ini" "datasets/mot16/$name/"
        done
    fi
fi
for name in MOT16-09 MOT16-08; do
    if [ -d "datasets/mot16/$name/img1" ]; then
        count="$(ls "datasets/mot16/$name/img1" | wc -l | tr -d ' ')"
        note "ok    datasets/mot16/$name/img1 ($count frames); the preflight checks the content digest"
    fi
done

# --------------------------------------------------------------------------- #
# MuPoTS-3D TS10 and TS11, from MPI for Informatics
# --------------------------------------------------------------------------- #
# The archive is the whole test set (5.6 GB, twenty sequences); only the two
# sequences of the people suite are extracted, into the directory the
# ground-truth evaluation (ADR-0004) reads, so one copy serves both.
MUPOTS_URL="https://gvv.mpi-inf.mpg.de/projects/SingleShotMultiPerson/MultiPersonTestSet.zip"
MUPOTS_SHA="$(sed -n 's/^original_sha256 = "\([0-9a-f]\{64\}\)"/\1/p' experiments/mupots-ts10.toml)"
MUPOTS_ZIP="datasets/_download/MultiPersonTestSet.zip"
MUPOTS_ROOT="datasets/mupots-3d/MultiPersonTestSet"
need_mupots=0
for seq in TS10 TS11; do
    if [ ! -f "$MUPOTS_ROOT/$seq/img_000000.jpg" ]; then need_mupots=1; fi
done
if [ "$need_mupots" = 1 ]; then
    if [ -z "$MUPOTS_TERMS" ]; then
        fail "MuPoTS-3D TS10/TS11 are absent. Read the terms at https://vcai.mpi-inf.mpg.de/projects/SingleShotMultiPerson/ (research only, no redistribution, cite Mehta et al. 3DV 2018 and 3DV 2017) and rerun with --accept-mupots-terms"
    else
        mkdir -p datasets/_download "$MUPOTS_ROOT"
        if [ ! -f "$MUPOTS_ZIP" ]; then
            note "fetch $MUPOTS_URL (5.6 GB); terms accepted by the operator"
            curl -fL --retry 3 -o "$MUPOTS_ZIP.part" "$MUPOTS_URL" && mv "$MUPOTS_ZIP.part" "$MUPOTS_ZIP"
        fi
        have="$(digest "$MUPOTS_ZIP")"
        if [ "$have" != "$MUPOTS_SHA" ]; then
            fail "$MUPOTS_ZIP hashes to $have, expected $MUPOTS_SHA; the archive is not the one the campaign used"
        else
            note "ok    $MUPOTS_ZIP ($have)"
            for seq in TS10 TS11; do
                if [ -f "$MUPOTS_ROOT/$seq/img_000000.jpg" ]; then continue; fi
                note "unzip MultiPersonTestSet/$seq"
                unzip -q -o "$MUPOTS_ZIP" "MultiPersonTestSet/$seq/*" -d datasets/mupots-3d
            done
        fi
    fi
fi
for seq in TS10 TS11; do
    if [ -f "$MUPOTS_ROOT/$seq/img_000000.jpg" ]; then
        count="$(ls "$MUPOTS_ROOT/$seq" | grep -c '\.jpg$' | tr -d ' ')"
        note "ok    $MUPOTS_ROOT/$seq ($count frames); the preflight checks the content digest"
    fi
done

if [ "$status" = 0 ]; then
    note "done: run 'make bench-preflight MACHINE=<id>' to verify every input against the experiments"
fi
exit "$status"
