#!/usr/bin/env bash
# Prepare a rented Linux box for the campaign (ADR-0006 §4.3). Run on the
# instance, as root, from the directory the bundle was pushed to:
#
#   bash setup-instance.sh <commit> <machine-id>
#
# What it does, in order, and each step is idempotent:
#   1. apt: ffmpeg, git, unzip, curl, ca-certificates, make (nothing else; torch
#      comes from the lockfile)
#   2. uv, the Python toolchain manager the project pins everything with
#   3. the repository from the bundle, checked out at the campaign commit
#   4. the environment from the lockfile: `uv sync --frozen --extra multihmr`
#   5. the footage: clips pushed beside this script, MOT16 from MOTChallenge,
#      the synthetic fixture clip (ffmpeg testsrc2) for the shakedown,
#      MuPoTS-3D TS10/TS11 from MPI if UNPROJECT_ACCEPT_MUPOTS_TERMS=1 is set
#      (the operator accepted the terms on their own machine; campaign.py
#      forwards the variable when asked with --accept-mupots-terms)
#   6. the preflight, which fetches the weights and checks the card
#
# Apart from the apt packages, uv and the tools' own caches, everything it
# writes is under $HOME/unproject-campaign.
set -euo pipefail

COMMIT="${1:?usage: setup-instance.sh <commit> <machine-id>}"
MACHINE="${2:?usage: setup-instance.sh <commit> <machine-id>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="$HOME/unproject-campaign"
REPO="$WORK/Unproject"

echo "== 1. system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq ffmpeg git unzip curl ca-certificates make >/dev/null
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || {
    echo "nvidia-smi failed: no card or no driver on this instance" >&2; exit 1; }

echo "== 2. uv"
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

echo "== 3. repository at $COMMIT"
mkdir -p "$WORK"
if [ ! -d "$REPO/.git" ]; then
    git clone -q "$HERE/unproject.bundle" "$REPO"
fi
git -C "$REPO" fetch -q "$HERE/unproject.bundle" "$COMMIT" || true
git -C "$REPO" checkout -q "$COMMIT"
git -C "$REPO" status --porcelain | grep -q . && { echo "tree is dirty after checkout" >&2; exit 1; }
echo "at $(git -C "$REPO" rev-parse --short HEAD), clean"

echo "== 4. environment from the lockfile"
cd "$REPO"
uv sync --frozen --extra multihmr
uv run --frozen python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available(), 'archs', torch.cuda.get_arch_list())"

echo "== 5. footage"
mkdir -p datasets
MUPOTS_FLAG=""
if [ "${UNPROJECT_ACCEPT_MUPOTS_TERMS:-}" = "1" ]; then MUPOTS_FLAG="--accept-mupots-terms"; fi
if [ -d "$HERE/clips" ]; then
    benchmarks/fetch-datasets.sh --from-dir "$HERE/clips" $MUPOTS_FLAG
else
    benchmarks/fetch-datasets.sh $MUPOTS_FLAG
fi
# The fixture clip of experiments/fixture.toml (README, "fixture"): a synthetic
# pattern with nobody in it, never committed; the shakedown's first suite
# runs on it before any campaign job does.
if [ ! -f datasets/fixture/clip.mp4 ]; then
    mkdir -p datasets/fixture
    ffmpeg -loglevel error -y -f lavfi -i "testsrc2=size=320x240:rate=10:duration=3" \
        -c:v libx264 -pix_fmt yuv420p datasets/fixture/clip.mp4
fi

echo "== 6. preflight on $MACHINE"
make bench-preflight MACHINE="$MACHINE" || {
    echo "preflight not ready; read the FAIL lines above" >&2; exit 1; }
echo "ready: run 'make bench-all MACHINE=$MACHINE' in $REPO"
