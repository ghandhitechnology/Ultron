#!/usr/bin/env bash
# Idempotent Cloud Agent bootstrap for the Ultron research trainer.
# Installs the two pieces missing from the default image (python3-venv, uv),
# then syncs Python deps from the committed lockfile and the TypeScript harness deps.
set -euo pipefail

# python3-venv is required for uv/pip to build the virtual environment.
if ! dpkg -s python3-venv >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends python3-venv
fi

# uv drives reproducible installs from the committed uv.lock. Install it to a
# global location so every agent shell can find it without PATH tweaks.
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh \
    | sudo env UV_INSTALL_DIR=/usr/local/bin UV_UNMANAGED_INSTALL=1 sh
fi

# Python environment (.venv) resolved from uv.lock, including the dev extra (pytest).
uv sync --extra dev

# TypeScript harness dependencies for `npm run check`.
npm install
