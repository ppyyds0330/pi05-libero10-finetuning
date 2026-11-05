#!/usr/bin/env bash
# Fetch the pinned upstream and apply the exact patch from the completed run.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-/root/autodl-tmp/pi-libero/openpi}"
PIN=215abfb217dbac7d5f1273282331b9b1866c0479
if [[ ! -e "$TARGET" ]]; then
  git clone --no-checkout https://github.com/Physical-Intelligence/openpi.git "$TARGET"
  git -C "$TARGET" checkout --detach "$PIN"
elif [[ "$(git -C "$TARGET" rev-parse HEAD)" != "$PIN" ]]; then
  echo "Existing checkout differs from the pinned commit; select a clean target." >&2
  exit 1
fi
git -C "$TARGET" submodule update --init --recursive
PATCH="$PROJECT_DIR/patches/openpi-libero10-lora.patch"
if git -C "$TARGET" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "The project patch is already applied."
else
  git -C "$TARGET" apply --check "$PATCH"
  git -C "$TARGET" apply "$PATCH"
fi
echo "Prepared: $TARGET"
echo "Read docs/reproduction.md before installing dependencies or training."
