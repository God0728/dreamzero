#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
echo "WARNING: scripts/train/unitree_dishwasher_train_wan22_320x176_v0-1.sh is deprecated."
echo "Use scripts/train/unitree_collect_blocks_1camera_camera0_train_wan22_384x512_v0-1.sh instead."

exec bash "$SCRIPT_DIR/unitree_collect_blocks_1camera_camera0_train_wan22_384x512_v0-1.sh" "$@"
