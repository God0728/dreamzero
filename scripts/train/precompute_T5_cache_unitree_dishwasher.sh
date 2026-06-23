#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
echo "WARNING: scripts/train/precompute_T5_cache_unitree_dishwasher.sh is deprecated."
echo "Use scripts/train/precompute_T5_cache_unitree_collect_blocks_1camera_camera0_384x512.sh instead."

exec bash "$SCRIPT_DIR/precompute_T5_cache_unitree_collect_blocks_1camera_camera0_384x512.sh" "$@"
