#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

SWEEP_FLOOR_DATA_ROOT=${SWEEP_FLOOR_DATA_ROOT:-"/mnt/raid0/unitree_brainco/gear_format/data_sweep_floor_lerobot_gear_50eps"}
CACHE_DIR=${CACHE_DIR:-/mnt/raid0/unitree_brainco/cache/dreamzero/t5_cache/unitree_sweep_floor_50eps}
NUM_GPUS=${NUM_GPUS:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export CUDA_VISIBLE_DEVICES
export NO_ALBUMENTATIONS_UPDATE=1
if [ -z "${PYTHON_EXEC:-}" ]; then
  if [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
    PYTHON_EXEC="$CONDA_PREFIX/bin/python"
  elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON_EXEC="$VIRTUAL_ENV/bin/python"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_EXEC=$(command -v python)
  else
    PYTHON_EXEC=$(command -v python3)
  fi
fi
echo "Using PYTHON_EXEC=$PYTHON_EXEC"

"$PYTHON_EXEC" - <<'PY'
import importlib.util
import sys

missing = [name for name in ("torch", "peft") if importlib.util.find_spec(name) is None]
if missing:
    print(
        "ERROR: missing Python package(s): " + ", ".join(missing) + "\n"
        "Install them in the same H200 environment used by this script, e.g.:\n"
        f"  {sys.executable} -m pip install peft==0.5.0",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY

WAN22_CKPT_DIR=${WAN22_CKPT_DIR:-"/mnt/raid0/dreamzero_models/Wan2.2-TI2V-5B"}
WAN21_CKPT_DIR=${WAN21_CKPT_DIR:-"/mnt/raid0/dreamzero_models/Wan2.1-I2V-14B-480P"}

resolve_hf_snapshot_dir() {
  local dir="$1"
  if [ -f "$dir/refs/main" ]; then
    local ref
    ref=$(cat "$dir/refs/main")
    if [ -d "$dir/snapshots/$ref" ]; then
      echo "$dir/snapshots/$ref"
      return 0
    fi
  fi
  echo "$dir"
}

WAN22_CKPT_DIR=$(resolve_hf_snapshot_dir "$WAN22_CKPT_DIR")
WAN21_CKPT_DIR=$(resolve_hf_snapshot_dir "$WAN21_CKPT_DIR")
TOKENIZER_DIR=${TOKENIZER_DIR:-"$WAN22_CKPT_DIR/google/umt5-xxl"}
if [ ! -f "$TOKENIZER_DIR/tokenizer.json" ] && [ ! -f "$TOKENIZER_DIR/spiece.model" ]; then
  echo "ERROR: tokenizer files missing under $TOKENIZER_DIR"
  echo "Set TOKENIZER_DIR to a local snapshot directory containing tokenizer.json or spiece.model."
  exit 1
fi

"$PYTHON_EXEC" -m torch.distributed.run \
  --nproc_per_node="$NUM_GPUS" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  scripts/data/precompute_t5_text_embeddings.py \
  data=dreamzero/unitree_sweep_floor_relative_wan22 \
  sweep_floor_data_root="$SWEEP_FLOOR_DATA_ROOT" \
  single_dataset_cls=groot.vla.data.dataset.lerobot_sharded.ShardedLeRobotPaddedLangActionChunkDatasetUnitree \
  model=dreamzero/vla \
  model/dreamzero/action_head=wan_flow_matching_action_tf_wan22_unitree_352x640_full \
  model/dreamzero/transform=dreamzero_cotrain \
  num_frames=33 \
  action_horizon=48 \
  num_views=3 \
  num_frame_per_block=2 \
  num_action_per_block=48 \
  num_state_per_block=1 \
  image_resolution_width=320 \
  image_resolution_height=176 \
  max_state_dim=64 \
  max_action_dim=64 \
  max_chunk_size=4 \
  dit_version="$WAN22_CKPT_DIR" \
  text_encoder_pretrained_path="$WAN22_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth" \
  image_encoder_pretrained_path="$WAN21_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  vae_pretrained_path="$WAN22_CKPT_DIR/Wan2.2_VAE.pth" \
  tokenizer_path="$TOKENIZER_DIR" \
  pretrained_model_path=null \
  text_embedding_cache_dir="$CACHE_DIR"
