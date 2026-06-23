#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

COLLECT_BLOCKS_DATA_ROOT=${COLLECT_BLOCKS_DATA_ROOT:-${DISHWASHER_DATA_ROOT:-"/mnt/raid0/jimmy/gear_format/walk_to_table_put_cups_and_rack_plates_in_dishwasher_lerobot_gear/train"}}
CACHE_DIR=${CACHE_DIR:-/mnt/raid0/dreamzero_models/t5_cache/unitree_collect_blocks_1camera_camera0_384x512_train}
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
        "Install them in the same environment used by this script, e.g.:\n"
        f"  {sys.executable} -m pip install peft==0.5.0",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY

if [ ! -d "$COLLECT_BLOCKS_DATA_ROOT" ]; then
  echo "ERROR: converted collect-blocks train dataset not found at $COLLECT_BLOCKS_DATA_ROOT"
  echo "Run scripts/data/convert_unitree_group_split_to_lerobot_gear.py first."
  exit 1
fi
for meta_file in info.json modality.json embodiment.json stats.json relative_stats_dreamzero.json tasks.jsonl episodes.jsonl; do
  if [ ! -f "$COLLECT_BLOCKS_DATA_ROOT/meta/$meta_file" ]; then
    echo "ERROR: $COLLECT_BLOCKS_DATA_ROOT/meta/$meta_file is missing."
    exit 1
  fi
done

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
TOKENIZER_DIR=${TOKENIZER_DIR:-"/mnt/raid0/dreamzero_models/umt5-xxl"}
if [ ! -f "$TOKENIZER_DIR/tokenizer.json" ] && [ ! -f "$TOKENIZER_DIR/spiece.model" ]; then
  echo "ERROR: tokenizer files missing under $TOKENIZER_DIR"
  echo "Set TOKENIZER_DIR to a local snapshot directory containing tokenizer.json or spiece.model."
  exit 1
fi

mkdir -p "$CACHE_DIR"

"$PYTHON_EXEC" -m torch.distributed.run \
  --nproc_per_node="$NUM_GPUS" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  scripts/data/precompute_t5_text_embeddings.py \
  data=dreamzero/unitree_collect_blocks_1camera_camera0_relative_wan22 \
  collect_blocks_data_root="$COLLECT_BLOCKS_DATA_ROOT" \
  single_dataset_cls=groot.vla.data.dataset.lerobot_sharded.ShardedLeRobotPaddedLangActionChunkDatasetUnitree \
  model=dreamzero/vla \
  model/dreamzero/action_head=wan_flow_matching_action_tf_wan22_unitree_384x512_full \
  model/dreamzero/transform=dreamzero_cotrain \
  num_frames=33 \
  action_horizon=48 \
  num_views=1 \
  num_frame_per_block=2 \
  num_action_per_block=48 \
  num_state_per_block=1 \
  image_resolution_width=512 \
  image_resolution_height=384 \
  max_state_dim=64 \
  max_action_dim=64 \
  max_chunk_size=4 \
  frame_seqlen=192 \
  dit_version="$WAN22_CKPT_DIR" \
  text_encoder_pretrained_path="$WAN22_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth" \
  image_encoder_pretrained_path="$WAN21_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  vae_pretrained_path="$WAN22_CKPT_DIR/Wan2.2_VAE.pth" \
  tokenizer_path="$TOKENIZER_DIR" \
  pretrained_model_path=null \
  text_embedding_cache_dir="$CACHE_DIR"
