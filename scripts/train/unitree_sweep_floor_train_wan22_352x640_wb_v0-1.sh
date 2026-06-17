#!/bin/bash
# Unitree sweep-floor Wan2.2-TI2V-5B full DiT fine-tune.
# Dataset must first be converted by scripts/data/convert_sweep_floor_to_lerobot_gear.py.
# State/action are one 60-D key: robot_q[:36] + hand[:12] + ee_state[:12].
# The 60-D action is trained as relative action against the matching current state.

set -euo pipefail
export HYDRA_FULL_ERROR=1

export NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=8
export OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
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

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

SWEEP_FLOOR_DATA_ROOT=${SWEEP_FLOOR_DATA_ROOT:-"/mnt/unitree_cpfs/ruixuan/datasets/gear_format/data_sweep_floor_lerobot_gear_100eps"}
OUTPUT_DIR=${OUTPUT_DIR:-"/mnt/unitree_cpfs/ruixuan/code/wm/dreamzero/checkpoints/unitree_sweep_floor_wan22_352x640_gbs8_v0.2"}
T5_CACHE_DIR=${T5_CACHE_DIR:-"/mnt/unitree_cpfs/ruixuan/cache/dreamzero/t5_cache/unitree_sweep_floor_100eps"}

NUM_GPUS=${NUM_GPUS:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((NUM_GPUS * PER_DEVICE_TRAIN_BATCH_SIZE))}

WAN22_CKPT_DIR=${WAN22_CKPT_DIR:-"/mnt/unitree_cpfs/ruixuan/cache/hf_cache/Wan2.2-TI2V-5B"}
WAN21_CKPT_DIR=${WAN21_CKPT_DIR:-"/mnt/unitree_cpfs/ruixuan/cache/hf_cache/models--Wan-AI--Wan2.1-I2V-14B-480P"}

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
MAX_STEPS=${MAX_STEPS:-50000}
SAVE_STEPS=${SAVE_STEPS:-2000}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-groot/vla/configs/deepspeed/zero1.json}
REPORT_TO=${REPORT_TO:-tensorboard}
TENSORBOARD_DIR=${TENSORBOARD_DIR:-"$OUTPUT_DIR/tensorboard"}

if [ ! -d "$SWEEP_FLOOR_DATA_ROOT" ]; then
  echo "ERROR: converted sweep-floor dataset not found at $SWEEP_FLOOR_DATA_ROOT"
  echo "Run: python3 scripts/data/convert_sweep_floor_to_lerobot_gear.py --force"
  exit 1
fi
for meta_file in info.json modality.json embodiment.json stats.json relative_stats_dreamzero.json tasks.jsonl episodes.jsonl; do
  if [ ! -f "$SWEEP_FLOOR_DATA_ROOT/meta/$meta_file" ]; then
    echo "ERROR: $SWEEP_FLOOR_DATA_ROOT/meta/$meta_file is missing."
    echo "Run: python3 scripts/data/convert_sweep_floor_to_lerobot_gear.py --force"
    exit 1
  fi
done
"$PYTHON_EXEC" - "$SWEEP_FLOOR_DATA_ROOT" <<'PY'
import json
import sys
from pathlib import Path

dataset_root = Path(sys.argv[1])
expected_dim = 60
info = json.loads((dataset_root / "meta" / "info.json").read_text())
features = info.get("features", {})
for key in ("observation.state.sweep_floor_control", "action.sweep_floor_control"):
    shape = features.get(key, {}).get("shape")
    if shape != [expected_dim]:
        print(
            f"ERROR: {key} shape is {shape}, expected [{expected_dim}]. "
            "Re-run scripts/data/convert_sweep_floor_to_lerobot_gear.py --force "
            "after updating the converter.",
            file=sys.stderr,
        )
        raise SystemExit(1)
PY
if [ ! -d "$T5_CACHE_DIR" ] || [ -z "$(ls -A "$T5_CACHE_DIR" 2>/dev/null)" ]; then
  echo "ERROR: T5 cache missing at $T5_CACHE_DIR"
  echo "Run: bash scripts/train/precompute_T5_cache_unitree_sweep_floor.sh"
  exit 1
fi
if [ ! -d "$WAN22_CKPT_DIR" ] || [ -z "$(ls -A "$WAN22_CKPT_DIR" 2>/dev/null)" ]; then
  echo "ERROR: Wan2.2 weights missing at $WAN22_CKPT_DIR"
  exit 1
fi
if [ ! -f "$WAN21_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" ]; then
  echo "ERROR: Wan2.1 CLIP image encoder missing under $WAN21_CKPT_DIR"
  exit 1
fi
if [ ! -d "$TOKENIZER_DIR" ] || [ -z "$(ls -A "$TOKENIZER_DIR" 2>/dev/null)" ]; then
  echo "ERROR: tokenizer missing at $TOKENIZER_DIR"
  exit 1
fi
if [ ! -f "$TOKENIZER_DIR/tokenizer.json" ] && [ ! -f "$TOKENIZER_DIR/spiece.model" ]; then
  echo "ERROR: tokenizer files missing under $TOKENIZER_DIR"
  echo "Set TOKENIZER_DIR to a local snapshot directory containing tokenizer.json or spiece.model."
  exit 1
fi

VISIBLE_GPUS=$("$PYTHON_EXEC" - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)
if [ "$VISIBLE_GPUS" -lt "$NUM_GPUS" ]; then
  echo "ERROR: NUM_GPUS=$NUM_GPUS but only $VISIBLE_GPUS CUDA device(s) are visible to Python."
  echo "Check CUDA_VISIBLE_DEVICES, or run with NUM_GPUS=$VISIBLE_GPUS."
  exit 1
fi

"$PYTHON_EXEC" -m torch.distributed.run \
  --nproc_per_node "$NUM_GPUS" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  groot/vla/experiment/experiment.py \
  report_to="$REPORT_TO" \
  data=dreamzero/unitree_sweep_floor_relative_wan22 \
  single_dataset_cls=groot.vla.data.dataset.lerobot_sharded.ShardedLeRobotSubLangSingleActionChunkDatasetUnitree \
  wandb_project=dreamzero \
  train_architecture=full \
  num_frames=33 \
  action_horizon=48 \
  num_views=3 \
  model=dreamzero/vla \
  model/dreamzero/action_head=wan_flow_matching_action_tf_wan22_unitree_352x640_full \
  model/dreamzero/transform=dreamzero_cotrain \
  num_frame_per_block=2 \
  num_action_per_block=48 \
  num_state_per_block=1 \
  frame_seqlen=220 \
  seed=42 \
  max_state_dim=64 \
  max_action_dim=64 \
  training_args.learning_rate="$LEARNING_RATE" \
  +training_args.logging_dir="$TENSORBOARD_DIR" \
  +training_args.logging_first_step=true \
  training_args.deepspeed="$DEEPSPEED_CONFIG" \
  save_steps="$SAVE_STEPS" \
  training_args.warmup_ratio=0.05 \
  output_dir="$OUTPUT_DIR" \
  per_device_train_batch_size="$PER_DEVICE_TRAIN_BATCH_SIZE" \
  global_batch_size="$GLOBAL_BATCH_SIZE" \
  max_steps="$MAX_STEPS" \
  weight_decay=1e-2 \
  training_args.adam_beta1=0.9 \
  training_args.adam_beta2=0.95 \
  save_total_limit=10 \
  upload_checkpoints=false \
  bf16=true \
  tf32=true \
  eval_bf16=true \
  dataloader_pin_memory=true \
  dataloader_num_workers=1 \
  image_resolution_width=320 \
  image_resolution_height=176 \
  save_lora_only=false \
  max_chunk_size=4 \
  save_strategy=steps \
  sweep_floor_data_root="$SWEEP_FLOOR_DATA_ROOT" \
  dit_version="$WAN22_CKPT_DIR" \
  text_encoder_pretrained_path="$WAN22_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth" \
  image_encoder_pretrained_path="$WAN21_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  vae_pretrained_path="$WAN22_CKPT_DIR/Wan2.2_VAE.pth" \
  tokenizer_path="$TOKENIZER_DIR" \
  pretrained_model_path=null \
  text_embedding_cache_dir="$T5_CACHE_DIR" \
  require_text_embedding_cache=true \
  action_head_cfg.config.load_text_encoder=false \
  action_head_cfg.config.torch_compile_dit_blocks=false \
  text_embedding_cache_runtime=model
