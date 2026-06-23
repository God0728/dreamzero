#!/bin/bash
# Unitree collect-blocks-into-basket Wan2.2-TI2V-5B 单相机(camera0, 384x512)全量 DiT 微调脚本。
# 数据需先用 scripts/data/convert_unitree_group_split_to_lerobot_gear.py 转换。
# 默认只训练 output_root/train；output_root/val 留给离线评估，不混入训练。
# state/action 为单一 60-D full-body key:
#   robot_q[:36] + hand[:12] + ee_state[:12]
# 其中 robot_q[:36] = floating base 7D + G1 全身 29 关节。

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
        "Install them in the same environment used by this script, e.g.:\n"
        f"  {sys.executable} -m pip install peft==0.5.0",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

DATASET_ROOT=${DATASET_ROOT:-"/mnt/raid0/jimmy/gear_format/walk_to_table_put_cups_and_rack_plates_in_dishwasher_lerobot_gear"}
COLLECT_BLOCKS_DATA_ROOT=${COLLECT_BLOCKS_DATA_ROOT:-${DISHWASHER_DATA_ROOT:-"$DATASET_ROOT/train"}}
COLLECT_BLOCKS_VAL_DATA_ROOT=${COLLECT_BLOCKS_VAL_DATA_ROOT:-${DISHWASHER_VAL_DATA_ROOT:-"$DATASET_ROOT/val"}}
OUTPUT_DIR=${OUTPUT_DIR:-"/mnt/raid0/dreamzero_checkpoints/unitree_collect_blocks_1camera_camera0_wan22_384x512_v0.1"}
T5_CACHE_DIR=${T5_CACHE_DIR:-"/mnt/raid0/dreamzero_models/t5_cache/unitree_collect_blocks_1camera_camera0_384x512_train"}

NUM_GPUS=${NUM_GPUS:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((NUM_GPUS * PER_DEVICE_TRAIN_BATCH_SIZE))}

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
MAX_STEPS=${MAX_STEPS:-50000}
SAVE_STEPS=${SAVE_STEPS:-2000}
SAVE_STRATEGY=${SAVE_STRATEGY:-steps}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-10}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-groot/vla/configs/deepspeed/zero1.json}
REPORT_TO=${REPORT_TO:-tensorboard}
TENSORBOARD_DIR=${TENSORBOARD_DIR:-"$OUTPUT_DIR/tensorboard"}
DATALOADER_PIN_MEMORY=${DATALOADER_PIN_MEMORY:-false}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-0}

# Optional continuation from a previous DreamZero checkpoint, e.g. stack-blocks:
#   PRETRAINED_MODEL_PATH=/mnt/raid0/dreamzero_checkpoints/.../checkpoint-50000 bash ...
PRETRAINED_MODEL_PATH=${PRETRAINED_MODEL_PATH:-null}
TASK_PROMPT=${TASK_PROMPT:-"A bimanual robotic task in a home living room environment. The robot looks down from a head-mounted camera at a light wood-grain desk with several objects: a green building block and an orange-yellow building block. A light-brown woven basket rests on a white side table to the right. The robot first turns to the side table, grasps the woven basket with the right hand, and lifts it. It then holds the basket suspended over the main desk with the right hand alone. The left hand reaches down, pinches the orange-yellow building block, lifts it, and drops it into the basket. The left hand then picks up the green building block and drops it into the basket as well. Finally, both hands carry the basket back to the white side table on the right and gently place it down. The motions are slow, deliberate, and highly smooth, with precise finger coordination and seamless weight transfer between hands."}

if [ ! -d "$COLLECT_BLOCKS_DATA_ROOT" ]; then
  echo "ERROR: converted collect-blocks train dataset not found at $COLLECT_BLOCKS_DATA_ROOT"
  echo "Run:"
  echo "  /home/unitree/miniconda3/envs/dreamzero/bin/python scripts/data/convert_unitree_group_split_to_lerobot_gear.py \\"
  echo "    --raw-root /mnt/raid0/jimmy/walk_to_table_put_cups_and_rack_plates_in_dishwasher \\"
  echo "    --output-root $DATASET_ROOT --default-task \"$TASK_PROMPT\" --val-last-n-episodes 1 --force"
  exit 1
fi
if [ ! -d "$COLLECT_BLOCKS_VAL_DATA_ROOT" ]; then
  echo "WARN: collect-blocks val dataset not found at $COLLECT_BLOCKS_VAL_DATA_ROOT"
  echo "      Training can continue, but offline validation will be unavailable."
fi
for meta_file in info.json modality.json embodiment.json stats.json relative_stats_dreamzero.json tasks.jsonl episodes.jsonl; do
  if [ ! -f "$COLLECT_BLOCKS_DATA_ROOT/meta/$meta_file" ]; then
    echo "ERROR: $COLLECT_BLOCKS_DATA_ROOT/meta/$meta_file is missing."
    exit 1
  fi
done

"$PYTHON_EXEC" - "$COLLECT_BLOCKS_DATA_ROOT" <<'PY'
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
            "Re-run scripts/data/convert_unitree_group_split_to_lerobot_gear.py --force.",
            file=sys.stderr,
        )
        raise SystemExit(1)
episodes = [line for line in (dataset_root / "meta" / "episodes.jsonl").read_text().splitlines() if line.strip()]
tasks = [json.loads(line)["task"] for line in (dataset_root / "meta" / "tasks.jsonl").read_text().splitlines() if line.strip()]
print(f"Train dataset: {len(episodes)} episodes, full-body control dim={expected_dim}")
print(f"Task prompt: {tasks[0] if tasks else '<missing>'}")
PY

if [ ! -d "$T5_CACHE_DIR" ] || [ -z "$(ls -A "$T5_CACHE_DIR" 2>/dev/null)" ]; then
  echo "ERROR: T5 cache missing at $T5_CACHE_DIR"
  echo "Run: bash scripts/train/precompute_T5_cache_unitree_collect_blocks_1camera_camera0_384x512.sh"
  exit 1
fi
"$PYTHON_EXEC" - "$COLLECT_BLOCKS_DATA_ROOT" "$T5_CACHE_DIR" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from groot.vla.model.dreamzero.transform.dreamzero_cotrain import format_dreamzero_prompt

dataset_root = Path(sys.argv[1])
cache_dir = Path(sys.argv[2])
tasks = [
    json.loads(line)["task"]
    for line in (dataset_root / "meta" / "tasks.jsonl").read_text().splitlines()
    if line.strip()
]
if not tasks:
    print(f"ERROR: no task prompt found in {dataset_root / 'meta' / 'tasks.jsonl'}", file=sys.stderr)
    raise SystemExit(1)

missing = []
for task in sorted(set(tasks)):
    prompt = format_dreamzero_prompt(
        task,
        embodiment_id=26,
        num_views=1,
        embodiment_tag_mapping={"unitree_g1_upper_body": 26},
        embodiment_tag="unitree_g1_upper_body",
    )
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    cache_path = cache_dir / f"{digest}.t5_len512.dreamzero_wan_t5.pt"
    print(f"T5 prompt cache: {digest} exists={cache_path.exists()}")
    if not cache_path.exists():
        missing.append(cache_path)

if missing:
    print("ERROR: required T5 prompt cache file(s) missing:", file=sys.stderr)
    for path in missing:
        print(f"  {path}", file=sys.stderr)
    print(
        "Run: bash scripts/train/precompute_T5_cache_unitree_collect_blocks_1camera_camera0_384x512.sh",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
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
  data=dreamzero/unitree_collect_blocks_1camera_camera0_relative_wan22 \
  single_dataset_cls=groot.vla.data.dataset.lerobot_sharded.ShardedLeRobotSubLangSingleActionChunkDatasetUnitree \
  wandb_project=dreamzero \
  train_architecture=full \
  num_frames=33 \
  action_horizon=48 \
  num_views=1 \
  model=dreamzero/vla \
  model/dreamzero/action_head=wan_flow_matching_action_tf_wan22_unitree_384x512_full \
  model/dreamzero/transform=dreamzero_cotrain \
  num_frame_per_block=2 \
  num_action_per_block=48 \
  num_state_per_block=1 \
  frame_seqlen=192 \
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
  save_total_limit="$SAVE_TOTAL_LIMIT" \
  upload_checkpoints=false \
  bf16=true \
  tf32=true \
  eval_bf16=true \
  dataloader_pin_memory="$DATALOADER_PIN_MEMORY" \
  dataloader_num_workers="$DATALOADER_NUM_WORKERS" \
  image_resolution_width=512 \
  image_resolution_height=384 \
  save_lora_only=false \
  max_chunk_size=4 \
  save_strategy="$SAVE_STRATEGY" \
  collect_blocks_data_root="$COLLECT_BLOCKS_DATA_ROOT" \
  dit_version="$WAN22_CKPT_DIR" \
  text_encoder_pretrained_path="$WAN22_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth" \
  image_encoder_pretrained_path="$WAN21_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  vae_pretrained_path="$WAN22_CKPT_DIR/Wan2.2_VAE.pth" \
  tokenizer_path="$TOKENIZER_DIR" \
  pretrained_model_path="$PRETRAINED_MODEL_PATH" \
  text_embedding_cache_dir="$T5_CACHE_DIR" \
  require_text_embedding_cache=true \
  action_head_cfg.config.load_text_encoder=false \
  action_head_cfg.config.torch_compile_dit_blocks=false \
  text_embedding_cache_runtime=model
