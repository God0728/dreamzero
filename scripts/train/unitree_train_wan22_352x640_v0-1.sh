#!/bin/bash
# Unitree real-robot upper-body Wan2.2-TI2V-5B from-Wan training, full DiT fine-tune.
# Temporal/data alignment follows Agibot-style 30FPS data:
#   - video frames sampled from 30FPS to 5FPS, num_frames=33
#   - action_horizon=48 and num_action_per_block=48
#   - per-view resize 320x176, 3-view composite 640x352, frame_seqlen=220
# No DreamZero/DROID checkpoint is loaded; action/state encoder and action decoder
# are initialized from scratch while Wan2.2 DiT/T5/VAE and Wan2.1 CLIP are loaded.

set -euo pipefail
export HYDRA_FULL_ERROR=1

export NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=8
export OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

T5_CACHE_DIR=${T5_CACHE_DIR:-/DATA/disk2/yuhangwang/data/WorldModel/dreamzero/t5_cache/unitree_3tasks}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

UNITREE_0415_ORGANIZE_STATIONERY_DATA_ROOT=${UNITREE_0415_ORGANIZE_STATIONERY_DATA_ROOT:-"/DATA/disk3/yuhangwang/data/Depth/raw/unitree_3tasks_v2.1/0415/organize_stationery"}
UNITREE_0415_WATER_PLANT_DATA_ROOT=${UNITREE_0415_WATER_PLANT_DATA_ROOT:-"/DATA/disk3/yuhangwang/data/Depth/raw/unitree_3tasks_v2.1/0415/water_plant"}
UNITREE_0415_COLLECT_EGGS_DATA_ROOT=${UNITREE_0415_COLLECT_EGGS_DATA_ROOT:-"/DATA/disk3/yuhangwang/data/Depth/raw/unitree_3tasks_v2.1/0415/collect_eggs"}
UNITREE_0508_COLLECT_EGGS_DATA_ROOT=${UNITREE_0508_COLLECT_EGGS_DATA_ROOT:-"/DATA/disk3/yuhangwang/data/Depth/raw/unitree_3tasks_v2.1/0508/collect_eggs"}
OUTPUT_DIR=${OUTPUT_DIR:-"/DATA/disk2/yuhangwang/data/WorldModel/dreamzero/trains/unitree_wan22_352x640_gbs8_v0.1"}

NUM_GPUS=${NUM_GPUS:-8}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((NUM_GPUS * PER_DEVICE_TRAIN_BATCH_SIZE))}

WAN22_CKPT_DIR=${WAN22_CKPT_DIR:-"/DATA/disk2/yuhangwang/model/WorldModel/dreamzero/Wan2.2-TI2V-5B"}
WAN21_CKPT_DIR=${WAN21_CKPT_DIR:-"/DATA/disk2/yuhangwang/model/WorldModel/dreamzero/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"/DATA/disk2/yuhangwang/model/WorldModel/dreamzero/umt5-xxl"}
MAX_STEPS=${MAX_STEPS:-100000}
SAVE_STEPS=${SAVE_STEPS:-10000}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-groot/vla/configs/deepspeed/zero1.json}
REPORT_TO=${REPORT_TO:-tensorboard}
TENSORBOARD_DIR=${TENSORBOARD_DIR:-"$OUTPUT_DIR/tensorboard"}

for dataset_root in \
  "$UNITREE_0415_ORGANIZE_STATIONERY_DATA_ROOT" \
  "$UNITREE_0415_WATER_PLANT_DATA_ROOT" \
  "$UNITREE_0415_COLLECT_EGGS_DATA_ROOT" \
  "$UNITREE_0508_COLLECT_EGGS_DATA_ROOT"
do
  if [ ! -d "$dataset_root" ]; then
    echo "ERROR: Unitree dataset not found at $dataset_root"
    exit 1
  fi
  for meta_file in modality.json embodiment.json relative_stats_dreamzero.json; do
    if [ ! -f "$dataset_root/meta/$meta_file" ]; then
      echo "ERROR: $dataset_root/meta/$meta_file is missing. Run convert_lerobot_to_gear.py first."
      exit 1
    fi
  done
done
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

# Per-view 320x176 becomes an Agibot-style 640x352 2x2 composite
# [head | right wrist; left wrist | black], matching frame_seqlen=220.
torchrun --nproc_per_node "$NUM_GPUS" --standalone groot/vla/experiment/experiment.py \
  report_to="$REPORT_TO" \
  data=dreamzero/unitree_upper_body_relative_wan22 \
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
  unitree_0415_organize_stationery_data_root="$UNITREE_0415_ORGANIZE_STATIONERY_DATA_ROOT" \
  unitree_0415_water_plant_data_root="$UNITREE_0415_WATER_PLANT_DATA_ROOT" \
  unitree_0415_collect_eggs_data_root="$UNITREE_0415_COLLECT_EGGS_DATA_ROOT" \
  unitree_0508_collect_eggs_data_root="$UNITREE_0508_COLLECT_EGGS_DATA_ROOT" \
  dit_version="$WAN22_CKPT_DIR" \
  text_encoder_pretrained_path="$WAN22_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth" \
  image_encoder_pretrained_path="$WAN21_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  vae_pretrained_path="$WAN22_CKPT_DIR/Wan2.2_VAE.pth" \
  tokenizer_path="$TOKENIZER_DIR" \
  pretrained_model_path=null \
  text_embedding_cache_dir="$T5_CACHE_DIR" \
  require_text_embedding_cache=true \
  action_head_cfg.config.load_text_encoder=false \
  text_embedding_cache_runtime=model

  # action_head_cfg.config.use_gradient_checkpointing=false
