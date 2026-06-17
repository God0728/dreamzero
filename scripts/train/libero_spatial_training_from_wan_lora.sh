#!/bin/bash

export HYDRA_FULL_ERROR=1

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

LIBERO_SPATIAL_DATA_ROOT=${LIBERO_SPATIAL_DATA_ROOT:-"/DATA/disk2/yuhangwang/data/WorldModel/libero-spatial-dreamzero"}
OUTPUT_DIR=${OUTPUT_DIR:-"/DATA/disk2/yuhangwang/data/WorldModel/dreamzero/trains/dreamzero_libero_spatial_from_wan_reset_heads_perdevicebs4"}

if [ -z "${NUM_GPUS:-}" ]; then
  NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
NUM_GPUS=${NUM_GPUS:-8}

WAN_CKPT_DIR=${WAN_CKPT_DIR:-"/DATA/disk2/yuhangwang/model/WorldModel/dreamzero/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"/DATA/disk2/yuhangwang/model/WorldModel/dreamzero/umt5-xxl"}
PRETRAINED_MODEL_PATH=${PRETRAINED_MODEL_PATH:-"/DATA/disk2/yuhangwang/model/WorldModel/dreamzero/DreamZero-DROID"}
MAX_STEPS=${MAX_STEPS:-40000}
SAVE_STEPS=${SAVE_STEPS:-10000}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-2}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG-groot/vla/configs/deepspeed/zero2.json}
REPORT_TO=${REPORT_TO:-tensorboard}
TENSORBOARD_DIR=${TENSORBOARD_DIR:-"$OUTPUT_DIR/tensorboard"}

if [ ! -d "$LIBERO_SPATIAL_DATA_ROOT" ]; then
  echo "ERROR: LIBERO-Spatial dataset not found at $LIBERO_SPATIAL_DATA_ROOT"
  exit 1
fi

if [ ! -f "$LIBERO_SPATIAL_DATA_ROOT/meta/embodiment.json" ]; then
  echo "ERROR: $LIBERO_SPATIAL_DATA_ROOT/meta/embodiment.json is missing"
  exit 1
fi

if [ ! -d "$WAN_CKPT_DIR" ] || [ -z "$(ls -A "$WAN_CKPT_DIR" 2>/dev/null)" ]; then
  echo "ERROR: Wan2.1 weights missing at $WAN_CKPT_DIR"
  echo "Check your proxy settings in ~/.bashrc before attempting any download."
  exit 1
fi

if [ ! -d "$TOKENIZER_DIR" ] || [ -z "$(ls -A "$TOKENIZER_DIR" 2>/dev/null)" ]; then
  echo "ERROR: tokenizer missing at $TOKENIZER_DIR"
  echo "Check your proxy settings in ~/.bashrc before attempting any download."
  exit 1
fi

if [ ! -d "$PRETRAINED_MODEL_PATH" ]; then
  echo "ERROR: DreamZero-DROID checkpoint missing at $PRETRAINED_MODEL_PATH"
  exit 1
fi

# Per-view 160x160 becomes a 160x320 side-by-side LIBERO composite, matching
# the Wan2.2 action-head target without an extra spatial rescale.
torchrun --nproc_per_node "$NUM_GPUS" --standalone groot/vla/experiment/experiment.py \
  report_to="$REPORT_TO" \
  data=dreamzero/libero_spatial_relative \
  single_dataset_cls=groot.vla.data.dataset.lerobot_sharded.ShardedLeRobotPaddedLangActionChunkDatasetDROID \
  wandb_project=dreamzero \
  train_architecture=lora \
  num_frames=33 \
  action_horizon=24 \
  num_views=2 \
  model=dreamzero/vla \
  model/dreamzero/action_head=wan_flow_matching_action_tf \
  model/dreamzero/transform=dreamzero_cotrain \
  num_frame_per_block=2 \
  num_action_per_block=24 \
  num_state_per_block=1 \
  seed=42 \
  training_args.learning_rate="$LEARNING_RATE" \
  +training_args.logging_dir="$TENSORBOARD_DIR" \
  +training_args.logging_first_step=true \
  training_args.deepspeed="$DEEPSPEED_CONFIG" \
  save_steps="$SAVE_STEPS" \
  training_args.warmup_ratio=0.05 \
  output_dir="$OUTPUT_DIR" \
  per_device_train_batch_size="$PER_DEVICE_TRAIN_BATCH_SIZE" \
  max_steps="$MAX_STEPS" \
  weight_decay=1e-5 \
  save_total_limit=5 \
  upload_checkpoints=false \
  bf16=true \
  tf32=true \
  eval_bf16=true \
  dataloader_pin_memory=true \
  dataloader_num_workers=4 \
  image_resolution_width=320 \
  image_resolution_height=352 \
  save_lora_only=true \
  max_chunk_size=4 \
  frame_seqlen=880 \
  save_strategy=steps \
  libero_spatial_data_root="$LIBERO_SPATIAL_DATA_ROOT" \
  dit_version="$WAN_CKPT_DIR" \
  text_encoder_pretrained_path="$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth" \
  image_encoder_pretrained_path="$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  vae_pretrained_path="$WAN_CKPT_DIR/Wan2.1_VAE.pth" \
  tokenizer_path="$TOKENIZER_DIR" \
  pretrained_model_path=null \
  ++action_head_cfg.config.skip_component_loading=false \
  ++action_head_cfg.config.defer_lora_injection=false
