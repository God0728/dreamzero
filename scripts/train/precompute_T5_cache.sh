#!/usr/bin/env bash
set -euo pipefail

cd /DATA/disk2/yuhangwang/code/WorldModel/dz/source/dreamzero

CACHE_DIR=${CACHE_DIR:-/DATA/disk2/yuhangwang/data/WorldModel/dreamzero/t5_cache/libero_all_wan22_160x320}
NUM_GPUS=${NUM_GPUS:-4}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export CUDA_VISIBLE_DEVICES

torchrun --standalone --nproc_per_node="$NUM_GPUS" \
  scripts/data/precompute_t5_text_embeddings.py \
  data=dreamzero/libero_all_relative \
  single_dataset_cls=groot.vla.data.dataset.lerobot_sharded.ShardedLeRobotPaddedLangActionChunkDatasetDROID \
  model=dreamzero/vla \
  model/dreamzero/action_head=wan_flow_matching_action_tf_wan22_libero_160x320_full \
  model/dreamzero/transform=dreamzero_cotrain \
  num_frames=33 \
  action_horizon=24 \
  num_views=2 \
  num_frame_per_block=2 \
  num_action_per_block=24 \
  num_state_per_block=1 \
  image_resolution_width=160 \
  image_resolution_height=160 \
  max_chunk_size=4 \
  libero_all_data_root=/DATA/disk2/yuhangwang/data/WorldModel/libero_mujoco3.3.2 \
  dit_version=/DATA/disk2/yuhangwang/model/WorldModel/dreamzero/Wan2.2-TI2V-5B \
  text_encoder_pretrained_path=/DATA/disk2/yuhangwang/model/WorldModel/dreamzero/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth \
  image_encoder_pretrained_path=/DATA/disk2/yuhangwang/model/WorldModel/dreamzero/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
  vae_pretrained_path=/DATA/disk2/yuhangwang/model/WorldModel/dreamzero/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
  tokenizer_path=/DATA/disk2/yuhangwang/model/WorldModel/dreamzero/umt5-xxl \
  pretrained_model_path=null \
  text_embedding_cache_dir="$CACHE_DIR"
