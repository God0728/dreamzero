```bash
下载模型

huggingface-cli download google/umt5-xxl --local-dir ./checkpoints/umt5-xxl
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./checkpoints/Wan2.1-I2V-14B-480P
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./Wan2.2-TI2V-5B

huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P \
  --local-dir /mnt/raid0/dreamzero_models/Wan2.1-I2V-14B-480P \
  --include "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"

转换数据集 数采 -> lerobot -> gear

python scripts/data/convert_stack_blocks_to_lerobot_gear.py \
  --raw-root /mnt/raid0/unitree_brainco/stack_blocks \
  --output-root /mnt/raid0/unitree_brainco/gear_format/stack_blocks_lerobot_gear_50eps --force

数据集路径

/mnt/raid0/unitree_brainco/gear_format/stack_blocks_lerobot_gear_50eps
挂载到
/home/unitree/jimmy/dreamzero/dataset/stack_blocks_lerobot_gear_50eps

模型权重文件路径

/mnt/raid0/dreamzero_models/Wan2.2-TI2V-5B
挂载到
/home/unitree/jimmy/dreamzero/checkpoints/Wan2.2-TI2V-5B

mkdir -p /home/unitree/jimmy/dreamzero/checkpoints/Wan2.2-TI2V-5B && mount --bind /mnt/raid0/dreamzero_models/Wan2.2-TI2V-5B /home/unitree/jimmy/dreamzero/checkpoints/Wan2.2-TI2V-5B && findmnt /home/unitree/jimmy/dreamzero/checkpoints/Wan2.2-TI2V-5B

mkdir -p /home/unitree/jimmy/dreamzero/checkpoints/umt5-xxl 
sudo mount --bind /mnt/raid0/dreamzero_models/umt5-xxl /home/unitree/jimmy/dreamzero/checkpoints/umt5-xxl && findmnt /home/unitree/jimmy/dreamzero/checkpoints/umt5-xxl


执行T5预计算

bash scripts/train/precompute_T5_cache_unitree_stack_blocks.sh


训练

问题
已经到 100% 20/20，loss 正常输出
崩在 save_model -> deepspeed clone_tensors_for_torch_save
同时出现 DataLoader worker killed by signal Killed（典型是保存阶段内存压力把 worker 连带杀掉

第20步保存触发
  → deepspeed zero1 把完整 5B 模型聚合 + clone 到 CPU 内存(~15-20GB)
  → 保存写盘耗时数分钟(单分片 5G)
  → 同时 dataloader worker(num_workers=1)在后台预取下一批
  → 预取触发新 shard 缓存:把整段视频帧解码进 RAM(可达数十 GB)
  → 保存 CPU 克隆 + shard 缓存内存叠加,瞬时尖峰
  → worker 进程被 SIGKILL
  → 主进程下一个 CPU 算子检测到 worker 死亡 → 抛 RuntimeError → 全体退出


解决方法：NUM_WORKERS = 0


第一次训练
REPORT_TO=wandb \
MAX_STEPS=50000 \
SAVE_STRATEGY=steps SAVE_STEPS=2000 \
DATALOADER_NUM_WORKERS=0 DATALOADER_PIN_MEMORY=false \
OUTPUT_DIR=/mnt/raid0/dreamzero_checkpoints/unitree_stack_blocks_wan22_320x176_v0.1_0616_50eps_run1 \
NUM_GPUS=8 \
bash scripts/train/unitree_stack_blocks_train_wan22_320x176_v0-1.sh


tensorboard --logdir /mnt/raid0/dreamzero_checkpoints/unitree_stack_blocks_wan22_320x176_v0.1_0616_50eps_run1/tensorboard --port 6006 --bind_all

六卡 
tmux new -s dz_run2

REPORT_TO=wandb \
MAX_STEPS=50000 \
SAVE_STRATEGY=steps SAVE_STEPS=2000 \
DATALOADER_NUM_WORKERS=0 DATALOADER_PIN_MEMORY=false \
OUTPUT_DIR=/mnt/raid0/dreamzero_checkpoints/unitree_stack_blocks_wan22_320x176_v0.1_0616_50eps_run2 \
NUM_GPUS=6 \
bash scripts/train/unitree_stack_blocks_train_wan22_320x176_v0-1.sh


dataset格式 7d state_base_pose + 29d robot_joint_q +12d ee_pos +12d hand_q 

