再对齐一下，验证方案：
验证一： 100条真机全身数据， DreamZero，单任务验证；
验证二： 100条真机全身数据 + 100条人体全身ego数据， DreamZero，单任务验证全身ego数据的加成；
验证三： 100条真机全身数据 + 100条人体全身ego数据 + 100条人体第三视角全身数据， DreamZero，单任务验证第三视角加成效果；


conda deactivate 

source ~/.bashrc


cd /mnt/unitree_cpfs/ruixuan/code/wm/dreamzero

conda deactivate 
source .venv/bin/activate


# 问题
Initialized dataset data_sweep_floor_lerobot_gear_100eps with EmbodimentTag.UNITREE_G1_UPPER_BODY

EmbodimentTag 需要更换

# 装库
uv pip install \
  --python /mnt/unitree_cpfs/ruixuan/code/wm/dreamzero/.venv \
  "numpy==1.26.4" \
  --link-mode copy

uv pip install \
  --python /mnt/unitree_cpfs/ruixuan/code/wm/dreamzero/.venv \
  "viser>=0.2.0,<1.0" \
  --link-mode copy

## 启动推理server client
MASTER_PORT=29617 .venv/bin/python scripts/inference/unitree_full_body/server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --model-path checkpoints/unitree_sweep_floor_wan22_352x640_gbs8_v0.2/checkpoint-32000 \
  --device cuda:0 \
  --prompt "pick up an apple" \
  --action-horizon 48 \
  --video-stride 6 \
  --eval-mode causal_gt \
  --return-video

export NO_ALBUMENTATIONS_UPDATE=1 
.venv/bin/python scripts/inference/unitree_full_body/client_viser.py 

# 转换数据集
python3 scripts/data/convert_sweep_floor_to_lerobot_gear.py --force

输出到 output-root
default="/mnt/unitree_cpfs/ruixuan/datasets/gear_format/data_sweep_floor_lerobot_gear_100eps",

# 编码文本 token
bash scripts/train/precompute_T5_cache_unitree_sweep_floor.sh

编码后token存储在：
```
(.venv) root@unitree-GM403-16:/mnt/unitree_cpfs/ruixuan/code/wm/dreamzero# find /mnt/unitree_cpfs/ruixuan/cache/dreamzero/t5_cache/unitree_sweep_floor_100eps -type f
/mnt/unitree_cpfs/ruixuan/cache/dreamzero/t5_cache/unitree_sweep_floor_100eps/95f9063b7499449b55ab0bed61999480bce44658928e3666a329070531f005e1.t5_len512.dreamzero_wan_t5.pt
```

# 训练开始
bash scripts/train/unitree_sweep_floor_train_wan22_352x640_wb_v0-1.sh

# 验证
MASTER_PORT=29617 python scripts/eval_unitree_sweep_floor_episode_stream.py \
  --model-path checkpoints/unitree_sweep_floor_wan22_352x640_gbs8_v0.2/checkpoint-8000 \
  --dataset-path /mnt/unitree_cpfs/ruixuan/datasets/gear_format/data_sweep_floor_lerobot_gear_100eps \
  --output-dir /data0/ruixuan/evals/unitree_sweep_floor/checkpoint-8000_ep0 \
  --episode-ids 0 \
  --device cuda:0 \
  --action-horizon 48 \
  --video-stride 6 \
  --video-fps 5


# 导出日志
tmux capture-pane -t dz -p -S -200 > ./tmp/tmux_last200.log

# 可视化log
/mnt/unitree_cpfs/ruixuan/envs/tb_envs/tb_env/bin/tensorboard \
  --logdir_spec sweep_floor_gbs8:/mnt/unitree_cpfs/ruixuan/code/wm/dreamzero/checkpoints/unitree_sweep_floor_wan22_352x640_gbs8_v0.1/tensorboard \
  --host 0.0.0.0 \
  --port 6006

## 多曲线可视化
/mnt/unitree_cpfs/ruixuan/envs/tb_envs/tb_env/bin/tensorboard \
  --logdir_spec "sweep_v0.1:/mnt/unitree_cpfs/ruixuan/code/wm/dreamzero/checkpoints/unitree_sweep_floor_wan22_352x640_gbs8_v0.1/tensorboard,sweep_v0.2:/mnt/unitree_cpfs/ruixuan/code/wm/dreamzero/checkpoints/unitree_sweep_floor_wan22_352x640_gbs8_v0.2/tensorboard" \
  --host 0.0.0.0 \
  --port 6006

或者方便选择曲线命名：

RUNS=(
  "sweep_v0.1:/mnt/unitree_cpfs/ruixuan/code/wm/dreamzero/checkpoints/unitree_sweep_floor_wan22_352x640_gbs8_v0.1/tensorboard"
  "sweep_v0.2:/mnt/unitree_cpfs/ruixuan/code/wm/dreamzero/checkpoints/unitree_sweep_floor_wan22_352x640_gbs8_v0.2/tensorboard"
)
LOGDIR_SPEC=$(IFS=, ; echo "${RUNS[*]}")
/mnt/unitree_cpfs/ruixuan/envs/tb_envs/tb_env/bin/tensorboard \
  --logdir_spec "$LOGDIR_SPEC" \
  --host 0.0.0.0 \
  --port 6006


# 参数调整
  weight_decay=1e-2 \ <----- 1e-5

- v0.0 (default): input=160x160, lr=1e-5, weight decay=1e-5, adam_beta1 = 0.95, adam_beta2 = 0.999
- v0.1: input=224x224, lr=1e-5, weight decay=1e-2, adam_beta1 = 0.9, adam_beta2 = 0.95
- v0.2: input=224x224, lr=1e-4, weight decay=1e-2, adam_beta1 = 0.9, adam_beta2 = 0.95

v0.2, gbs=8,steps=40k 77.5%    steps=100k 94%


# 挂载checkpoints evals
mkdir -p /mnt/unitree_cpfs/ruixuan/code/wm/dreamzero/checkpoints
mkdir -p /data0/ruixuan/dreamzero/checkpoints
mount --bind \
  /data0/ruixuan/dreamzero/checkpoints \
  /mnt/unitree_cpfs/ruixuan/code/wm/dreamzero/checkpoints


mount --bind \
  /data0/ruixuan/evals \
  /mnt/unitree_cpfs/ruixuan/code/wm/dreamzero/evals

## Action Dim 设计
53dim: 29 + 12(hand qpos) + 12 (ee_state)
ee_state 是否转为 rot6d


## 临时tensorboard安装
在ryan-jumpboard执行：

uv venv /mnt/unitree_cpfs/ruixuan/envs/tb_envs/tb_env \
  --python 3.11.11 \
  --seed \
  --relocatable \
  --link-mode copy

ls -l /mnt/unitree_cpfs/ruixuan/envs/tb_envs/tb_env/bin/python

uv pip install \
  --python /mnt/unitree_cpfs/ruixuan/envs/tb_envs/tb_env/bin/python \
  "setuptools<81" \
  tensorboard \
  --link-mode copy

/mnt/unitree_cpfs/ruixuan/envs/tb_envs/tb_env/bin/tensorboard \
  --logdir /mnt/unitree_cpfs/ruixuan/code/wm/dreamzero/checkpoints/unitree_sweep_floor_wan22_352x640_gbs8_v0.1 \
  --host 0.0.0.0 \
  --port 6006



## 日志
```
Successfully dumped metadata                     
Set global batch size to 1, set gradient accumulation steps to 1                                  
Current global step: 0                           
Creating custom train dataloader                                                                                                                                                                     
train dataloader length: 1048576000                                                                                                                                                                  
eval dataloader length: no eval dataloader                                                                                                                                                           
train dataset length: 1048576000                                                                                                                                                                     
GPU memory before training: 1.430511474609375e-06 GB                                                                                                                                                 
No initial actions to dump                                                                                                                                                                           
Current global step: 0                                                                                                                                                                               
Creating custom train dataloader                                                                                                                                                                     
  0%|                                                                                                                                                 | 0/50000 [00:00<?, ?it/s]Caching shard        
Cached shard in 33.36 seconds                                                                     
Rank 0, Worker 0: Wait for shard 7 in dataset 0 in 33.36 seconds                                                                                                                                     


videos torch.Size([1, 3, 33, 352, 640])


Loading relative action stats from /mnt/unitree_cpfs/ruixuan/datasets/gear_format/data_sweep_floor_lerobot_gear_100eps/meta/relative_stats_dreamzero.json                                                                                                                                              
relative_action_per_horizon False                                                                                                                                                                                                                                                                      
Using relative stats for sweep_floor_control: {'max': array([1.63403201, 0.795084  , 0.08231503, 0.42469031, 0.201628  ,                                                                                                                                                                               
       0.14576781, 0.8319416 , 0.78795153, 0.6835584 , 0.85410845,                                                                                                                                                                                                                                     
       1.08325815, 0.5663892 , 0.31838897, 0.77288854, 0.4699114 ,                                                                                                                                                                                                                                     
       0.86081707, 1.12822211, 0.55214286, 0.36093938, 0.53687614,                                                                                                                                                                                                                                     
       0.50534707, 0.38557115, 1.39537632, 0.84506404, 1.22259796,                                                                                                                                                                                                                                     
       1.38696516, 1.05580175, 0.28319028, 0.90748918, 1.        ,                                                                                                                                                                                                                                     
       0.99699998, 1.        , 1.        , 1.        , 1.        ,                                                                                                                                                                                                                                     
       1.        , 1.        , 1.        , 1.        , 1.        ,                                                                                                                                                                                                                                     
       1.        , 0.33200586, 0.40934649, 0.37967139, 1.5323118 ,                                                                                                                                                                                                                                     
       1.69935131, 1.94485319, 0.37803778, 0.43775544, 0.40889999,                                                                                                                                                                                                                                     
       1.9970572 , 1.50097835, 1.71642613]), 'min': array([-2.1228621 , -2.77099657, -0.08462358, -0.3893171 , -0.16549598,                                                                                                                                                                            
       -0.27291125, -0.81655383, -0.72497517, -0.4358061 , -0.8255344 ,                                                                                                                                                                                                                                
       -0.82302517, -0.68556732, -0.34280729, -0.56737268, -0.54282159,                                                                                                                                                                                                                                
       -0.93595111, -0.99047911, -0.55114836, -0.23899522, -0.5961163 ,                                                                                                                                                                                                                                
       -0.47054341, -0.49442059, -1.189852  , -0.68546808, -0.86266279,                                                                                                                                                                                                                                
       -1.04431462, -1.15187871, -0.33032823, -0.7150991 , -0.68800002,                                                                                                                              
       -0.86900002, -0.70499998, -0.91399997, -0.92400002, -0.99900001,                                                                                                                              
       -0.71100003, -0.86400002, -0.85699999, -0.94599998, -0.99699998,                                                                                                                              
       -1.        , -0.29283395, -0.35858193, -0.4219273 , -1.27933717,                                                                                                                              
       -1.38074374, -1.58427739, -0.42638475, -0.54979628, -0.28099674,                                                                                                                              
       -1.46087015, -1.93370283, -1.89118445]), 'mean': array([ 2.66395773e-02,  1.63930607e-02,  1.10294881e-04,  4.39469651e-03,                                                                   
       -4.16608180e-03,  4.40915969e-03, -2.82730344e-03,  7.68377488e-03,                                                                                                                           
        9.49948009e-03,  2.19472341e-03, -7.63353799e-03, -3.17325218e-02,                                                                                                                           
       -1.10206574e-02, -4.54606065e-03, -6.05139800e-02,  6.10479039e-04,                                                                                                                           
        2.55160287e-02,  6.55424529e-03,  8.85566016e-02,  8.40331639e-03,                                                                                                                           
        1.73308689e-03, -3.13605055e-02, -6.96864900e-02,  2.56989602e-02,                                                                                                                           
       -2.12405443e-03, -1.61373772e-02,  8.80497023e-03, -4.53401247e-02,                                                                                                                           
        4.58772642e-03,  3.55986131e-01,  3.92572988e-01,  3.32573154e-01,                                                                                                                           
        1.79965965e-01,  2.38263889e-01,  2.49025871e-01,  3.47836809e-01,                                                                                                                           
        3.72374904e-01,  2.96615485e-01,  1.82085143e-01,  1.26053457e-01,                                                                                                                           
        1.49728631e-01,  7.00765309e-04,  1.60435102e-03,  4.02174948e-02,                                                                                                                           
        4.14563319e-03, -1.49218783e-01, -1.13848101e-02,  5.21145910e-03,                                                                                                                           
        1.78847254e-03,  3.85371145e-02,  3.97611077e-03, -1.56549160e-01,                                                                                                                           
        2.52927541e-02]), 'std': array([0.14522969, 0.07629449, 0.00699464, 0.06477369, 0.02425982,                                                                                                  
       0.03279452, 0.11507718, 0.13335663, 0.07733547, 0.10234425,                                                                                                                                   
       0.12713283, 0.07500044, 0.06180996, 0.12130704, 0.08535275,                                                                                                                                   
       0.09404003, 0.13215843, 0.08991823, 0.06375008, 0.06430603,                                                                                                                                   
       0.06917611, 0.08864289, 0.19706218, 0.09730994, 0.13647869,                                                                                                                                   
       0.14564604, 0.13882733, 0.05356129, 0.09852282, 0.20940055,                                                                                                                                   
       0.24181174, 0.19917757, 0.1767087 , 0.17713403, 0.19374161,                                                                                                                                   
       0.20525235, 0.26109542, 0.20712045, 0.18539528, 0.18711802,                                                                                                                                   
       0.1893826 , 0.0427761 , 0.04499707, 0.05580688, 0.20028914,                                                                                                                                   
       0.21576773, 0.22725512, 0.06165884, 0.06436236, 0.0508367 ,                                                                                                                                   
       0.21062828, 0.21819406, 0.22512148]), 'q01': array([-0.16828724, -0.13725423, -0.02190548, -0.19252006, -0.06848343,                                                                          
       -0.13783645, -0.39104805, -0.34534252, -0.18296108, -0.2639653 ,                                                                                                                              
       -0.34735346, -0.22030423, -0.17391681, -0.32382083, -0.25909253,                                                                                                                              
       -0.30152025, -0.39109398, -0.22753776, -0.08362573, -0.18526866,                                                                                                                              
       -0.18793523, -0.34951387, -0.62539546, -0.21948206, -0.34950433,                                                                                                                              
       -0.44680706, -0.43884002, -0.17153136, -0.29252395,  0.        ,                                                                                                                              
       -0.064     ,  0.        ,  0.        ,  0.        ,  0.        ,                                                                                                                              
        0.        , -0.14300001,  0.        ,  0.        ,  0.        ,                                                                                                                              
        0.        , -0.11034073, -0.11935816, -0.1050947 , -0.55594013,                                                                                                                              
       -0.6715185 , -0.68929718, -0.20731094, -0.22738636, -0.11369401,                                                                                                                              
       -0.57042771, -0.83717843, -0.72028165]), 'q99': array([0.83107534, 0.33091859, 0.02499877, 0.20339934, 0.06918754,                                                                            
       0.07194196, 0.45569753, 0.42153408, 0.24209227, 0.37212828,                                                                                                                                   
       0.55215013, 0.18234337, 0.14492249, 0.3569403 , 0.18604179,                                                                                                                                                           
       0.30431645, 0.54796777, 0.24849984, 0.22654925, 0.19216734,
       0.17724482, 0.17197062, 0.4930493 , 0.31632343, 0.42525169,                                                                                                                                                           
       0.39086368, 0.39539926, 0.11402674, 0.2650697 , 1.        ,                      
       0.98299998, 1.        , 1.        , 1.        , 1.        ,                      
       1.        , 0.99199998, 1.        , 1.        , 1.        ,                                                                                                               
       1.        , 0.12384539, 0.13154981, 0.20827134, 0.63396327,                      
       0.50339675, 0.66237665, 0.17574897, 0.20507545, 0.17557697,                      
       0.73106975, 0.57572079, 0.68632872])}                

```