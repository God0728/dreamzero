# Unitree Full-Body Inference Server

这个目录提供一个 Unitree 全身任务模型的常驻推理服务和一个数据集重放可视化客户端。
代码不绑定 sweep-floor 任务；sweep-floor 只是在当前仓库里已有的一份示例数据/ checkpoint。

## 结构

```text
server.py
  常驻 websocket 推理服务，加载 DreamZero checkpoint，接收图像/state/prompt，返回 48-step 的 60D action chunk。

client_viser.py
  LeRobot/GEAR 数据集重放客户端，发送 color_0/color_2/color_3 和低维 state，
  使用 viser + scripts/inference/assets 里的 URDF 显示预测关节动作、输入图像和预测图像。

client_real_robot.py
  真机推理客户端。从机器人读取三路相机 + 60D state，双缓冲预取下一段 chunk，
  控制线程按固定频率平滑下发完整 36D robot_q（base7 + 29 关节）+ 12 手部命令（不阻塞于推理延迟）。
```

## Server

启动：

```bash
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
```

`--return-video` 会额外返回 decoded predicted video，方便可视化；实时控制时可以去掉以降低通信量。

## Client

数据集重放 + 可视化：

```bash
.venv/bin/python scripts/inference/unitree_full_body/client_viser.py \
  --host 127.0.0.1 \
  --port 8000 \
  --dataset-path /mnt/unitree_cpfs/ruixuan/datasets/gear_format/data_sweep_floor_lerobot_gear_100eps \
  --episode-id 0 \
  --stride 48
```

如果要放大 Viser 里的 GT/PRED 视频窗口，改命令行参数：

```bash
--image-width 0.9
```

对应代码默认值在 `client_viser.py` 里的：

```python
DEFAULT_IMAGE_WIDTH = 0.72
```

默认是交互式播放：

```text
1. 打开 http://localhost:8081
2. 点击 Start / Resume (s)，或在启动 client 的终端按 `s`：开始顺序推理并播放
3. client 会按 stride=48 依次推理 chunk，并把 action/video 对齐缓存到同一个时间轴
4. 播放时 action 按 30Hz，图像按 5Hz 更新
5. 点击 Pause (space)，或在终端按空格：暂停推理和播放
6. 拖动 Timeline frame：回放已经推理并缓存的动作和 GT/PRED 图像
7. 再按 `s` 或 Start / Resume：从当前时间轴位置继续顺序播放；需要新 chunk 时会继续推理
```

完整 episode 推理完成后，Viser 程序不会退出；完整时间轴仍可拖拽回放。

时间轴上会保留 chunk/cache 事件提示：

```text
红色边框：旧 chunk 结束，新 chunk 开始
绿色边框：新 chunk 开始时触发 current_start_frame >= local_attn_size，模型清 KV/cross-attn cache
```

边框会套住上排 GT 三个窗口和下排 PRED 三个窗口。默认按当前训练配置估计：

```bash
--local-attn-size 9      # max_chunk_size=4, num_frame_per_block=2 -> 4*2+1
--num-frame-per-block 2
```

如果模型配置变了，需要同步改这两个参数，才能让绿色 cache-reset 标记和模型内部 reset 对齐。

如果想连续自动跑完整 episode：

```bash
.venv/bin/python scripts/inference/unitree_full_body/client_viser.py \
  --host 127.0.0.1 \
  --port 8000 \
  --dataset-path /mnt/unitree_cpfs/ruixuan/datasets/gear_format/data_sweep_floor_lerobot_gear_100eps \
  --episode-id 0 \
  --stride 48 \
  --auto-play
```

上面的 dataset path 是一个示例数据集。当前 Viser client 兼容两种 state/action layout：

```text
53D: robot_q[:29] + hand[:12] + ee_state[:12]
60D: robot_q[:36] + hand[:12] + ee_state[:12]
```

如果 server metadata 里没有正确的 `action_dim`，可以显式传：

```bash
--control-dim 53
# 或
--control-dim 60
```

viser 默认开在：

```text
http://localhost:8081
```

如果没有安装 viser，可以加：

```bash
--no-viser
```

## Websocket 消息协议

连接后 server 会先发送 metadata。客户端 infer 请求使用 msgpack-numpy：

```python
{
    "endpoint": "infer",
    "color_0": np.ndarray(H, W, 3),  # head camera, RGB uint8
    "color_2": np.ndarray(H, W, 3),  # left wrist, RGB uint8
    "color_3": np.ndarray(H, W, 3),  # right wrist, RGB uint8
    "state": np.ndarray(60,),        # robot_q_current[:36] + hand_state[:12] + ee_state[:12]
    "prompt": "<task instruction>",
}
```

`state` 规范格式是 60D（与训练 `dataset/meta/modality.json` 的 `end=60` 一致）。
也接受 53D 的去 base 版本（29 关节 + 12 手 + 12 ee），server 会在前面补 7 维零
还原成 60D（base 用相对动作，不参与真机执行）。

也可以用模型 key：

```text
video.head_stereo_left
video.wrist_left
video.wrist_right
state.sweep_floor_control
```

返回：

```python
{
    "ok": True,
    "action": np.ndarray(48, 60),          # 完整 60D = robot_q_desired[:36] + hand_cmd[:12] + ee_state[:12]
    "action.sweep_floor_control": np.ndarray(48, 60),
    "robot_q_desired": np.ndarray(48, 36),  # base_pose(7) + 29 关节
    "robot_joint_cmd": np.ndarray(48, 29),  # 去 base 的 29 actuated 关节（真机直接执行）
    "hand_cmd": np.ndarray(48, 12),         # 左手 6 + 右手 6（真机直接执行）
    "ee_state": np.ndarray(48, 12),         # 仅信息用，不执行
    "control": np.ndarray(48, 53),          # 去 base 的 53D = 29 关节 + 12 手 + 12 ee
    "pred_video": np.ndarray(T, H, W, 3),   # 只有 --return-video 时存在
}
```

返回的 `action` 已经是 `GrootSimPolicy.unapply()` 之后的动作：先从 q0-q99 统计量反归一化；
如果 checkpoint 的训练配置启用了 relative action，`unapply()` 会用请求里的当前 `state`
把 relative delta 加回绝对量。因此 client/viser 端不要再手动加 state，直接把
`robot_q_desired` 解释为绝对 root pose + body 关节命令，把 `hand_cmd` 解释为反归一化后的手部命令。

模型返回的 60D `action` 规范布局（与 `datasetinfo.md` / `dataset/meta/modality.json` 一致）：

```text
[0:36]   robot_q_desired = base_pose(7) + 29 关节
  [0:3]    root xyz
  [3:7]    root wxyz
  [7:36]   29 actuated body 关节 = 腿(12) + 腰(3) + 双臂(14)
[36:48]  hand_cmd = 左手(6) + 右手(6)
[48:60]  ee_state = 左 xyzrpy(6) + 右 xyzrpy(6)（仅信息用）
```

真机执行：完整 `robot_q_desired[0:36]`（base7 + 29 关节）整体喂给 WBC `SecureMotionInferencer`（与遥操喂 GMR raw_qpos 一致，base 不能用固定值）+ `hand_cmd`（12）下发手部；`ee_state`（12）仅信息用、不执行。

Viser 可视化是另一回事：URDF 只画出 body 前 22 个 actuated DOF，所以 viser 内部只取
`robot_q_desired[7:29]` 画到 URDF，剩余关节保持默认值。这是可视化选择，不是动作维度的真实划分。

## Viser 可视化映射

`client_viser.py` 会把每次推理返回的 48 步 action 按 30Hz 连续可视化到
`scripts/inference/assets/g1_29dof_mode_15_brainco_hand.urdf`：

```text
robot_q_desired[0:3]  -> /unitree_g1 root position
robot_q_desired[3:7]  -> /unitree_g1 root orientation, wxyz
robot_q_desired[7:29] -> 机器人 body 前 22 个 actuated DOF
hand_cmd[0:6]         -> 左 BrainCo hand 的 16DOF URDF 中 [0, 1, 4, 7, 10, 13]
hand_cmd[6:12]        -> 右 BrainCo hand 的 16DOF URDF 中 [0, 1, 4, 7, 10, 13]
```

Viser scene 会创建 `/floor` 地平面网格；`/unitree_g1` 是 URDF root frame，会跟随
模型输出的 `xyz+wxyz` 移动和旋转。模型没有预测的 body 后 7 个 DOF 不写入 URDF，
保持默认值。

viser/yourdfpy 的 `update_cfg()` 使用 51 个 actuated joints；URDF 里的 mimic joints
会自动跟随，不要直接写进 cfg。actuated joint 顺序中，body 22D 映射为：

```text
[0:22]
```

左手 6D 映射到全局 actuated joint index：

```text
[22, 23, 25, 27, 29, 31]
```

右手 6D 映射到全局 actuated joint index：

```text
[40, 41, 43, 45, 47, 49]
```

viser web 端会显示并随 playback 更新：

```text
上排 GT:   数据集真值 color_0 / color_2 / color_3
下排 PRED: 模型预测 head / right_wrist / left_wrist
```

预测图像只有 server 启动时加 `--return-video` 才会返回。

## 真实机器人客户端接入

直接用 `client_real_robot.py`（双缓冲预取，推理延迟不阻塞控制）：

```bash
# 先用 dummy 机器人验证整条链路（无硬件）：
.venv/bin/python scripts/inference/unitree_full_body/client_real_robot.py \
  --host 127.0.0.1 --port 8000 \
  --prompt "stack the blocks" \
  --robot dummy \
  --control-hz 30 \
  --action-horizon 48 \
  --replan-stride 24
```

接入真机直接用 `--robot unitree_g1`。`UnitreeG1RobotInterface` 已对接遥操采集栈
`/home/unitree/jimmy/wbc_pico_record`（即录制 60D 训练数据的同一套代码），并通过
`unitree_sdk2py` DDS 驱动机器人：

```bash
.venv/bin/python scripts/inference/unitree_full_body/client_real_robot.py \
  --host 127.0.0.1 --port 8000 \
  --prompt "stack the blocks" \
  --robot unitree_g1 \
  --net-interface enp5s0 \
  --eef brainco \
  --image-server-address 192.168.123.164 \
  --wbc-repo /home/unitree/jimmy/wbc_pico_record \
  --control-hz 30 --action-horizon 48 --replan-stride 24
```

数据流（与录制脚本一一对应）：

```text
相机   : ZMQ ImageClient -> color_0(头部左目) / color_2(左腕) / color_3(右腕)
state  : rt/lowstate LowState(29 关节 + IMU) + 手部状态 + FK 末端 -> 60D
机器人 out: robot_q_desired[0:36](base7+关节29) -> WBC SecureMotionInferencer(zero-rotation 校准) -> rt/fsm/teleop/cmd(FSM 504)
手   out: hand_cmd(12) -> 手控制器 set_hand_targets
```

要点：
* 29 关节顺序 = 腿(12) + 腰(3) + 左臂(7) + 右臂(7)，与 `datasetinfo.md` robot_q[7:36] 一致。
* base(7) 用固定站立位姿喂给 WBC（按部署约定只执行 29 关节），ee_state(12) 不执行。
* IMU 四元数按"相对第一帧"变换（与录制 `_apply_recording_transform_to_quat` 一致），匹配训练分布。
* WBC 推理线程按 `--wbc-rate-hz`(默认 60Hz) 跟踪最新 setpoint，与控制频率解耦。
* `reset()` 切到 FSM 504 并启动 WBC 线程；`close()` 切回 FSM 801 并松手。

如需接入其它机器人，实现 `RobotInterface` 三个方法即可，保持发给 server 的字段一致：

```text
color_0 -> 头部相机左目
color_2 -> 腕部左侧
color_3 -> 腕部右侧
state   -> robot_q_current[:36] + hand_state[:12] + ee_state[:12]   # 60D
```

* `read_observation()`：返回三路 RGB 帧 + 60D state。
* `command_joints_hand(joint_cmd_29, hand_cmd_12)`：下发一个 setpoint。

推理与控制解耦：后台推理线程在当前 chunk 还在播放时预取下一段，控制线程按
`--control-hz` 平滑输出；`--replan-stride` 控制每执行多少步后切换到新 chunk。
若新 chunk 未就绪，控制线程保持最后一个 setpoint，不会跳变。

server 内部会维护 `action_horizon + 1` 帧历史，并按 `video_stride=6` 取窗口：

```text
[t-48, t-42, ..., t]
```

开头历史不足时会用第一帧补齐。


# system.set_fsm(504) #进入遥操模式   system.set_fsm(801)进入走跑模式


raw_qpos: GMR 重映射得到的 qpos (36维: pos(3) + quat(4) + joints(29))