# DreamZero 真机客户端

这个目录是轻量版真机 client，用在连接 Unitree G1 的本地电脑上。前提是这台电脑已经能正常运行 `wbc_pico_record` 遥操/WBC 栈。

大模型 server 仍然运行在远端或云端推理机上。本 client 只负责读取机器人和相机状态，通过 websocket 请求远端模型推理，然后把返回的动作下发给 WBC 和手部控制器。

## 文件说明

- `client_real_robot.py`: G1 真机控制 client。
- `policy_client.py`: 轻量 websocket client，不依赖 `openpi_client`。
- `msgpack_numpy.py`: 本地 numpy/msgpack 序列化工具，协议与 server 匹配。

## 拷贝位置

推荐把整个目录拷到连接 G1 的本地电脑：

```text
wbc_pico_record/
  dreamzero_client/
    client_real_robot.py
    policy_client.py
    msgpack_numpy.py
```

如果放在 `wbc_pico_record/` 下面，client 会自动把上一级目录识别为 `--wbc-repo`。如果放在别的位置，也可以手动传 `--wbc-repo`。

## Python 依赖

使用原本能跑 `wbc_pico_record` 遥操的 Python 环境即可。如果缺 websocket/msgpack 依赖，再安装：

```bash
pip install websockets msgpack numpy scipy
```

机器人侧依赖，例如 `unitree_sdk2py`、手部控制器、相机 ZMQ、`utils.inference.SecureMotionInferencer`，应该来自已有的 `wbc_pico_record` 环境。

## 启动真机 Client

先在远端推理机启动 DreamZero server。然后在连接 G1 的本地电脑上运行：

```bash
cd /path/to/wbc_pico_record
python dreamzero_client/client_real_robot.py \
  --host 10.0.8.192 \
  --port 8000 \
  --prompt "Stack the blocks in the order of red, green and yellow" \
  --robot unitree_g1 \
  --net-interface enp5s0 \
  --eef brainco \
  --image-server-address 192.168.123.164 \
  --wbc-repo /home/unitree/wbc_pico_record \
  --control-hz 30 \
  --action-horizon 48 \
  --replan-stride 24
```

如果 `dreamzero_client/` 不在 `wbc_pico_record/` 下面，需要显式指定：

```bash
--wbc-repo /path/to/wbc_pico_record
```

## 无硬件网络测试

只测试 websocket 和远端 server，不连接机器人：

```bash
python dreamzero_client/client_real_robot.py \
  --robot dummy \
  --host <SERVER_IP> \
  --port 8000 \
  --prompt "stack the blocks" \
  --max-steps 10
```

## 单 Chunk 调试模式

第一次上真机建议使用手动单 chunk 模式：

```bash
python dreamzero_client/client_real_robot.py \
  --host <SERVER_IP> \
  --port 8000 \
  --prompt "Stack the blocks in the order of red, green and yellow" \
  --robot unitree_g1 \
  --manual-step \
  --net-interface enp5s0 \
  --eef brainco
```

终端按键：

- `Enter` 或 `n`: 推理一个 48-step chunk，并播放这个 chunk。
- `e` 或 `x`: 软件急停，停止 WBC 发布并切回安全 FSM。
- `q`: 正常退出。

## Dry Run 模式

`--dry-run` 不会切 FSM，不会启动 WBC 发布，也不会下发手部命令。它只读取 observation、请求远端模型、执行安全检查。

```bash
python dreamzero_client/client_real_robot.py \
  --host <SERVER_IP> \
  --port 8000 \
  --prompt "Stack the blocks in the order of red, green and yellow" \
  --robot unitree_g1 \
  --manual-step \
  --dry-run
```

## 安全保护

默认开启安全检查：

- 如果第一个 waypoint 距离当前机器人状态太远，直接拒绝并触发软件急停。
- 对 chunk 内每一步的 base xyz、身体关节、手部命令做限幅。
- 如果 base rotation 单步变化太大，则保持上一帧旋转。

常用阈值：

```bash
--max-initial-joint-delta 0.75 \
--max-step-joint-delta 0.12 \
--max-step-base-xyz-delta 0.03
```

这些阈值偏保守，首跑建议先架起机器人空载观察 root pose 和关节命令，再落地。

## G1 本地服务

示例启动顺序：

```bash
ssh g1
cd /home/unitree/jimmy/wbc_pico_record
bash check.sh

# 启动手和图像服务
bash G1_setup.sh
python image_server/image_server.py
```
