# stack_blocks → LeRobot-v2/GEAR 转换数据契约 Checklist

本文件记录 DreamZero(GR00T VLA)训练管线**实际读取数据的代码契约**,用于保证
`scripts/data/convert_stack_blocks_to_lerobot_gear.py` 产出的数据集能被模型直接训练。

所有结论均来自对以下代码的逐行核对(非推测):
- 加载器:`groot/vla/data/dataset/lerobot.py`(`LeRobotSingleDataset`)
- Schema:`groot/vla/data/schema/lerobot.py`、`groot/vla/data/schema/embodiment_tags.py`
- 训练配置:`groot/vla/configs/data/dreamzero/unitree_sweep_floor_relative_wan22.yaml`
- 训练脚本:`scripts/train/unitree_sweep_floor_train_wan22_352x640_wb_v0-1.sh`
- 模板转换器(已验证 94% 成功率):`scripts/data/convert_sweep_floor_to_lerobot_gear.py`

---

## 0. 原始数据(stack_blocks)实测结构

每个 episode 是一对文件:`<batch>/episode_XXXX.mcap` + `episode_XXXX_recomputed_ee_fullbody.json`。
本管线**只用 mcap**(60-D 方案与 sweep_floor 完全同构),不使用 recomputed json。

mcap 仅 2 个 JSON topic:
- `/episode/meta`(1 条):`text.goal` = 任务描述(如 "Stack the blocks in the order of red, green and yellow"),`info.image` = 640×480@30fps。
- `/whole_body/frame`(N 条,本例 1331):每帧含
  - `colors.color_0/1/2/3`:各为 640×480 JPEG RGB,字段 `data` 是 **base64** 编码的 jpeg 字节。
    - color_0 = 头部立体左目,color_1 = 头部立体右目(弃用),color_2 = 左腕,color_3 = 右腕。
  - `states`:`robot_q_current[36]` + `hand_state[12]` + `ee_state[12]`
  - `actions`:`robot_q_desired[36]` + `hand_cmd[12]` + `ee_state[12]`

> 与 sweep_floor 的 `data.json` 帧字段**完全一致**,因此可直接复用其 60-D 方案与训练配置。

---

## 1. 目标目录结构(加载器硬编码,必须逐字匹配)

加载器中的文件名常量(`groot/vla/data/dataset/lerobot.py` L31-L43):

```
<output_root>/
  data/chunk-000/episode_000000.parquet      # data_path: data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet
  videos/chunk-000/episode_000000_observation.images.color_0.mp4
  videos/chunk-000/episode_000000_observation.images.color_2.mp4
  videos/chunk-000/episode_000000_observation.images.color_3.mp4
  meta/info.json
  meta/modality.json
  meta/embodiment.json
  meta/stats.json
  meta/relative_stats_dreamzero.json
  meta/tasks.jsonl
  meta/episodes.jsonl
```

- [ ] `data_path` / `video_path` / `chunks_size` 来自 `info.json`,由加载器
  `_get_data_path_pattern` / `_get_video_path_pattern` / `_get_chunk_size` 直接读取。
- [ ] 视频文件名中的 `{video_key}` = modality.json 里 video 项的 `original_key`
  (即 `observation.images.color_0` 等),见 `get_video_path` L1437。

## 2. parquet 列名契约(`get_state_or_action` L1493 / `get_video` L1441)

加载器读取的列(缺一不可):
- [ ] `observation.state.<CONTROL_KEY>`:整条 60-D 向量(object 列,每行一个 np.float32 数组)。
      加载器做 `np.stack(df[le_key])` 后按 `[:, start:end]` 切片。`le_key` = modality.json
      state 项的 `original_key`。
- [ ] `action.<CONTROL_KEY>`:整条 60-D 向量,同上。
- [ ] `timestamp`:**视频对齐的关键**。`get_video` 用该列做 `get_frames_by_timestamps`,
      因此 parquet 行数 = 视频帧数,且 `timestamp[i] = i / fps`,视频以同样 fps 写入。
- [ ] `frame_index`:用于 `action.task_progress`(可选特性)与调试。
- [ ] `episode_index`、`task_index`、`annotation.task`、`index`、`next.done`:沿用模板写入。

> 默认 `CONTROL_KEY = sweep_floor_control`,以便**零改动复用**现有 Hydra 配置与 embodiment。

## 3. modality.json(`LeRobotModalityMetadata` 校验,schema L100)

- [ ] `state.<CONTROL_KEY>`:`original_key=observation.state.<CONTROL_KEY>`、`start=0`、`end=60`、
      `absolute=true`、`rotation_type=null`、`dtype=float32`。
- [ ] `action.<CONTROL_KEY>`:`original_key=action.<CONTROL_KEY>`、`start=0`、`end=60`、同上。
- [ ] `video`:3 项,key 必须与训练配置 `modality_keys` 一致:
      `head_stereo_left→observation.images.color_0`、`wrist_left→color_2`、`wrist_right→color_3`。
- [ ] `annotation`:`task_index→task_index`、`task→annotation.task`。
- [ ] `_check_integrity`(L1227)会对配置里每个 `modality_key` 调 `get_key_meta`,
      key 不存在即报错。务必让 modality.json 的 key 覆盖训练配置所有 key。

## 4. embodiment.json + EmbodimentTag

- [ ] `{"robot_type": "unitree_g1_upper_body", "embodiment_tag": "unitree_g1_upper_body"}`。
- [ ] `unitree_g1_upper_body` 已在 `EmbodimentTag` 枚举中注册(embodiment_tags.py 末尾)。
      `EmbodimentTag(embodiment_tag)` 不在枚举内会直接抛错。

## 5. stats.json(`_get_lerobot_stats_meta` L408)

- [ ] 以**原始列名**为 key:`observation.state.<CONTROL_KEY>`、`action.<CONTROL_KEY>`。
- [ ] 每个值含 `mean/std/min/max/q01/q99`(`DatasetStatisticalValues` 全部必填)。
- [ ] `use_global_metadata=false` 时从数据集自身 `meta/stats.json` 读取(本管线即此)。
      若文件缺失,加载器会回退到从 parquet 现算,但我们直接写好。

## 6. relative_stats_dreamzero.json(`_get_lerobot_relative_stats_meta` L459)

- [ ] 以 **action 子 key** 为顶层 key:`<CONTROL_KEY>`(即去掉 `action.` 前缀)。
- [ ] 值含 `mean/std/min/max/q01/q99`。
- [ ] 训练配置 `relative_action: true` + `relative_action_keys: [<CONTROL_KEY>]` 时被读取;
      相对动作 = `action[anchor:anchor+H] - state[anchor]`,H = action 的 delta_indices 长度
      (sweep_floor 用 48)。文件存在则直接 load,否则现算并保存到数据集 meta。

## 7. tasks.jsonl / episodes.jsonl

- [ ] `tasks.jsonl`:每行 `{"task_index": i, "task": "<文本>"}`;加载器 `set_index("task_index")`。
- [ ] `episodes.jsonl`:每行 `{"episode_index": i, "tasks": [...], "length": L}`;
      `_get_trajectories`(L1122)读取 `episode_index` 与 `length`,**必须与 parquet 行数一致**。

## 8. 视频编码

- [ ] 每相机一支 mp4,libx264,`macro_block_size=16`。640 与 480 均可被 16 整除,**不触发缩放**。
- [ ] 帧顺序按帧 `idx` 升序;每帧写入 = parquet 一行,保证 `timestamp` 对齐。
- [ ] 训练时由 Hydra transform 再裁剪/缩放到 320×176(见配置),转换阶段保持原生 640×480。

## 9. 训练侧对接(`unitree_sweep_floor_relative_wan22.yaml`)

- [ ] `use_global_metadata: false`、`relative_action: true`、`relative_action_keys: [sweep_floor_control]`。
- [ ] `max_state_dim/max_action_dim: 64`:60-D 在训练时由 padded dataset 自动补零到 64,转换器无需处理。
- [ ] `modality_config_unitree_g1_upper_body` 的 key 与本数据集 modality.json **逐一对应**。
- [ ] 复用方式二选一:
      1. 直接把该配置里的 `sweep_floor_data_root` 指向 stack_blocks 输出目录(CONTROL_KEY 保持默认);
      2. 或复制一份 `unitree_stack_blocks_relative_wan22.yaml` 并改 `*_data_root`(若改了 CONTROL_KEY,
         则配置里所有 `*_control` key 也要同步改名,否则 `_check_integrity` 报错)。

## 10. 训练前自检(沿用训练脚本的校验)

- [ ] `meta/` 下 7 个文件齐全(info/modality/embodiment/stats/relative_stats/tasks/episodes)。
- [ ] `info.json.features["observation.state.<CONTROL_KEY>"].shape == [60]`,action 同。
- [ ] 每 episode:parquet 行数 == 各 mp4 帧数 == `episodes.jsonl` 的 `length`。
- [ ] T5 文本缓存需另跑 `precompute_T5_cache_*`(按 task 文本生成,独立于本转换)。
