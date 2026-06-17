#!/usr/bin/env python3
"""将 Unitree stack_blocks(mcap 原始日志)转换为 LeRobot-v2/GEAR 格式。

原始数据布局(实测):
  raw_root/
    2026_06_08_robot_7297/                       # 一个或多个批次目录
      episode_0000.mcap                          # 原始日志(含图像 + 状态/动作)
      episode_0000_recomputed_ee_fullbody.json   # 重算 EE(本脚本不使用)
      ...
    7297/                                        # 相机标定(忽略)
    labels.json                                  # 审核状态索引(忽略)

每个 mcap 仅 2 个 JSON topic:
  /episode/meta      -> text.goal(任务文本)、info.image(640x480@30fps)
  /whole_body/frame  -> 每帧 colors.color_0..3(640x480 base64-jpeg)+ states + actions

与 sweep_floor 完全同构,沿用单一 60-D 控制向量(state 与 action 共用一个 key):
  state.<control>  = robot_q_current[:36] + hand_state[:12] + ee_state[:12]
  action.<control> = robot_q_desired[:36] + hand_cmd[:12]   + ee_state[:12]

匹配的 state/action key 让 DreamZero 用标准的 action-minus-state 逻辑计算相对动作。

数据契约详见 docs/STACK_BLOCKS_CONVERSION_CHECKLIST.md。
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shutil
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from mcap.reader import make_reader


# 3 路相机:与 sweep_floor 对齐(弃用头部立体右目 color_1 以保证正向迁移)
VIDEO_KEYS = {
    "head_stereo_left": "observation.images.color_0",
    "wrist_left": "observation.images.color_2",
    "wrist_right": "observation.images.color_3",
}
# 新列名 -> mcap colors 中的原始相机 key
RAW_CAMERA_KEYS = {
    "observation.images.color_0": "color_0",
    "observation.images.color_2": "color_2",
    "observation.images.color_3": "color_3",
}

ROBOT_Q_DIM = 36
HAND_DIM = 12
EE_DIM = 12
CONTROL_DIM = ROBOT_Q_DIM + HAND_DIM + EE_DIM  # 60

META_TOPIC = "/episode/meta"
FRAME_TOPIC = "/whole_body/frame"


def _configure_imageio_ffmpeg() -> None:
    """系统无 ffmpeg 时,回退到 imageio-ffmpeg 自带的二进制。"""

    if os.environ.get("IMAGEIO_FFMPEG_EXE"):
        return
    try:
        import imageio_ffmpeg

        os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass


def _as_vector(value: Any, *, name: str, dim: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < dim:
        raise ValueError(f"{name} 维度为 {arr.size},至少需要 {dim}")
    return arr[:dim].astype(np.float32)


def _state_vector(frame: dict[str, Any]) -> np.ndarray:
    states = frame["states"]
    return np.concatenate(
        [
            _as_vector(states["robot_q_current"], name="states.robot_q_current", dim=ROBOT_Q_DIM),
            _as_vector(states["hand_state"], name="states.hand_state", dim=HAND_DIM),
            _as_vector(states["ee_state"], name="states.ee_state", dim=EE_DIM),
        ],
        axis=0,
    ).astype(np.float32)


def _action_vector(frame: dict[str, Any]) -> np.ndarray:
    actions = frame["actions"]
    return np.concatenate(
        [
            _as_vector(actions["robot_q_desired"], name="actions.robot_q_desired", dim=ROBOT_Q_DIM),
            _as_vector(actions["hand_cmd"], name="actions.hand_cmd", dim=HAND_DIM),
            _as_vector(actions["ee_state"], name="actions.ee_state", dim=EE_DIM),
        ],
        axis=0,
    ).astype(np.float32)


def _episode_sort_key(path: Path) -> tuple[str, int]:
    """按 (批次目录名, episode 序号) 排序,保证跨批次稳定有序。"""
    try:
        idx = int(path.stem.split("_")[-1])
    except ValueError:
        idx = 10**12
    return (path.parent.name, idx)


def _find_episode_mcaps(raw_root: Path) -> list[Path]:
    """递归查找所有 episode_*.mcap(任意批次子目录下)。"""
    return sorted(raw_root.glob("*/episode_*.mcap"), key=_episode_sort_key)


def _decode_jpeg(color_entry: dict[str, Any]) -> np.ndarray:
    raw = base64.b64decode(color_entry["data"])
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))


def _read_mcap_episode(mcap_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """读取单个 mcap,返回 (meta, frames)。

    frames 按帧 idx 升序排序;每帧含 idx / states / actions / colors(原始 base64 jpeg)。
    图像以压缩 jpeg 字节保留(每帧 ~50KB),解码推迟到写视频时,避免占用大量内存。
    """
    meta: dict[str, Any] = {}
    frames: list[dict[str, Any]] = []
    with mcap_path.open("rb") as f:
        reader = make_reader(f)
        for _schema, channel, message in reader.iter_messages():
            obj = json.loads(message.data)
            if channel.topic == META_TOPIC:
                meta = obj
            elif channel.topic == FRAME_TOPIC:
                frames.append(obj)
    if not frames:
        raise RuntimeError(f"{mcap_path} 中未找到 {FRAME_TOPIC} 消息")
    frames.sort(key=lambda fr: int(fr.get("idx", 0)))
    return meta, frames


def _task_text(meta: dict[str, Any], default: str) -> str:
    text = meta.get("text") or {}
    goal = str(text.get("goal") or "").strip()
    desc = str(text.get("desc") or "").strip()
    placeholder = {"task description", "none", "null", ""}
    if goal and desc and desc.lower() not in placeholder:
        return f"{goal}. {desc}"
    return goal or desc or default


def _write_videos(
    *,
    frames: list[dict[str, Any]],
    video_dir: Path,
    new_ep_idx: int,
    fps: int,
) -> tuple[int, int]:
    """为 3 路相机各写一支 mp4,逐帧追加。返回 (width, height)。"""
    writers = {}
    for original_key in RAW_CAMERA_KEYS:
        out_path = video_dir / f"episode_{new_ep_idx:06d}_{original_key}.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            writers[original_key] = imageio.get_writer(
                out_path, fps=fps, codec="libx264", macro_block_size=16
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "imageio 找不到 ffmpeg。请在运行环境安装系统 ffmpeg,"
                "或安装 Python 包 imageio-ffmpeg。"
            ) from exc

    width = height = None
    try:
        for frame in frames:
            colors = frame["colors"]
            for original_key, raw_camera_key in RAW_CAMERA_KEYS.items():
                img = _decode_jpeg(colors[raw_camera_key])
                height, width = int(img.shape[0]), int(img.shape[1])
                writers[original_key].append_data(img)
    finally:
        for writer in writers.values():
            writer.close()
    if width is None or height is None:
        raise ValueError(f"episode_{new_ep_idx:06d} 未写入任何视频帧")
    return width, height


def _stats(arrays: list[np.ndarray]) -> dict[str, list[float]]:
    data = np.concatenate(arrays, axis=0).astype(np.float64)
    return {
        "mean": np.mean(data, axis=0).tolist(),
        "std": np.std(data, axis=0).tolist(),
        "min": np.min(data, axis=0).tolist(),
        "max": np.max(data, axis=0).tolist(),
        "q01": np.quantile(data, 0.01, axis=0).tolist(),
        "q99": np.quantile(data, 0.99, axis=0).tolist(),
    }


def _relative_stats(
    state_arrays: list[np.ndarray],
    action_arrays: list[np.ndarray],
    *,
    action_horizon: int,
    control_key: str,
) -> dict[str, dict[str, list[float]]]:
    all_relative: list[np.ndarray] = []
    for states, actions in zip(state_arrays, action_arrays):
        usable = len(actions) - (int(action_horizon) - 1)
        for anchor in range(max(usable, 0)):
            ref_state = states[anchor]
            all_relative.append(actions[anchor : anchor + int(action_horizon)] - ref_state)
    if not all_relative:
        raise ValueError("无相对动作样本;请检查 episode 长度 / action_horizon")
    rel = np.concatenate(all_relative, axis=0).astype(np.float64)
    return {control_key: _stats([rel])}


def convert(args: argparse.Namespace) -> None:
    _configure_imageio_ffmpeg()
    control_key = args.control_key
    state_column = f"observation.state.{control_key}"
    action_column = f"action.{control_key}"

    raw_root = Path(args.raw_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not raw_root.exists():
        raise FileNotFoundError(raw_root)
    if output_root.exists() and args.force:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    mcap_paths = _find_episode_mcaps(raw_root)[: int(args.max_episodes)]
    if not mcap_paths:
        raise RuntimeError(f"{raw_root} 下未找到 */episode_*.mcap")

    data_dir = output_root / "data" / "chunk-000"
    video_dir = output_root / "videos" / "chunk-000"
    meta_dir = output_root / "meta"
    for d in (data_dir, video_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    state_arrays: list[np.ndarray] = []
    action_arrays: list[np.ndarray] = []
    tasks: dict[str, int] = {}
    episodes_meta: list[dict[str, Any]] = []
    video_width = int(args.width)
    video_height = int(args.height)

    for new_ep_idx, mcap_path in enumerate(tqdm(mcap_paths, desc="转换 episodes")):
        meta, frames = _read_mcap_episode(mcap_path)
        task = _task_text(meta, args.default_task)
        task_idx = tasks.setdefault(task, len(tasks))

        rows = []
        ep_states = []
        ep_actions = []
        for frame_idx, frame in enumerate(frames):
            state = _state_vector(frame)
            action = _action_vector(frame)
            ep_states.append(state)
            ep_actions.append(action)
            rows.append(
                {
                    "episode_index": new_ep_idx,
                    "frame_index": frame_idx,
                    "timestamp": float(frame_idx) / float(args.fps),
                    "next.done": frame_idx == len(frames) - 1,
                    "index": frame_idx,
                    "task_index": task_idx,
                    "annotation.task": task,
                    state_column: state,
                    action_column: action,
                }
            )

        pd.DataFrame(rows).to_parquet(data_dir / f"episode_{new_ep_idx:06d}.parquet")
        state_arrays.append(np.stack(ep_states, axis=0))
        action_arrays.append(np.stack(ep_actions, axis=0))
        episodes_meta.append(
            {"episode_index": new_ep_idx, "tasks": [task], "length": len(rows)}
        )

        video_width, video_height = _write_videos(
            frames=frames,
            video_dir=video_dir,
            new_ep_idx=new_ep_idx,
            fps=int(args.fps),
        )

    total_frames = int(sum(len(x) for x in state_arrays))

    info = {
        "codebase_version": "v2.0",
        "robot_type": "unitree_g1_upper_body",
        "total_episodes": len(mcap_paths),
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_videos": len(mcap_paths) * len(RAW_CAMERA_KEYS),
        "chunks_size": 1000,
        "fps": int(args.fps),
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/episode_{episode_index:06d}_{video_key}.mp4",
        "features": {
            state_column: {
                "dtype": "float32",
                "shape": [CONTROL_DIM],
                "names": [f"{control_key}_{i}" for i in range(CONTROL_DIM)],
            },
            action_column: {
                "dtype": "float32",
                "shape": [CONTROL_DIM],
                "names": [f"{control_key}_{i}" for i in range(CONTROL_DIM)],
            },
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
            "annotation.task": {"dtype": "string", "shape": [1]},
        },
    }
    for original_key in RAW_CAMERA_KEYS:
        info["features"][original_key] = {
            "dtype": "video",
            "shape": [video_height, video_width, 3],
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": int(args.fps),
                "video.height": video_height,
                "video.width": video_width,
                "video.channels": 3,
            },
        }

    modality = {
        "state": {
            control_key: {
                "original_key": state_column,
                "start": 0,
                "end": CONTROL_DIM,
                "rotation_type": None,
                "absolute": True,
                "dtype": "float32",
                "range": None,
            }
        },
        "action": {
            control_key: {
                "original_key": action_column,
                "start": 0,
                "end": CONTROL_DIM,
                "rotation_type": None,
                "absolute": True,
                "dtype": "float32",
                "range": None,
            }
        },
        "video": {
            "head_stereo_left": {"original_key": "observation.images.color_0"},
            "wrist_left": {"original_key": "observation.images.color_2"},
            "wrist_right": {"original_key": "observation.images.color_3"},
        },
        "annotation": {
            "task_index": {"original_key": "task_index"},
            "task": {"original_key": "annotation.task"},
        },
    }

    stats = {
        state_column: _stats(state_arrays),
        action_column: _stats(action_arrays),
    }
    relative_stats = _relative_stats(
        state_arrays,
        action_arrays,
        action_horizon=int(args.action_horizon),
        control_key=control_key,
    )

    (meta_dir / "info.json").write_text(json.dumps(info, indent=4))
    (meta_dir / "modality.json").write_text(json.dumps(modality, indent=4))
    (meta_dir / "embodiment.json").write_text(
        json.dumps(
            {"robot_type": "unitree_g1_upper_body", "embodiment_tag": "unitree_g1_upper_body"},
            indent=4,
        )
    )
    (meta_dir / "stats.json").write_text(json.dumps(stats, indent=4))
    (meta_dir / "relative_stats_dreamzero.json").write_text(json.dumps(relative_stats, indent=4))

    with (meta_dir / "tasks.jsonl").open("w") as f:
        for task, idx in sorted(tasks.items(), key=lambda item: item[1]):
            f.write(json.dumps({"task_index": idx, "task": task}) + "\n")
    with (meta_dir / "episodes.jsonl").open("w") as f:
        for episode in episodes_meta:
            f.write(json.dumps(episode) + "\n")

    print(f"已转换 {len(mcap_paths)} 个 episode({total_frames} 帧)-> {output_root}")
    print(f"state/action key: {control_key}, dim={CONTROL_DIM}")
    print(f"视频: {len(RAW_CAMERA_KEYS)} 路 @ {video_width}x{video_height}, fps={args.fps}")
    print("相对动作统计: meta/relative_stats_dreamzero.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--raw-root", default="/mnt/raid0/unitree_brainco/stack_blocks")
    parser.add_argument(
        "--output-root",
        default="/mnt/raid0/unitree_brainco/gear_format/stack_blocks_lerobot_gear_50eps",
    )
    parser.add_argument("--max-episodes", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--action-horizon", type=int, default=48)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    # 默认沿用 sweep_floor 的 key,以零改动复用现有 Hydra 配置与 embodiment。
    # 若改名,务必同步修改对应的 data/dreamzero/*.yaml 中所有 *_control key。
    parser.add_argument("--control-key", default="sweep_floor_control")
    parser.add_argument("--default-task", default="stack the blocks")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
