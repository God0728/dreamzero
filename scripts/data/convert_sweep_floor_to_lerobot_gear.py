#!/usr/bin/env python3
"""Convert Unitree sweep-floor raw episodes to LeRobot-v2/GEAR metadata.

Raw layout expected:
  raw_root/
    episode_0001/
      data.json
      colors/000000_color_0.jpg
      colors/000000_color_2.jpg
      colors/000000_color_3.jpg
      ...

The converted dataset uses a single 60-D low-dimensional control vector for
both state and action:
  state.sweep_floor_control  = robot_q_current[:36] + hand_state[:12] + ee_state[:12]
  action.sweep_floor_control = robot_q_desired[:36] + hand_cmd[:12] + ee_state[:12]

This matching state/action key lets DreamZero compute relative actions with its
standard action-minus-state chunk logic.
"""

from __future__ import annotations

import argparse
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


VIDEO_KEYS = {
    "video.head_stereo_left": "observation.images.color_0",
    "video.wrist_left": "observation.images.color_2",
    "video.wrist_right": "observation.images.color_3",
}
RAW_CAMERA_KEYS = {
    "observation.images.color_0": "color_0",
    "observation.images.color_2": "color_2",
    "observation.images.color_3": "color_3",
}
CONTROL_KEY = "sweep_floor_control"
ROBOT_Q_DIM = 36
HAND_DIM = 12
EE_DIM = 12
CONTROL_DIM = ROBOT_Q_DIM + HAND_DIM + EE_DIM


def _configure_imageio_ffmpeg() -> None:
    """Use the Python-package ffmpeg binary when system ffmpeg is unavailable."""

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
        raise ValueError(f"{name} has dim {arr.size}, expected at least {dim}")
    return arr[:dim].astype(np.float32)


def _state_vector(step: dict[str, Any]) -> np.ndarray:
    states = step["states"]
    return np.concatenate(
        [
            _as_vector(states["robot_q_current"], name="states.robot_q_current", dim=ROBOT_Q_DIM),
            _as_vector(states["hand_state"], name="states.hand_state", dim=HAND_DIM),
            _as_vector(states["ee_state"], name="states.ee_state", dim=EE_DIM),
        ],
        axis=0,
    ).astype(np.float32)


def _action_vector(step: dict[str, Any]) -> np.ndarray:
    actions = step["actions"]
    return np.concatenate(
        [
            _as_vector(actions["robot_q_desired"], name="actions.robot_q_desired", dim=ROBOT_Q_DIM),
            _as_vector(actions["hand_cmd"], name="actions.hand_cmd", dim=HAND_DIM),
            _as_vector(actions["ee_state"], name="actions.ee_state", dim=EE_DIM),
        ],
        axis=0,
    ).astype(np.float32)


def _episode_sort_key(path: Path) -> int:
    try:
        return int(path.name.split("_")[-1])
    except ValueError:
        return 10**12


def _load_raw_episode(path: Path) -> dict[str, Any]:
    data_path = path / "data.json"
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    return json.loads(data_path.read_text())


def _task_text(raw: dict[str, Any], default: str) -> str:
    text = raw.get("text") or {}
    goal = str(text.get("goal") or "").strip()
    desc = str(text.get("desc") or "").strip()
    if goal and desc and desc.lower() not in {"task description", "none", "null"}:
        return f"{goal}. {desc}"
    return goal or desc or default


def _write_video(
    *,
    raw_episode_dir: Path,
    steps: list[dict[str, Any]],
    raw_camera_key: str,
    output_path: Path,
    fps: int,
) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        writer = imageio.get_writer(output_path, fps=fps, codec="libx264", macro_block_size=16)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "imageio could not find ffmpeg. Install ffmpeg on this machine or install "
            "the Python package imageio-ffmpeg in the environment that runs this script."
        ) from exc
    width = height = None
    try:
        for step in steps:
            rel = step["colors"][raw_camera_key]
            img_path = raw_episode_dir / rel
            frame = np.asarray(Image.open(img_path).convert("RGB"))
            height, width = int(frame.shape[0]), int(frame.shape[1])
            writer.append_data(frame)
    finally:
        writer.close()
    if width is None or height is None:
        raise ValueError(f"No frames written for {raw_episode_dir} {raw_camera_key}")
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
) -> dict[str, dict[str, list[float]]]:
    all_relative: list[np.ndarray] = []
    for states, actions in zip(state_arrays, action_arrays):
        usable = len(actions) - (int(action_horizon) - 1)
        for anchor in range(max(usable, 0)):
            ref_state = states[anchor]
            all_relative.append(actions[anchor : anchor + int(action_horizon)] - ref_state)
    if not all_relative:
        raise ValueError("No relative action samples; check episode lengths/action_horizon")
    rel = np.concatenate(all_relative, axis=0).astype(np.float64)
    return {CONTROL_KEY: _stats([rel])}


def convert(args: argparse.Namespace) -> None:
    _configure_imageio_ffmpeg()
    raw_root = Path(args.raw_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not raw_root.exists():
        raise FileNotFoundError(raw_root)
    if output_root.exists() and args.force:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    raw_episode_dirs = sorted(
        [p for p in raw_root.glob("episode_*") if (p / "data.json").exists()],
        key=_episode_sort_key,
    )[: int(args.max_episodes)]
    if not raw_episode_dirs:
        raise RuntimeError(f"No episode_*/data.json found under {raw_root}")

    data_dir = output_root / "data" / "chunk-000"
    video_dir = output_root / "videos" / "chunk-000"
    meta_dir = output_root / "meta"
    data_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    state_arrays: list[np.ndarray] = []
    action_arrays: list[np.ndarray] = []
    tasks: dict[str, int] = {}
    episodes_meta: list[dict[str, Any]] = []
    video_width = int(args.width)
    video_height = int(args.height)

    for new_ep_idx, ep_dir in enumerate(tqdm(raw_episode_dirs, desc="Converting episodes")):
        raw = _load_raw_episode(ep_dir)
        steps = raw["data"]
        task = _task_text(raw, args.default_task)
        task_idx = tasks.setdefault(task, len(tasks))

        rows = []
        ep_states = []
        ep_actions = []
        for frame_idx, step in enumerate(steps):
            state = _state_vector(step)
            action = _action_vector(step)
            ep_states.append(state)
            ep_actions.append(action)
            rows.append(
                {
                    "episode_index": new_ep_idx,
                    "frame_index": frame_idx,
                    "timestamp": float(frame_idx) / float(args.fps),
                    "next.done": frame_idx == len(steps) - 1,
                    "index": frame_idx,
                    "task_index": task_idx,
                    "annotation.task": task,
                    "observation.state.sweep_floor_control": state,
                    "action.sweep_floor_control": action,
                }
            )

        pd.DataFrame(rows).to_parquet(data_dir / f"episode_{new_ep_idx:06d}.parquet")
        state_arrays.append(np.stack(ep_states, axis=0))
        action_arrays.append(np.stack(ep_actions, axis=0))
        episodes_meta.append({"episode_index": new_ep_idx, "tasks": [task], "length": len(rows)})

        for original_key, raw_camera_key in RAW_CAMERA_KEYS.items():
            out_video = video_dir / f"episode_{new_ep_idx:06d}_{original_key}.mp4"
            video_width, video_height = _write_video(
                raw_episode_dir=ep_dir,
                steps=steps,
                raw_camera_key=raw_camera_key,
                output_path=out_video,
                fps=int(args.fps),
            )

    total_frames = int(sum(len(x) for x in state_arrays))
    info = {
        "codebase_version": "v2.0",
        "robot_type": "unitree_g1_upper_body",
        "total_episodes": len(raw_episode_dirs),
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_videos": len(raw_episode_dirs) * len(RAW_CAMERA_KEYS),
        "chunks_size": 1000,
        "fps": int(args.fps),
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/episode_{episode_index:06d}_{video_key}.mp4",
        "features": {
            "observation.state.sweep_floor_control": {
                "dtype": "float32",
                "shape": [CONTROL_DIM],
                "names": [f"sweep_floor_control_{i}" for i in range(CONTROL_DIM)],
            },
            "action.sweep_floor_control": {
                "dtype": "float32",
                "shape": [CONTROL_DIM],
                "names": [f"sweep_floor_control_{i}" for i in range(CONTROL_DIM)],
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
            CONTROL_KEY: {
                "original_key": "observation.state.sweep_floor_control",
                "start": 0,
                "end": CONTROL_DIM,
                "rotation_type": None,
                "absolute": True,
                "dtype": "float32",
                "range": None,
            }
        },
        "action": {
            CONTROL_KEY: {
                "original_key": "action.sweep_floor_control",
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
        "observation.state.sweep_floor_control": _stats(state_arrays),
        "action.sweep_floor_control": _stats(action_arrays),
    }
    relative_stats = _relative_stats(
        state_arrays,
        action_arrays,
        action_horizon=int(args.action_horizon),
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

    print(f"Converted {len(raw_episode_dirs)} episodes ({total_frames} frames) to {output_root}")
    print(f"State/action key: {CONTROL_KEY}, dim={CONTROL_DIM}")
    print("Relative stats: meta/relative_stats_dreamzero.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--raw-root", default="/mnt/unitree_cpfs/ruixuan/datasets/data_sweep_floor")
    parser.add_argument(
        "--output-root",
        default="/mnt/unitree_cpfs/ruixuan/datasets/gear_format/data_sweep_floor_lerobot_gear_100eps_fix_action",
    )
    parser.add_argument("--max-episodes", type=int, default=100)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--action-horizon", type=int, default=48)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--default-task", default="sweep the floor")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
