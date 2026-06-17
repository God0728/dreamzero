#!/usr/bin/env python3
"""Episode-level DreamZero Unitree open-loop eval via the Unitree EEF server logic.

This evaluator builds an in-process fake robotdeploy client: for each selected
Unitree LeRobot-v2 episode it streams observations chunk-by-chunk into
``eval_utils.serve_unitree_dreamzero_eef.UnitreeDreamZeroEEFPolicy``.  The server
policy records predicted action chunks and predicted video latents.  After the
whole episode, this script reads those server artifacts, compares against GT, and
writes episode-level action plots plus pred/GT side-by-side videos.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any, Iterable

# Avoid noisy HuggingFace tokenizers fork warnings during offline eval setup.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import cv2
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

DREAMZERO_ROOT = Path(__file__).resolve().parents[1]
if str(DREAMZERO_ROOT) not in sys.path:
    sys.path.insert(0, str(DREAMZERO_ROOT))

from eval_utils import serve_unitree_dreamzero_eef as unitree_server
from groot.vla.data.dataset.lerobot import LeRobotSingleDataset, ModalityConfig
from groot.vla.data.schema import EmbodimentTag


UNITREE_VIDEO_KEYS = [item.split("=", 1)[0] for item in unitree_server.DEFAULT_VIEW_MAP]
UNITREE_STATE_KEYS = [item.split("=", 1)[0] for item in unitree_server.DEFAULT_STATE_MAP]
UNITREE_ACTION_KEYS = list(unitree_server.DEFAULT_MODEL_ACTION_KEYS)
ROBOT_ACTION_KEYS = list(unitree_server.DEFAULT_ROBOT_ACTION_KEYS)


def _parse_map(items: Iterable[str]) -> dict[str, str]:
    return unitree_server._parse_key_map(items)  # noqa: SLF001 - intentionally reuses server contract.


def _select_prompt(prompt: str, *, mode: str, rng: random.Random) -> str:
    variants = [part.strip() for part in str(prompt).split("@") if part.strip()]
    if not variants:
        return str(prompt)
    if mode == "random":
        return rng.choice(variants)
    return variants[0]


def _read_episode_tasks(dataset_path: Path) -> dict[int, str]:
    tasks_by_episode: dict[int, str] = {}
    episode_path = dataset_path / "meta" / "episodes.jsonl"
    if episode_path.exists():
        for line in episode_path.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            tasks = item.get("tasks") or []
            if tasks:
                tasks_by_episode[int(item["episode_index"])] = str(tasks[0])

    task_index_to_text: dict[int, str] = {}
    tasks_path = dataset_path / "meta" / "tasks.jsonl"
    if tasks_path.exists():
        for line in tasks_path.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            task_index_to_text[int(item["task_index"])] = str(item["task"])

    for episode_id in list(tasks_by_episode):
        if tasks_by_episode[episode_id]:
            continue
        traj = dataset_path / "data" / f"chunk-{episode_id // 1000:03d}" / f"episode_{episode_id:06d}.parquet"
        if not traj.exists():
            continue
        try:
            import pandas as pd

            df = pd.read_parquet(traj, columns=["task_index"])
            task_idx = int(df["task_index"].iloc[0])
            tasks_by_episode[episode_id] = task_index_to_text.get(task_idx, "")
        except Exception:
            pass
    return tasks_by_episode


def _load_dataset(dataset_path: str, *, video_backend: str, action_horizon: int) -> LeRobotSingleDataset:
    modality_configs = {
        "video": ModalityConfig(delta_indices=[0], modality_keys=UNITREE_VIDEO_KEYS),
        "state": ModalityConfig(delta_indices=[0], modality_keys=UNITREE_STATE_KEYS),
        "action": ModalityConfig(delta_indices=list(range(int(action_horizon))), modality_keys=UNITREE_ACTION_KEYS),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.task_index"]),
    }
    return LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_configs,
        embodiment_tag=EmbodimentTag.UNITREE_G1_UPPER_BODY,
        transforms=None,
        use_global_metadata=False,
        video_backend=video_backend,
        discard_bad_trajectories=False,
    )


def _episode_length(dataset: LeRobotSingleDataset, episode_id: int) -> int:
    for eid, length in zip(dataset.trajectory_ids, dataset.trajectory_lengths):
        if int(eid) == int(episode_id):
            return int(length)
    raise ValueError(f"episode_id={episode_id} not found; available starts={list(map(int, dataset.trajectory_ids[:10]))}")


def _resolve_episode_ids(dataset: LeRobotSingleDataset, args: argparse.Namespace) -> list[int]:
    available = [int(eid) for eid in dataset.trajectory_ids]
    if args.episode_ids:
        missing = [eid for eid in args.episode_ids if eid not in available]
        if missing:
            raise ValueError(f"episode ids not found: {missing}; available count={len(available)}")
        return [int(eid) for eid in args.episode_ids]
    count = len(available) if args.episode_num is None else min(int(args.episode_num), len(available))
    return available[:count]


def _get_step_data(dataset: LeRobotSingleDataset, episode_id: int, keys: Iterable[str], indices: np.ndarray) -> dict[str, Any]:
    return dataset.get_step_data(
        int(episode_id),
        {key: np.asarray(indices, dtype=int).copy() for key in keys},
    )


def _historical_video_indices(step: int, *, action_horizon: int, video_stride: int) -> np.ndarray:
    return np.arange(step - action_horizon, step + 1, video_stride, dtype=int)


def _build_fake_obs(
    dataset: LeRobotSingleDataset,
    *,
    episode_id: int,
    episode_length: int,
    anchor: int,
    prompt: str,
    view_map: dict[str, str],
    state_map: dict[str, str],
    action_horizon: int,
    max_context_chunk_num: int,
    video_stride: int,
    client_preprocess: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if anchor <= 0:
        video_indices = np.asarray([0], dtype=int)
    else:
        # Match eval_unitree_open_loop.py exactly: each non-initial causal call
        # receives the previous 48-step 30FPS action block downsampled to 5FPS,
        # including the current replan boundary frame:
        # [step-48, step-42, ..., step] for the default Unitree 48/6 setup.
        video_indices = _historical_video_indices(
            anchor,
            action_horizon=action_horizon,
            video_stride=video_stride,
        )
        video_indices = np.clip(video_indices, 0, max(episode_length - 1, 0)).astype(int)
    state_index = np.asarray([min(max(anchor, 0), episode_length - 1)], dtype=int)

    video_data = _get_step_data(dataset, episode_id, view_map.keys(), video_indices)
    state_data = _get_step_data(dataset, episode_id, state_map.keys(), state_index)

    obs: dict[str, Any] = {"prompt": prompt}
    for model_key, client_key in view_map.items():
        frames = np.asarray(video_data[model_key], dtype=np.uint8)
        if client_preprocess:
            resize_hw = client_preprocess.get("resize_hw")
            crop_scale = client_preprocess.get("center_crop_scale")
            if resize_hw:
                frames = np.stack(
                    [
                        _resize_frame(
                            _center_crop(frame[..., :3], float(crop_scale) if crop_scale is not None else None),
                            height=int(resize_hw[0]),
                            width=int(resize_hw[1]),
                        )
                        for frame in frames
                    ],
                    axis=0,
                ).astype(np.uint8)
        obs[client_key] = frames
    for model_key, client_key in state_map.items():
        obs[client_key] = np.asarray(state_data[model_key], dtype=np.float64)
    return obs


def _center_crop(frame: np.ndarray, scale: float | None) -> np.ndarray:
    if scale is None or scale >= 0.999:
        return frame
    h, w = frame.shape[:2]
    ch = max(1, int(round(h * float(scale))))
    cw = max(1, int(round(w * float(scale))))
    y0 = max((h - ch) // 2, 0)
    x0 = max((w - cw) // 2, 0)
    return frame[y0 : y0 + ch, x0 : x0 + cw]


def _resize_frame(frame: np.ndarray, *, height: int, width: int) -> np.ndarray:
    if frame.shape[:2] == (height, width):
        return frame
    return cv2.resize(frame, (int(width), int(height)), interpolation=cv2.INTER_LINEAR)


def _compose_unitree_views(
    view_frames: dict[str, np.ndarray],
    *,
    video_keys: list[str],
    per_view_height: int,
    per_view_width: int,
    center_crop_scale: float | None,
) -> np.ndarray:
    available = [view_frames[key] for key in video_keys if key in view_frames]
    if not available:
        raise RuntimeError("No Unitree video frames available")
    num_frames = min(frames.shape[0] for frames in available)
    output = np.zeros((num_frames, 2 * per_view_height, 2 * per_view_width, 3), dtype=np.uint8)

    def put(view_index: int, row: int, col: int) -> None:
        if view_index >= len(video_keys) or video_keys[view_index] not in view_frames:
            return
        frames = np.asarray(view_frames[video_keys[view_index]], dtype=np.uint8)[:num_frames]
        y0, x0 = row * per_view_height, col * per_view_width
        for t, frame in enumerate(frames):
            cropped = _center_crop(frame[..., :3], center_crop_scale)
            resized = _resize_frame(cropped, height=per_view_height, width=per_view_width)
            output[t, y0 : y0 + per_view_height, x0 : x0 + per_view_width] = resized

    put(0, 0, 0)  # head
    put(1, 1, 0)  # left wrist
    put(2, 0, 1)  # right wrist
    return output


def _gt_video_episode(
    dataset: LeRobotSingleDataset,
    *,
    episode_id: int,
    episode_length: int,
    pred_frame_count: int,
    video_stride: int,
    preprocess: dict[str, Any],
) -> np.ndarray:
    # Match the decoded AR sequence length where possible: N chunks -> 8N+1 frames.
    indices = np.arange(0, episode_length, int(video_stride), dtype=int)
    if indices.size == 0 or indices[0] != 0:
        indices = np.concatenate([np.asarray([0], dtype=int), indices])
    indices = indices[:pred_frame_count]
    data = _get_step_data(dataset, episode_id, UNITREE_VIDEO_KEYS, indices)
    resize_hw = preprocess.get("resize_hw") or [176, 320]
    center_crop_scale = preprocess.get("center_crop_scale", 0.95)
    return _compose_unitree_views(
        {key: np.asarray(data[key], dtype=np.uint8) for key in UNITREE_VIDEO_KEYS},
        video_keys=UNITREE_VIDEO_KEYS,
        per_view_height=int(resize_hw[0]),
        per_view_width=int(resize_hw[1]),
        center_crop_scale=float(center_crop_scale) if center_crop_scale is not None else None,
    )


def _gt_actions_episode(
    dataset: LeRobotSingleDataset,
    *,
    episode_id: int,
    episode_length: int,
) -> tuple[np.ndarray, dict[str, int], list[str]]:
    indices = np.arange(0, episode_length, dtype=int)
    data = _get_step_data(dataset, episode_id, UNITREE_ACTION_KEYS, indices)
    arrays: list[np.ndarray] = []
    dims: dict[str, int] = {}
    robot_keys: list[str] = []
    for model_key in UNITREE_ACTION_KEYS:
        robot_key = unitree_server.MODEL_TO_ROBOT_ACTION_KEY[model_key]
        arr = np.asarray(data[model_key], dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        arrays.append(arr)
        dims[robot_key] = int(arr.shape[-1])
        robot_keys.append(robot_key)
    return np.concatenate(arrays, axis=-1), dims, robot_keys


def _load_pred_actions(npz_path: str, *, target_len: int) -> np.ndarray:
    payload = np.load(npz_path)
    pred = np.asarray(payload["action_concat"], dtype=np.float32)
    if pred.shape[0] < target_len:
        pad = np.repeat(pred[-1:], target_len - pred.shape[0], axis=0)
        pred = np.concatenate([pred, pad], axis=0)
    return pred[:target_len]


def _train_cfg_get(policy: Any, key: str, default: Any) -> Any:
    train_cfg = getattr(policy, "train_cfg", {}) or {}
    if hasattr(train_cfg, "get"):
        return train_cfg.get(key, default)
    return getattr(train_cfg, key, default)


def _relative_action_info(policy: Any) -> dict[str, Any]:
    keys = _train_cfg_get(policy, "relative_action_keys", [])
    return {
        "relative_action": bool(_train_cfg_get(policy, "relative_action", False)),
        "relative_action_per_horizon": bool(_train_cfg_get(policy, "relative_action_per_horizon", False)),
        "relative_action_keys": [str(key) for key in list(keys)],
    }


def _validate_relative_action_inputs(
    *,
    relative_action_info: dict[str, Any],
    state_keys: Iterable[str],
) -> None:
    """Fail early if relative pose actions cannot be converted back to absolute."""

    if not (
        relative_action_info["relative_action"]
        or relative_action_info["relative_action_per_horizon"]
    ):
        return
    available = set(state_keys)
    missing = [
        f"state.{key}"
        for key in relative_action_info["relative_action_keys"]
        if f"state.{key}" not in available
    ]
    if missing:
        raise ValueError(
            "Relative-action Unitree eval requires matching current state keys so "
            f"GrootSimPolicy.unapply can convert deltas back to absolute actions; "
            f"missing={missing}, available={sorted(available)}"
        )


def _pad_ylim(low: float, high: float, *, pad_frac: float) -> tuple[float, float]:
    low = float(low)
    high = float(high)
    if not np.isfinite(low) or not np.isfinite(high):
        return -1.0, 1.0
    if high < low:
        low, high = high, low
    span = high - low
    if span < 1e-9:
        center = 0.5 * (low + high)
        span = max(abs(center) * 0.1, 1e-3)
        low, high = center - 0.5 * span, center + 0.5 * span
    pad = max(span * pad_frac, 1e-6)
    return low - pad, high + pad


def _action_dim_names(action_keys: list[str], dims: dict[str, int]) -> list[str]:
    names: list[str] = []
    for key in action_keys:
        for idx in range(dims[key]):
            names.append(f"{key.replace('action.', '')}[{idx}]")
    return names


def _load_dataset_action_ylims(
    dataset: LeRobotSingleDataset,
    *,
    action_keys: list[str],
    dim_names: list[str],
    pad_frac: float,
) -> dict[str, tuple[float, float]]:
    stats = dataset.lerobot_stats_meta
    bounds: dict[str, tuple[float, float]] = {}
    name_idx = 0
    for model_key in UNITREE_ACTION_KEYS:
        subkey = model_key.replace("action.", "")
        meta = dataset.lerobot_modality_meta.action.get(subkey)
        if meta is None:
            name_idx += unitree_server.MODEL_ACTION_DIMS[model_key]
            continue
        original_key = meta.original_key or model_key
        stat = stats.get(original_key) if hasattr(stats, "get") else None
        if stat is None:
            name_idx += unitree_server.MODEL_ACTION_DIMS[model_key]
            continue
        stat_dict = stat.model_dump() if hasattr(stat, "model_dump") else dict(stat)
        q01 = np.asarray(stat_dict.get("q01"), dtype=np.float32)
        q99 = np.asarray(stat_dict.get("q99"), dtype=np.float32)
        q01 = q01[int(meta.start) : int(meta.end)]
        q99 = q99[int(meta.start) : int(meta.end)]
        for low, high in zip(q01, q99):
            if name_idx < len(dim_names):
                bounds[dim_names[name_idx]] = _pad_ylim(float(low), float(high), pad_frac=pad_frac)
            name_idx += 1
    return bounds


def _sample_action_ylims(
    *,
    pred: np.ndarray,
    gt: np.ndarray,
    dim_names: list[str],
    mode: str,
    pad_frac: float,
) -> dict[str, tuple[float, float]]:
    if mode == "auto":
        return {}
    values = np.concatenate([pred, gt], axis=0)
    bounds: dict[str, tuple[float, float]] = {}
    for dim, name in enumerate(dim_names):
        vals = values[:, dim]
        if mode == "robust":
            low, high = np.nanpercentile(vals, [1, 99])
        else:
            low, high = np.nanmin(vals), np.nanmax(vals)
        bounds[name] = _pad_ylim(float(low), float(high), pad_frac=pad_frac)
    return bounds


def _merge_ylims(
    base: dict[str, tuple[float, float]],
    *,
    pred: np.ndarray,
    gt: np.ndarray,
    dim_names: list[str],
    include: str,
    pad_frac: float,
) -> dict[str, tuple[float, float]]:
    if include == "none":
        return dict(base)
    values = gt if include == "gt" else np.concatenate([pred, gt], axis=0)
    merged = dict(base)
    for dim, name in enumerate(dim_names):
        low = float(np.nanmin(values[:, dim]))
        high = float(np.nanmax(values[:, dim]))
        if name in merged:
            low = min(low, merged[name][0])
            high = max(high, merged[name][1])
        merged[name] = _pad_ylim(low, high, pad_frac=pad_frac)
    return merged


def save_action_plot(
    path: Path,
    *,
    pred: np.ndarray,
    gt: np.ndarray,
    dim_names: list[str],
    title: str,
    ylims: dict[str, tuple[float, float]],
    ylim_label: str,
) -> None:
    dim = gt.shape[-1]
    ncols = min(4, dim)
    nrows = int(math.ceil(dim / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 2.6 * nrows), squeeze=False)
    fig.suptitle(title, fontsize=12)
    x = np.arange(gt.shape[0])
    for i in range(dim):
        ax = axes[i // ncols][i % ncols]
        ax.plot(x, gt[:, i], label="gt", linewidth=1.2)
        ax.plot(x, pred[:, i], label="pred", linewidth=1.0)
        ax.set_title(dim_names[i], fontsize=8)
        if dim_names[i] in ylims:
            ax.set_ylim(*ylims[dim_names[i]])
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)
    for i in range(dim, nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)
    fig.text(0.995, 0.005, f"ylim={ylim_label}", ha="right", va="bottom", fontsize=8)
    fig.tight_layout(rect=[0, 0.01, 1, 0.96])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _resize_video(frames: np.ndarray, *, height: int, width: int) -> np.ndarray:
    if frames.shape[1:3] == (height, width):
        return frames
    return np.stack([_resize_frame(frame, height=height, width=width) for frame in frames], axis=0).astype(np.uint8)


def save_video_compare(path: Path, *, pred: np.ndarray, gt: np.ndarray, fps: int) -> None:
    n = min(pred.shape[0], gt.shape[0])
    pred = pred[:n]
    gt = gt[:n]
    if gt.shape[1:3] != pred.shape[1:3]:
        gt = _resize_video(gt, height=pred.shape[1], width=pred.shape[2])
    font = cv2.FONT_HERSHEY_SIMPLEX
    frames = []
    for p, g in zip(pred, gt):
        canvas = np.concatenate([p, g], axis=1)
        scale = max(min(canvas.shape[:2]) / 500.0, 0.5)
        thick = max(1, int(round(scale * 2)))
        cv2.putText(canvas, "PRED", (12, 28), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
        cv2.putText(canvas, "GT", (p.shape[1] + 12, 28), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
        frames.append(canvas)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=int(fps), codec="libx264")


def video_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    n = min(pred.shape[0], gt.shape[0])
    pred = pred[:n]
    gt = gt[:n]
    if gt.shape[1:3] != pred.shape[1:3]:
        gt = _resize_video(gt, height=pred.shape[1], width=pred.shape[2])
    pred_f = pred.astype(np.float32) / 255.0
    gt_f = gt.astype(np.float32) / 255.0
    mse = float(np.mean((pred_f - gt_f) ** 2))
    mae = float(np.mean(np.abs(pred_f - gt_f)))
    psnr = float("inf") if mse <= 1e-12 else float(20.0 * math.log10(1.0 / math.sqrt(mse)))
    return {"video_mae": mae, "video_psnr": psnr, "video_frames": int(n)}


def aggregate_episode_metrics(summaries: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    metric_keys = sorted(
        {
            key
            for summary in summaries
            for key, value in (summary.get("metrics") or {}).items()
            if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)
        }
    )
    aggregated: dict[str, dict[str, float]] = {}
    for key in metric_keys:
        values = [
            float(summary["metrics"][key])
            for summary in summaries
            if key in (summary.get("metrics") or {})
        ]
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        aggregated[key] = {
            "mean": float(np.mean(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
        }
    return aggregated


def _load_pred_video_frames(artifacts: dict[str, Any]) -> tuple[np.ndarray, str]:
    """Load raw decoded prediction frames from the server's temporary npy."""

    frames_npy = artifacts.get("pred_video_frames_npy")
    if frames_npy and Path(frames_npy).exists():
        return np.load(frames_npy), str(frames_npy)

    video_json = artifacts.get("pred_video_json")
    if video_json and Path(video_json).exists():
        payload = json.loads(Path(video_json).read_text())
        frames_npy = payload.get("frames_npy")
        if frames_npy and Path(frames_npy).exists():
            return np.load(frames_npy), str(frames_npy)

    raise RuntimeError(f"Server did not produce required predicted-video npy: {artifacts}")


def _delete_pred_video_npy(path: str, artifacts: dict[str, Any]) -> None:
    npy_path = Path(path)
    if npy_path.suffix != ".npy":
        raise RuntimeError(f"Refusing to delete non-npy predicted-video source: {npy_path}")
    npy_path.unlink()
    artifacts["pred_video_frames_npy_deleted_after_compare"] = True
    artifacts["pred_video_frames_npy_deleted_path"] = str(npy_path)
    artifact_json = artifacts.get("pred_video_json")
    if artifact_json and Path(artifact_json).exists():
        payload = json.loads(Path(artifact_json).read_text())
        payload["frames_npy_deleted_after_compare"] = True
        payload["frames_npy_deleted_path"] = str(npy_path)
        Path(artifact_json).write_text(json.dumps(payload, indent=2))
    session_dir = artifacts.get("session_dir")
    artifacts_json = Path(session_dir) / "artifacts.json" if session_dir else None
    if artifacts_json and artifacts_json.exists():
        artifacts_json.write_text(json.dumps(artifacts, indent=2))


def build_server_args(args: argparse.Namespace, *, server_output_dir: Path) -> argparse.Namespace:
    subsequent_frames = int(args.action_horizon // args.video_stride + 1)
    argv = [
        "--checkpoint",
        args.model_path,
        "--device",
        args.device,
        "--action-horizon",
        str(args.action_horizon),
        "--obs-chunk-size",
        str(args.max_context_chunk_num * args.action_horizon + 1),
        "--initial-frames",
        "1",
        "--subsequent-frames",
        str(subsequent_frames),
        "--include-current-boundary",
        "--client-image-color-order",
        "rgb",
        "--output-dir",
        str(server_output_dir),
        "--predicted-video-fps",
        str(args.video_fps),
        "--log-level",
        args.log_level,
        "--save-predicted-video-npy",
    ]
    if args.prompt:
        argv.extend(["--prompt", args.prompt])
    if args.tokenizer_path_override:
        argv.extend(["--tokenizer-path-override", args.tokenizer_path_override])
    for override in args.model_config_override or []:
        argv.extend(["--model-config-override", override])
    if args.skip_img_transform:
        argv.append("--skip-img-transform")
    if args.no_predicted_video_watermark:
        argv.append("--no-predicted-video-watermark")
    if not args.log_latency:
        argv.append("--no-log-latency")
    return unitree_server.parse_args(argv)


def evaluate(args: argparse.Namespace) -> None:
    os.environ["ENABLE_DIT_CACHE"] = "true" if args.enable_dit_cache else "false"
    if args.attention_backend:
        os.environ["ATTENTION_BACKEND"] = args.attention_backend
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    server_output_dir = output_dir / "server_artifacts"

    server_args = build_server_args(args, server_output_dir=server_output_dir)
    runtime = unitree_server._init_runtime(  # noqa: SLF001 - reuse exact serving runtime.
        server_args.device,
        server_args.distributed_timeout_seconds,
        dynamo_recompile_limit=server_args.torch_dynamo_recompile_limit,
        dynamo_cache_size_limit=server_args.torch_dynamo_cache_size_limit,
    )
    server_args.device = runtime["device"]
    if runtime["distributed"] and runtime["rank"] != 0:
        server_args.output_dir = None
        server_args.save_predictions = False
    policy_cls = unitree_server.DistributedUnitreeDreamZeroEEFPolicy if runtime["distributed"] else unitree_server.UnitreeDreamZeroEEFPolicy
    policy_kwargs: dict[str, Any] = {
        "args": server_args,
        "device_mesh": runtime["device_mesh"],
        "world_size": runtime["world_size"],
    }
    if runtime["distributed"]:
        policy_kwargs["signal_group"] = runtime["signal_group"]
    policy = policy_cls(**policy_kwargs)
    try:
        if args.cfg_scale is not None:
            policy.policy.trained_model.action_head.cfg_scale = float(args.cfg_scale)
        if runtime["distributed"] and float(policy.policy.trained_model.action_head.cfg_scale) == 1.0:
            raise ValueError("2-GPU CFG-parallel mode requires cfg_scale != 1.0.")
        if runtime["distributed"] and runtime["rank"] != 0:
            policy.worker_loop()
            return

        _validate_relative_action_inputs(
            relative_action_info=_relative_action_info(policy.policy),
            state_keys=policy.state_map.keys(),
        )

        dataset = _load_dataset(args.dataset_path, video_backend=args.video_backend, action_horizon=args.action_horizon)
        tasks_by_episode = _read_episode_tasks(Path(args.dataset_path))
        episode_ids = _resolve_episode_ids(dataset, args)
        view_map = policy.view_map
        state_map = policy.state_map
        preprocess = unitree_server._infer_client_video_preprocess_from_checkpoint(  # noqa: SLF001
            args.model_path,
            server_args.embodiment_tag,
            view_map.keys(),
        ) or {"center_crop_scale": 0.95, "resize_hw": [176, 320]}
        client_preprocess = policy.client_image_preprocess if args.skip_img_transform else None

        summaries = []
        for ep_index, episode_id in enumerate(episode_ids):
            episode_length = _episode_length(dataset, episode_id)
            prompt_raw = args.prompt or tasks_by_episode.get(int(episode_id), "")
            prompt = _select_prompt(prompt_raw, mode=args.prompt_variant, rng=rng)
            session_alias = f"episode_{int(episode_id):06d}"
            policy.reset({"session_id": session_alias, "prompt": prompt})

            num_chunks = int(math.ceil(episode_length / args.action_horizon))
            for chunk_idx in range(num_chunks):
                anchor = chunk_idx * args.action_horizon
                obs = _build_fake_obs(
                    dataset,
                    episode_id=episode_id,
                    episode_length=episode_length,
                    anchor=anchor,
                    prompt=prompt,
                    view_map=view_map,
                    state_map=state_map,
                    action_horizon=args.action_horizon,
                    max_context_chunk_num=args.max_context_chunk_num,
                    video_stride=args.video_stride,
                    client_preprocess=client_preprocess,
                )
                policy.get_action(obs)
                if (chunk_idx + 1) % max(args.log_every, 1) == 0:
                    print(f"ep={episode_id} chunk={chunk_idx + 1}/{num_chunks}")

            artifacts = policy.flush_outputs()
            if "pred_action_npz" not in artifacts:
                raise RuntimeError(f"Server did not produce required episode artifacts: {artifacts}")
            episode_dir = output_dir / f"episode_{int(episode_id):06d}"
            episode_dir.mkdir(parents=True, exist_ok=True)

            gt_action, dims, robot_keys = _gt_actions_episode(
                dataset,
                episode_id=episode_id,
                episode_length=episode_length,
            )
            pred_action = _load_pred_actions(artifacts["pred_action_npz"], target_len=gt_action.shape[0])
            action_mse = float(np.mean((pred_action - gt_action) ** 2))
            action_mae = float(np.mean(np.abs(pred_action - gt_action)))
            dim_names = _action_dim_names(robot_keys, dims)
            if args.action_ylim_mode == "dataset-p01-p99":
                base_ylims = _load_dataset_action_ylims(
                    dataset,
                    action_keys=robot_keys,
                    dim_names=dim_names,
                    pad_frac=args.action_ylim_pad_frac,
                )
                ylims = _merge_ylims(
                    base_ylims,
                    pred=pred_action,
                    gt=gt_action,
                    dim_names=dim_names,
                    include=args.action_ylim_include_sample,
                    pad_frac=args.action_ylim_pad_frac,
                ) if base_ylims else _sample_action_ylims(
                    pred=pred_action,
                    gt=gt_action,
                    dim_names=dim_names,
                    mode="robust",
                    pad_frac=args.action_ylim_pad_frac,
                )
                ylim_label = "dataset-p01-p99" if base_ylims else "robust"
            else:
                ylims = _sample_action_ylims(
                    pred=pred_action,
                    gt=gt_action,
                    dim_names=dim_names,
                    mode=args.action_ylim_mode,
                    pad_frac=args.action_ylim_pad_frac,
                )
                ylim_label = args.action_ylim_mode
            action_plot = episode_dir / "action_pred_vs_gt.png"
            save_action_plot(
                action_plot,
                pred=pred_action,
                gt=gt_action,
                dim_names=dim_names,
                title=f"Unitree episode {episode_id} action | MSE={action_mse:.5g}",
                ylims=ylims,
                ylim_label=ylim_label,
            )

            pred_video, pred_video_source = _load_pred_video_frames(artifacts)
            gt_video = _gt_video_episode(
                dataset,
                episode_id=episode_id,
                episode_length=episode_length,
                pred_frame_count=pred_video.shape[0],
                video_stride=args.video_stride,
                preprocess=preprocess,
            )
            vmetrics = video_metrics(pred_video, gt_video)
            compare_video = episode_dir / "video_pred_gt_side_by_side.mp4"
            save_video_compare(compare_video, pred=pred_video, gt=gt_video, fps=args.video_fps)
            _delete_pred_video_npy(pred_video_source, artifacts)

            summary = {
                "episode_id": int(episode_id),
                "episode_length": int(episode_length),
                "num_chunks": int(num_chunks),
                "prompt": prompt,
                "prompt_raw": prompt_raw,
                "metrics": {
                    "action_mse": action_mse,
                    "action_mae": action_mae,
                    **vmetrics,
                },
                "artifacts": {
                    "server": artifacts,
                    "action_plot": str(action_plot),
                    "video_compare_mp4": str(compare_video),
                    "pred_video_source": pred_video_source,
                },
                "config": {
                    "max_context_chunk_num": int(args.max_context_chunk_num),
                    "action_horizon": int(args.action_horizon),
                    "video_stride": int(args.video_stride),
                    "preprocess": preprocess,
                },
            }
            (episode_dir / "summary.json").write_text(json.dumps(summary, indent=2))
            summaries.append(summary)
            print(
                f"episode={episode_id} action_mse={action_mse:.6g} "
                f"video_mae={vmetrics['video_mae']:.6g} artifacts={episode_dir}"
            )

        (output_dir / "summary.json").write_text(
            json.dumps(
                {
                    "metrics": aggregate_episode_metrics(summaries),
                    "episodes": summaries,
                },
                indent=2,
            )
        )
    finally:
        if not (runtime["distributed"] and runtime["rank"] != 0):
            policy.shutdown()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-path", "--model_path", dest="model_path", required=True)
    parser.add_argument("--dataset-path", "--dataset_path", dest="dataset_path", required=True)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episode-ids", "--episode_ids", dest="episode_ids", nargs="*", type=int, default=None)
    parser.add_argument("--episode-num", "--episode_num", dest="episode_num", type=int, default=1)
    parser.add_argument("--max-context-chunk-num", "--max_context_chunk_num", dest="max_context_chunk_num", type=int, default=3)
    parser.add_argument("--action-horizon", "--action_horizon", dest="action_horizon", type=int, default=48)
    parser.add_argument("--video-stride", "--video_stride", dest="video_stride", type=int, default=6)
    parser.add_argument("--video-fps", "--video_fps", dest="video_fps", type=int, default=5)
    parser.add_argument("--video-backend", "--video_backend", dest="video_backend", default="decord")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-variant", choices=["first", "random"], default="first")
    parser.add_argument("--cfg-scale", "--cfg_scale", dest="cfg_scale", type=float, default=None)
    parser.add_argument("--tokenizer-path-override", default=None)
    parser.add_argument("--model-config-override", action="append", default=[])
    parser.add_argument("--enable-dit-cache", "--enable_dit_cache", dest="enable_dit_cache", action="store_true")
    parser.add_argument("--attention-backend", "--attention_backend", dest="attention_backend", default="TE")
    parser.add_argument("--skip-img-transform", action="store_true")
    parser.add_argument("--no-predicted-video-watermark", action="store_true")
    parser.add_argument("--log-latency", action="store_true", default=False)
    parser.add_argument("--log-every", "--log_every", dest="log_every", type=int, default=1)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--action-ylim-mode",
        "--action_ylim_mode",
        dest="action_ylim_mode",
        choices=["dataset-p01-p99", "robust", "minmax", "auto"],
        default="dataset-p01-p99",
    )
    parser.add_argument(
        "--action-ylim-include-sample",
        "--action_ylim_include_sample",
        dest="action_ylim_include_sample",
        choices=["none", "gt", "pred-gt"],
        default="gt",
    )
    parser.add_argument("--action-ylim-pad-frac", "--action_ylim_pad_frac", dest="action_ylim_pad_frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.max_context_chunk_num < 1:
        raise ValueError("--max-context-chunk-num must be >= 1")
    if args.action_horizon <= 0:
        raise ValueError("--action-horizon must be > 0")
    if args.video_stride <= 0:
        raise ValueError("--video-stride must be > 0")
    if args.action_horizon % args.video_stride != 0:
        raise ValueError("--video-stride must exactly divide --action-horizon for exact boundary-inclusive video windows")
    if args.action_ylim_pad_frac < 0:
        raise ValueError("--action-ylim-pad-frac must be >= 0")
    evaluate(args)


if __name__ == "__main__":
    main()
