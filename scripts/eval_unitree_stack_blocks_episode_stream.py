#!/usr/bin/env python3
"""Episode eval for Unitree stack-blocks full-body (60D) DreamZero checkpoints.

This script evaluates a full-body 60D checkpoint on complete LeRobot/GEAR
episodes.  It streams each episode chunk-by-chunk, predicts action chunks and
video latents, then writes action curves and pred/GT comparison videos.

The 60D ``sweep_floor_control`` layout is:
    [0:7]   base_pose  (xyz3 + quat_wxyz4)         -- not executed on real robot
    [7:36]  robot_q    (29 joints: legs12 + waist3 + larm7 + rarm7)
    [36:48] hand        (left6 + right6)
    [48:60] ee_state    (left xyzrpy6 + right xyzrpy6) -- not executed on real robot

Adapted from ``eval_unitree_sweep_floor_episode_stream.py`` (the only change of
substance is ``ACTION_DIM`` 53 -> 60 and the per-dim names).
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import json
import math
import os
from pathlib import Path
import random
import socket
import sys
from typing import Any, Iterable

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import cv2
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange
from tianshou.data import Batch

DREAMZERO_ROOT = Path(__file__).resolve().parents[1]
if str(DREAMZERO_ROOT) not in sys.path:
    sys.path.insert(0, str(DREAMZERO_ROOT))

from eval_utils.serve_unitree_dreamzero_eef import (  # noqa: E402
    _configure_torch_dynamo,
    _infer_client_video_preprocess_from_checkpoint,
)
from groot.vla.data.dataset.lerobot import LeRobotSingleDataset, ModalityConfig  # noqa: E402
from groot.vla.data.schema import EmbodimentTag  # noqa: E402
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy  # noqa: E402


VIDEO_KEYS = ["video.head_stereo_left", "video.wrist_left", "video.wrist_right"]
STATE_KEY = "state.sweep_floor_control"
ACTION_KEY = "action.sweep_floor_control"
LANGUAGE_KEY = "annotation.task_index"
ACTION_DIM = 60

# 60D control layout boundaries.
BASE_DIM = 7
JOINT_DIM = 29
HAND_DIM = 12
EE_DIM = 12


def _maybe_init_dist(timeout_seconds: int = 600) -> None:
    if dist.is_available() and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        if "MASTER_PORT" not in os.environ:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", 0))
                os.environ["MASTER_PORT"] = str(sock.getsockname()[1])
        dist.init_process_group(
            backend="gloo",
            world_size=1,
            rank=0,
            timeout=_datetime.timedelta(seconds=int(timeout_seconds)),
        )


def _read_episode_tasks(dataset_path: Path) -> dict[int, str]:
    tasks_by_episode: dict[int, str] = {}
    path = dataset_path / "meta" / "episodes.jsonl"
    if not path.exists():
        return tasks_by_episode
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        tasks = item.get("tasks") or []
        if tasks:
            tasks_by_episode[int(item["episode_index"])] = str(tasks[0])
    return tasks_by_episode


def _load_dataset(dataset_path: str, *, video_backend: str, action_horizon: int) -> LeRobotSingleDataset:
    modality_configs = {
        "video": ModalityConfig(delta_indices=[0], modality_keys=VIDEO_KEYS),
        "state": ModalityConfig(delta_indices=[0], modality_keys=[STATE_KEY]),
        "action": ModalityConfig(delta_indices=list(range(int(action_horizon))), modality_keys=[ACTION_KEY]),
        "language": ModalityConfig(delta_indices=[0], modality_keys=[LANGUAGE_KEY]),
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
    raise ValueError(f"episode_id={episode_id} not found")


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
    per_view_height: int,
    per_view_width: int,
    center_crop_scale: float | None,
) -> np.ndarray:
    available = [view_frames[key] for key in VIDEO_KEYS if key in view_frames]
    if not available:
        raise RuntimeError("No Unitree video frames available")
    num_frames = min(frames.shape[0] for frames in available)
    output = np.zeros((num_frames, 2 * per_view_height, 2 * per_view_width, 3), dtype=np.uint8)

    def put(key: str, row: int, col: int) -> None:
        if key not in view_frames:
            return
        frames = np.asarray(view_frames[key], dtype=np.uint8)[:num_frames]
        y0, x0 = row * per_view_height, col * per_view_width
        for t, frame in enumerate(frames):
            cropped = _center_crop(frame[..., :3], center_crop_scale)
            resized = _resize_frame(cropped, height=per_view_height, width=per_view_width)
            output[t, y0 : y0 + per_view_height, x0 : x0 + per_view_width] = resized

    put(VIDEO_KEYS[0], 0, 0)
    put(VIDEO_KEYS[2], 0, 1)
    put(VIDEO_KEYS[1], 1, 0)
    return output


def _build_model_obs(
    dataset: LeRobotSingleDataset,
    *,
    episode_id: int,
    episode_length: int,
    anchor: int,
    prompt: str,
    action_horizon: int,
    video_stride: int,
) -> dict[str, Any]:
    if anchor <= 0:
        video_indices = np.asarray([0], dtype=int)
    else:
        video_indices = _historical_video_indices(anchor, action_horizon=action_horizon, video_stride=video_stride)
        video_indices = np.clip(video_indices, 0, max(episode_length - 1, 0)).astype(int)
    state_index = np.asarray([min(max(anchor, 0), episode_length - 1)], dtype=int)

    video_data = _get_step_data(dataset, episode_id, VIDEO_KEYS, video_indices)
    state_data = _get_step_data(dataset, episode_id, [STATE_KEY], state_index)

    obs: dict[str, Any] = {LANGUAGE_KEY: prompt}
    for key in VIDEO_KEYS:
        frames = np.asarray(video_data[key], dtype=np.uint8)
        obs[key] = frames[0] if frames.shape[0] == 1 else frames
    obs[STATE_KEY] = np.asarray(state_data[STATE_KEY], dtype=np.float32).reshape(1, ACTION_DIM)
    return obs


def _reset_action_head_state(policy: GrootSimPolicy) -> None:
    action_head = getattr(getattr(policy, "trained_model", None), "action_head", None)
    if action_head is None:
        return
    for attr, value in {
        "current_start_frame": 0,
        "language": None,
        "clip_feas": None,
        "ys": None,
        "kv_cache1": None,
        "kv_cache_neg": None,
        "crossattn_cache": None,
        "crossattn_cache_neg": None,
    }.items():
        if hasattr(action_head, attr):
            setattr(action_head, attr, value)


def _extract_action(act: Any, *, horizon: int) -> np.ndarray:
    if isinstance(act, dict) and ACTION_KEY in act:
        arr = act[ACTION_KEY]
    elif isinstance(act, dict) and "action" in act:
        arr = act["action"]
    elif hasattr(act, ACTION_KEY):
        arr = getattr(act, ACTION_KEY)
    else:
        try:
            arr = act[ACTION_KEY]
        except Exception as exc:
            raise KeyError(f"Could not find {ACTION_KEY!r} or concat action in model output") from exc
    if torch.is_tensor(arr):
        arr = arr.detach().cpu().float().numpy()
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[-1] < ACTION_DIM:
        raise ValueError(f"Expected action [T,>={ACTION_DIM}], got {arr.shape}")
    arr = arr[:, :ACTION_DIM]
    if arr.shape[0] < horizon:
        arr = np.concatenate([arr, np.repeat(arr[-1:], horizon - arr.shape[0], axis=0)], axis=0)
    return arr[:horizon].astype(np.float32)


def _video_pred_starts_new_segment(policy: GrootSimPolicy, video_pred: Any, seen_chunks: int) -> bool:
    if video_pred is None or seen_chunks <= 0:
        return False
    action_head = getattr(getattr(policy, "trained_model", None), "action_head", None)
    num_frame_per_block = int(getattr(action_head, "num_frame_per_block", 1) or 1)
    current_start_frame = int(getattr(action_head, "current_start_frame", -1) or -1)
    shape = getattr(video_pred, "shape", ())
    latent_count = int(shape[2]) if len(shape) > 2 else -1
    return current_start_frame == 1 + num_frame_per_block or latent_count == 1 + num_frame_per_block


def _decode_video_chunks(policy: GrootSimPolicy, chunks: list[tuple[Any, bool]]) -> np.ndarray:
    action_head = policy.trained_model.action_head
    segments: list[list[Any]] = []
    current: list[Any] = []
    for tensor, starts_new in chunks:
        if starts_new and current:
            segments.append(current)
            current = []
        current.append(tensor)
    if current:
        segments.append(current)

    decoded: list[np.ndarray] = []
    with torch.inference_mode():
        for segment_index, segment in enumerate(segments):
            latents = torch.cat([tensor.detach().cpu().contiguous() for tensor in segment], dim=2).to(policy.device)
            frames = action_head.vae.decode(
                latents,
                tiled=action_head.tiled,
                tile_size=(action_head.tile_size_height, action_head.tile_size_width),
                tile_stride=(action_head.tile_stride_height, action_head.tile_stride_width),
            )
            frames = rearrange(frames, "B C T H W -> B T H W C")[0]
            frames = ((frames.float() + 1.0) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
            if segment_index > 0 and frames.shape[0] > 0:
                frames = frames[1:]
            decoded.append(frames)
    return np.concatenate(decoded, axis=0) if decoded else np.zeros((0, 1, 1, 3), dtype=np.uint8)


def _gt_video_episode(
    dataset: LeRobotSingleDataset,
    *,
    episode_id: int,
    episode_length: int,
    pred_frame_count: int,
    video_stride: int,
    preprocess: dict[str, Any],
) -> np.ndarray:
    indices = np.arange(0, episode_length, int(video_stride), dtype=int)
    if indices.size == 0 or indices[0] != 0:
        indices = np.concatenate([np.asarray([0], dtype=int), indices])
    indices = indices[:pred_frame_count]
    data = _get_step_data(dataset, episode_id, VIDEO_KEYS, indices)
    resize_hw = preprocess.get("resize_hw") or [176, 320]
    center_crop_scale = preprocess.get("center_crop_scale", 0.95)
    return _compose_unitree_views(
        {key: np.asarray(data[key], dtype=np.uint8) for key in VIDEO_KEYS},
        per_view_height=int(resize_hw[0]),
        per_view_width=int(resize_hw[1]),
        center_crop_scale=float(center_crop_scale) if center_crop_scale is not None else None,
    )


def _gt_actions_episode(dataset: LeRobotSingleDataset, *, episode_id: int, episode_length: int) -> np.ndarray:
    data = _get_step_data(dataset, episode_id, [ACTION_KEY], np.arange(0, episode_length, dtype=int))
    return np.asarray(data[ACTION_KEY], dtype=np.float32).reshape(episode_length, ACTION_DIM)


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
    frames = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    for p, g in zip(pred, gt):
        canvas = np.concatenate([p, g], axis=1)
        cv2.putText(canvas, "PRED", (12, 28), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "GT", (p.shape[1] + 12, 28), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        frames.append(canvas)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=int(fps), codec="libx264")


def _action_dim_names() -> list[str]:
    names = [f"base_pose[{i}]" for i in range(BASE_DIM)]
    names += [f"robot_q_desired[{i}]" for i in range(JOINT_DIM)]
    names += [f"hand_cmd[{i}]" for i in range(HAND_DIM)]
    names += [f"ee_state[{i}]" for i in range(EE_DIM)]
    return names


def _action_group_mse(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    diff2 = (pred - gt) ** 2
    return {
        "base_pose": float(np.mean(diff2[:, 0:BASE_DIM])),
        "robot_q": float(np.mean(diff2[:, BASE_DIM : BASE_DIM + JOINT_DIM])),
        "hand": float(np.mean(diff2[:, BASE_DIM + JOINT_DIM : BASE_DIM + JOINT_DIM + HAND_DIM])),
        "ee_state": float(np.mean(diff2[:, BASE_DIM + JOINT_DIM + HAND_DIM :])),
    }


def save_action_plot(path: Path, *, pred: np.ndarray, gt: np.ndarray, title: str) -> None:
    dim_names = _action_dim_names()
    dim = gt.shape[-1]
    ncols = 4
    nrows = int(math.ceil(dim / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 2.3 * nrows), squeeze=False)
    fig.suptitle(title, fontsize=12)
    x = np.arange(gt.shape[0])
    for i in range(dim):
        ax = axes[i // ncols][i % ncols]
        ax.plot(x, gt[:, i], label="gt", linewidth=1.0)
        ax.plot(x, pred[:, i], label="pred", linewidth=0.9)
        ax.set_title(dim_names[i], fontsize=8)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)
    for i in range(dim, nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)
    fig.tight_layout(rect=[0, 0.01, 1, 0.96])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _video_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
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


def _select_prompt(prompt: str, *, mode: str, rng: random.Random) -> str:
    variants = [part.strip() for part in str(prompt).split("@") if part.strip()]
    if not variants:
        return str(prompt)
    return rng.choice(variants) if mode == "random" else variants[0]


def evaluate(args: argparse.Namespace) -> None:
    dynamo_config = _configure_torch_dynamo(
        torch,
        recompile_limit=args.torch_dynamo_recompile_limit,
        cache_size_limit=args.torch_dynamo_cache_size_limit,
    )
    if dynamo_config:
        print(f"torch._dynamo config: {dynamo_config}")

    # Some model internals use plain `.to("cuda")`; bind current CUDA device first
    # so those tensors land on the user-requested GPU rather than cuda:0.
    if str(args.device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested device={args.device}, but CUDA is not available")
        target_device = torch.device(args.device)
        torch.cuda.set_device(target_device)
        print(f"Using CUDA device: {torch.cuda.current_device()} ({torch.cuda.get_device_name(torch.cuda.current_device())})")

    _maybe_init_dist()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    overrides = list(args.model_config_override or [])
    if not any(item.startswith("action_head_cfg.config.load_text_encoder=") for item in overrides):
        overrides.append("action_head_cfg.config.load_text_encoder=false")
    if not any(item.startswith("action_head_cfg.config.torch_compile_dit_blocks=") for item in overrides):
        overrides.append("action_head_cfg.config.torch_compile_dit_blocks=false")
    if args.t5_cache_dir and not any(item.startswith("text_embedding_cache_dir=") for item in overrides):
        overrides.append(f"text_embedding_cache_dir={args.t5_cache_dir}")
    if args.t5_cache_dir and not any(item.startswith("require_text_embedding_cache=") for item in overrides):
        overrides.append("require_text_embedding_cache=true")
    if args.t5_cache_dir and not any(item.startswith("text_embedding_cache_runtime=") for item in overrides):
        overrides.append("text_embedding_cache_runtime=model")

    print(f"Loading checkpoint: {args.model_path}")
    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag.UNITREE_G1_UPPER_BODY,
        model_path=args.model_path,
        device=args.device,
        model_config_overrides=overrides,
        tokenizer_path_override=args.tokenizer_path_override,
        skip_assert_delta_indices=True,
        skip_img_transform=False,
    )
    policy.trained_model.action_head.cfg_scale = float(args.cfg_scale)

    dataset = _load_dataset(args.dataset_path, video_backend=args.video_backend, action_horizon=args.action_horizon)
    tasks_by_episode = _read_episode_tasks(Path(args.dataset_path))
    episode_ids = _resolve_episode_ids(dataset, args)
    preprocess = _infer_client_video_preprocess_from_checkpoint(
        args.model_path,
        EmbodimentTag.UNITREE_G1_UPPER_BODY.value,
        VIDEO_KEYS,
    ) or {"center_crop_scale": 0.95, "resize_hw": [176, 320]}

    summaries: list[dict[str, Any]] = []
    for episode_id in episode_ids:
        episode_length = _episode_length(dataset, episode_id)
        prompt = _select_prompt(args.prompt or tasks_by_episode.get(int(episode_id), ""), mode=args.prompt_variant, rng=rng)
        if not prompt:
            raise ValueError(f"No prompt found for episode {episode_id}; pass --prompt")
        _reset_action_head_state(policy)

        pred_chunks: list[np.ndarray] = []
        video_chunks: list[tuple[Any, bool]] = []
        last_latent_video: torch.Tensor | None = None
        num_chunks = int(math.ceil(episode_length / args.action_horizon))
        for chunk_idx in range(num_chunks):
            anchor = chunk_idx * args.action_horizon
            obs = _build_model_obs(
                dataset,
                episode_id=episode_id,
                episode_length=episode_length,
                anchor=anchor,
                prompt=prompt,
                action_horizon=args.action_horizon,
                video_stride=args.video_stride,
            )
            if args.eval_mode == "teacher_forcing":
                _reset_action_head_state(policy)
                latent_video = None
            elif args.eval_mode == "open_loop":
                latent_video = last_latent_video
            else:
                latent_video = None

            with torch.inference_mode():
                result_batch, video_pred = policy.lazy_joint_forward_causal(Batch(obs=obs), latent_video=latent_video)
            pred_chunks.append(_extract_action(result_batch.act, horizon=args.action_horizon))
            if video_pred is not None:
                starts_new = args.eval_mode == "teacher_forcing" or _video_pred_starts_new_segment(
                    policy,
                    video_pred,
                    len(video_chunks),
                )
                video_chunks.append((video_pred.detach().cpu().contiguous(), starts_new))
                last_latent_video = video_pred.detach()
            if (chunk_idx + 1) % max(args.log_every, 1) == 0:
                print(f"episode={episode_id} mode={args.eval_mode} chunk={chunk_idx + 1}/{num_chunks}")

        episode_dir = output_dir / f"episode_{int(episode_id):06d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        pred_action = np.concatenate(pred_chunks, axis=0)[:episode_length]
        gt_action = _gt_actions_episode(dataset, episode_id=episode_id, episode_length=episode_length)
        np.savez_compressed(
            episode_dir / "actions.npz",
            pred=pred_action,
            gt=gt_action,
            action_key=ACTION_KEY,
            dim_names=np.asarray(_action_dim_names(), dtype=object),
        )
        action_mse = float(np.mean((pred_action - gt_action) ** 2))
        action_mae = float(np.mean(np.abs(pred_action - gt_action)))
        group_mse = _action_group_mse(pred_action, gt_action)
        save_action_plot(
            episode_dir / "action_pred_vs_gt.png",
            pred=pred_action,
            gt=gt_action,
            title=(
                f"stack-blocks episode {episode_id} action | MSE={action_mse:.5g} "
                f"| q={group_mse['robot_q']:.4g} hand={group_mse['hand']:.4g}"
            ),
        )

        pred_video = _decode_video_chunks(policy, video_chunks)
        np.save(episode_dir / "pred_video_frames.npy", pred_video)
        imageio.mimsave(episode_dir / "pred_video.mp4", list(pred_video), fps=int(args.video_fps), codec="libx264")
        gt_video = _gt_video_episode(
            dataset,
            episode_id=episode_id,
            episode_length=episode_length,
            pred_frame_count=pred_video.shape[0],
            video_stride=args.video_stride,
            preprocess=preprocess,
        )
        save_video_compare(episode_dir / "video_pred_gt_side_by_side.mp4", pred=pred_video, gt=gt_video, fps=args.video_fps)
        metrics = {
            "action_mse": action_mse,
            "action_mae": action_mae,
            "action_group_mse": group_mse,
            **_video_metrics(pred_video, gt_video),
        }
        summary = {
            "episode_id": int(episode_id),
            "episode_length": int(episode_length),
            "num_chunks": int(num_chunks),
            "prompt": prompt,
            "metrics": metrics,
            "artifacts": {
                "actions_npz": str(episode_dir / "actions.npz"),
                "action_plot": str(episode_dir / "action_pred_vs_gt.png"),
                "pred_video_mp4": str(episode_dir / "pred_video.mp4"),
                "pred_video_frames_npy": str(episode_dir / "pred_video_frames.npy"),
                "video_compare_mp4": str(episode_dir / "video_pred_gt_side_by_side.mp4"),
            },
            "config": {
                "eval_mode": args.eval_mode,
                "action_horizon": int(args.action_horizon),
                "video_stride": int(args.video_stride),
                "preprocess": preprocess,
                "model_config_override": overrides,
            },
        }
        (episode_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        summaries.append(summary)
        print(
            f"episode={episode_id} action_mse={action_mse:.6g} "
            f"q_mse={group_mse['robot_q']:.6g} hand_mse={group_mse['hand']:.6g} "
            f"video_mae={metrics['video_mae']:.6g} artifacts={episode_dir}"
        )

    (output_dir / "summary.json").write_text(json.dumps({"episodes": summaries}, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-path", "--model_path", dest="model_path", required=True)
    parser.add_argument("--dataset-path", "--dataset_path", dest="dataset_path", required=True)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episode-ids", "--episode_ids", dest="episode_ids", nargs="*", type=int, default=None)
    parser.add_argument("--episode-num", "--episode_num", dest="episode_num", type=int, default=1)
    parser.add_argument("--action-horizon", "--action_horizon", dest="action_horizon", type=int, default=48)
    parser.add_argument("--video-stride", "--video_stride", dest="video_stride", type=int, default=6)
    parser.add_argument("--video-fps", "--video_fps", dest="video_fps", type=int, default=5)
    parser.add_argument("--video-backend", "--video_backend", dest="video_backend", default="decord")
    parser.add_argument(
        "--eval-mode",
        "--eval_mode",
        dest="eval_mode",
        choices=["causal_gt", "teacher_forcing", "open_loop"],
        default="open_loop",
        help=(
            "causal_gt keeps model KV/cache across chunks while reading GT video windows; "
            "teacher_forcing resets cache each chunk and reads GT video windows; "
            "open_loop feeds the previous predicted latent video into the next chunk."
        ),
    )
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-variant", choices=["first", "random"], default="first")
    parser.add_argument("--cfg-scale", "--cfg_scale", dest="cfg_scale", type=float, default=1.0)
    parser.add_argument(
        "--t5-cache-dir",
        "--t5_cache_dir",
        dest="t5_cache_dir",
        default="/mnt/raid0/dreamzero_models/t5_cache/unitree_stack_blocks_50eps",
    )
    parser.add_argument("--tokenizer-path-override", default=None)
    parser.add_argument("--model-config-override", action="append", default=[])
    parser.add_argument("--log-every", "--log_every", dest="log_every", type=int, default=1)
    parser.add_argument("--torch-dynamo-recompile-limit", type=int, default=800)
    parser.add_argument("--torch-dynamo-cache-size-limit", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.action_horizon <= 0:
        raise ValueError("--action-horizon must be > 0")
    if args.video_stride <= 0:
        raise ValueError("--video-stride must be > 0")
    if args.action_horizon % args.video_stride != 0:
        raise ValueError("--video-stride must exactly divide --action-horizon")
    evaluate(args)


if __name__ == "__main__":
    main()
