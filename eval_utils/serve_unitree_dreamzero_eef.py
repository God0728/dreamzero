#!/usr/bin/env python3
"""Serve DreamZero Unitree-G1 EEF-RPY policy for the unmodified robotdeploy client.

This file intentionally implements the websocket protocol used by
``unitree_rl.policy.policy_client.WebsocketClientPolicy`` in robotdeploy:

- On websocket connection, send msgpack metadata containing ``data_keys``,
  ``state_chunk_size`` and ``action_chunk_size``.
- Receive msgpack messages shaped as ``{"type": "policy_reset"}`` or
  ``{"type": "get_action", "obs": obs}``.
- Return per-key action chunks shaped ``[1, T, D]`` for EEF-RPY control:
  ``action.left_ee_rpy``, ``action.right_ee_rpy``, ``action.left_gripper`` and
  ``action.right_gripper``.

Internally the Unitree DreamZero checkpoint uses dataset/model keys named
``*_ee_pose_gripper_base``.  The unmodified robotdeploy client uses
``*_ee_rpy`` keys for the same 6D ``[x, y, z, roll, pitch, yaw]`` values, so
this server intentionally renames between the two key contracts at the
websocket boundary.

The prompt is configured server-side via ``--prompt`` or ``--prompt-file`` so
no robotdeploy client code needs to change.

For CFG-parallel DreamZero inference, launch with ``torchrun``:

```
torchrun --nproc_per_node=2 source/dreamzero/eval_utils/serve_unitree_dreamzero_eef.py \
  --checkpoint /path/to/checkpoint --prompt "..." --port 8165
```

Rank 0 owns the websocket protocol. Rank 1 stays in a worker loop and
participates in ``lazy_joint_forward_causal`` through DreamZero's
``device_mesh`` path, matching ``socket_test_optimized_AR.py``.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as _datetime
import http
import json
import logging
import os
import pickle
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Iterable

import numpy as np

DREAMZERO_ROOT = Path(__file__).resolve().parents[1]
if str(DREAMZERO_ROOT) not in sys.path:
    sys.path.insert(0, str(DREAMZERO_ROOT))

DEFAULT_VIEW_MAP = (
    "video.head_stereo_left=observation.images.cam_left_high",
    "video.wrist_left=observation.images.cam_left_wrist",
    "video.wrist_right=observation.images.cam_right_wrist",
)
DEFAULT_STATE_MAP = (
    "state.left_ee_pose_gripper_base=observation.state.left_ee_rpy",
    "state.right_ee_pose_gripper_base=observation.state.right_ee_rpy",
    "state.left_gripper=observation.state.left_gripper",
    "state.right_gripper=observation.state.right_gripper",
)
DEFAULT_MODEL_ACTION_KEYS = (
    "action.left_ee_pose_gripper_base",
    "action.right_ee_pose_gripper_base",
    "action.left_gripper",
    "action.right_gripper",
)
DEFAULT_ROBOT_ACTION_KEYS = (
    "action.left_ee_rpy",
    "action.right_ee_rpy",
    "action.left_gripper",
    "action.right_gripper",
)
MODEL_TO_ROBOT_ACTION_KEY = {
    "action.left_ee_pose_gripper_base": "action.left_ee_rpy",
    "action.right_ee_pose_gripper_base": "action.right_ee_rpy",
    "action.left_gripper": "action.left_gripper",
    "action.right_gripper": "action.right_gripper",
}
MODEL_ACTION_DIMS = {
    "action.left_ee_pose_gripper_base": 6,
    "action.right_ee_pose_gripper_base": 6,
    "action.left_gripper": 1,
    "action.right_gripper": 1,
}
ROBOT_ACTION_DIMS = {
    "action.left_ee_rpy": 6,
    "action.right_ee_rpy": 6,
    "action.left_gripper": 1,
    "action.right_gripper": 1,
}
LEGACY_ROBOT_STATE_TO_MODEL_STATE = {
    "state.left_ee_rpy": "state.left_ee_pose_gripper_base",
    "state.right_ee_rpy": "state.right_ee_pose_gripper_base",
}
DEFAULT_TEXT_ENCODER_OVERRIDE = "action_head_cfg.config.load_text_encoder=true"
IMAGE_TRANSPORT_SENTINEL = "__dreamzero_image_transport__"
LOG = logging.getLogger(__name__)
SIGNAL_INFER = 0
SIGNAL_SHUTDOWN = 1
SIGNAL_RESET = 2
INTERNAL_RECORD_PREDICTION_KEY = "__dreamzero_record_prediction__"


def _safe_path_part(value: Any, fallback: str) -> str:
    text = str(value or fallback)
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)
    return cleaned.strip("._-") or fallback


def _timestamp_session_id(requested: Any = None) -> str:
    stamp = _datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    suffix = _safe_path_part(requested, "session")
    return f"{stamp}_{suffix}"


def _flatten_action_chunk(value: Any, *, key: str) -> np.ndarray:
    arr = _to_numpy(value).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    elif arr.ndim == 1:
        dim = ROBOT_ACTION_DIMS.get(key, arr.shape[-1])
        arr = arr.reshape(-1, dim)
    if arr.ndim != 2:
        raise ValueError(f"{key} action chunk must be [T,D] or [B,T,D], got {arr.shape}")
    return np.ascontiguousarray(arr)


def _action_dim_names(action_keys: Iterable[str], dims: dict[str, int]) -> list[str]:
    names: list[str] = []
    for key in action_keys:
        base = key.replace("action.", "")
        for idx in range(int(dims[key])):
            names.append(f"{base}.{idx}")
    return names


def _save_action_curve(
    path: Path,
    *,
    actions: np.ndarray,
    dim_names: list[str],
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dim = int(actions.shape[-1])
    ncols = min(4, max(dim, 1))
    nrows = int(np.ceil(dim / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 2.6 * nrows), squeeze=False)
    fig.suptitle(title, fontsize=12)
    x = np.arange(actions.shape[0])
    for idx in range(dim):
        ax = axes[idx // ncols][idx % ncols]
        ax.plot(x, actions[:, idx], label="pred", linewidth=1.0)
        ax.set_title(dim_names[idx] if idx < len(dim_names) else f"dim{idx}", fontsize=8)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=7)
    for idx in range(dim, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    fig.tight_layout(rect=[0, 0.01, 1, 0.96])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _add_video_watermarks(frames: np.ndarray, chunks: list[tuple[Any, int]]) -> np.ndarray:
    """Annotate decoded prediction frames by causal chunk, matching LIBERO style."""

    if not chunks:
        return frames
    import cv2

    out = np.ascontiguousarray(frames.copy())
    total, height, width = out.shape[:3]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(min(height, width) / 360.0, 0.45)
    thick = max(1, int(round(scale * 2)))
    pad = max(4, int(round(6 * scale)))
    margin = max(6, int(round(8 * scale)))
    latent_counts = [int(tensor.shape[2]) for tensor, _ in chunks]
    total_latents = max(sum(latent_counts), 1)
    wan_map = 4 * sum(latent_counts) - 3 == total
    frame_start = 0
    latent_cumsum = 0
    for index, ((_, decision), latent_count) in enumerate(zip(chunks, latent_counts)):
        latent_cumsum += latent_count
        if wan_map:
            frame_end = frame_start + (4 * latent_count - 3 if index == 0 else 4 * latent_count)
        else:
            frame_end = total if index == len(chunks) - 1 else round(total * latent_cumsum / total_latents)
        label = f"CHUNK {decision:03d}"
        (tw, th), base = cv2.getTextSize(label, font, scale, thick)
        x0 = max(width - tw - 2 * pad - margin, 0)
        y0 = max(height - th - base - 2 * pad - margin, 0)
        x1 = min(width - margin, width)
        y1 = min(height - margin, height)
        for frame in out[frame_start:min(frame_end, total)]:
            roi = frame[y0:y1, x0:x1]
            cv2.addWeighted(roi, 0.38, roi, 0.0, 0, dst=roi)
            cv2.putText(
                frame,
                label,
                (x0 + pad, y1 - pad - base),
                font,
                scale,
                (255, 255, 255),
                thick,
                cv2.LINE_AA,
            )
        frame_start = min(frame_end, total)
    return out


def _parse_key_map(items: Iterable[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in items:
        if "=" in item:
            left, right = item.split("=", 1)
        elif ":" in item:
            left, right = item.split(":", 1)
        else:
            raise ValueError(f"Invalid key map {item!r}; expected MODEL_KEY=CLIENT_KEY")
        left = left.strip()
        right = right.strip()
        if not left or not right:
            raise ValueError(f"Invalid empty key in map item {item!r}")
        mapping[left] = right
    if not mapping:
        raise ValueError("key map cannot be empty")
    return mapping


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


def _validate_relative_action_state_keys(
    *,
    relative_action_info: dict[str, Any],
    state_keys: Iterable[str],
) -> None:
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
            "Unitree relative-action serving requires matching current state keys "
            f"for delta-to-absolute conversion; missing={missing}, available={sorted(available)}"
        )


def _normalize_unitree_state_map(mapping: dict[str, str]) -> dict[str, str]:
    """Accept old EEF-RPY model key overrides but feed checkpoint keys."""

    normalized = dict(mapping)
    for legacy_key, model_key in LEGACY_ROBOT_STATE_TO_MODEL_STATE.items():
        if legacy_key not in normalized:
            continue
        client_key = normalized.pop(legacy_key)
        if model_key in normalized and normalized[model_key] != client_key:
            raise ValueError(
                f"Conflicting state-map entries for {legacy_key!r} and {model_key!r}: "
                f"{client_key!r} vs {normalized[model_key]!r}"
            )
        normalized[model_key] = client_key
        LOG.warning(
            "state-map used legacy model key %s; renamed it to checkpoint key %s",
            legacy_key,
            model_key,
        )
    return normalized


def _resolve_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        prompt = Path(args.prompt_file).expanduser().read_text().strip()
        if args.prompt and args.prompt.strip() and args.prompt.strip() != prompt:
            raise ValueError("Use either --prompt or --prompt-file, not both with different values")
        return prompt
    return (args.prompt or "").strip()


def _maybe_init_single_process_dist(timeout_seconds: int = 600) -> None:
    # GrootSimPolicy calls torch.distributed.get_rank() during construction.
    import torch.distributed as dist

    if dist.is_available() and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(
            backend="gloo",
            world_size=1,
            rank=0,
            timeout=_datetime.timedelta(seconds=int(timeout_seconds)),
        )


def _configure_torch_dynamo(
    torch_module: Any,
    *,
    recompile_limit: int,
    cache_size_limit: int,
) -> dict[str, Any]:
    """Match the official optimized AR server's higher Dynamo recompile budget.

    The DreamZero flow scheduler has ``@torch.compile(..., dynamic=False)`` on
    its UniPC update functions.  AR inference legitimately hits several tensor
    rank/shape variants (video latents vs action tensors, first frame vs later
    chunks), so PyTorch's default ``recompile_limit=8`` can abort serving before
    the first action chunk finishes.  ``socket_test_optimized_AR.py`` raises the
    recompile limit to 800 for the same reason; do that before model import/load.
    """

    dynamo_config = getattr(getattr(torch_module, "_dynamo", None), "config", None)
    if dynamo_config is None:
        return {}
    applied: dict[str, Any] = {}
    for name, value in {
        "recompile_limit": int(recompile_limit),
        "cache_size_limit": int(cache_size_limit),
        "accumulated_cache_size_limit": int(cache_size_limit),
        "accumulated_recompile_limit": max(int(recompile_limit) * 2, int(cache_size_limit)),
    }.items():
        if hasattr(dynamo_config, name):
            setattr(dynamo_config, name, value)
            applied[name] = getattr(dynamo_config, name)
    return applied


def _init_runtime(
    device: str,
    timeout_seconds: int,
    *,
    dynamo_recompile_limit: int,
    dynamo_cache_size_limit: int,
) -> dict[str, Any]:
    """Initialize DreamZero serving runtime.

    Single-process mode keeps the existing behavior. ``torchrun`` mode creates
    an NCCL process group plus a CUDA device mesh so the DreamZero action head
    can split CFG work across two ranks; a separate Gloo group carries small
    rank-0 control messages and serialized model observations.
    """

    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    dynamo_config = _configure_torch_dynamo(
        torch,
        recompile_limit=dynamo_recompile_limit,
        cache_size_limit=dynamo_cache_size_limit,
    )
    timeout = _datetime.timedelta(seconds=int(timeout_seconds))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        if world_size > 2:
            raise ValueError("Unitree DreamZero EEF serving supports at most 2 GPUs for CFG parallelism")
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed Unitree DreamZero EEF serving requires CUDA")
        torch.cuda.set_device(local_rank)
        runtime_device = f"cuda:{local_rank}"
        if not dist.is_initialized():
            try:
                dist.init_process_group(
                    backend="nccl",
                    device_id=torch.device(runtime_device),
                    timeout=timeout,
                )
            except TypeError:
                dist.init_process_group(backend="nccl", timeout=timeout)
        device_mesh = init_device_mesh(
            device_type="cuda",
            mesh_shape=(world_size,),
            mesh_dim_names=("ip",),
        )
        signal_group = dist.new_group(backend="gloo", timeout=timeout)
        return {
            "world_size": world_size,
            "rank": rank,
            "local_rank": local_rank,
            "device": runtime_device,
            "device_mesh": device_mesh,
            "signal_group": signal_group,
            "distributed": True,
            "dynamo_config": dynamo_config,
        }

    _maybe_init_single_process_dist(timeout_seconds)
    return {
        "world_size": 1,
        "rank": 0,
        "local_rank": 0,
        "device": device,
        "device_mesh": None,
        "signal_group": None,
        "distributed": False,
        "dynamo_config": dynamo_config,
    }


def _load_dreamzero_policy(
    *,
    checkpoint: str,
    embodiment_tag: str,
    device: str,
    device_mesh: Any = None,
    model_config_overrides: list[str],
    tokenizer_path_override: str | None,
    skip_assert_delta_indices: bool,
    skip_img_transform: bool,
):
    _maybe_init_single_process_dist()

    from groot.vla.data.schema import EmbodimentTag
    from groot.vla.model.n1_5.sim_policy import GrootSimPolicy

    try:
        embodiment = EmbodimentTag(embodiment_tag)
    except ValueError as exc:
        valid = [item.value for item in EmbodimentTag]
        raise ValueError(f"Unknown embodiment tag {embodiment_tag!r}; valid values include {valid}") from exc

    return GrootSimPolicy(
        embodiment_tag=embodiment,
        model_path=checkpoint,
        device=device,
        model_config_overrides=model_config_overrides,
        tokenizer_path_override=tokenizer_path_override,
        skip_assert_delta_indices=skip_assert_delta_indices,
        skip_img_transform=skip_img_transform,
        device_mesh=device_mesh,
    )


def _to_numpy(value: Any) -> np.ndarray:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().float().numpy()
    except Exception:  # pragma: no cover - torch may be unavailable during static checks
        pass
    return np.asarray(value)


def _latest_vector(value: Any, *, key: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim >= 2:
        arr = arr.reshape(-1, arr.shape[-1])[-1]
    if not np.isfinite(arr).all():
        raise ValueError(f"{key} contains non-finite values: {arr}")
    return np.ascontiguousarray(arr.reshape(1, -1)).copy()


def _decode_jpeg_image_payload(value: dict[str, Any], *, key: str) -> list[np.ndarray]:
    if value.get(IMAGE_TRANSPORT_SENTINEL) != "jpeg":
        raise ValueError(f"{key} has unsupported image transport payload: {value.get(IMAGE_TRANSPORT_SENTINEL)!r}")
    encoded_frames = value.get("frames")
    if not isinstance(encoded_frames, (list, tuple)):
        raise ValueError(f"{key} jpeg transport payload must contain a list of frames")

    import cv2

    frames: list[np.ndarray] = []
    for idx, encoded in enumerate(encoded_frames):
        if isinstance(encoded, memoryview):
            encoded = encoded.tobytes()
        if isinstance(encoded, bytearray):
            encoded = bytes(encoded)
        if not isinstance(encoded, (bytes, np.ndarray)):
            raise ValueError(f"{key} jpeg frame {idx} has invalid type {type(encoded)!r}")
        buf = np.frombuffer(encoded, dtype=np.uint8) if isinstance(encoded, bytes) else np.asarray(encoded, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"{key} jpeg frame {idx} failed to decode")
        frames.append(np.ascontiguousarray(frame[..., :3]).copy())

    expected_shape = value.get("shape")
    if expected_shape is not None:
        expected = tuple(int(x) for x in expected_shape)
        if len(expected) == 3:
            expected = (1, *expected)
        actual = (len(frames), *frames[0].shape) if frames else (0,)
        if len(expected) == 4 and actual != expected:
            raise ValueError(f"{key} decoded jpeg shape {actual} does not match declared shape {expected}")
    return frames


def _as_frames(value: Any, *, key: str) -> list[np.ndarray]:
    if isinstance(value, dict) and value.get(IMAGE_TRANSPORT_SENTINEL):
        return _decode_jpeg_image_payload(value, key=key)

    arr = np.asarray(value)
    if arr.ndim == 3:
        frames = [arr]
    elif arr.ndim == 4:
        frames = [arr[i] for i in range(arr.shape[0])]
    else:
        raise ValueError(f"{key} must be HWC or THWC image data, got shape {arr.shape}")
    output = []
    for frame in frames:
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[-1] not in (1, 3, 4):
            raise ValueError(f"{key} frame must be HWC image data, got shape {frame.shape}")
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        output.append(np.ascontiguousarray(frame[..., :3]).copy())
    return output


def _resize_frame(frame: np.ndarray, size_hw: tuple[int, int] | None) -> np.ndarray:
    if size_hw is None:
        return frame
    height, width = size_hw
    if frame.shape[0] == height and frame.shape[1] == width:
        return frame
    import cv2

    return cv2.resize(frame, (int(width), int(height)), interpolation=cv2.INTER_AREA)


def _infer_client_video_preprocess_from_checkpoint(
    checkpoint: str,
    embodiment_tag: str,
    model_view_keys: Iterable[str],
) -> dict[str, Any]:
    """Read deterministic eval crop/resize from the checkpoint training config.

    This lets the server publish the image preprocessing contract through
    websocket metadata, so robotdeploy clients do not need hard-coded per-model
    image sizes.  If the config cannot be read, return an empty dict and let the
    caller decide whether that is fatal.
    """

    conf_path = Path(checkpoint).expanduser() / "experiment_cfg" / "conf.yaml"
    if not conf_path.exists():
        return {}
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(conf_path)
        transforms = cfg.transforms[embodiment_tag].transforms
    except Exception as exc:
        LOG.warning("Could not inspect %s for client image preprocessing metadata: %s", conf_path, exc)
        return {}

    wanted = set(model_view_keys)
    crop_scale: float | None = None
    resize_hw: tuple[int, int] | None = None
    for transform in transforms:
        target = str(transform.get("_target_", ""))
        apply_to = set(transform.get("apply_to", []) or [])
        if apply_to and not (apply_to & wanted):
            continue
        if target.endswith("VideoCrop") and "scale" in transform:
            crop_scale = float(transform.scale)
        elif target.endswith("VideoResize") and "height" in transform and "width" in transform:
            resize_hw = (int(transform.height), int(transform.width))
    result: dict[str, Any] = {}
    if crop_scale is not None:
        result["center_crop_scale"] = crop_scale
    if resize_hw is not None:
        result["resize_hw"] = list(resize_hw)
    return result


def _action_value(act: Any, key: str) -> Any:
    if isinstance(act, dict) and key in act:
        return act[key]
    try:
        return act[key]
    except Exception:
        pass
    if hasattr(act, key):
        return getattr(act, key)
    raise KeyError(key)


def _contains_action_key(act: Any, key: str) -> bool:
    if isinstance(act, dict):
        return key in act
    try:
        act[key]
        return True
    except Exception:
        return hasattr(act, key)


def _normalize_chunk(value: Any, *, key: str, horizon: int, dim: int) -> np.ndarray:
    arr = _to_numpy(value).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    elif arr.ndim == 1:
        if arr.size % dim != 0:
            raise ValueError(f"{key} flat action size {arr.size} is not divisible by dim {dim}")
        arr = arr.reshape(-1, dim)
    if arr.ndim != 2 or arr.shape[-1] != dim:
        raise ValueError(f"{key} must have shape [T,{dim}] or [B,T,{dim}], got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{key} contains non-finite values")
    if arr.shape[0] < horizon:
        pad = np.repeat(arr[-1:], horizon - arr.shape[0], axis=0)
        arr = np.concatenate([arr, pad], axis=0)
    elif arr.shape[0] > horizon:
        arr = arr[:horizon]
    return arr.reshape(1, horizon, dim).astype(np.float32)


class UnitreePredictionRecorder:
    """Session-scoped recorder for Unitree DreamZero predicted video/action."""

    def __init__(
        self,
        output_root: str | None,
        *,
        fps: int = 5,
        watermark: bool = True,
        save_video_npy: bool = False,
    ):
        self.output_root = Path(output_root).expanduser() if output_root else None
        self.fps = int(fps)
        self.watermark = bool(watermark)
        self.save_video_npy = bool(save_video_npy)
        self.session_id: str | None = None
        self.session_dir: Path | None = None
        self.video_chunks: list[tuple[Any, int, bool]] = []
        self.action_chunks: list[tuple[dict[str, np.ndarray], int]] = []
        self.last_artifacts: dict[str, Any] = {}

    @property
    def enabled(self) -> bool:
        return self.output_root is not None

    def start_session(self, requested_session_id: Any = None) -> str | None:
        self.clear_buffers()
        self.last_artifacts = {}
        if self.output_root is None:
            self.session_id = None
            self.session_dir = None
            return None
        self.session_id = _timestamp_session_id(requested_session_id)
        self.session_dir = self.output_root / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        (self.session_dir / "session.json").write_text(
            json.dumps({"session_id": self.session_id, "created_at": self.session_id[:22]}, indent=2)
        )
        return self.session_id

    def clear_buffers(self) -> None:
        self.video_chunks.clear()
        self.action_chunks.clear()

    def append(
        self,
        *,
        actions: dict[str, np.ndarray],
        video_pred: Any,
        decision: int,
        starts_new_video_segment: bool = False,
    ) -> None:
        if self.output_root is None:
            return
        action_copy = {
            key: _flatten_action_chunk(value, key=key).copy()
            for key, value in actions.items()
        }
        self.action_chunks.append((action_copy, int(decision)))
        if video_pred is not None:
            try:
                tensor = video_pred.detach().cpu().contiguous()
            except AttributeError:
                tensor = video_pred
            self.video_chunks.append((tensor, int(decision), bool(starts_new_video_segment)))

    def _save_actions(self) -> dict[str, Any]:
        assert self.session_dir is not None
        if not self.action_chunks:
            return {}
        keys = list(DEFAULT_ROBOT_ACTION_KEYS)
        by_key: dict[str, np.ndarray] = {}
        dims: dict[str, int] = {}
        for key in keys:
            chunks = [chunk[key] for chunk, _ in self.action_chunks if key in chunk]
            if not chunks:
                continue
            by_key[key] = np.concatenate(chunks, axis=0).astype(np.float32)
            dims[key] = int(by_key[key].shape[-1])
        if not by_key:
            return {}

        concat = np.concatenate([by_key[key] for key in keys if key in by_key], axis=-1)
        npz_payload = {
            "action_concat": concat,
            "chunk_decisions": np.asarray([decision for _, decision in self.action_chunks], dtype=np.int32),
        }
        for key, value in by_key.items():
            npz_payload[_safe_path_part(key, key).replace(".", "_")] = value
        npz_path = self.session_dir / "pred_actions.npz"
        np.savez_compressed(npz_path, **npz_payload)

        dim_names = _action_dim_names([key for key in keys if key in by_key], dims)
        plot_path = self.session_dir / "pred_actions.png"
        _save_action_curve(
            plot_path,
            actions=concat,
            dim_names=dim_names,
            title=f"Predicted Unitree actions | {self.session_id}",
        )
        json_path = self.session_dir / "pred_actions.json"
        json_path.write_text(
            json.dumps(
                {
                    "path": str(npz_path),
                    "plot_path": str(plot_path),
                    "num_steps": int(concat.shape[0]),
                    "action_keys": [key for key in keys if key in by_key],
                    "dims": dims,
                    "chunks": [int(decision) for _, decision in self.action_chunks],
                },
                indent=2,
            )
        )
        return {
            "pred_action_npz": str(npz_path),
            "pred_action_plot": str(plot_path),
            "pred_action_json": str(json_path),
            "pred_action_steps": int(concat.shape[0]),
        }

    def _save_video(self, policy: Any) -> dict[str, Any]:
        assert self.session_dir is not None
        if not self.video_chunks:
            return {}
        import imageio.v2 as imageio
        import torch
        from einops import rearrange

        action_head = policy.trained_model.action_head
        latent_segments: list[list[tuple[Any, int, bool]]] = []
        current_segment: list[tuple[Any, int, bool]] = []
        for item in self.video_chunks:
            _, _, starts_new_segment = item
            if starts_new_segment and current_segment:
                latent_segments.append(current_segment)
                current_segment = []
            current_segment.append(item)
        if current_segment:
            latent_segments.append(current_segment)

        decoded_segments: list[np.ndarray] = []
        with torch.inference_mode():
            for segment_index, segment in enumerate(latent_segments):
                latents = torch.cat([tensor for tensor, _, _ in segment], dim=2).to(policy.device)
                segment_frames = action_head.vae.decode(
                    latents,
                    tiled=action_head.tiled,
                    tile_size=(action_head.tile_size_height, action_head.tile_size_width),
                    tile_stride=(action_head.tile_stride_height, action_head.tile_stride_width),
                )
                segment_frames = rearrange(segment_frames, "B C T H W -> B T H W C")[0]
                segment_frames = ((segment_frames.float() + 1.0) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
                # A hard-reset chunk includes a fresh boundary latent. Decode it as
                # a new Wan VAE segment, then drop the duplicated boundary frame
                # before concatenating raw frames back onto the episode video.
                if segment_index > 0 and segment_frames.shape[0] > 0:
                    segment_frames = segment_frames[1:]
                decoded_segments.append(segment_frames)
        frames = np.concatenate(decoded_segments, axis=0)
        raw_npy_path = None
        if self.save_video_npy:
            raw_npy_path = self.session_dir / "pred_video_frames.npy"
            np.save(raw_npy_path, frames)
        watermark_chunks = [(tensor, decision) for tensor, decision, _ in self.video_chunks]
        frames_for_mp4 = _add_video_watermarks(frames, watermark_chunks) if self.watermark else frames
        mp4_path = self.session_dir / "pred_video.mp4"
        imageio.mimsave(mp4_path, list(frames_for_mp4), fps=self.fps, codec="libx264")
        json_path = self.session_dir / "pred_video.json"
        video_json: dict[str, Any] = {
            "path": str(mp4_path),
            "num_frames": int(frames.shape[0]),
            "fps": int(self.fps),
            "latent_shapes": [list(tensor.shape) for tensor, _, _ in self.video_chunks],
            "chunks": [int(decision) for _, decision, _ in self.video_chunks],
            "latent_segments": [
                [int(decision) for _, decision, _ in segment]
                for segment in latent_segments
            ],
            "dropped_segment_boundary_frames": max(len(latent_segments) - 1, 0),
            "watermark": bool(self.watermark),
            "save_predicted_video_npy": bool(raw_npy_path is not None),
        }
        if raw_npy_path is not None:
            video_json["frames_npy"] = str(raw_npy_path)
        json_path.write_text(
            json.dumps(video_json, indent=2)
        )
        result = {
            "pred_video_mp4": str(mp4_path),
            "pred_video_json": str(json_path),
            "pred_video_frames": int(frames.shape[0]),
        }
        if raw_npy_path is not None:
            result["pred_video_frames_npy"] = str(raw_npy_path)
        return result

    def flush(self, policy: Any) -> dict[str, Any]:
        if self.output_root is None or self.session_dir is None:
            self.clear_buffers()
            return {}
        if not self.action_chunks and not self.video_chunks:
            return dict(self.last_artifacts)
        artifacts: dict[str, Any] = {
            "session_id": self.session_id,
            "session_dir": str(self.session_dir),
        }
        try:
            artifacts.update(self._save_actions())
        except Exception as exc:  # noqa: BLE001
            LOG.warning("failed to save Unitree predicted actions: %s", exc)
            artifacts["pred_action_error"] = str(exc)
        try:
            artifacts.update(self._save_video(policy))
        except Exception as exc:  # noqa: BLE001
            LOG.warning("failed to save Unitree predicted video: %s", exc)
            artifacts["pred_video_error"] = str(exc)
        self.last_artifacts = artifacts
        if self.session_dir is not None:
            (self.session_dir / "artifacts.json").write_text(json.dumps(artifacts, indent=2))
        self.clear_buffers()
        return artifacts


class UnitreeDreamZeroEEFPolicy:
    def __init__(self, args: argparse.Namespace, *, device_mesh: Any = None, world_size: int = 1):
        self.args = args
        self.world_size = int(world_size)
        self.prompt = _resolve_prompt(args)
        if not self.prompt:
            LOG.warning("No prompt configured; use --prompt or --prompt-file for language-conditioned checkpoints")
        self.view_map = _parse_key_map(args.view_map or DEFAULT_VIEW_MAP)
        self.state_map = _normalize_unitree_state_map(_parse_key_map(args.state_map or DEFAULT_STATE_MAP))
        self.model_action_keys = tuple(args.action_keys or DEFAULT_MODEL_ACTION_KEYS)
        if self.model_action_keys == DEFAULT_ROBOT_ACTION_KEYS:
            LOG.warning(
                "--action-keys was set to robotdeploy response keys %s; "
                "using checkpoint model keys %s internally instead",
                DEFAULT_ROBOT_ACTION_KEYS,
                DEFAULT_MODEL_ACTION_KEYS,
            )
            self.model_action_keys = DEFAULT_MODEL_ACTION_KEYS
        if set(self.model_action_keys) != set(DEFAULT_MODEL_ACTION_KEYS):
            raise ValueError(
                "Unitree DreamZero EEF mode expects checkpoint action keys "
                f"{DEFAULT_MODEL_ACTION_KEYS}, got {self.model_action_keys}"
            )
        self.robot_action_keys = DEFAULT_ROBOT_ACTION_KEYS
        self.action_horizon = int(args.action_horizon)
        self.obs_chunk_size = int(args.obs_chunk_size or (self.action_horizon + 1))
        self.initial_frames = int(args.initial_frames)
        self.subsequent_frames = int(args.subsequent_frames)
        self.include_current_boundary = bool(args.include_current_boundary)
        self.image_resize = tuple(args.resize_images) if args.resize_images else None
        self.client_image_color_order = str(args.client_image_color_order).lower()
        self.client_image_preprocess = self._resolve_client_image_preprocess()
        self.client_jpeg_quality = int(args.client_jpeg_quality or 0)
        self.policy = _load_dreamzero_policy(
            checkpoint=args.checkpoint,
            embodiment_tag=args.embodiment_tag,
            device=args.device,
            device_mesh=device_mesh,
            model_config_overrides=list(args.model_config_override or []),
            tokenizer_path_override=args.tokenizer_path_override,
            skip_assert_delta_indices=args.skip_assert_delta_indices,
            skip_img_transform=args.skip_img_transform,
        )
        if args.cfg_scale is not None:
            self.policy.trained_model.action_head.cfg_scale = float(args.cfg_scale)
            LOG.info("Set action_head.cfg_scale=%s", self.policy.trained_model.action_head.cfg_scale)
        if self.world_size > 1 and float(self.policy.trained_model.action_head.cfg_scale) == 1.0:
            raise ValueError("2-GPU CFG-parallel Unitree serving requires cfg_scale != 1.0.")
        self.relative_action_info = _relative_action_info(self.policy)
        _validate_relative_action_state_keys(
            relative_action_info=self.relative_action_info,
            state_keys=self.state_map.keys(),
        )
        self.is_first_request = True
        self.infer_count = 0
        save_predictions = bool(args.save_predictions) if args.save_predictions is not None else bool(args.output_dir)
        output_root = args.output_dir if save_predictions else None
        self.recorder = UnitreePredictionRecorder(
            output_root,
            fps=int(args.predicted_video_fps),
            watermark=bool(args.predicted_video_watermark),
            save_video_npy=bool(args.save_predicted_video_npy),
        )
        self.current_session_id = None
        self.current_session_request = None
        self._start_new_recording_session("server_start")
        self._reset_action_head_state()

    def _resolve_client_image_preprocess(self) -> dict[str, Any]:
        if self.args.client_resize_images:
            if not self.args.skip_img_transform:
                raise ValueError("--client-resize-images must be used with --skip-img-transform to avoid double image resize")
            resize_hw = [int(self.args.client_resize_images[0]), int(self.args.client_resize_images[1])]
            preprocess: dict[str, Any] = {"resize_hw": resize_hw}
            if self.args.client_center_crop_scale is not None:
                preprocess["center_crop_scale"] = float(self.args.client_center_crop_scale)
            return preprocess

        if self.args.client_center_crop_scale is not None and not self.args.skip_img_transform:
            raise ValueError("--client-center-crop-scale must be used with --skip-img-transform to avoid double image crop")

        if not self.args.skip_img_transform:
            return {}

        preprocess = _infer_client_video_preprocess_from_checkpoint(
            self.args.checkpoint,
            self.args.embodiment_tag,
            self.view_map.keys(),
        )
        if self.args.client_center_crop_scale is not None:
            preprocess["center_crop_scale"] = float(self.args.client_center_crop_scale)
        if not preprocess.get("resize_hw"):
            raise ValueError(
                "--skip-img-transform requires the server to publish client-side image preprocessing, "
                "but no VideoResize could be inferred from the checkpoint. Pass --client-resize-images HEIGHT WIDTH."
            )
        return preprocess

    def _client_subsequent_image_indices(self) -> list[int]:
        history_len = max(int(self.obs_chunk_size), 1)
        if history_len <= 1:
            return [0]
        offsets = self._sample_offsets(self.subsequent_frames)
        block_end = history_len - 1 if self.include_current_boundary else max(history_len - 2, 0)
        block_span = self.action_horizon if self.include_current_boundary else self.action_horizon - 1
        block_start = max(0, block_end - int(block_span))
        return [min(block_start + int(offset), block_end) for offset in offsets]

    def _observation_transport_metadata(self) -> dict[str, Any]:
        image_keys = list(self.view_map.values())
        state_keys = list(self.state_map.values())
        image_preprocess = {
            "enabled": bool(self.client_image_preprocess),
            "per_key": {key: dict(self.client_image_preprocess) for key in image_keys}
            if self.client_image_preprocess
            else {},
        }
        if self.client_image_preprocess:
            image_preprocess.update(self.client_image_preprocess)

        image_encoding: dict[str, Any] = {"type": "raw"}
        if self.client_jpeg_quality > 0:
            image_encoding = {
                "type": "jpeg",
                "quality": int(np.clip(self.client_jpeg_quality, 1, 100)),
                "format": "jpg",
            }

        return {
            "version": 1,
            "enabled": True,
            "history_size": int(self.obs_chunk_size),
            "image_keys": image_keys,
            "state_keys": state_keys,
            "initial_image_indices": [-1],
            "subsequent_image_indices": self._client_subsequent_image_indices(),
            "initial_state_indices": [-1],
            "subsequent_state_indices": [-1],
            "image_preprocess": image_preprocess,
            "image_encoding": image_encoding,
            "client_image_color_order": self.client_image_color_order,
            "model_image_color_order": "rgb",
            "server_skip_img_transform": bool(self.args.skip_img_transform),
        }

    @property
    def metadata(self) -> dict[str, Any]:
        observation_keys = sorted(set(self.view_map.values()) | set(self.state_map.values()))
        return {
            "policy_path": str(Path(self.args.checkpoint).expanduser()),
            "policy_type": "dreamzero_unitree_eef_rpy",
            "data_keys": [*observation_keys, *self.robot_action_keys],
            "state_chunk_size": int(self.obs_chunk_size),
            "obs_chunk_size": int(self.obs_chunk_size),
            "action_chunk_size": int(self.action_horizon),
            "model_view_keys": list(self.view_map.keys()),
            "model_state_keys": list(self.state_map.keys()),
            "model_action_keys": list(self.model_action_keys),
            "robot_action_keys": list(self.robot_action_keys),
            "model_to_robot_action_key": dict(MODEL_TO_ROBOT_ACTION_KEY),
            "embodiment_tag": self.args.embodiment_tag,
            "initial_frames": int(self.initial_frames),
            "subsequent_frames": int(self.subsequent_frames),
            "include_current_boundary": bool(self.include_current_boundary),
            "prompt": self.prompt,
            "world_size": int(self.world_size),
            "cfg_parallel": bool(self.world_size > 1),
            "cfg_scale": float(self.policy.trained_model.action_head.cfg_scale),
            "relative_action": dict(self.relative_action_info),
            "output_dir": str(self.recorder.output_root) if self.recorder.output_root else None,
            "current_session_id": self.current_session_id,
            "observation_transport": self._observation_transport_metadata(),
        }

    def _reset_action_head_state(self) -> None:
        action_head = getattr(getattr(self.policy, "trained_model", None), "action_head", None)
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

    def _local_reset(self) -> None:
        self.is_first_request = True
        self.infer_count = 0
        self._reset_action_head_state()

    def flush_outputs(self) -> dict[str, Any]:
        return self.recorder.flush(self.policy)

    def _start_new_recording_session(self, requested_session_id: Any = None) -> str | None:
        self.current_session_id = self.recorder.start_session(requested_session_id)
        self.current_session_request = None if requested_session_id is None else str(requested_session_id)
        return self.current_session_id

    def reset(self, reset_info: dict[str, Any] | None = None) -> dict[str, Any]:
        reset_info = reset_info or {}
        flushed = self.flush_outputs()
        if "prompt" in reset_info and reset_info["prompt"] is not None:
            self.prompt = str(reset_info["prompt"]).strip()
        requested_session_id = reset_info.get("session_id")
        self._start_new_recording_session(requested_session_id or "reset")
        self._local_reset()
        return {
            "status": "reset successful",
            "policy_path": str(Path(self.args.checkpoint).expanduser()),
            "action_horizon": int(self.action_horizon),
            "prompt": self.prompt,
            "world_size": int(self.world_size),
            "cfg_parallel": bool(self.world_size > 1),
            "cfg_scale": float(self.policy.trained_model.action_head.cfg_scale),
            "session_id": self.current_session_id,
            "server_artifacts_last_flush": flushed,
            "output_dir": str(self.recorder.output_root) if self.recorder.output_root else None,
        }

    def _sample_offsets(self, window_len: int) -> list[int]:
        if window_len <= 1:
            return [0]
        if self.action_horizon == 48 and window_len == 9:
            return [0, 6, 12, 18, 24, 30, 36, 42, 48]
        if self.action_horizon == 48 and window_len == 8:
            return [0, 6, 12, 18, 24, 30, 36, 42]
        if self.action_horizon == 24 and window_len == 9:
            return [0, 3, 6, 9, 12, 15, 18, 21, 24]
        if self.action_horizon == 24 and window_len == 8:
            return [0, 3, 6, 9, 12, 15, 18, 21]
        if self.action_horizon == 24 and window_len == 4:
            return [0, 7, 15, 23]
        return np.rint(np.linspace(0, self.action_horizon - 1, window_len)).astype(int).tolist()

    def _select_video_window(self, frames: list[np.ndarray], *, source_key: str) -> np.ndarray:
        if not frames:
            raise ValueError(f"No frames for {source_key}")
        window_len = self.initial_frames if self.is_first_request else self.subsequent_frames
        window_len = max(int(window_len), 1)
        frames = [_resize_frame(frame, self.image_resize) for frame in frames]
        if self.is_first_request or window_len == 1:
            return np.ascontiguousarray(np.asarray(frames[-1], dtype=np.uint8)).copy()

        # Optimized robotdeploy clients can send exactly the sampled model
        # context frames instead of the full 30 FPS action-history window.  In
        # that case, consume the provided frames directly; the legacy path below
        # is only for full-history payloads.
        if len(frames) <= window_len:
            selected = list(frames)
            while len(selected) < window_len:
                selected.insert(0, selected[0])
            return np.ascontiguousarray(np.stack(selected, axis=0).astype(np.uint8)).copy()

        # The unmodified robotdeploy client sends the latest state_chunk_size
        # frames. By default, sample the just-completed action block and exclude
        # the current replan boundary frame, matching the LIBERO eval adapter.
        offsets = self._sample_offsets(window_len)
        block_end = len(frames) - 1 if self.include_current_boundary else max(len(frames) - 2, 0)
        block_span = self.action_horizon if self.include_current_boundary else self.action_horizon - 1
        block_start = max(0, block_end - int(block_span))
        selected = []
        for offset in offsets:
            idx = min(block_start + int(offset), block_end)
            selected.append(frames[idx])
        while len(selected) < window_len:
            selected.insert(0, selected[0])
        return np.ascontiguousarray(np.stack(selected, axis=0).astype(np.uint8)).copy()

    def _to_model_rgb_frames(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        if self.client_image_color_order == "rgb":
            return frames
        if self.client_image_color_order != "bgr":
            raise ValueError(f"Unsupported client image color order: {self.client_image_color_order!r}")
        # Unitree robotdeploy's ImageClient uses cv2.imdecode(..., IMREAD_COLOR),
        # so live camera observations arrive as BGR.  DreamZero/LeRobot training
        # transforms consume RGB images; swap channels at the websocket boundary
        # before any model-side crop/resize to keep real-robot eval aligned with
        # training and avoid pink objects appearing blue in predicted videos.
        return [np.ascontiguousarray(frame[..., ::-1]).copy() for frame in frames]

    def _model_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        built: dict[str, Any] = {}
        for model_key, client_key in self.view_map.items():
            if client_key not in obs:
                raise KeyError(f"Missing image key {client_key!r}; received keys={sorted(obs)}")
            frames = _as_frames(obs[client_key], key=client_key)
            built[model_key] = self._select_video_window(self._to_model_rgb_frames(frames), source_key=client_key)
        for model_key, client_key in self.state_map.items():
            if client_key not in obs:
                raise KeyError(f"Missing state key {client_key!r}; received keys={sorted(obs)}")
            built[model_key] = _latest_vector(obs[client_key], key=client_key)
        built["annotation.task_index"] = str(obs.get("prompt") or self.prompt)
        return built

    def _split_concat_action(self, act: Any) -> dict[str, np.ndarray]:
        concat = None
        for key in ("action", "actions"):
            if _contains_action_key(act, key):
                concat = _to_numpy(_action_value(act, key)).astype(np.float32)
                break
        if concat is None:
            raise KeyError(
                "DreamZero output did not contain Unitree model action keys, robotdeploy action keys, "
                "or a concat 'action'/'actions' tensor. "
                f"Expected model keys: {DEFAULT_MODEL_ACTION_KEYS}; "
                f"robot response keys: {DEFAULT_ROBOT_ACTION_KEYS}"
            )
        if concat.ndim == 3:
            concat = concat[0]
        if concat.ndim == 1:
            concat = concat.reshape(1, -1)
        if concat.ndim != 2 or concat.shape[-1] < 14:
            raise ValueError(f"Concat EEF action must have shape [T,>=14], got {concat.shape}")
        return {
            "action.left_ee_pose_gripper_base": concat[:, 0:6],
            "action.right_ee_pose_gripper_base": concat[:, 6:12],
            "action.left_gripper": concat[:, 12:13],
            "action.right_gripper": concat[:, 13:14],
        }

    def _extract_actions(self, act: Any) -> dict[str, np.ndarray]:
        if all(_contains_action_key(act, key) for key in self.model_action_keys):
            model_chunks = {
                key: _normalize_chunk(
                    _action_value(act, key),
                    key=key,
                    horizon=self.action_horizon,
                    dim=MODEL_ACTION_DIMS[key],
                )
                for key in self.model_action_keys
            }
            chunks = {
                MODEL_TO_ROBOT_ACTION_KEY[key]: value
                for key, value in model_chunks.items()
            }
        elif all(_contains_action_key(act, key) for key in self.robot_action_keys):
            # Keep a compatibility path for older checkpoints/tests that may
            # already return robotdeploy-facing EEF-RPY keys.
            chunks = {
                key: _normalize_chunk(
                    _action_value(act, key),
                    key=key,
                    horizon=self.action_horizon,
                    dim=ROBOT_ACTION_DIMS[key],
                )
                for key in self.robot_action_keys
            }
        else:
            split = self._split_concat_action(act)
            chunks = {
                MODEL_TO_ROBOT_ACTION_KEY[key]: _normalize_chunk(
                    value,
                    key=key,
                    horizon=self.action_horizon,
                    dim=MODEL_ACTION_DIMS[key],
                )
                for key, value in split.items()
            }
        return chunks

    def _forward_model(self, model_obs: dict[str, Any]) -> Any:
        from tianshou.data import Batch
        import torch

        with torch.inference_mode():
            return self.policy.lazy_joint_forward_causal(Batch(obs=model_obs))

    def _actions_and_video_from_model_obs(self, model_obs: dict[str, Any]) -> tuple[dict[str, np.ndarray], Any]:
        result_batch, video_pred = self._forward_model(model_obs)
        return self._extract_actions(result_batch.act), video_pred

    def _video_pred_starts_new_segment(self, video_pred: Any) -> bool:
        if video_pred is None or not self.recorder.video_chunks:
            return False
        action_head = getattr(getattr(self.policy, "trained_model", None), "action_head", None)
        num_frame_per_block = int(getattr(action_head, "num_frame_per_block", 1) or 1)
        current_start_frame = int(getattr(action_head, "current_start_frame", -1) or -1)
        shape = getattr(video_pred, "shape", ())
        latent_count = int(shape[2]) if len(shape) > 2 else -1
        return (
            current_start_frame == 1 + num_frame_per_block
            or latent_count == 1 + num_frame_per_block
        )

    def _log_request_shapes(self, model_obs: dict[str, Any]) -> None:
        if self.args.print_request_shapes:
            shape_summary = {k: list(np.asarray(v).shape) for k, v in model_obs.items() if hasattr(v, "shape")}
            LOG.info("request shapes=%s prompt=%r", shape_summary, self.prompt)

    def _record_infer_complete(self, actions: dict[str, np.ndarray], started_at: float) -> None:
        self.is_first_request = False
        self.infer_count += 1
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        if self.args.log_latency:
            stats = {
                key: {
                    "shape": list(value.shape),
                    "min": float(np.nanmin(value)),
                    "max": float(np.nanmax(value)),
                    "absmax": float(np.nanmax(np.abs(value))),
                }
                for key, value in actions.items()
            }
            LOG.info("infer#%d latency=%.1fms stats=%s", self.infer_count, elapsed_ms, json.dumps(stats))

    def get_action(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        started_at = time.perf_counter()
        requested_session_id = obs.get("session_id")
        if (
            requested_session_id is not None
            and str(requested_session_id) != str(self.current_session_id)
            and str(requested_session_id) != str(getattr(self, "current_session_request", ""))
        ):
            self.flush_outputs()
            self._start_new_recording_session(requested_session_id)
            self._local_reset()
        model_obs = self._model_obs(obs)
        self._log_request_shapes(model_obs)
        actions, video_pred = self._actions_and_video_from_model_obs(model_obs)
        if bool(obs.get(INTERNAL_RECORD_PREDICTION_KEY, True)):
            self.recorder.append(
                actions=actions,
                video_pred=video_pred,
                decision=self.infer_count + 1,
                starts_new_video_segment=self._video_pred_starts_new_segment(video_pred),
            )
        self._record_infer_complete(actions, started_at)
        return actions

    def shutdown(self) -> None:
        self.flush_outputs()


class DistributedUnitreeDreamZeroEEFPolicy(UnitreeDreamZeroEEFPolicy):
    """Rank-0 websocket policy plus nonzero-rank CFG-parallel worker loop."""

    def __init__(self, *, signal_group: Any, **kwargs: Any):
        self.signal_group = signal_group
        super().__init__(**kwargs)

    def _broadcast_signal(self, signal_value: int) -> None:
        import torch
        import torch.distributed as dist

        tensor = torch.tensor([signal_value], dtype=torch.int32)
        dist.broadcast(tensor, src=0, group=self.signal_group)

    def _broadcast_model_obs(self, model_obs: dict[str, Any]) -> None:
        import torch
        import torch.distributed as dist

        payload = pickle.dumps(model_obs, protocol=pickle.HIGHEST_PROTOCOL)
        size_tensor = torch.tensor([len(payload)], dtype=torch.int64)
        dist.broadcast(size_tensor, src=0, group=self.signal_group)
        data_array = np.frombuffer(payload, dtype=np.uint8).copy()
        data_tensor = torch.from_numpy(data_array).clone()
        dist.broadcast(data_tensor, src=0, group=self.signal_group)

    def _recv_model_obs(self) -> dict[str, Any]:
        import torch
        import torch.distributed as dist

        size_tensor = torch.zeros(1, dtype=torch.int64)
        dist.broadcast(size_tensor, src=0, group=self.signal_group)
        data_tensor = torch.empty(int(size_tensor.item()), dtype=torch.uint8)
        dist.broadcast(data_tensor, src=0, group=self.signal_group)
        return pickle.loads(data_tensor.numpy().tobytes())

    def reset(self, reset_info: dict[str, Any] | None = None) -> dict[str, Any]:
        result = super().reset(reset_info)
        self._broadcast_signal(SIGNAL_RESET)
        return result

    def get_action(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        import torch.distributed as dist

        started_at = time.perf_counter()
        requested_session_id = obs.get("session_id")
        if (
            requested_session_id is not None
            and str(requested_session_id) != str(self.current_session_id)
            and str(requested_session_id) != str(getattr(self, "current_session_request", ""))
        ):
            self.flush_outputs()
            self._start_new_recording_session(requested_session_id)
            self._local_reset()
            self._broadcast_signal(SIGNAL_RESET)
        model_obs = self._model_obs(obs)
        self._log_request_shapes(model_obs)
        self._broadcast_signal(SIGNAL_INFER)
        self._broadcast_model_obs(model_obs)
        dist.barrier()
        actions, video_pred = self._actions_and_video_from_model_obs(model_obs)
        dist.barrier()
        if bool(obs.get(INTERNAL_RECORD_PREDICTION_KEY, True)):
            self.recorder.append(
                actions=actions,
                video_pred=video_pred,
                decision=self.infer_count + 1,
                starts_new_video_segment=self._video_pred_starts_new_segment(video_pred),
            )
        self._record_infer_complete(actions, started_at)
        return actions

    def worker_loop(self) -> None:
        import torch
        import torch.distributed as dist

        rank = dist.get_rank()
        LOG.info("Worker rank=%d entering Unitree DreamZero EEF CFG-parallel loop", rank)
        while True:
            signal_tensor = torch.zeros(1, dtype=torch.int32)
            dist.broadcast(signal_tensor, src=0, group=self.signal_group)
            signal = int(signal_tensor.item())
            if signal == SIGNAL_SHUTDOWN:
                LOG.info("Worker rank=%d received SHUTDOWN", rank)
                return
            if signal == SIGNAL_RESET:
                LOG.info("Worker rank=%d received RESET", rank)
                self._local_reset()
                continue
            if signal != SIGNAL_INFER:
                raise RuntimeError(f"Worker rank={rank} received unknown distributed signal {signal}")
            model_obs = self._recv_model_obs()
            dist.barrier()
            self._forward_model(model_obs)
            dist.barrier()
            self.is_first_request = False
            self.infer_count += 1

    def shutdown(self) -> None:
        import torch.distributed as dist

        if dist.is_initialized() and dist.get_rank() == 0:
            self._broadcast_signal(SIGNAL_SHUTDOWN)
        super().shutdown()


class RobotdeployWebsocketPolicyServer:
    def __init__(self, policy: UnitreeDreamZeroEEFPolicy, *, host: str, port: int):
        self.policy = policy
        self.host = host
        self.port = int(port)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        import websockets
        import websockets.asyncio.server as websocket_server
        import websockets.frames
        from openpi_client import msgpack_numpy

        async def health_check(connection, request):
            if getattr(request, "path", None) == "/healthz":
                return connection.respond(http.HTTPStatus.OK, "OK\n")
            return None

        async def handler(websocket):
            LOG.info("Connection from %s opened", getattr(websocket, "remote_address", None))
            packer = msgpack_numpy.Packer()
            await websocket.send(packer.pack(self.policy.metadata))
            while True:
                try:
                    recv_start = time.perf_counter()
                    raw_msg = await websocket.recv()
                    recv_ms = (time.perf_counter() - recv_start) * 1000.0
                    request_bytes = len(raw_msg) if isinstance(raw_msg, (bytes, bytearray, memoryview)) else len(str(raw_msg).encode("utf-8"))
                    unpack_start = time.perf_counter()
                    msg = msgpack_numpy.unpackb(raw_msg)
                    unpack_ms = (time.perf_counter() - unpack_start) * 1000.0
                    msg_type = msg.get("type")
                    policy_start = time.perf_counter()
                    if msg_type == "get_action":
                        response = self.policy.get_action(msg["obs"])
                    elif msg_type == "policy_reset":
                        response = self.policy.reset(msg)
                    elif msg_type in {"policy_train", "policy_save", "policy_load", "update_dataset"}:
                        response = {"status": f"ignored unsupported request {msg_type}"}
                    else:
                        raise ValueError(f"Invalid message type: {msg_type}")
                    policy_ms = (time.perf_counter() - policy_start) * 1000.0
                    pack_start = time.perf_counter()
                    packed_response = packer.pack(response)
                    pack_ms = (time.perf_counter() - pack_start) * 1000.0
                    send_start = time.perf_counter()
                    await websocket.send(packed_response)
                    send_ms = (time.perf_counter() - send_start) * 1000.0
                    if self.policy.args.profile_rpc:
                        response_bytes = len(packed_response)
                        LOG.info(
                            "rpc_profile type=%s request=%.2fMiB response=%.3fMiB "
                            "recv_wait=%.1fms unpack=%.1fms policy=%.1fms pack=%.1fms send=%.1fms",
                            msg_type,
                            request_bytes / (1024 * 1024),
                            response_bytes / (1024 * 1024),
                            recv_ms,
                            unpack_ms,
                            policy_ms,
                            pack_ms,
                            send_ms,
                        )
                except websockets.ConnectionClosed:
                    LOG.info("Connection from %s closed", getattr(websocket, "remote_address", None))
                    break
                except Exception:
                    await websocket.send(traceback.format_exc())
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error. Traceback included in previous frame.",
                    )
                    raise

        try:
            server_cm = websocket_server.serve(
                handler,
                self.host,
                self.port,
                compression=None,
                max_size=None,
                process_request=health_check,
                ping_interval=None,
                ping_timeout=None,
            )
        except TypeError:
            server_cm = websocket_server.serve(
                handler,
                self.host,
                self.port,
                compression=None,
                max_size=None,
                ping_interval=None,
                ping_timeout=None,
            )
        async with server_cm as server:
            LOG.info("Serving robotdeploy-compatible Unitree DreamZero policy on %s:%d", self.host, self.port)
            await server.serve_forever()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve DreamZero Unitree G1 EEF-RPY policy for the unmodified robotdeploy client")
    parser.add_argument("--checkpoint", required=True, help="DreamZero checkpoint/model directory")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8165)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--distributed-timeout-seconds",
        type=int,
        default=int(os.environ.get("DREAMZERO_DISTRIBUTED_TIMEOUT_SECONDS", "3600")),
        help="torch.distributed timeout for torchrun CFG-parallel serving",
    )
    parser.add_argument(
        "--torch-dynamo-recompile-limit",
        type=int,
        default=int(os.environ.get("DREAMZERO_TORCH_DYNAMO_RECOMPILE_LIMIT", "800")),
        help=(
            "PyTorch Dynamo recompile limit for DreamZero AR serving. "
            "The official optimized AR server uses 800 to avoid first-chunk scheduler recompilation aborts."
        ),
    )
    parser.add_argument(
        "--torch-dynamo-cache-size-limit",
        type=int,
        default=int(os.environ.get("DREAMZERO_TORCH_DYNAMO_CACHE_SIZE_LIMIT", "1000")),
        help="PyTorch Dynamo cache-size budget for DreamZero AR serving.",
    )
    parser.add_argument("--embodiment-tag", default="unitree_g1_upper_body")
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", default=None, help="Server-side language prompt")
    prompt_group.add_argument("--prompt-file", default=None, help="File containing the server-side language prompt")
    parser.add_argument("--action-horizon", type=int, default=48)
    parser.add_argument("--obs-chunk-size", type=int, default=None, help="Frames requested from client; default action_horizon+1")
    parser.add_argument("--initial-frames", type=int, default=1)
    parser.add_argument("--subsequent-frames", type=int, default=8)
    parser.add_argument(
        "--include-current-boundary",
        action="store_true",
        help="Sample video windows including the current replan boundary frame; default samples completed block only.",
    )
    parser.add_argument("--resize-images", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"), default=None)
    parser.add_argument(
        "--client-image-color-order",
        choices=("bgr", "rgb"),
        default="bgr",
        help=(
            "Color order of image arrays/JPEG frames received from robotdeploy. "
            "Unitree live cameras are OpenCV BGR; the server converts them to "
            "RGB for DreamZero. Offline LeRobot replay should pass rgb."
        ),
    )
    parser.add_argument(
        "--client-resize-images",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        default=None,
        help=(
            "Publish a client-side per-view resize target in websocket metadata. "
            "Use with --skip-img-transform to move deterministic image crop/resize to robotdeploy."
        ),
    )
    parser.add_argument(
        "--client-center-crop-scale",
        type=float,
        default=None,
        help=(
            "Optional deterministic center-crop scale published for client-side preprocessing. "
            "If omitted with --skip-img-transform, it is inferred from checkpoint transforms when possible."
        ),
    )
    parser.add_argument(
        "--client-jpeg-quality",
        type=int,
        default=0,
        help="Optional client-side JPEG quality in [1,100]; 0 disables JPEG and keeps raw uint8 arrays.",
    )
    parser.add_argument("--view-map", nargs="*", default=list(DEFAULT_VIEW_MAP), help="MODEL_VIDEO_KEY=CLIENT_IMAGE_KEY entries")
    parser.add_argument("--state-map", nargs="*", default=list(DEFAULT_STATE_MAP), help="MODEL_STATE_KEY=CLIENT_STATE_KEY entries")
    parser.add_argument(
        "--action-keys",
        nargs="*",
        default=list(DEFAULT_MODEL_ACTION_KEYS),
        help=(
            "Checkpoint/model EEF action keys; normally leave unchanged. "
            "The websocket response is always converted to robotdeploy EEF-RPY keys."
        ),
    )
    parser.add_argument(
        "--model-config-override",
        action="append",
        default=[DEFAULT_TEXT_ENCODER_OVERRIDE],
        help="Dotlist override passed to GrootSimPolicy; repeatable.",
    )
    parser.add_argument("--tokenizer-path-override", default=None)
    parser.add_argument("--skip-assert-delta-indices", action="store_true")
    parser.add_argument("--skip-img-transform", action="store_true")
    parser.add_argument(
        "--cfg-scale",
        "--cfg_scale",
        dest="cfg_scale",
        type=float,
        default=None,
        help="Override action_head.cfg_scale. Use 1.0 on single GPU to disable CFG and skip the negative-prompt DiT pass.",
    )
    parser.add_argument("--log-latency", action="store_true", default=True)
    parser.add_argument("--no-log-latency", dest="log_latency", action="store_false")
    parser.add_argument("--print-request-shapes", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional server-side root for predicted artifacts. Each reset/session "
            "gets a timestamped session-id subdirectory containing pred video/action files."
        ),
    )
    parser.add_argument(
        "--save-predictions",
        dest="save_predictions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable predicted action/video recording. Defaults to true when --output-dir is set.",
    )
    parser.add_argument("--predicted-video-fps", type=int, default=5)
    parser.add_argument(
        "--save-predicted-video-npy",
        action="store_true",
        default=False,
        help=(
            "Also save raw decoded predicted video frames as pred_video_frames.npy. "
            "Disabled by default because the artifact is large; offline eval can "
            "enable it temporarily and delete it after comparison rendering."
        ),
    )
    parser.add_argument(
        "--no-predicted-video-watermark",
        dest="predicted_video_watermark",
        action="store_false",
        default=True,
    )
    parser.add_argument(
        "--profile-rpc",
        action="store_true",
        help="Log websocket request/response sizes plus recv/unpack/policy/pack/send timings.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)s %(message)s",
        force=True,
    )
    runtime = _init_runtime(
        args.device,
        args.distributed_timeout_seconds,
        dynamo_recompile_limit=args.torch_dynamo_recompile_limit,
        dynamo_cache_size_limit=args.torch_dynamo_cache_size_limit,
    )
    if runtime["distributed"] and args.cfg_scale is not None and abs(float(args.cfg_scale) - 1.0) < 1e-9:
        raise ValueError("cfg_scale=1.0 disables CFG and is only supported with single-GPU Unitree serving.")
    args.device = runtime["device"]
    if runtime["distributed"] and runtime["rank"] != 0:
        args.output_dir = None
        args.save_predictions = False
    policy_cls = DistributedUnitreeDreamZeroEEFPolicy if runtime["distributed"] else UnitreeDreamZeroEEFPolicy
    policy_kwargs: dict[str, Any] = {
        "args": args,
        "device_mesh": runtime["device_mesh"],
        "world_size": runtime["world_size"],
    }
    if runtime["distributed"]:
        policy_kwargs["signal_group"] = runtime["signal_group"]
    policy = policy_cls(**policy_kwargs)
    LOG.info(
        "Starting Unitree DreamZero EEF-RPY server host=%s port=%d checkpoint=%s embodiment=%s "
        "action_horizon=%d obs_chunk_size=%d prompt=%r view_map=%s state_map=%s "
        "observation_transport=%s output_dir=%s session_id=%s world_size=%d rank=%d device=%s cfg_scale=%s dynamo_config=%s",
        args.host,
        args.port,
        args.checkpoint,
        args.embodiment_tag,
        policy.action_horizon,
        policy.obs_chunk_size,
        policy.prompt,
        policy.view_map,
        policy.state_map,
        policy.metadata.get("observation_transport"),
        policy.metadata.get("output_dir"),
        policy.metadata.get("current_session_id"),
        runtime["world_size"],
        runtime["rank"],
        runtime["device"],
        policy.policy.trained_model.action_head.cfg_scale,
        runtime.get("dynamo_config", {}),
    )
    if runtime["distributed"] and runtime["rank"] != 0:
        policy.worker_loop()
        return
    try:
        RobotdeployWebsocketPolicyServer(policy, host=args.host, port=args.port).serve_forever()
    finally:
        policy.shutdown()


if __name__ == "__main__":
    main()
