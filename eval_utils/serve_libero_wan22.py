"""Serve DreamZero WAN2.2 for LIBERO with the official WAN22 server shape.

Client requests carry LIBERO keys (video.image, video.wrist_image, state.*, annotation.task_index).
The server owns action horizon, replan cadence, observation window policy, rotation policy metadata,
and optional predicted-video recording.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from tianshou.data import Batch

_dynamo = torch._dynamo.config
for _name, _value in {
    "cache_size_limit": 1000,
    "recompile_limit": 800,
    "accumulated_cache_size_limit": 1000,
    "accumulated_recompile_limit": 2000,
}.items():
    if hasattr(_dynamo, _name):
        setattr(_dynamo, _name, _value)

DREAMZERO_ROOT = Path(__file__).resolve().parents[1]
if str(DREAMZERO_ROOT) not in sys.path:
    sys.path.insert(0, str(DREAMZERO_ROOT))

from openpi_client.base_policy import BasePolicy

from eval_utils.libero_wan22_config import DEFAULT_PROFILE, LiberoWan22Profile, profile_with_overrides
from eval_utils.policy_server import PolicyServerConfig, WebsocketPolicyServer

LOG = logging.getLogger(__name__)
SIGNAL_INFER = 0
SIGNAL_SHUTDOWN = 1
SIGNAL_RESET = 2


@dataclasses.dataclass
class LiberoWan22ServerConfig(PolicyServerConfig):
    default_profile: str = DEFAULT_PROFILE.name
    profile_metadata: dict[str, Any] = dataclasses.field(default_factory=DEFAULT_PROFILE.metadata)
    action_horizon: int = DEFAULT_PROFILE.action_horizon
    replan_steps: int = DEFAULT_PROFILE.replan_steps
    initial_frames: int = DEFAULT_PROFILE.obs_window_policy.initial_frames
    subsequent_frames: int = DEFAULT_PROFILE.obs_window_policy.subsequent_frames
    include_current_boundary: bool = DEFAULT_PROFILE.obs_window_policy.include_current_boundary
    wait_steps: int = DEFAULT_PROFILE.wait_steps
    max_steps_by_benchmark: dict[str, int] = dataclasses.field(
        default_factory=lambda: dict(DEFAULT_PROFILE.max_steps_by_benchmark)
    )
    save_predicted_videos: bool = DEFAULT_PROFILE.save_predicted_videos
    predicted_video_output_root: str | None = None
    attention_backend: str | None = None
    enable_tensorrt: bool = False
    enable_dit_cache: bool = False
    world_size: int = 1


def _safe(value: Any, fallback: str) -> str:
    text = str(value or fallback)
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)
    return cleaned or fallback


def _resize_frame(frame: np.ndarray, resolution: tuple[int, int]) -> np.ndarray:
    h, w = resolution
    arr = np.asarray(frame, dtype=np.uint8)
    if arr.shape[:2] != (h, w):
        arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(arr)


def _checkpoint_resolution(policy: Any, fallback: tuple[int, int]) -> tuple[int, int]:
    eval_transform = getattr(policy, "eval_transform", None)
    for transform in getattr(eval_transform, "transforms", []):
        resolutions = getattr(transform, "original_resolutions", None)
        if resolutions:
            width, height = next(iter(resolutions.values()))
            return (int(height), int(width))
    return fallback


def _iter_frames(value: Any, resolution: tuple[int, int]) -> list[np.ndarray]:
    arr = np.asarray(value, dtype=np.uint8)
    if arr.ndim == 3:
        return [_resize_frame(arr, resolution)]
    if arr.ndim == 4:
        return [_resize_frame(frame, resolution) for frame in arr]
    raise ValueError(f"expected HWC or THWC image payload, got shape {arr.shape}")


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _action_value(act: Any, key: str) -> Any:
    try:
        return act[key]
    except (KeyError, TypeError, AttributeError, IndexError):
        if hasattr(act, key):
            return getattr(act, key)
        raise KeyError(key)


def _reset_action_head_state(policy: GrootSimPolicy) -> None:
    action_head = getattr(getattr(policy, "trained_model", None), "action_head", None)
    if action_head is None:
        return
    for name, value in {
        "current_start_frame": 0,
        "language": None,
        "clip_feas": None,
        "ys": None,
        "kv_cache1": None,
        "kv_cache_neg": None,
        "crossattn_cache": None,
        "crossattn_cache_neg": None,
    }.items():
        if hasattr(action_head, name):
            setattr(action_head, name, value)


def _add_watermarks(frames: np.ndarray, chunks: list[tuple[torch.Tensor, int]]) -> np.ndarray:
    if not chunks:
        return frames
    out = np.ascontiguousarray(frames)
    total, height, width = out.shape[:3]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(min(height, width) / 360.0, 0.45)
    thick = max(1, int(round(scale * 2)))
    pad = max(4, int(round(6 * scale)))
    margin = max(6, int(round(8 * scale)))
    latent_counts = [int(t.shape[2]) for t, _ in chunks]
    wan_map = 4 * sum(latent_counts) - 3 == total
    frame_start = 0
    latent_cumsum = 0
    for index, ((_, decision), latent_count) in enumerate(zip(chunks, latent_counts)):
        latent_cumsum += latent_count
        if wan_map:
            frame_end = frame_start + (4 * latent_count - 3 if index == 0 else 4 * latent_count)
        else:
            frame_end = total if index == len(chunks) - 1 else round(total * latent_cumsum / max(sum(latent_counts), 1))
        label = f"CHUNK {decision:03d}"
        (tw, th), base = cv2.getTextSize(label, font, scale, thick)
        x0 = max(width - tw - 2 * pad - margin, 0)
        y0 = max(height - th - base - 2 * pad - margin, 0)
        x1 = min(width - margin, width)
        y1 = min(height - margin, height)
        for frame in out[frame_start:min(frame_end, total)]:
            roi = frame[y0:y1, x0:x1]
            cv2.addWeighted(roi, 0.38, roi, 0.0, 0, dst=roi)
            cv2.putText(frame, label, (x0 + pad, y1 - pad - base), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
        frame_start = min(frame_end, total)
    return out


class PredictedVideoRecorder:
    def __init__(self, output_root: str | None, *, fps: int, watermark: bool = True):
        self.output_root = Path(output_root) if output_root else None
        self.fps = int(fps)
        self.watermark = bool(watermark)
        self.chunks: list[tuple[torch.Tensor, int, bool]] = []
        self.last_path: str | None = None

    def clear(self) -> None:
        self.chunks.clear()

    @staticmethod
    def _truncate(tensor: torch.Tensor, action_steps: int, replan_steps: int) -> torch.Tensor:
        if tensor.ndim < 3 or tensor.shape[2] <= 1 or action_steps <= 0:
            return tensor
        plan_steps = min(max(int(replan_steps), 1), int(action_steps))
        keep = (int(tensor.shape[2]) * plan_steps + action_steps // 2) // action_steps
        keep = min(max(keep, 1), int(tensor.shape[2]))
        return tensor[:, :, :keep].contiguous()

    def append(
        self,
        video_pred: torch.Tensor | None,
        *,
        decision: int,
        action_steps: int,
        replan_steps: int,
        starts_new_segment: bool = False,
    ) -> None:
        if self.output_root is None or video_pred is None:
            return
        tensor = self._truncate(video_pred.detach().cpu(), action_steps, replan_steps)
        self.chunks.append((tensor, int(decision), bool(starts_new_segment)))

    def flush(
        self,
        policy: GrootSimPolicy,
        *,
        eval_run_name: str | None,
        profile_name: str,
        benchmark_name: str | None,
        task_id: int | None,
        episode_index: int | None,
    ) -> str | None:
        if self.output_root is None or not self.chunks or episode_index is None:
            self.clear()
            return None
        output_dir = (
            self.output_root
            / _safe(eval_run_name, "eval_run")
            / "profiles"
            / _safe(profile_name, "profile")
            / _safe(benchmark_name, "benchmark")
            / (f"task_{int(task_id):02d}" if task_id is not None else "task_unknown")
            / f"episode_{int(episode_index):03d}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "predicted.mp4"
        try:
            from einops import rearrange

            action_head = policy.trained_model.action_head
            latent_segments: list[list[tuple[torch.Tensor, int, bool]]] = []
            current_segment: list[tuple[torch.Tensor, int, bool]] = []
            for item in self.chunks:
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
                    segment_frames = ((segment_frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
                    # A WAN hard-reset chunk starts a fresh 1+4N VAE segment.
                    # Drop the duplicated boundary frame for all non-initial
                    # segments before concatenating episode-level raw frames.
                    if segment_index > 0 and segment_frames.shape[0] > 0:
                        segment_frames = segment_frames[1:]
                    decoded_segments.append(segment_frames)
            frames = np.concatenate(decoded_segments, axis=0)
            if self.watermark:
                watermark_chunks = [(tensor, decision) for tensor, decision, _ in self.chunks]
                frames = _add_watermarks(frames, watermark_chunks)
            imageio.mimsave(output_path, list(frames), fps=self.fps, codec="libx264")
            (output_dir / "predicted.json").write_text(
                json.dumps(
                    {
                        "path": str(output_path),
                        "num_frames": int(frames.shape[0]),
                        "chunks": [int(decision) for _, decision, _ in self.chunks],
                        "latent_segments": [
                            [int(decision) for _, decision, _ in segment]
                            for segment in latent_segments
                        ],
                        "dropped_segment_boundary_frames": max(len(latent_segments) - 1, 0),
                    },
                    indent=2,
                )
            )
            self.last_path = str(output_path)
            return self.last_path
        except Exception as exc:  # noqa: BLE001
            LOG.warning("failed to save predicted video: %s", exc)
            return None
        finally:
            self.clear()


class DreamZeroLiberoWan22Policy(BasePolicy):
    def __init__(
        self,
        policy: GrootSimPolicy,
        profile: LiberoWan22Profile,
        *,
        predicted_video_output_root: str | None,
        predicted_video_watermark: bool,
    ):
        self.policy = policy
        self.profile = profile
        self.buffers = {key: [] for key in profile.view_keys}
        self.is_first_call = True
        self.current_session_id: str | None = None
        self.eval_run_name: str | None = None
        self.benchmark_name: str | None = None
        self.task_id: int | None = None
        self.episode_index: int | None = None
        self.infer_count = 0
        self.recorder = PredictedVideoRecorder(
            predicted_video_output_root if profile.save_predicted_videos else None,
            fps=profile.predicted_video_fps,
            watermark=predicted_video_watermark,
        )

    def _reset_buffers(self) -> None:
        self.buffers = {key: [] for key in self.profile.view_keys}
        self.is_first_call = True
        self.infer_count = 0
        _reset_action_head_state(self.policy)

    def _check_profile(self, requested: Any) -> None:
        if requested not in (None, "", self.profile.name):
            raise ValueError(
                f"server is authoritative for profile {self.profile.name!r}; "
                f"client requested {requested!r}"
            )

    def _flush(self) -> str | None:
        return self.recorder.flush(
            self.policy,
            eval_run_name=self.eval_run_name,
            profile_name=self.profile.name,
            benchmark_name=self.benchmark_name,
            task_id=self.task_id,
            episode_index=self.episode_index,
        )

    def _sample_offsets(self, window_len: int) -> list[int]:
        if window_len <= 1:
            return [0]
        if self.profile.action_horizon == 24 and window_len == 9:
            return [0, 3, 6, 9, 12, 15, 18, 21, 24]
        if self.profile.action_horizon == 24 and window_len == 8:
            return [0, 3, 6, 9, 12, 15, 18, 21]
        if self.profile.action_horizon == 24 and window_len == 4:
            return [0, 7, 15, 23]
        end = (
            int(self.profile.action_horizon)
            if self.profile.obs_window_policy.include_current_boundary
            else int(self.profile.action_horizon) - 1
        )
        return np.rint(np.linspace(0, max(end, 0), window_len)).astype(int).tolist()

    def _select_video_window(self, frames: list[np.ndarray], *, window_len: int, source_key: str) -> np.ndarray:
        if not frames:
            raise ValueError(f"No frames for {source_key}")
        if self.is_first_call or window_len <= 1:
            return np.ascontiguousarray(np.asarray(frames[-1], dtype=np.uint8)).copy()

        # Optimized clients may send exactly the sampled model frames; consume
        # those directly.  If a client sends a full raw action-history block,
        # sample it with the same exact offsets used in training.
        if len(frames) <= window_len:
            selected = list(frames)
            while len(selected) < window_len:
                selected.insert(0, selected[0])
            return np.ascontiguousarray(np.stack(selected, axis=0).astype(np.uint8)).copy()

        offsets = self._sample_offsets(window_len)
        include_boundary = bool(self.profile.obs_window_policy.include_current_boundary)
        block_end = len(frames) - 1 if include_boundary else max(len(frames) - 2, 0)
        block_span = int(self.profile.action_horizon) if include_boundary else int(self.profile.action_horizon) - 1
        block_start = max(0, block_end - max(block_span, 0))
        selected = [frames[min(block_start + int(offset), block_end)] for offset in offsets]
        return np.ascontiguousarray(np.stack(selected, axis=0).astype(np.uint8)).copy()

    def _model_obs(self, request: dict[str, Any]) -> dict[str, Any]:
        obs = request.get("obs", request)
        window_policy = self.profile.obs_window_policy
        window_len = window_policy.initial_frames if self.is_first_call else window_policy.subsequent_frames
        built: dict[str, Any] = {}
        for key in self.profile.view_keys:
            if key not in obs:
                raise KeyError(f"request missing {key!r}; received {sorted(obs.keys())}")
            request_frames = _iter_frames(obs[key], self.profile.raw_image_resolution)
            if len(request_frames) <= 1:
                self.buffers[key].extend(request_frames)
                frames = list(self.buffers[key])
            else:
                frames = request_frames
            built[key] = self._select_video_window(frames, window_len=window_len, source_key=key)
        for key in self.profile.state_keys:
            built[key] = np.asarray(obs[key], dtype=np.float64)
        built["annotation.task_index"] = request.get("prompt") or obs.get("annotation.task_index", "")
        return built

    def _actions(self, result_batch: Any) -> np.ndarray:
        act = result_batch.act
        ee_delta = _to_numpy(_action_value(act, "action.ee_delta_pose")).astype(np.float32)
        gripper = _to_numpy(_action_value(act, "action.gripper")).astype(np.float32)
        if ee_delta.ndim == 1:
            ee_delta = ee_delta.reshape(1, -1)
        if gripper.ndim == 1:
            gripper = gripper.reshape(-1, 1)
        return np.concatenate([ee_delta, gripper[:, :1]], axis=-1).astype(np.float32)

    def _video_pred_starts_new_segment(self, video_pred: torch.Tensor | None) -> bool:
        if video_pred is None or not self.recorder.chunks:
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

    def reset(self, reset_info: dict[str, Any]) -> dict[str, Any]:
        self._check_profile(reset_info.get("profile"))
        flushed = self._flush()
        self.current_session_id = reset_info.get("session_id")
        self.eval_run_name = reset_info.get("eval_run_name", self.eval_run_name)
        self.benchmark_name = reset_info.get("benchmark", self.benchmark_name)
        self.task_id = None if reset_info.get("task_id") is None else int(reset_info["task_id"])
        self.episode_index = None if reset_info.get("episode_index") is None else int(reset_info["episode_index"])
        self._reset_buffers()
        return {
            "status": "reset successful",
            "profile_name": self.profile.name,
            "server_artifacts": {
                "predicted_video_path_last_flush": flushed,
                "predicted_video_output_root": (
                    str(self.recorder.output_root) if self.recorder.output_root else None
                ),
            },
        }

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        self._check_profile(request.get("profile"))
        session_id = request.get("session_id")
        if session_id is not None and session_id != self.current_session_id:
            self._flush()
            self.current_session_id = session_id
            self.eval_run_name = request.get("eval_run_name", self.eval_run_name)
            self.benchmark_name = request.get("benchmark", self.benchmark_name)
            self.task_id = None if request.get("task_id") is None else int(request["task_id"])
            self.episode_index = None if request.get("episode_index") is None else int(request["episode_index"])
            self._reset_buffers()
        model_obs = self._model_obs(request)
        with torch.inference_mode():
            result_batch, video_pred = self.policy.lazy_joint_forward_causal(Batch(obs=model_obs))
        actions = self._actions(result_batch)
        self.is_first_call = False
        self.infer_count += 1
        self.recorder.append(
            video_pred if self.profile.save_predicted_videos else None,
            decision=self.infer_count,
            action_steps=int(actions.shape[0]),
            replan_steps=self.profile.replan_steps,
            starts_new_segment=self._video_pred_starts_new_segment(video_pred),
        )
        stats = {
            "shape": list(actions.shape),
            "min": float(np.nanmin(actions)),
            "max": float(np.nanmax(actions)),
            "absmax": float(np.nanmax(np.abs(actions))),
            "has_nan": bool(np.isnan(actions).any()),
            "has_inf": bool(np.isinf(actions).any()),
        }
        return {
            "actions": actions,
            "action_horizon": int(actions.shape[0]),
            "replan_steps": int(self.profile.replan_steps),
            "obs_window_policy": self.profile.obs_window_policy.label(),
            "include_current_boundary": bool(self.profile.obs_window_policy.include_current_boundary),
            "profile_name": self.profile.name,
            "resolved_profile": self.profile.name,
            "server_artifacts": {
                "predicted_video_output_root": (
                    str(self.recorder.output_root) if self.recorder.output_root else None
                )
            },
            "stats": stats,
        }

    def shutdown(self) -> None:
        self._flush()


class DistributedDreamZeroLiberoWan22Policy(DreamZeroLiberoWan22Policy):
    def __init__(self, *, signal_group: Any, **kwargs: Any):
        self.signal_group = signal_group
        super().__init__(**kwargs)

    def _broadcast_signal(self, signal_value: int) -> None:
        tensor = torch.tensor([signal_value], dtype=torch.int32)
        dist.broadcast(tensor, src=0, group=self.signal_group)

    def _broadcast_request(self, request: dict[str, Any]) -> None:
        payload = pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL)
        size = torch.tensor([len(payload)], dtype=torch.int64)
        data = torch.tensor(list(payload), dtype=torch.uint8)
        dist.broadcast(size, src=0, group=self.signal_group)
        dist.broadcast(data, src=0, group=self.signal_group)

    def _recv_request(self) -> dict[str, Any]:
        size = torch.zeros(1, dtype=torch.int64)
        dist.broadcast(size, src=0, group=self.signal_group)
        data = torch.empty(int(size.item()), dtype=torch.uint8)
        dist.broadcast(data, src=0, group=self.signal_group)
        return pickle.loads(bytes(data.tolist()))

    def reset(self, reset_info: dict[str, Any]) -> dict[str, Any]:
        result = super().reset(reset_info)
        self._broadcast_signal(SIGNAL_RESET)
        return result

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        self._broadcast_signal(SIGNAL_INFER)
        self._broadcast_request(request)
        dist.barrier()
        result = super().infer(request)
        dist.barrier()
        return result

    def worker_loop(self) -> None:
        while True:
            signal = torch.zeros(1, dtype=torch.int32)
            dist.broadcast(signal, src=0, group=self.signal_group)
            value = int(signal.item())
            if value == SIGNAL_SHUTDOWN:
                return
            if value == SIGNAL_RESET:
                self._reset_buffers()
                continue
            if value != SIGNAL_INFER:
                raise RuntimeError(f"unknown distributed signal {value}")
            request = self._recv_request()
            dist.barrier()
            super().infer(request)
            dist.barrier()

    def shutdown(self) -> None:
        if dist.get_rank() == 0:
            super().shutdown()
            self._broadcast_signal(SIGNAL_SHUTDOWN)


def _init_runtime() -> dict[str, Any]:
    timeout = _dt.timedelta(seconds=int(os.environ.get("DREAMZERO_DISTRIBUTED_TIMEOUT_SECONDS", "3600")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("distributed LIBERO WAN22 serving requires CUDA")
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        if not dist.is_initialized():
            try:
                dist.init_process_group(backend="nccl", device_id=torch.device(device), timeout=timeout)
            except TypeError:
                dist.init_process_group(backend="nccl", timeout=timeout)
        return {
            "world_size": world_size,
            "rank": rank,
            "device": device,
            "device_mesh": init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=("ip",)),
            "signal_group": dist.new_group(backend="gloo", timeout=timeout),
            "distributed": True,
        }
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            rank=0,
            world_size=1,
            timeout=timeout,
        )
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    return {
        "world_size": 1,
        "rank": 0,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "device_mesh": init_device_mesh("cuda", mesh_shape=(1,), mesh_dim_names=("ip",)) if torch.cuda.is_available() else None,
        "signal_group": None,
        "distributed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DreamZero WAN22 LIBERO policy server")
    parser.add_argument("--model-path", "--checkpoint", dest="model_path", required=True)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--embodiment-tag", default="libero_sim")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--profile-name", default=DEFAULT_PROFILE.name)
    parser.add_argument("--raw-image-resolution", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"), default=None)
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--replan-steps", type=int, default=None)
    parser.add_argument("--wait-steps", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--obs-window-policy", default=None, help="Frame window such as 1->9")
    parser.add_argument(
        "--include-current-boundary",
        dest="include_current_boundary",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether subsequent video windows include the current replan boundary frame.",
    )
    parser.add_argument("--attention-backend", default=None)
    parser.add_argument("--enable-tensorrt", action="store_true", default=False)
    parser.add_argument("--enable-dit-cache", action="store_true", default=False)
    parser.add_argument(
        "--cfg-scale",
        "--cfg_scale",
        dest="cfg_scale",
        type=float,
        default=None,
        help="Override action_head.cfg_scale. Use 1.0 on single GPU to disable CFG and skip the negative-prompt DiT pass.",
    )
    parser.add_argument(
        "--save-predicted-videos",
        dest="save_predicted_videos",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--pre-rotate-images-for-policy",
        dest="pre_rotate_images_for_policy",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--rotate-rollout-frames-for-video",
        dest="rotate_rollout_frames_for_video",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--output-dir", default=None, help="Server-side predicted-video root")
    parser.add_argument("--predicted-video-fps", type=int, default=None)
    parser.add_argument(
        "--no-predicted-video-watermark",
        dest="predicted_video_watermark",
        action="store_false",
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s", force=True)
    if args.attention_backend:
        os.environ["ATTENTION_BACKEND"] = args.attention_backend
    os.environ["ENABLE_TENSORRT"] = "true" if args.enable_tensorrt else "false"
    os.environ["ENABLE_DIT_CACHE"] = "true" if args.enable_dit_cache else "false"
    profile = profile_with_overrides(
        DEFAULT_PROFILE,
        name=args.profile_name,
        raw_image_resolution=args.raw_image_resolution,
        action_horizon=args.action_horizon,
        replan_steps=args.replan_steps,
        wait_steps=args.wait_steps,
        max_steps=args.max_steps,
        obs_window_policy=args.obs_window_policy,
        include_current_boundary=args.include_current_boundary,
        save_predicted_videos=args.save_predicted_videos,
        pre_rotate_images_for_policy=args.pre_rotate_images_for_policy,
        rotate_rollout_frames_for_video=args.rotate_rollout_frames_for_video,
        predicted_video_fps=args.predicted_video_fps,
    )
    output_root = args.output_dir
    if output_root is None and profile.save_predicted_videos:
        checkpoint_name = Path(args.model_path.rstrip("/")).name
        output_root = str(Path(args.model_path).resolve().parent / f"libero_wan22_predictions_{checkpoint_name}")
    from groot.vla.data.schema import EmbodimentTag
    from groot.vla.model.n1_5.sim_policy import GrootSimPolicy

    runtime = _init_runtime()
    if runtime["world_size"] > 2:
        raise ValueError("LIBERO eval v2 WAN22 serving supports at most 2 GPUs for CFG parallelism.")
    if runtime["distributed"] and args.cfg_scale is not None and abs(float(args.cfg_scale) - 1.0) < 1e-9:
        raise ValueError("cfg_scale=1.0 disables CFG and is only supported with single-GPU serving in this path.")
    if runtime["distributed"] and runtime["rank"] != 0:
        output_root = None
    LOG.info("Loading DreamZero WAN22 LIBERO policy from %s", args.model_path)
    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag(args.embodiment_tag),
        model_path=args.model_path,
        tokenizer_path_override=args.tokenizer_path,
        device=runtime["device"],
        device_mesh=runtime["device_mesh"],
        model_config_overrides=["action_head_cfg.config.load_text_encoder=true"],
    )
    if args.cfg_scale is not None:
        policy.trained_model.action_head.cfg_scale = float(args.cfg_scale)
        LOG.info("Set action_head.cfg_scale=%s", policy.trained_model.action_head.cfg_scale)
    if args.raw_image_resolution is None:
        profile = dataclasses.replace(
            profile,
            raw_image_resolution=_checkpoint_resolution(policy, profile.raw_image_resolution),
        )
        LOG.info("Using checkpoint LIBERO video resolution %s", profile.raw_image_resolution)
    wrapper_kwargs = {
        "policy": policy,
        "profile": profile,
        "predicted_video_output_root": output_root,
        "predicted_video_watermark": args.predicted_video_watermark,
    }
    if runtime["distributed"]:
        wrapper = DistributedDreamZeroLiberoWan22Policy(
            signal_group=runtime["signal_group"],
            **wrapper_kwargs,
        )
    else:
        wrapper = DreamZeroLiberoWan22Policy(**wrapper_kwargs)
    config = LiberoWan22ServerConfig(
        image_resolution=profile.raw_image_resolution,
        needs_wrist_camera=True,
        n_external_cameras=1,
        needs_stereo_camera=False,
        needs_session_id=True,
        action_space="cartesian_position",
        default_profile=profile.name,
        profile_metadata=profile.metadata(),
        action_horizon=profile.action_horizon,
        replan_steps=profile.replan_steps,
        initial_frames=profile.obs_window_policy.initial_frames,
        subsequent_frames=profile.obs_window_policy.subsequent_frames,
        include_current_boundary=profile.obs_window_policy.include_current_boundary,
        wait_steps=profile.wait_steps,
        max_steps_by_benchmark=dict(profile.max_steps_by_benchmark),
        save_predicted_videos=profile.save_predicted_videos,
        predicted_video_output_root=output_root,
        attention_backend=args.attention_backend,
        enable_tensorrt=args.enable_tensorrt,
        enable_dit_cache=args.enable_dit_cache,
        world_size=runtime["world_size"],
    )
    if runtime["distributed"] and runtime["rank"] != 0:
        LOG.info("Worker rank=%d entering LIBERO WAN22 v2 worker loop", runtime["rank"])
        wrapper.worker_loop()
        return
    LOG.info(
        "Serving LIBERO WAN22 on %s:%d profile=%s resolution=%s replan=%d obs=%s include_boundary=%s tensorrt=%s dit_cache=%s world_size=%d",
        args.host,
        args.port,
        profile.name,
        profile.raw_image_resolution,
        profile.replan_steps,
        profile.obs_window_policy.label(),
        profile.obs_window_policy.include_current_boundary,
        args.enable_tensorrt,
        args.enable_dit_cache,
        runtime["world_size"],
    )
    try:
        WebsocketPolicyServer(wrapper, config, host=args.host, port=args.port).serve_forever()
    finally:
        wrapper.shutdown()


if __name__ == "__main__":
    main()
