"""Experimental LIBERO WAN2.2 server with lightweight KV-cache sliding.

This script keeps the production LIBERO v2 server untouched.  It explores the
issue where the WAN action head hard-resets when ``current_start_frame`` reaches
``local_attn_size``.  Instead of replaying the whole local window, it keeps the
existing KV cache, keeps ``current_start_frame`` monotonic for RoPE coherence,
recomputes only each inference call's anchor CLIP features and first-frame latent
conditioning, then lets the DiT attention implementation crop KV naturally when
appending new tokens.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange
from tianshou.data import Batch

DREAMZERO_ROOT = Path(__file__).resolve().parents[1]
if str(DREAMZERO_ROOT) not in sys.path:
    sys.path.insert(0, str(DREAMZERO_ROOT))

from eval_utils import serve_libero_wan22 as v2
from eval_utils.libero_wan22_config import DEFAULT_PROFILE, profile_with_overrides
from eval_utils.policy_server import WebsocketPolicyServer

LOG = logging.getLogger(__name__)
DISABLED_HARD_RESET_LOCAL_ATTN_SIZE = 10**12


@dataclasses.dataclass
class SlidingWindowLiberoWan22ServerConfig(v2.LiberoWan22ServerConfig):
    sliding_window_kv: bool = True
    sliding_window_profile_json: str | None = None


def _copy_obs(obs: dict[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            copied[key] = value.copy()
        elif torch.is_tensor(value):
            copied[key] = value.detach().clone()
        else:
            copied[key] = value
    return copied


def _anchor_obs(obs: dict[str, Any], view_keys: tuple[str, ...]) -> dict[str, Any]:
    anchored = _copy_obs(obs)
    for key in view_keys:
        arr = np.asarray(anchored[key], dtype=np.uint8)
        if arr.ndim == 4:
            index = -1 if arr.shape[0] in (4, 9) else 0
            anchored[key] = arr[index:].copy() if index < 0 else arr[index : index + 1].copy()
        elif arr.ndim == 3:
            anchored[key] = arr[None].copy()
        else:
            raise ValueError(f"expected HWC or THWC for {key}, got {arr.shape}")
    return anchored


def _sync_if_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _unsqueeze_dict_values(data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            out[key] = np.expand_dims(value, axis=0)
        elif isinstance(value, list):
            out[key] = np.array(value)
        elif torch.is_tensor(value):
            out[key] = value.unsqueeze(0)
        elif isinstance(value, str):
            out[key] = np.array([value])
        else:
            out[key] = value
    return out


class SlidingWindowDreamZeroLiberoWan22Policy(v2.DreamZeroLiberoWan22Policy):
    def __init__(
        self,
        *args: Any,
        sliding_window_profile_json: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.sliding_window_profile_json = Path(sliding_window_profile_json) if sliding_window_profile_json else None
        self.sliding_events: list[dict[str, Any]] = []
        self.sliding_recompute_count = 0
        self.sliding_recompute_seconds_sum = 0.0
        self.infer_seconds_sum = 0.0
        self.infer_seconds_count = 0
        self.original_model_local_attn_size: int | None = None
        self._disable_local_attn_hard_reset()

    def _action_head(self) -> Any:
        return getattr(getattr(self.policy, "trained_model", None), "action_head", None)

    def _cache_attention_state(self) -> dict[str, int]:
        action_head = self._action_head()
        model = getattr(action_head, "model", None)
        blocks = getattr(model, "blocks", None) or []
        self_attn = getattr(blocks[0], "self_attn", None) if blocks else None
        return {
            "cache_local_attn_size": int(getattr(self_attn, "local_attn_size", -1) or -1),
            "max_attention_size": int(getattr(self_attn, "max_attention_size", -1) or -1),
            "frame_seqlen": int(getattr(self_attn, "frame_seqlen", -1) or -1),
        }

    def _disable_local_attn_hard_reset(self) -> None:
        action_head = self._action_head()
        model = getattr(action_head, "model", None)
        if model is None:
            return
        current = int(getattr(model, "local_attn_size", -1) or -1)
        if current <= 0 or current >= DISABLED_HARD_RESET_LOCAL_ATTN_SIZE:
            return
        self.original_model_local_attn_size = current
        setattr(model, "local_attn_size", DISABLED_HARD_RESET_LOCAL_ATTN_SIZE)
        LOG.info(
            "Disabled action-head local_attn hard reset: model.local_attn_size %d -> %d; "
            "attention cache crop still uses block self_attn.max_attention_size",
            current,
            DISABLED_HARD_RESET_LOCAL_ATTN_SIZE,
        )

    def _context_state(self) -> dict[str, int]:
        action_head = self._action_head()
        model = getattr(action_head, "model", None)
        state = {
            "current_start_frame": int(getattr(action_head, "current_start_frame", 0) or 0),
            "model_local_attn_size": int(getattr(model, "local_attn_size", -1) or -1),
            "original_model_local_attn_size": int(self.original_model_local_attn_size or -1),
            "num_frame_per_block": max(int(getattr(action_head, "num_frame_per_block", 1) or 1), 1),
        }
        state.update(self._cache_attention_state())
        return state

    def _can_recompute_anchor(self) -> bool:
        state = self._context_state()
        action_head = self._action_head()
        return (
            action_head is not None
            and state["current_start_frame"] > 0
            and getattr(action_head, "kv_cache1", None) is not None
            and getattr(action_head, "clip_feas", None) is not None
            and getattr(action_head, "ys", None) is not None
        )

    def _normalized_anchor_input(self, anchor_obs: dict[str, Any]) -> dict[str, Any]:
        batch = Batch(obs=anchor_obs)
        if not self.policy._check_state_is_batched(batch.obs):
            batch.obs = _unsqueeze_dict_values(batch.obs)
        batch = self.policy.apply(batch)
        normalized = batch.normalized_obs
        if isinstance(normalized, Batch):
            normalized = normalized.__getstate__()
        for key, value in list(normalized.items()):
            if torch.is_tensor(value) and value.dtype == torch.float32 and self.policy.eval_bf16:
                normalized[key] = value.to(dtype=torch.bfloat16)
        return normalized

    def _prepare_anchor_video(self, data: dict[str, Any]) -> torch.Tensor:
        action_head = self._action_head()
        device = torch.device(action_head._device)
        videos = rearrange(data["images"], "b t h w c -> b c t h w")
        if videos.dtype == torch.uint8:
            videos = videos.to(device=device, non_blocking=True).float() / 255.0
            videos = videos.to(dtype=action_head.dtype)
            bsz, channels, frames, height, width = videos.shape
            videos = videos.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channels, height, width)
            videos = action_head.normalize_video(videos)
            videos = videos.reshape(bsz, frames, channels, height, width).permute(0, 2, 1, 3, 4)
            videos = videos.to(dtype=action_head.dtype)
        else:
            videos = videos.to(device=device, dtype=action_head.dtype, non_blocking=True)
        videos = videos.to(dtype=torch.bfloat16)

        target_h = getattr(action_head.config, "target_video_height", None)
        target_w = getattr(action_head.config, "target_video_width", None)
        if target_h is None or target_w is None:
            if getattr(action_head.model, "frame_seqlen", None) in (50, 55):
                target_h, target_w = 176, 320
            else:
                target_h, target_w = None, None
        if target_h is not None and target_w is not None:
            _, _, _, height, width = videos.shape
            if (height, width) != (target_h, target_w):
                bsz, channels, frames, _, _ = videos.shape
                videos = torch.nn.functional.interpolate(
                    videos.reshape(bsz * frames, channels, height, width),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                ).reshape(bsz, channels, frames, target_h, target_w)
        return videos[:, :, :1].transpose(1, 2)

    def _recompute_window_anchor(self, model_obs: dict[str, Any]) -> dict[str, Any]:
        action_head = self._action_head()
        if action_head is None:
            return {"enabled": True, "recomputed": False, "reason": "missing_action_head"}

        before = self._context_state()
        if not self._can_recompute_anchor():
            return {"enabled": True, "recomputed": False, "reason": "not_initialized", "before": before}

        anchor_obs = _anchor_obs(model_obs, tuple(self.profile.view_keys))
        _sync_if_cuda()
        started_at = time.perf_counter()
        with torch.inference_mode():
            data = self._normalized_anchor_input(anchor_obs)
            anchor_image = self._prepare_anchor_video(data)
            _, _, _, height, width = anchor_image.shape
            clip_feas, ys, _ = action_head.encode_image(anchor_image, action_head.num_frames, height, width)
            action_head.clip_feas = clip_feas.to(dtype=anchor_image.dtype)
            action_head.ys = ys.to(dtype=anchor_image.dtype)
        _sync_if_cuda()
        elapsed = time.perf_counter() - started_at
        after = self._context_state()

        self.sliding_recompute_count += 1
        self.sliding_recompute_seconds_sum += elapsed
        event = {
            "enabled": True,
            "recomputed": True,
            "rope_mode": "absolute_monotonic",
            "recompute_seconds": elapsed,
            "before": before,
            "after": after,
            "recompute_count": self.sliding_recompute_count,
        }
        self.sliding_events.append(event)
        self.sliding_events = self.sliding_events[-200:]
        LOG.info(
            "sliding-window anchor recomputed in %.3fs with monotonic RoPE before=%s after=%s",
            elapsed,
            before,
            after,
        )
        return event

    def _sliding_summary(self, last_event: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "enabled": True,
            "mode": "absolute_rope_anchor_recompute_keep_kv",
            "hard_reset_disabled": True,
            "recompute_count": int(self.sliding_recompute_count),
            "recompute_seconds_sum": float(self.sliding_recompute_seconds_sum),
            "recompute_seconds_mean": (
                self.sliding_recompute_seconds_sum / self.sliding_recompute_count
                if self.sliding_recompute_count
                else 0.0
            ),
            "infer_seconds_count": int(self.infer_seconds_count),
            "infer_seconds_sum": float(self.infer_seconds_sum),
            "infer_seconds_mean": self.infer_seconds_sum / self.infer_seconds_count if self.infer_seconds_count else 0.0,
            "context_state": self._context_state(),
            "last_event": last_event or {},
        }

    def _write_profile_json(self) -> None:
        if self.sliding_window_profile_json is None:
            return
        payload = self._sliding_summary()
        payload["events"] = self.sliding_events
        self.sliding_window_profile_json.parent.mkdir(parents=True, exist_ok=True)
        self.sliding_window_profile_json.write_text(json.dumps(payload, indent=2))

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
        slide_event = self._recompute_window_anchor(model_obs) if self._can_recompute_anchor() else None
        _sync_if_cuda()
        infer_started_at = time.perf_counter()
        with torch.inference_mode():
            result_batch, video_pred = self.policy.lazy_joint_forward_causal(Batch(obs=model_obs))
        _sync_if_cuda()
        infer_seconds = time.perf_counter() - infer_started_at
        self.infer_seconds_sum += infer_seconds
        self.infer_seconds_count += 1

        actions = self._actions(result_batch)
        self.is_first_call = False
        self.infer_count += 1
        self.recorder.append(
            video_pred if self.profile.save_predicted_videos else None,
            decision=self.infer_count,
            action_steps=int(actions.shape[0]),
            replan_steps=self.profile.replan_steps,
        )
        stats = {
            "shape": list(actions.shape),
            "min": float(np.nanmin(actions)),
            "max": float(np.nanmax(actions)),
            "absmax": float(np.nanmax(np.abs(actions))),
            "has_nan": bool(np.isnan(actions).any()),
            "has_inf": bool(np.isinf(actions).any()),
            "infer_seconds": float(infer_seconds),
            "sliding_window": self._sliding_summary(slide_event),
        }
        self._write_profile_json()
        return {
            "actions": actions,
            "action_horizon": int(actions.shape[0]),
            "replan_steps": int(self.profile.replan_steps),
            "obs_window_policy": self.profile.obs_window_policy.label(),
            "include_current_boundary": bool(self.profile.obs_window_policy.include_current_boundary),
            "profile_name": self.profile.name,
            "resolved_profile": self.profile.name,
            "server_artifacts": {
                "predicted_video_output_root": str(self.recorder.output_root) if self.recorder.output_root else None,
                "sliding_window_profile_json": (
                    str(self.sliding_window_profile_json) if self.sliding_window_profile_json else None
                ),
            },
            "stats": stats,
        }

    def shutdown(self) -> None:
        super().shutdown()
        self._write_profile_json()


class SlidingWindowDistributedDreamZeroLiberoWan22Policy(SlidingWindowDreamZeroLiberoWan22Policy):
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
        self._broadcast_signal(v2.SIGNAL_RESET)
        return result

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        self._broadcast_signal(v2.SIGNAL_INFER)
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
            if value == v2.SIGNAL_SHUTDOWN:
                return
            if value == v2.SIGNAL_RESET:
                self._reset_buffers()
                continue
            if value != v2.SIGNAL_INFER:
                raise RuntimeError(f"unknown distributed signal {value}")
            request = self._recv_request()
            dist.barrier()
            super().infer(request)
            dist.barrier()

    def shutdown(self) -> None:
        if dist.get_rank() == 0:
            super().shutdown()
            self._broadcast_signal(v2.SIGNAL_SHUTDOWN)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experimental lightweight sliding-window LIBERO WAN22 policy server")
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
    parser.add_argument("--save-predicted-videos", dest="save_predicted_videos", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pre-rotate-images-for-policy", dest="pre_rotate_images_for_policy", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--rotate-rollout-frames-for-video", dest="rotate_rollout_frames_for_video", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--output-dir", default=None, help="Server-side predicted-video root")
    parser.add_argument("--predicted-video-fps", type=int, default=None)
    parser.add_argument("--no-predicted-video-watermark", dest="predicted_video_watermark", action="store_false", default=True)
    parser.add_argument("--sliding-window-profile-json", default=None)
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
        output_root = str(Path(args.model_path).resolve().parent / f"libero_wan22_sliding_predictions_{checkpoint_name}")

    from groot.vla.data.schema import EmbodimentTag
    from groot.vla.model.n1_5.sim_policy import GrootSimPolicy

    runtime = v2._init_runtime()
    if runtime["world_size"] > 2:
        raise ValueError("LIBERO eval v2 sliding-window serving supports at most 2 GPUs for CFG parallelism.")
    if runtime["distributed"] and args.cfg_scale is not None and abs(float(args.cfg_scale) - 1.0) < 1e-9:
        raise ValueError("cfg_scale=1.0 disables CFG and is only supported with single-GPU serving in this path.")
    if runtime["distributed"] and runtime["rank"] != 0:
        output_root = None
        args.sliding_window_profile_json = None
    LOG.info("Loading DreamZero WAN22 LIBERO lightweight sliding-window policy from %s", args.model_path)
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
        profile = dataclasses.replace(profile, raw_image_resolution=v2._checkpoint_resolution(policy, profile.raw_image_resolution))
        LOG.info("Using checkpoint LIBERO video resolution %s", profile.raw_image_resolution)
    wrapper_kwargs = {
        "policy": policy,
        "profile": profile,
        "predicted_video_output_root": output_root,
        "predicted_video_watermark": args.predicted_video_watermark,
        "sliding_window_profile_json": args.sliding_window_profile_json,
    }
    if runtime["distributed"]:
        wrapper = SlidingWindowDistributedDreamZeroLiberoWan22Policy(signal_group=runtime["signal_group"], **wrapper_kwargs)
    else:
        wrapper = SlidingWindowDreamZeroLiberoWan22Policy(**wrapper_kwargs)
    config = SlidingWindowLiberoWan22ServerConfig(
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
        sliding_window_profile_json=args.sliding_window_profile_json,
    )
    if runtime["distributed"] and runtime["rank"] != 0:
        LOG.info("Worker rank=%d entering LIBERO WAN22 lightweight sliding-window worker loop", runtime["rank"])
        wrapper.worker_loop()
        return
    LOG.info(
        "Serving lightweight sliding-window LIBERO WAN22 on %s:%d profile=%s resolution=%s replan=%d obs=%s include_boundary=%s world_size=%d",
        args.host,
        args.port,
        profile.name,
        profile.raw_image_resolution,
        profile.replan_steps,
        profile.obs_window_policy.label(),
        profile.obs_window_policy.include_current_boundary,
        runtime["world_size"],
    )
    try:
        WebsocketPolicyServer(wrapper, config, host=args.host, port=args.port).serve_forever()
    finally:
        wrapper.shutdown()


if __name__ == "__main__":
    main()
