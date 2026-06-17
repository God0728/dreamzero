#!/usr/bin/env python3
"""Websocket inference server for Unitree full-body DreamZero checkpoints.

The server keeps the DreamZero policy loaded, maintains a short RGB frame
history for each camera, and serves 53-D Unitree full-body action chunks:

    robot_q_desired[:29] + hand_cmd[:12] + ee_state[:12]

Client messages use msgpack-numpy and follow the same endpoint convention as
``eval_utils.policy_server``:

    {"endpoint": "infer", "color_0": HWC uint8, "color_2": HWC uint8,
     "color_3": HWC uint8, "state": (53,), "prompt": "..."}

The response contains ``action`` with shape ``[action_horizon, 53]`` and,
optionally, decoded ``pred_video`` frames.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import deque
import datetime as _datetime
import logging
import os
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

DREAMZERO_ROOT = Path(__file__).resolve().parents[3]
if str(DREAMZERO_ROOT) not in sys.path:
    sys.path.insert(0, str(DREAMZERO_ROOT))

LOG = logging.getLogger("unitree_full_body_server")

VIDEO_KEYS = ["video.head_stereo_left", "video.wrist_left", "video.wrist_right"]
CLIENT_IMAGE_ALIASES = {
    "video.head_stereo_left": ("video.head_stereo_left", "color_0", "head", "head_stereo_left"),
    "video.wrist_left": ("video.wrist_left", "color_2", "wrist_left", "left_wrist"),
    "video.wrist_right": ("video.wrist_right", "color_3", "wrist_right", "right_wrist"),
}
STATE_KEY = "state.sweep_floor_control"
ACTION_KEY = "action.sweep_floor_control"
ACTION_DIM = 53
ROBOT_Q_DIM = 29
HAND_DIM = 12
EE_DIM = 12


def _maybe_init_dist(timeout_seconds: int = 600) -> None:
    import torch.distributed as dist

    if dist.is_available() and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29617")
        dist.init_process_group(
            backend="gloo",
            world_size=1,
            rank=0,
            timeout=_datetime.timedelta(seconds=int(timeout_seconds)),
        )


def _reset_action_head_state(policy: Any) -> None:
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


def _as_rgb_frame(value: Any, *, key: str, color_order: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 4:
        arr = arr[-1]
    if arr.ndim != 3 or arr.shape[-1] not in (1, 3, 4):
        raise ValueError(f"{key} must be HWC or THWC image data, got shape={arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    arr = np.ascontiguousarray(arr[..., :3])
    if color_order.lower() == "bgr":
        arr = arr[..., ::-1].copy()
    elif color_order.lower() != "rgb":
        raise ValueError(f"Unsupported image color order {color_order!r}")
    return arr


def _as_state(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < ACTION_DIM:
        raise ValueError(f"state must have at least {ACTION_DIM} values, got {arr.size}")
    if not np.isfinite(arr[:ACTION_DIM]).all():
        raise ValueError("state contains non-finite values")
    return arr[:ACTION_DIM].reshape(1, ACTION_DIM)


def _find_first(obs: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in obs:
            return obs[key]
    raise KeyError(f"Missing any of keys={keys}; received={sorted(obs)}")


def _extract_action(act: Any, *, horizon: int) -> np.ndarray:
    if isinstance(act, dict) and ACTION_KEY in act:
        arr = act[ACTION_KEY]
    elif isinstance(act, dict) and "action" in act:
        arr = act["action"]
    else:
        arr = act[ACTION_KEY]
    try:
        import torch

        if torch.is_tensor(arr):
            arr = arr.detach().cpu().float().numpy()
    except Exception:
        pass
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[-1] < ACTION_DIM:
        raise ValueError(f"Expected action dim >= {ACTION_DIM}, got {arr.shape}")
    if arr.shape[0] < horizon:
        arr = np.concatenate([arr, np.repeat(arr[-1:], horizon - arr.shape[0], axis=0)], axis=0)
    return arr[:horizon, :ACTION_DIM].astype(np.float32)


def _decode_video(policy: Any, video_pred: Any) -> np.ndarray | None:
    if video_pred is None:
        return None
    import torch
    from einops import rearrange

    action_head = policy.trained_model.action_head
    with torch.inference_mode():
        latents = video_pred.detach().to(policy.device)
        frames = action_head.vae.decode(
            latents,
            tiled=action_head.tiled,
            tile_size=(action_head.tile_size_height, action_head.tile_size_width),
            tile_stride=(action_head.tile_stride_height, action_head.tile_stride_width),
        )
        frames = rearrange(frames, "B C T H W -> B T H W C")[0]
        return ((frames.float() + 1.0) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)


class UnitreeFullBodyPolicyServer:
    def __init__(self, args: argparse.Namespace):
        import torch
        from eval_utils.serve_unitree_dreamzero_eef import _configure_torch_dynamo
        from groot.vla.data.schema import EmbodimentTag
        from groot.vla.model.n1_5.sim_policy import GrootSimPolicy

        dynamo = _configure_torch_dynamo(
            torch,
            recompile_limit=args.torch_dynamo_recompile_limit,
            cache_size_limit=args.torch_dynamo_cache_size_limit,
        )
        if dynamo:
            LOG.info("torch._dynamo config: %s", dynamo)
        _maybe_init_dist(args.distributed_timeout_seconds)

        overrides = list(args.model_config_override or [])
        defaults = [
            "action_head_cfg.config.load_text_encoder=false",
            "action_head_cfg.config.torch_compile_dit_blocks=false",
            f"text_embedding_cache_dir={args.t5_cache_dir}",
            "require_text_embedding_cache=true",
            "text_embedding_cache_runtime=model",
        ]
        for item in defaults:
            key = item.split("=", 1)[0]
            if not any(existing.startswith(key + "=") for existing in overrides):
                overrides.append(item)

        LOG.info("Loading checkpoint %s", args.model_path)
        self.policy = GrootSimPolicy(
            embodiment_tag=EmbodimentTag.UNITREE_G1_UPPER_BODY,
            model_path=args.model_path,
            device=args.device,
            model_config_overrides=overrides,
            tokenizer_path_override=args.tokenizer_path_override,
            skip_assert_delta_indices=True,
            skip_img_transform=False,
        )
        self.policy.trained_model.action_head.cfg_scale = float(args.cfg_scale)
        self.args = args
        self.histories = {key: deque(maxlen=args.action_horizon + 1) for key in VIDEO_KEYS}
        self.last_latent_video = None
        self.request_index = 0
        _reset_action_head_state(self.policy)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "policy": "unitree_full_body_dreamzero",
            "model_path": self.args.model_path,
            "image_keys": {
                "head": "color_0",
                "wrist_left": "color_2",
                "wrist_right": "color_3",
            },
            "state_key": "state or state.sweep_floor_control",
            "action_key": ACTION_KEY,
            "action_dim": ACTION_DIM,
            "action_layout": {
                "robot_q_desired": [0, ROBOT_Q_DIM],
                "hand_cmd": [ROBOT_Q_DIM, ROBOT_Q_DIM + HAND_DIM],
                "ee_state": [ROBOT_Q_DIM + HAND_DIM, ACTION_DIM],
            },
            "action_horizon": self.args.action_horizon,
            "video_stride": self.args.video_stride,
            "eval_mode": self.args.eval_mode,
        }

    def reset(self, _: dict[str, Any] | None = None) -> dict[str, Any]:
        for history in self.histories.values():
            history.clear()
        self.last_latent_video = None
        self.request_index = 0
        _reset_action_head_state(self.policy)
        return {"ok": True, "message": "server state reset"}

    def _video_window(self, key: str) -> np.ndarray:
        history = self.histories[key]
        if not history:
            raise RuntimeError(f"No frame history for {key}")
        if len(history) <= 1:
            return history[-1]
        needed = self.args.action_horizon + 1
        frames = list(history)
        if len(frames) < needed:
            frames = [frames[0]] * (needed - len(frames)) + frames
        window = frames[-needed :: self.args.video_stride]
        if len(window) == 1:
            return window[0]
        return np.stack(window, axis=0)

    def _build_model_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        for model_key, aliases in CLIENT_IMAGE_ALIASES.items():
            frame = _as_rgb_frame(
                _find_first(obs, aliases),
                key=model_key,
                color_order=self.args.client_image_color_order,
            )
            self.histories[model_key].append(frame)

        state_value = obs.get(STATE_KEY, obs.get("state"))
        if state_value is None:
            raise KeyError(f"Missing state; provide 'state' or {STATE_KEY!r}")
        prompt = str(obs.get("prompt") or self.args.prompt).strip()
        if not prompt:
            raise ValueError("Missing prompt; pass --prompt or include prompt in request")

        model_obs = {STATE_KEY: _as_state(state_value), "annotation.task_index": prompt}
        for key in VIDEO_KEYS:
            model_obs[key] = self._video_window(key)
        return model_obs

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        import torch
        from tianshou.data import Batch

        if self.args.eval_mode == "teacher_forcing":
            _reset_action_head_state(self.policy)
            latent_video = None
        elif self.args.eval_mode == "open_loop":
            latent_video = self.last_latent_video
        else:
            latent_video = None

        model_obs = self._build_model_obs(obs)
        with torch.inference_mode():
            result_batch, video_pred = self.policy.lazy_joint_forward_causal(
                Batch(obs=model_obs),
                latent_video=latent_video,
            )
        action = _extract_action(result_batch.act, horizon=self.args.action_horizon)
        if video_pred is not None:
            self.last_latent_video = video_pred.detach()
        self.request_index += 1

        response: dict[str, Any] = {
            "ok": True,
            "request_index": self.request_index,
            "action": action,
            ACTION_KEY: action,
            "robot_q_desired": action[:, :ROBOT_Q_DIM],
            "hand_cmd": action[:, ROBOT_Q_DIM : ROBOT_Q_DIM + HAND_DIM],
            "ee_state": action[:, ROBOT_Q_DIM + HAND_DIM : ACTION_DIM],
        }
        if self.args.return_video:
            pred_video = _decode_video(self.policy, video_pred)
            if pred_video is not None:
                response["pred_video"] = pred_video
        return response


class MsgpackWebsocketServer:
    def __init__(self, policy: UnitreeFullBodyPolicyServer, *, host: str, port: int):
        self.policy = policy
        self.host = host
        self.port = port

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        from openpi_client import msgpack_numpy
        import websockets.asyncio.server as websocket_server
        import websockets.frames

        packer = msgpack_numpy.Packer()

        async def handler(websocket):
            LOG.info("Connection opened from %s", getattr(websocket, "remote_address", None))
            await websocket.send(packer.pack(self.policy.metadata))
            while True:
                try:
                    obs = msgpack_numpy.unpackb(await websocket.recv())
                    endpoint = obs.pop("endpoint", "infer")
                    if endpoint == "reset":
                        response = self.policy.reset(obs)
                    elif endpoint == "metadata":
                        response = self.policy.metadata
                    elif endpoint == "infer":
                        response = self.policy.infer(obs)
                    else:
                        raise ValueError(f"Unknown endpoint {endpoint!r}")
                    await websocket.send(packer.pack(response))
                except websockets.ConnectionClosed:
                    LOG.info("Connection closed from %s", getattr(websocket, "remote_address", None))
                    break
                except Exception:
                    await websocket.send(traceback.format_exc())
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error",
                    )
                    raise

        async with websocket_server.serve(
            handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
        ) as server:
            LOG.info("Serving Unitree full-body policy on ws://%s:%d", self.host, self.port)
            await server.serve_forever()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt", default="", help="Default task prompt. Clients may override it per request.")
    parser.add_argument("--action-horizon", type=int, default=48)
    parser.add_argument("--video-stride", type=int, default=6)
    parser.add_argument("--eval-mode", choices=["causal_gt", "teacher_forcing", "open_loop"], default="causal_gt")
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--return-video", action="store_true")
    parser.add_argument("--client-image-color-order", choices=["rgb", "bgr"], default="rgb")
    parser.add_argument(
        "--t5-cache-dir",
        default="/mnt/unitree_cpfs/ruixuan/cache/dreamzero/t5_cache/unitree_sweep_floor_100eps",
    )
    parser.add_argument("--tokenizer-path-override", default=None)
    parser.add_argument("--model-config-override", action="append", default=[])
    parser.add_argument("--distributed-timeout-seconds", type=int, default=600)
    parser.add_argument("--torch-dynamo-recompile-limit", type=int, default=800)
    parser.add_argument("--torch-dynamo-cache-size-limit", type=int, default=800)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args()
    policy = UnitreeFullBodyPolicyServer(args)
    MsgpackWebsocketServer(policy, host=args.host, port=args.port).serve_forever()


if __name__ == "__main__":
    main()
