#!/usr/bin/env python3
"""Profile robotdeploy DreamZero websocket payload phases.

Synthetic profiler for the Unitree DreamZero deployment transport. It measures
msgpack request size plus pack/unpack CPU time for the four payload contracts:

- baseline: legacy 49-frame, 480x640, raw uint8 image history for every camera.
- phase1: send only the server-required sampled model frames plus latest state.
- phase2: phase1 + deterministic client-side center-crop/resize to the training
  per-view resolution (default 176x320 for the Unitree 352x640 checkpoint).
- phase3: phase2 + optional JPEG compression.

It does not connect to the robot or inference server.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from typing import Any

import numpy as np


IMAGE_TRANSPORT_SENTINEL = "__dreamzero_image_transport__"
DEFAULT_IMAGE_KEYS = (
    "observation.images.cam_left_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
DEFAULT_STATE_DIMS = {
    "observation.state.left_ee_rpy": 6,
    "observation.state.right_ee_rpy": 6,
    "observation.state.left_gripper": 1,
    "observation.state.right_gripper": 1,
}


def _mib(nbytes: int) -> float:
    return nbytes / (1024 * 1024)


def _array_nbytes(value: Any) -> int:
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, dict):
        return sum(_array_nbytes(v) for v in value.values())
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    if isinstance(value, (list, tuple)):
        return sum(_array_nbytes(v) for v in value)
    return 0


def _load_msgpack_numpy():
    try:
        from openpi_client import msgpack_numpy

        return msgpack_numpy
    except Exception:
        from unitree_rl.policy import msgpack_numpy  # type: ignore

        return msgpack_numpy


def _center_crop(frame: np.ndarray, scale: float | None) -> np.ndarray:
    if scale is None or scale <= 0 or scale >= 1:
        return frame
    height, width = frame.shape[:2]
    crop_h = max(1, int(round(height * float(scale))))
    crop_w = max(1, int(round(width * float(scale))))
    y0 = max((height - crop_h) // 2, 0)
    x0 = max((width - crop_w) // 2, 0)
    return frame[y0 : y0 + crop_h, x0 : x0 + crop_w]


def _resize_frame(frame: np.ndarray, *, height: int, width: int) -> np.ndarray:
    if frame.shape[:2] == (height, width):
        return frame
    import cv2

    interpolation = cv2.INTER_AREA if height < frame.shape[0] or width < frame.shape[1] else cv2.INTER_LINEAR
    return cv2.resize(frame, (int(width), int(height)), interpolation=interpolation)


def _encode_jpeg_stack(frames: np.ndarray, *, quality: int) -> dict[str, Any]:
    import cv2

    quality = int(np.clip(quality, 1, 100))
    encoded_frames = []
    for frame in frames:
        ok, encoded = cv2.imencode(
            ".jpg",
            np.ascontiguousarray(frame),
            [cv2.IMWRITE_JPEG_QUALITY, quality],
        )
        if not ok:
            raise RuntimeError("JPEG encode failed")
        encoded_frames.append(encoded.tobytes())
    return {
        IMAGE_TRANSPORT_SENTINEL: "jpeg",
        "format": "jpg",
        "quality": quality,
        "dtype": "uint8",
        "shape": list(frames.shape),
        "frames": encoded_frames,
    }


def _sample_offsets(action_horizon: int, window_len: int) -> list[int]:
    if window_len <= 1:
        return [0]
    if action_horizon == 48 and window_len == 8:
        return [0, 6, 12, 18, 24, 30, 36, 42]
    if action_horizon == 24 and window_len == 8:
        return [0, 3, 6, 9, 12, 15, 18, 21]
    if action_horizon == 24 and window_len == 4:
        return [0, 7, 15, 23]
    return np.rint(np.linspace(0, action_horizon - 1, window_len)).astype(int).tolist()


def _selected_indices(*, history: int, action_horizon: int, window_len: int, include_current_boundary: bool) -> list[int]:
    if window_len <= 1:
        return [history - 1]
    block_end = history - 1 if include_current_boundary else max(history - 2, 0)
    block_start = max(0, block_end - (action_horizon - 1))
    return [min(block_start + offset, block_end) for offset in _sample_offsets(action_horizon, window_len)]


def _make_history(
    *,
    frames: int,
    height: int,
    width: int,
    camera_count: int,
    image_pattern: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    images: dict[str, np.ndarray] = {}
    states: dict[str, np.ndarray] = {}
    if image_pattern == "random":
        rng = np.random.default_rng(0)
        base = rng.integers(0, 256, size=(frames, height, width, 3), dtype=np.uint8)
    else:
        base = np.zeros((frames, height, width, 3), dtype=np.uint8)
        yy = np.arange(height, dtype=np.uint16).reshape(1, height, 1)
        xx = np.arange(width, dtype=np.uint16).reshape(1, 1, width)
        for i in range(frames):
            base[i, :, :, 0] = (xx + i) % 256
            base[i, :, :, 1] = (yy + i * 3) % 256
            base[i, :, :, 2] = ((xx // 2 + yy // 2) + i * 7) % 256
    for camera_idx, key in enumerate(DEFAULT_IMAGE_KEYS[:camera_count]):
        images[key] = (base + camera_idx).astype(np.uint8)
    for key, dim in DEFAULT_STATE_DIMS.items():
        states[key] = np.arange(frames * dim, dtype=np.float32).reshape(frames, dim)
    return images, states


def _build_obs(args: argparse.Namespace, phase: str, images: dict[str, np.ndarray], states: dict[str, np.ndarray]) -> dict[str, Any]:
    obs: dict[str, Any] = {}

    if phase == "baseline":
        obs.update(images)
        obs.update(states)
        return obs

    indices = _selected_indices(
        history=args.history_frames,
        action_horizon=args.action_horizon,
        window_len=args.subsequent_frames,
        include_current_boundary=args.include_current_boundary,
    )
    for key, frames in images.items():
        selected = frames[indices]
        if phase in {"phase2", "phase3"}:
            processed = []
            for frame in selected:
                cropped = _center_crop(frame, args.crop_scale)
                processed.append(_resize_frame(cropped, height=args.resize_height, width=args.resize_width))
            selected = np.stack(processed, axis=0).astype(np.uint8)
        if phase == "phase3":
            obs[key] = _encode_jpeg_stack(selected, quality=args.jpeg_quality)
        else:
            obs[key] = selected
    for key, values in states.items():
        obs[key] = values[-1:]
    return obs


def _profile_case(args: argparse.Namespace, phase: str, msgpack_numpy: Any) -> dict[str, Any]:
    images, states = _make_history(
        frames=args.history_frames,
        height=args.height,
        width=args.width,
        camera_count=args.camera_count,
        image_pattern=args.image_pattern,
    )
    packer = msgpack_numpy.Packer()

    build_times = []
    pack_times = []
    unpack_times = []
    packed_size = 0
    raw_mib = 0.0
    for repeat_idx in range(args.repeats):
        t0 = time.perf_counter()
        obs = _build_obs(args, phase, images, states)
        build_times.append((time.perf_counter() - t0) * 1000.0)
        raw_mib = _mib(_array_nbytes(obs))
        payload = {"type": "get_action", "obs": obs}

        t0 = time.perf_counter()
        packed = packer.pack(payload)
        pack_times.append((time.perf_counter() - t0) * 1000.0)
        packed_size = len(packed)

        t0 = time.perf_counter()
        msgpack_numpy.unpackb(packed)
        unpack_times.append((time.perf_counter() - t0) * 1000.0)
        if repeat_idx + 1 < args.repeats:
            del obs, payload, packed

    result = {
        "phase": phase,
        "raw_mib": raw_mib,
        "packed_mib": _mib(packed_size),
        "build_ms_mean": statistics.mean(build_times),
        "pack_ms_mean": statistics.mean(pack_times),
        "unpack_ms_mean": statistics.mean(unpack_times),
    }
    gc.collect()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["baseline", "phase1", "phase2", "phase3", "all"], default="all")
    parser.add_argument("--history-frames", type=int, default=49)
    parser.add_argument("--subsequent-frames", type=int, default=8)
    parser.add_argument("--action-horizon", type=int, default=48)
    parser.add_argument("--include-current-boundary", action="store_true")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--resize-height", type=int, default=176)
    parser.add_argument("--resize-width", type=int, default=320)
    parser.add_argument("--crop-scale", type=float, default=0.95)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--camera-count", type=int, default=3)
    parser.add_argument("--image-pattern", choices=["gradient", "random"], default="random")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a human-readable table")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    msgpack_numpy = _load_msgpack_numpy()
    phases = ["baseline", "phase1", "phase2", "phase3"] if args.phase == "all" else [args.phase]
    rows = [_profile_case(args, phase, msgpack_numpy) for phase in phases]
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return

    print(
        f"{'phase':<10} {'raw MiB':>10} {'packed MiB':>12} "
        f"{'build ms':>10} {'pack ms':>10} {'unpack ms':>10} {'packed vs base':>15}"
    )
    baseline = rows[0]["packed_mib"] if rows else 0.0
    for row in rows:
        ratio = row["packed_mib"] / baseline if baseline else 1.0
        print(
            f"{row['phase']:<10} {row['raw_mib']:10.2f} {row['packed_mib']:12.2f} "
            f"{row['build_ms_mean']:10.1f} {row['pack_ms_mean']:10.1f} "
            f"{row['unpack_ms_mean']:10.1f} {ratio:14.3f}x"
        )


if __name__ == "__main__":
    main()
