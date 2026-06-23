#!/usr/bin/env python3
"""Validate Unitree full-body websocket server outputs on a local dataset.

This is a pre-real-robot safety smoke test.  It reads real dataset frames/state,
sends them to the already-running cloud/server.py websocket endpoint, and checks
that the returned action chunk has the expected schema, finite values, plausible
first-step deltas, and smooth enough consecutive waypoints.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

DREAMZERO_ROOT = Path(__file__).resolve().parents[3]
if str(DREAMZERO_ROOT) not in sys.path:
    sys.path.insert(0, str(DREAMZERO_ROOT))

from eval_utils.policy_client import WebsocketClientPolicy  # noqa: E402
from groot.vla.data.dataset.lerobot import LeRobotSingleDataset, ModalityConfig  # noqa: E402
from groot.vla.data.schema import EmbodimentTag  # noqa: E402

LOG = logging.getLogger("validate_unitree_server_dataset")

VIDEO_KEYS = ["video.head_stereo_left", "video.wrist_left", "video.wrist_right"]
STATE_KEY = "state.sweep_floor_control"
LANGUAGE_KEY = "annotation.task_index"
ACTION_KEY = "action.sweep_floor_control"

ACTION_DIM = 60
ROBOT_Q_DIM = 36
BASE_DIM = 7
HAND_DIM = 12
EXEC_DIM = ROBOT_Q_DIM + HAND_DIM

BASE_XYZ = slice(0, 3)
BASE_QUAT = slice(3, 7)
JOINT = slice(BASE_DIM, ROBOT_Q_DIM)
HAND = slice(ROBOT_Q_DIM, ROBOT_Q_DIM + HAND_DIM)


def _load_dataset(dataset_path: str) -> LeRobotSingleDataset:
    modality_configs = {
        "video": ModalityConfig(delta_indices=[0], modality_keys=VIDEO_KEYS),
        "state": ModalityConfig(delta_indices=[0], modality_keys=[STATE_KEY]),
        "language": ModalityConfig(delta_indices=[0], modality_keys=[LANGUAGE_KEY]),
    }
    return LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_configs,
        embodiment_tag=EmbodimentTag.UNITREE_G1_UPPER_BODY,
        transforms=None,
        use_global_metadata=False,
        video_backend="decord",
        discard_bad_trajectories=False,
    )


def _episode_length(dataset: LeRobotSingleDataset, episode_id: int) -> int:
    for eid, length in zip(dataset.trajectory_ids, dataset.trajectory_lengths):
        if int(eid) == int(episode_id):
            return int(length)
    raise ValueError(f"episode_id={episode_id} not found")


def _read_episode_prompt(dataset_path: Path, episode_id: int, fallback: str) -> str:
    path = dataset_path / "meta" / "episodes.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if int(item.get("episode_index", -1)) == int(episode_id):
                tasks = item.get("tasks") or []
                if tasks:
                    return str(tasks[0])
    return fallback


def _get_step(dataset: LeRobotSingleDataset, episode_id: int, step: int) -> dict[str, Any]:
    keys = VIDEO_KEYS + [STATE_KEY]
    return dataset.get_step_data(
        int(episode_id),
        {key: np.asarray([int(step)], dtype=int) for key in keys},
    )


def _first_frame(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise ValueError(f"expected HWC image, got shape={arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr[..., :3])


def _parse_action(response: dict[str, Any], *, horizon: int) -> np.ndarray:
    if "action" in response:
        action = np.asarray(response["action"], dtype=np.float32)
    elif ACTION_KEY in response:
        action = np.asarray(response[ACTION_KEY], dtype=np.float32)
    elif "robot_q_desired" in response and "hand_cmd" in response:
        robot_q = np.asarray(response["robot_q_desired"], dtype=np.float32)
        hand = np.asarray(response["hand_cmd"], dtype=np.float32)
        pad = np.zeros((robot_q.shape[0], ACTION_DIM - EXEC_DIM), dtype=np.float32)
        action = np.concatenate([robot_q[:, :ROBOT_Q_DIM], hand[:, :HAND_DIM], pad], axis=1)
    else:
        raise RuntimeError(f"response has no action fields; keys={sorted(response)}")
    if action.ndim == 3:
        action = action[0]
    if action.ndim != 2:
        raise RuntimeError(f"expected action [H,D], got {action.shape}")
    if action.shape[0] < horizon or action.shape[1] < ACTION_DIM:
        raise RuntimeError(f"expected action at least [{horizon},{ACTION_DIM}], got {action.shape}")
    return action[:horizon, :ACTION_DIM].astype(np.float32)


def _quat_angle_delta(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    q1 = q1 / np.clip(np.linalg.norm(q1, axis=-1, keepdims=True), 1e-9, None)
    q2 = q2 / np.clip(np.linalg.norm(q2, axis=-1, keepdims=True), 1e-9, None)
    dot = np.abs(np.sum(q1 * q2, axis=-1))
    dot = np.clip(dot, -1.0, 1.0)
    return (2.0 * np.arccos(dot)).astype(np.float32)


def _max_abs(arr: np.ndarray) -> float:
    return float(np.max(np.abs(arr))) if arr.size else 0.0


def _summarize_action(action: np.ndarray, state: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    executable = action[:, :EXEC_DIM]
    first = action[0]
    step_delta = np.diff(executable, axis=0)

    initial_base_angle_delta = float(_quat_angle_delta(first[BASE_QUAT][None], state[BASE_QUAT][None])[0])
    step_base_angle_delta = _quat_angle_delta(action[:-1, BASE_QUAT], action[1:, BASE_QUAT])
    summary: dict[str, Any] = {
        "action_shape": list(action.shape),
        "finite": bool(np.isfinite(action).all()),
        "action_min": float(np.nanmin(action)),
        "action_max": float(np.nanmax(action)),
        "action_abs_max": _max_abs(action),
        "initial_base_xyz_delta": _max_abs(first[BASE_XYZ] - state[BASE_XYZ]),
        "initial_base_angle_delta": initial_base_angle_delta,
        "initial_joint_delta": _max_abs(first[JOINT] - state[JOINT]),
        "initial_hand_delta": _max_abs(first[HAND] - state[HAND]),
        "max_step_base_xyz_delta": _max_abs(step_delta[:, BASE_XYZ]),
        "max_step_base_angle_delta": float(np.max(step_base_angle_delta)) if step_base_angle_delta.size else 0.0,
        "max_step_joint_delta": _max_abs(step_delta[:, JOINT]),
        "max_step_hand_delta": _max_abs(step_delta[:, HAND]),
    }
    checks = {
        "finite": summary["finite"],
        "initial_base_xyz_delta": summary["initial_base_xyz_delta"] <= args.max_initial_base_xyz_delta,
        "initial_base_angle_delta": summary["initial_base_angle_delta"] <= args.max_initial_base_angle_delta,
        "initial_joint_delta": summary["initial_joint_delta"] <= args.max_initial_joint_delta,
        "initial_hand_delta": summary["initial_hand_delta"] <= args.max_initial_hand_delta,
        "max_step_base_xyz_delta": summary["max_step_base_xyz_delta"] <= args.max_step_base_xyz_delta,
        "max_step_base_angle_delta": summary["max_step_base_angle_delta"] <= args.max_step_base_angle_delta,
        "max_step_joint_delta": summary["max_step_joint_delta"] <= args.max_step_joint_delta,
        "max_step_hand_delta": summary["max_step_hand_delta"] <= args.max_step_hand_delta,
    }
    summary["checks"] = checks
    summary["ok"] = bool(all(checks.values()))
    return summary


def _make_request(dataset: LeRobotSingleDataset, *, episode_id: int, step: int, prompt: str) -> tuple[dict[str, Any], np.ndarray]:
    data = _get_step(dataset, episode_id, step)
    frames = [_first_frame(data[key]) for key in VIDEO_KEYS]
    state = np.asarray(data[STATE_KEY], dtype=np.float32).reshape(-1)
    if state.size < ACTION_DIM:
        raise ValueError(f"state must have at least {ACTION_DIM} dims, got {state.size}")
    request = {
        "color_0": frames[0],
        "color_2": frames[1],
        "color_3": frames[2],
        "state": state[:ACTION_DIM],
        "prompt": prompt,
    }
    return request, state[:ACTION_DIM]


def _default_steps(episode_length: int, count: int) -> list[int]:
    if count <= 1:
        return [0]
    stop = max(0, min(episode_length - 1, 24 * (count - 1)))
    return np.linspace(0, stop, num=count).round().astype(int).tolist()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dataset-path", default="dataset/stack_blocks_lerobot_gear_50eps")
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--steps", type=int, nargs="*", default=None, help="Dataset frame indices to test.")
    parser.add_argument("--num-steps", type=int, default=4, help="Used only when --steps is omitted.")
    parser.add_argument("--prompt", default="", help="Default: read from dataset meta/episodes.jsonl.")
    parser.add_argument("--action-horizon", type=int, default=48)
    parser.add_argument("--latency-budget", type=float, default=0.8, help="Warn if an infer call exceeds this many seconds.")
    parser.add_argument("--output-json", default="/tmp/dreamzero_server_dataset_validation.json")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when any safety check or latency budget fails.")
    parser.add_argument("--log-level", default="INFO")

    parser.add_argument("--max-initial-joint-delta", type=float, default=0.75)
    parser.add_argument("--max-initial-hand-delta", type=float, default=1.0)
    parser.add_argument("--max-initial-base-xyz-delta", type=float, default=0.30)
    parser.add_argument("--max-initial-base-angle-delta", type=float, default=1.0)
    parser.add_argument("--max-step-joint-delta", type=float, default=0.12)
    parser.add_argument("--max-step-hand-delta", type=float, default=0.25)
    parser.add_argument("--max-step-base-xyz-delta", type=float, default=0.03)
    parser.add_argument("--max-step-base-angle-delta", type=float, default=0.25)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    dataset_path = Path(args.dataset_path).expanduser()
    dataset = _load_dataset(str(dataset_path))
    episode_length = _episode_length(dataset, args.episode_id)
    prompt = args.prompt.strip() or _read_episode_prompt(dataset_path, args.episode_id, "")
    if not prompt:
        raise ValueError("No prompt supplied and no task found in dataset metadata")
    steps = args.steps if args.steps is not None and len(args.steps) > 0 else _default_steps(episode_length, args.num_steps)
    steps = [max(0, min(int(step), episode_length - 1)) for step in steps]

    client = WebsocketClientPolicy(host=args.host, port=args.port)
    metadata = client.get_server_metadata()
    LOG.info("server metadata: %s", metadata)
    try:
        client.reset({})
    except Exception:
        LOG.warning("server reset failed; continuing", exc_info=True)

    results = []
    for step in steps:
        request, state = _make_request(dataset, episode_id=args.episode_id, step=step, prompt=prompt)
        t0 = time.perf_counter()
        response = client.infer(request)
        latency = time.perf_counter() - t0
        action = _parse_action(response, horizon=args.action_horizon)
        summary = _summarize_action(action, state, args)
        summary.update(
            {
                "episode_id": int(args.episode_id),
                "step": int(step),
                "latency_s": float(latency),
                "latency_ok": bool(latency <= args.latency_budget),
                "response_keys": sorted(response.keys()),
            }
        )
        results.append(summary)
        status = "OK" if summary["ok"] else "FAIL"
        latency_status = "OK" if summary["latency_ok"] else "SLOW"
        LOG.info(
            "step=%d %s latency=%.3fs(%s) initial_joint=%.4f step_joint=%.4f initial_hand=%.4f step_hand=%.4f",
            step,
            status,
            latency,
            latency_status,
            summary["initial_joint_delta"],
            summary["max_step_joint_delta"],
            summary["initial_hand_delta"],
            summary["max_step_hand_delta"],
        )

    report = {
        "ok": bool(all(item["ok"] for item in results)),
        "latency_ok": bool(all(item["latency_ok"] for item in results)),
        "dataset_path": str(dataset_path),
        "episode_id": int(args.episode_id),
        "episode_length": int(episode_length),
        "prompt": prompt,
        "server_metadata": metadata,
        "thresholds": {
            "latency_budget": args.latency_budget,
            "max_initial_joint_delta": args.max_initial_joint_delta,
            "max_initial_hand_delta": args.max_initial_hand_delta,
            "max_initial_base_xyz_delta": args.max_initial_base_xyz_delta,
            "max_initial_base_angle_delta": args.max_initial_base_angle_delta,
            "max_step_joint_delta": args.max_step_joint_delta,
            "max_step_hand_delta": args.max_step_hand_delta,
            "max_step_base_xyz_delta": args.max_step_base_xyz_delta,
            "max_step_base_angle_delta": args.max_step_base_angle_delta,
        },
        "results": results,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    LOG.info("wrote report: %s", output_path)

    if args.strict and (not report["ok"] or not report["latency_ok"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
