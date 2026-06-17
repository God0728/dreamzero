#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio.v2 as imageio
import numpy as np
import websockets.sync.client
from openpi_client import msgpack_numpy

DREAMZERO_ROOT = Path(__file__).resolve().parents[1]
if str(DREAMZERO_ROOT) not in sys.path:
    sys.path.insert(0, str(DREAMZERO_ROOT))


def _resolve_libero_root() -> Path | None:
    candidates = []
    if os.environ.get("LIBERO_ROOT"):
        candidates.append(Path(os.environ["LIBERO_ROOT"]).expanduser())
    candidates.extend(
        [
            DREAMZERO_ROOT.parent / "LIBERO",
            DREAMZERO_ROOT.parent.parent / "LIBERO",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


LIBERO_ROOT = _resolve_libero_root()
if LIBERO_ROOT is not None and str(LIBERO_ROOT) not in sys.path:
    sys.path.insert(0, str(LIBERO_ROOT))

from eval_utils.libero_wan22_config import (  # noqa: E402
    VIDEO_KEYS,
    LiberoWan22Profile,
    profile_from_metadata,
    resolve_benchmark_names,
)

benchmark = None
LIBERO_INIT_STATES_ROOT = None
get_libero_path = None
OffScreenRenderEnv = None
T = None


def _load_libero() -> None:
    global benchmark, LIBERO_INIT_STATES_ROOT, get_libero_path, OffScreenRenderEnv, T
    if benchmark is not None:
        return
    import libero.libero as _libero_pkg
    from libero.libero import benchmark as _benchmark
    from libero.libero import get_libero_path as _get_libero_path
    from libero.libero.envs import OffScreenRenderEnv as _OffScreenRenderEnv
    from robosuite.utils import transform_utils as _T

    benchmark = _benchmark
    libero_pkg_root = Path(_libero_pkg.__file__).resolve().parent
    LIBERO_INIT_STATES_ROOT = Path(os.environ.get("LIBERO_INIT_STATES_ROOT", libero_pkg_root / "init_files"))
    get_libero_path = _get_libero_path
    OffScreenRenderEnv = _OffScreenRenderEnv
    T = _T

LOG = logging.getLogger(__name__)
PING_INTERVAL_SECS = 60
PING_TIMEOUT_SECS = 600
CONNECT_RETRY_SECS = 5
CONNECT_LOG_INTERVAL_SECS = 30


def _connect_websocket(uri: str):
    kwargs = {
        "compression": None,
        "max_size": None,
        "ping_interval": PING_INTERVAL_SECS,
        "ping_timeout": PING_TIMEOUT_SECS,
    }
    try:
        return websockets.sync.client.connect(uri, **kwargs, proxy=None)
    except TypeError:
        return websockets.sync.client.connect(uri, **kwargs)


class Wan22ServerClient:
    def __init__(self, host: str, port: int):
        self.uri = f"ws://{host}:{port}"
        self.packer = msgpack_numpy.Packer()
        self.ws = None
        started_at = time.monotonic()
        next_log_at = started_at
        attempt = 0
        while True:
            attempt += 1
            try:
                self.ws = _connect_websocket(self.uri)
                self.metadata = msgpack_numpy.unpackb(self.ws.recv())
                waited = time.monotonic() - started_at
                if attempt > 1:
                    LOG.info("connected to LIBERO eval v2 server at %s after %.1fs", self.uri, waited)
                return
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                if self.ws is not None:
                    try:
                        self.ws.close()
                    except Exception:
                        pass
                    self.ws = None
                now = time.monotonic()
                if now >= next_log_at:
                    LOG.info(
                        "waiting for LIBERO eval v2 websocket server at %s "
                        "(attempt=%d, last_error=%s: %s)",
                        self.uri,
                        attempt,
                        type(exc).__name__,
                        exc,
                    )
                    next_log_at = now + CONNECT_LOG_INTERVAL_SECS
                time.sleep(CONNECT_RETRY_SECS)

    def close(self) -> None:
        self.ws.close()

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.ws.send(self.packer.pack(payload))
        response = self.ws.recv()
        if isinstance(response, str):
            raise RuntimeError(response)
        return msgpack_numpy.unpackb(response)

    def reset(self, **payload: Any) -> dict[str, Any]:
        payload["endpoint"] = "reset"
        return self._request(payload)

    def infer(self, **payload: Any) -> dict[str, Any]:
        payload["endpoint"] = "infer"
        return self._request(payload)


class ActionQueue:
    def __init__(self, replan_steps: int):
        self.default_replan_steps = max(int(replan_steps), 1)
        self.actions: list[np.ndarray] = []
        self.executed_since_replan = 0
        self.replan_steps = self.default_replan_steps
        self.total_replans = 0
        self.discarded_actions = 0

    def needs_replan(self) -> bool:
        return not self.actions or self.executed_since_replan >= self.replan_steps

    def push(self, actions: np.ndarray, *, replan_steps: int) -> dict[str, Any]:
        arr = np.asarray(actions, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] < 7:
            raise ValueError(f"expected action chunk [T, >=7], got {arr.shape}")
        if self.actions:
            self.discarded_actions += len(self.actions)
        self.replan_steps = min(max(int(replan_steps), 1), int(arr.shape[0]))
        self.actions = [row[:7].copy() for row in arr]
        self.executed_since_replan = 0
        self.total_replans += 1
        return {"chunk_steps": int(arr.shape[0]), "replan_steps": int(self.replan_steps)}

    def pop(self) -> np.ndarray:
        if not self.actions:
            raise RuntimeError("action queue is empty")
        self.executed_since_replan += 1
        return self.actions.pop(0)

    def snapshot(self) -> dict[str, Any]:
        return {
            "queued_actions": len(self.actions),
            "replan_steps": int(self.replan_steps),
            "executed_since_replan": int(self.executed_since_replan),
            "total_replans": int(self.total_replans),
            "discarded_actions": int(self.discarded_actions),
        }


class FrameWindow:
    def __init__(self, profile: LiberoWan22Profile):
        self.profile = profile
        self.action_block_steps = max(int(profile.action_horizon), 1)
        self.first_request = True
        self.buffers = {key: [] for key in VIDEO_KEYS}
        self.last_window_info: dict[str, Any] = {}

    @staticmethod
    def rotate(frame: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(np.asarray(frame, dtype=np.uint8)[::-1, ::-1])

    def observe(self, obs: dict[str, Any]) -> None:
        mapping = {"video.image": "agentview_image", "video.wrist_image": "robot0_eye_in_hand_image"}
        for model_key, env_key in mapping.items():
            frame = np.asarray(obs[env_key], dtype=np.uint8)
            if self.profile.pre_rotate_images_for_policy:
                frame = self.rotate(frame)
            self.buffers[model_key].append(frame)

    def _sample_offsets(self, window_len: int) -> list[int]:
        if window_len <= 1:
            return [0]
        if self.action_block_steps == 24 and window_len == 9:
            return [0, 3, 6, 9, 12, 15, 18, 21, 24]
        if self.action_block_steps == 24 and window_len == 4:
            return [0, 7, 15, 23]
        if self.action_block_steps == 24 and window_len == 8:
            return [0, 3, 6, 9, 12, 15, 18, 21]
        if window_len == 4:
            return np.rint(np.linspace(0, self.action_block_steps - 1, window_len)).astype(int).tolist()
        end = (
            self.action_block_steps
            if self.profile.obs_window_policy.include_current_boundary
            else self.action_block_steps - 1
        )
        return np.rint(np.linspace(0, max(end, 0), window_len)).astype(int).tolist()

    def _indices(self, num_frames: int, window_len: int) -> list[int]:
        if num_frames <= 0:
            raise RuntimeError("no frames recorded")
        if self.first_request or window_len <= 1:
            return [num_frames - 1]
        include_boundary = bool(self.profile.obs_window_policy.include_current_boundary)
        block_end = num_frames - 1 if include_boundary else max(num_frames - 2, 0)
        block_span = self.action_block_steps if include_boundary else self.action_block_steps - 1
        block_start = max(0, block_end - max(block_span, 0))
        return [min(block_start + offset, block_end) for offset in self._sample_offsets(window_len)]

    def build(self, obs: dict[str, Any], prompt: str) -> dict[str, Any]:
        if any(not frames for frames in self.buffers.values()):
            self.observe(obs)
        window_policy = self.profile.obs_window_policy
        window_len = window_policy.initial_frames if self.first_request else window_policy.subsequent_frames
        payload: dict[str, Any] = {}
        self.last_window_info = {}
        for key in VIDEO_KEYS:
            indices = self._indices(len(self.buffers[key]), window_len)
            frames = [self.buffers[key][index] for index in indices]
            payload[key] = frames[0] if len(frames) == 1 else np.stack(frames, axis=0).astype(np.uint8)
            self.last_window_info[key] = {
                "indices": [int(index) for index in indices],
                "recorded_frames": len(self.buffers[key]),
                "window_len": int(window_len),
                "action_block_steps": int(self.action_block_steps),
                "first_request": bool(self.first_request),
                "include_current_boundary": bool(window_policy.include_current_boundary),
                "sample_offsets": self._sample_offsets(window_len),
            }
        payload.update(extract_state(obs))
        payload["annotation.task_index"] = prompt
        return payload

    def step_complete(self) -> None:
        self.first_request = False


def extract_state(obs: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        "state.ee_pos": np.asarray(obs["robot0_eef_pos"], dtype=np.float64).reshape(1, -1),
        "state.ee_ori": np.asarray(T.quat2axisangle(obs["robot0_eef_quat"]), dtype=np.float64).reshape(1, -1),
        "state.gripper_pos": np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64).reshape(1, -1),
    }


def dummy_action() -> list[float]:
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def map_gripper(action: np.ndarray) -> np.ndarray:
    mapped = np.asarray(action, dtype=np.float32).copy()
    mapped[-1] = -1.0 if float(mapped[-1]) >= 0.5 else 1.0
    return mapped


def rollout_frame(obs: dict[str, Any], profile: LiberoWan22Profile) -> np.ndarray:
    frames = []
    for key in ("agentview_image", "robot0_eye_in_hand_image"):
        frame = np.asarray(obs[key], dtype=np.uint8)
        if profile.rotate_rollout_frames_for_video:
            frame = FrameWindow.rotate(frame)
        frames.append(frame)
    return np.concatenate(frames, axis=1)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def save_video(path: Path, frames: list[np.ndarray], fps: int) -> str | None:
    if not frames:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, codec="libx264")
    return str(path)


def episode_dir(
    base: Path,
    profile: LiberoWan22Profile,
    eval_run: str,
    benchmark_name: str,
    task_id: int,
    episode_index: int,
) -> Path:
    return (
        base
        / eval_run
        / "profiles"
        / profile.name
        / benchmark_name
        / f"task_{task_id:02d}"
        / f"episode_{episode_index:03d}"
    )


def bddl_file_path(task: Any) -> str:
    if get_libero_path is None:
        raise RuntimeError("LIBERO get_libero_path is not loaded")
    return os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)


def make_env(task: Any, profile: LiberoWan22Profile, args: argparse.Namespace) -> Any:
    kwargs: dict[str, Any] = {
        "bddl_file_name": bddl_file_path(task),
        "camera_heights": int(profile.raw_image_resolution[0]),
        "camera_widths": int(profile.raw_image_resolution[1]),
    }
    return OffScreenRenderEnv(**kwargs)


def load_init_states(task: Any) -> Any:
    if LIBERO_INIT_STATES_ROOT is None:
        raise RuntimeError("LIBERO is not loaded")
    import torch

    return torch.load(LIBERO_INIT_STATES_ROOT / task.problem_folder / task.init_states_file)


def init_episode(env: Any, init_state: Any, profile: LiberoWan22Profile) -> dict[str, Any]:
    obs = env.reset()
    maybe_obs = env.set_init_state(init_state)
    if maybe_obs is not None:
        obs = maybe_obs
    for _ in range(profile.wait_steps):
        obs, _, _, _ = env.step(dummy_action())
    if obs is None:
        raise RuntimeError("failed to initialize LIBERO observation")
    return obs


def infer_payload(
    *,
    obs_payload: dict[str, Any],
    session_id: str,
    eval_run_name: str,
    benchmark_name: str,
    task_id: int,
    episode_index: int,
    prompt: str,
    profile: LiberoWan22Profile,
) -> dict[str, Any]:
    return {
        "obs": obs_payload,
        "session_id": session_id,
        "eval_run_name": eval_run_name,
        "benchmark": benchmark_name,
        "task_id": int(task_id),
        "episode_index": int(episode_index),
        "prompt": prompt,
        "profile": profile.name,
    }


def reset_server(
    client: Wan22ServerClient,
    *,
    session_id: str,
    eval_run_name: str,
    benchmark_name: str,
    task_id: int,
    episode_index: int,
    prompt: str,
    profile: LiberoWan22Profile,
) -> dict[str, Any]:
    return client.reset(
        session_id=session_id,
        eval_run_name=eval_run_name,
        benchmark=benchmark_name,
        task_id=int(task_id),
        episode_index=int(episode_index),
        prompt=prompt,
        profile=profile.name,
    )


def summarize_actions(actions: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(actions, dtype=np.float32)
    return {
        "shape": list(arr.shape),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "absmax": float(np.nanmax(np.abs(arr))),
        "has_nan": bool(np.isnan(arr).any()),
        "has_inf": bool(np.isinf(arr).any()),
    }


def evaluate_episode(
    *,
    task_suite: Any,
    task_id: int,
    episode_index: int,
    client: Wan22ServerClient,
    profile: LiberoWan22Profile,
    args: argparse.Namespace,
    benchmark_name: str,
    env: Any | None = None,
    init_states: Any | None = None,
) -> dict[str, Any]:
    task = task_suite.get_task(task_id)
    if init_states is None:
        init_states = load_init_states(task)
    owns_env = env is None
    if env is None:
        env = make_env(task, profile, args)
        env.seed(args.seed)
    session_id = f"{profile.name}-{benchmark_name}-task{task_id:02d}-ep{episode_index:03d}-{uuid.uuid4().hex[:8]}"
    reset_server(
        client,
        session_id=session_id,
        eval_run_name=args.eval_run_name,
        benchmark_name=benchmark_name,
        task_id=task_id,
        episode_index=episode_index,
        prompt=task.language,
        profile=profile,
    )
    obs = init_episode(env, init_states[episode_index % len(init_states)], profile)
    frames = [rollout_frame(obs, profile)] if profile.save_rollout_videos else []
    window = FrameWindow(profile)
    window.observe(obs)
    queue = ActionQueue(profile.replan_steps)
    max_steps = profile.max_steps_for_benchmark(benchmark_name)
    chunks: list[dict[str, Any]] = []
    success = False
    env_steps = 0
    infer_times: list[float] = []
    try:
        for _ in range(max_steps):
            if queue.needs_replan():
                request_obs = window.build(obs, task.language)
                start = time.perf_counter()
                response = client.infer(
                    **infer_payload(
                        obs_payload=request_obs,
                        session_id=session_id,
                        eval_run_name=args.eval_run_name,
                        benchmark_name=benchmark_name,
                        task_id=task_id,
                        episode_index=episode_index,
                        prompt=task.language,
                        profile=profile,
                    )
                )
                infer_times.append(time.perf_counter() - start)
                actions = np.asarray(response["actions"], dtype=np.float32)
                load = queue.push(actions, replan_steps=int(response.get("replan_steps", profile.replan_steps)))
                window.step_complete()
                chunks.append(
                    {
                        "actions": summarize_actions(actions),
                        "queue_load": load,
                        "frame_window": window.last_window_info,
                        "stats": response.get("stats", {}),
                    }
                )
            obs, _, done, _ = env.step(map_gripper(queue.pop()).tolist())
            env_steps += 1
            window.observe(obs)
            if profile.save_rollout_videos:
                frames.append(rollout_frame(obs, profile))
            if bool(done):
                success = True
                break
    finally:
        if owns_env:
            env.close()
    out_dir = episode_dir(Path(args.output_dir), profile, args.eval_run_name, benchmark_name, task_id, episode_index)
    rollout_path = save_video(out_dir / "rollout.mp4", frames, args.video_fps) if profile.save_rollout_videos else None
    predicted_path = None
    if profile.save_predicted_videos:
        flush = reset_server(
            client,
            session_id=f"{session_id}-flush",
            eval_run_name=args.eval_run_name,
            benchmark_name=benchmark_name,
            task_id=task_id,
            episode_index=episode_index,
            prompt=task.language,
            profile=profile,
        )
        predicted_path = (flush.get("server_artifacts") or {}).get("predicted_video_path_last_flush")
    result = {
        "episode_index": int(episode_index),
        "session_id": session_id,
        "success": bool(success),
        "num_env_steps": int(env_steps),
        "num_decisions": len(chunks),
        "wait_steps": int(profile.wait_steps),
        "max_steps": int(max_steps),
        "queue": queue.snapshot(),
        "chunks": chunks,
        "rollout_video_path": rollout_path,
        "predicted_video_path": predicted_path,
        "infer_time_seconds": {
            "sum": float(np.sum(infer_times)),
            "mean": float(np.mean(infer_times)) if infer_times else 0.0,
            "max": float(np.max(infer_times)) if infer_times else 0.0,
        },
    }
    write_json(out_dir / "result.json", result)
    LOG.info(
        "episode complete benchmark=%s task=%d ep=%d success=%s env_steps=%d replans=%d infer_mean=%.3fs infer_max=%.3fs",
        benchmark_name,
        task_id,
        episode_index,
        success,
        result["num_env_steps"],
        result["num_decisions"],
        result["infer_time_seconds"]["mean"],
        result["infer_time_seconds"]["max"],
    )
    return result


def evaluate_task(
    task_suite: Any,
    task_id: int,
    client: Wan22ServerClient,
    profile: LiberoWan22Profile,
    args: argparse.Namespace,
    *,
    benchmark_name: str,
) -> dict[str, Any]:
    task = task_suite.get_task(task_id)
    init_states = load_init_states(task)
    episodes = [
        evaluate_episode(
            task_suite=task_suite,
            task_id=task_id,
            episode_index=i,
            client=client,
            profile=profile,
            args=args,
            benchmark_name=benchmark_name,
            init_states=init_states,
        )
        for i in range(args.n_eval)
    ]
    result = {
        "benchmark": benchmark_name,
        "task_id": int(task_id),
        "task_name": task.name,
        "language": task.language,
        "profile_name": profile.name,
        "success_rate": sum(int(ep["success"]) for ep in episodes) / max(len(episodes), 1),
        "episode_results": episodes,
    }
    task_dir = (
        Path(args.output_dir)
        / args.eval_run_name
        / "profiles"
        / profile.name
        / benchmark_name
        / f"task_{task_id:02d}"
    )
    write_json(task_dir / "task_result.json", result)
    return result


def resolve_task_ids(task_suite: Any, requested: list[int] | None) -> list[int]:
    total = int(task_suite.get_num_tasks())
    task_ids = requested if requested else list(range(total))
    bad = [task_id for task_id in task_ids if task_id < 0 or task_id >= total]
    if bad:
        raise ValueError(f"task ids out of range: {bad}; num_tasks={total}")
    return [int(task_id) for task_id in task_ids]


def summarize_task_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    episodes = [ep for result in results for ep in result["episode_results"]]
    successes = sum(int(ep["success"]) for ep in episodes)
    return {
        "num_successes": int(successes),
        "success_rate": successes / max(len(episodes), 1),
        "mean_task_success_rate": float(np.mean([r["success_rate"] for r in results])) if results else 0.0,
        "num_tasks": len(results),
        "num_episodes": len(episodes),
    }


def summarize_infer_time_seconds(results: list[dict[str, Any]]) -> dict[str, Any]:
    episodes = [ep for result in results for ep in result["episode_results"]]
    total = float(sum(float(ep.get("infer_time_seconds", {}).get("sum", 0.0)) for ep in episodes))
    count = int(sum(int(ep.get("num_decisions", 0)) for ep in episodes))
    maximum = float(max((float(ep.get("infer_time_seconds", {}).get("max", 0.0)) for ep in episodes), default=0.0))
    return {
        "sum": total,
        "mean": total / count if count > 0 else 0.0,
        "max": maximum,
        "count": count,
    }


def write_aggregate(
    output_dir: Path,
    profile: LiberoWan22Profile,
    args: argparse.Namespace,
    benchmark_name: str,
    results: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    summary = summarize_task_results(results)
    write_json(
        output_dir
        / args.eval_run_name
        / "profiles"
        / profile.name
        / benchmark_name
        / "aggregate_results.json",
        {
            "benchmark": benchmark_name,
            "profile_name": profile.name,
            "eval_run_name": args.eval_run_name,
            "summary": summary,
            "task_results": results,
            "server_metadata": metadata,
        },
    )


def write_all_benchmark_aggregate(
    output_dir: Path,
    profile: LiberoWan22Profile,
    args: argparse.Namespace,
    benchmark_results: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> Path:
    all_task_results = [task for item in benchmark_results for task in item["task_results"]]
    benchmark_summaries = [
        {
            "benchmark": item["benchmark"],
            "summary": summarize_task_results(item["task_results"]),
            "task_results": item["task_results"],
        }
        for item in benchmark_results
    ]
    summary = summarize_task_results(all_task_results)
    summary["suite_success_rates"] = {
        item["benchmark"]: item["summary"]["success_rate"] for item in benchmark_summaries
    }
    summary["profile_time_seconds"] = {
        "infer_time_seconds": summarize_infer_time_seconds(all_task_results)
    }
    output_path = output_dir / args.eval_run_name / "profiles" / profile.name / "aggregate_results.json"
    write_json(
        output_path,
        {
            "benchmark": args.benchmark,
            "benchmarks": [item["benchmark"] for item in benchmark_results],
            "profile_name": profile.name,
            "eval_run_name": args.eval_run_name,
            "summary": summary,
            "benchmark_results": benchmark_summaries,
            "server_metadata": metadata,
        },
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standalone DreamZero WAN22 LIBERO eval client")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-run-name", default="default_run")
    parser.add_argument("--benchmark", default="libero_spatial")
    parser.add_argument("--task-ids", nargs="*", type=int, default=None)
    parser.add_argument("--n-eval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-videos", dest="save_videos", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--video-fps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s", force=True)
    _load_libero()
    client = Wan22ServerClient(args.host, args.port)
    try:
        profile = profile_from_metadata(client.metadata, save_rollout_videos=args.save_videos)
        if args.video_fps is None:
            args.video_fps = profile.rollout_video_fps
        benchmark_dict = benchmark.get_benchmark_dict()
        names = resolve_benchmark_names(args.benchmark)
        benchmark_results: list[dict[str, Any]] = []
        for benchmark_name in names:
            if benchmark_name not in benchmark_dict:
                raise ValueError(f"unsupported benchmark {benchmark_name!r}; available={sorted(benchmark_dict)}")
            task_suite = benchmark_dict[benchmark_name]()
            task_ids = resolve_task_ids(task_suite, args.task_ids)
            if args.dry_run:
                args.n_eval = 1
                task_ids = task_ids[:1]
            results = [
                evaluate_task(task_suite, task_id, client, profile, args, benchmark_name=benchmark_name)
                for task_id in task_ids
            ]
            write_aggregate(Path(args.output_dir), profile, args, benchmark_name, results, client.metadata)
            benchmark_results.append({"benchmark": benchmark_name, "task_results": results})
            LOG.info("benchmark=%s profile=%s tasks=%s done", benchmark_name, profile.name, task_ids)
            if args.dry_run:
                break
        if args.benchmark == "libero_all" and len(benchmark_results) == len(names):
            output_path = write_all_benchmark_aggregate(
                Path(args.output_dir),
                profile,
                args,
                benchmark_results,
                client.metadata,
            )
            LOG.info("wrote libero_all aggregate results to %s", output_path)
    finally:
        client.close()


if __name__ == "__main__":
    main()
