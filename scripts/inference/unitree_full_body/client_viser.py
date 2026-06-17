#!/usr/bin/env python3
"""Dataset replay client with image and viser URDF visualization.

This client is intentionally small and replaceable: the dataset replay loop can
be swapped for a real robot loop that sends the same message fields to
``server.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import threading
import time
from typing import Any

import numpy as np

DREAMZERO_ROOT = Path(__file__).resolve().parents[3]
if str(DREAMZERO_ROOT) not in sys.path:
    sys.path.insert(0, str(DREAMZERO_ROOT))

from eval_utils.policy_client import WebsocketClientPolicy  # noqa: E402
from groot.vla.data.dataset.lerobot import LeRobotSingleDataset, ModalityConfig  # noqa: E402
from groot.vla.data.schema import EmbodimentTag  # noqa: E402

LOG = logging.getLogger("unitree_full_body_client")

VIDEO_KEYS = ["video.head_stereo_left", "video.wrist_left", "video.wrist_right"]
STATE_KEY = "state.sweep_floor_control"
LANGUAGE_KEY = "annotation.task_index"
DEFAULT_CONTROL_DIM = 53
ROOT_POSE_DIM = 7
PREDICTED_BODY_DOF = 22
PADDED_BODY_DOF = 7
LEGACY_ROBOT_Q_DIM = ROOT_POSE_DIM + PREDICTED_BODY_DOF
FULL_ROBOT_Q_DIM = LEGACY_ROBOT_Q_DIM + PADDED_BODY_DOF
HAND_CMD_DIM = 12
URDF_CFG_DIM = 51
DEFAULT_IMAGE_WIDTH = 0.72

# ViserUrdf/yourdfpy update_cfg uses actuated joints only. Mimic joints in the
# URDF are driven automatically and must not be included in the cfg vector.
# Actuated joint order:
#   0:22   lower body + waist + left arm
#   22:33  left BrainCo hand actuated joints
#   33:40  right arm
#   40:51  right BrainCo hand actuated joints
BODY_Q_TO_URDF_INDICES = list(range(PREDICTED_BODY_DOF))
# 6DOF hand cmd maps to BrainCo 16DOF URDF local indices [0, 1, 4, 7, 10, 13].
# After removing mimic joints, those become local actuated indices [0, 1, 3, 5, 7, 9].
BRAINCO_HAND_CMD_ACTUATED_INDICES = [0, 1, 3, 5, 7, 9]
LEFT_HAND_URDF_BASE = 22
RIGHT_HAND_URDF_BASE = 40


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
            item = json.loads(line)
            if int(item.get("episode_index", -1)) == int(episode_id):
                tasks = item.get("tasks") or []
                if tasks:
                    return str(tasks[0])
    return fallback


def _get_step(dataset: LeRobotSingleDataset, episode_id: int, step: int) -> dict[str, Any]:
    keys = VIDEO_KEYS + [STATE_KEY]
    data = dataset.get_step_data(
        int(episode_id),
        {key: np.asarray([int(step)], dtype=int) for key in keys},
    )
    return data


def _first_frame(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr[..., :3])


class ViserRobot:
    def __init__(self, *, enabled: bool, urdf_path: Path, port: int, image_width: float):
        self.enabled = enabled
        self.server = None
        self.robot = None
        self.root_frame = None
        self.image_handles: dict[str, Any] = {}
        self.image_layouts: dict[str, tuple[np.ndarray, float, float]] = {}
        self.border_handles: dict[str, Any] = {}
        self.border_styles: dict[str, str] = {}
        self.label_handles: dict[str, Any] = {}
        self.status_handle = None
        self.banner_handle = None
        self.cached_handle = None
        self.timeline_slider = None
        self._updating_timeline = False
        self.image_width = float(image_width)
        if not enabled:
            return
        try:
            import viser
            import viser.extras
        except Exception as exc:
            LOG.warning("viser is not available; URDF visualization disabled: %s", exc)
            self.enabled = False
            return

        self.server = viser.ViserServer(port=port)
        self.server.scene.add_grid(
            "/floor",
            width=8.0,
            height=8.0,
            width_segments=16,
            height_segments=16,
            plane="xy",
            cell_color=(210, 210, 210),
            cell_thickness=0.8,
            cell_size=0.25,
            section_color=(120, 120, 120),
            section_thickness=1.5,
            section_size=1.0,
            position=(0.0, 0.0, 0.0),
        )
        self.root_frame = self.server.scene.add_frame(
            "/unitree_g1",
            show_axes=True,
            axes_length=0.35,
            axes_radius=0.012,
            origin_radius=0.035,
        )
        self.robot = viser.extras.ViserUrdf(
            self.server,
            urdf_or_path=urdf_path,
            root_node_name="/unitree_g1",
        )
        LOG.info("viser running on http://localhost:%d", port)

    def add_controls(self, *, on_start: Any, on_pause: Any, on_seek: Any, max_frame: int) -> None:
        if not self.enabled or self.server is None:
            return
        self.banner_handle = self.server.gui.add_markdown(
            "## READY\n"
            "**Press `s` / Start to run. Press space / Pause to stop.**\n\n"
            "Cached chunks stay on the timeline. Drag the slider to replay cached frames."
        )
        self.status_handle = self.server.gui.add_text("Status", "READY: press Start / terminal s", disabled=True)
        self.cached_handle = self.server.gui.add_text("Cached", "none", disabled=True)
        self.timeline_slider = self.server.gui.add_slider(
            "Timeline frame",
            min=0,
            max=max(0, int(max_frame)),
            step=1,
            initial_value=0,
            hint="Drag to replay frames that have already been inferred and cached.",
        )
        start_button = self.server.gui.add_button("Start / Resume (s)", hint="Run sequential inference and playback.")
        pause_button = self.server.gui.add_button("Pause (space)", hint="Pause inference and playback.")

        @start_button.on_click
        def _(_: Any) -> None:
            on_start()

        @pause_button.on_click
        def _(_: Any) -> None:
            on_pause()

        @self.timeline_slider.on_update
        def _(_: Any) -> None:
            if self._updating_timeline:
                return
            on_seek(int(self.timeline_slider.value))

    def set_status(self, text: str) -> None:
        if self.status_handle is not None:
            try:
                self.status_handle.value = text
            except Exception:
                pass
        if self.banner_handle is not None:
            try:
                self.banner_handle.content = (
                    f"## {text.upper()}\n"
                    "Use **Start / s** to run, **Pause / space** to pause. "
                    "Drag **Timeline frame** to replay cached frames."
                )
            except Exception:
                pass

    def set_timeline(self, *, frame: int, cached_until: int) -> None:
        if hasattr(self, "cached_handle") and self.cached_handle is not None:
            try:
                if cached_until < 0:
                    self.cached_handle.value = "none"
                else:
                    self.cached_handle.value = f"0..{cached_until}"
            except Exception:
                pass
        if hasattr(self, "timeline_slider") and self.timeline_slider is not None:
            try:
                self._updating_timeline = True
                self.timeline_slider.value = int(max(frame, 0))
            except Exception:
                pass
            finally:
                self._updating_timeline = False

    def _set_scene_image(self, name: str, image: np.ndarray, *, row: int, col: int, width: float | None = None) -> None:
        if not self.enabled or self.server is None:
            return
        if width is None:
            width = self.image_width
        image = np.asarray(image, dtype=np.uint8)
        height = max(width * image.shape[0] / max(image.shape[1], 1), 1e-3)
        col_spacing = width * 1.22
        row_spacing = max(height * 1.55, 0.5)
        position = (float(col) * col_spacing - col_spacing, 1.45, 2.1 - float(row) * row_spacing)
        try:
            handle = self.image_handles.get(name)
            if handle is not None and hasattr(handle, "image"):
                handle.image = image
                return
            handle = self.server.scene.add_image(
                name=name,
                image=image,
                render_width=width,
                render_height=height,
                position=position,
                wxyz=(0.7071, 0.7071, 0.0, 0.0),
            )
            self.image_handles[name] = handle
            self.image_layouts[name] = (np.asarray(position, dtype=np.float32), float(width), float(height))
        except Exception as exc:
            LOG.warning("Could not update viser image %s: %s", name, exc)

    def _set_image_borders(self, style: str | None) -> None:
        if not self.enabled or self.server is None:
            return
        color_by_style = {
            "chunk": (255, 40, 40),
            "cache_reset": (20, 220, 80),
        }
        color = color_by_style.get(style or "")
        line_width = 5.0 if style == "chunk" else 7.0
        for name, (position, width, height) in self.image_layouts.items():
            border_name = f"/image_borders{name}"
            handle = self.border_handles.get(border_name)
            current_style = self.border_styles.get(border_name)
            if color is None:
                if handle is not None and hasattr(handle, "visible"):
                    handle.visible = False
                continue
            pad = 0.025
            w = width + pad
            h = height + pad
            z = 0.004
            points = np.asarray(
                [
                    [[-w / 2, -h / 2, z], [w / 2, -h / 2, z]],
                    [[w / 2, -h / 2, z], [w / 2, h / 2, z]],
                    [[w / 2, h / 2, z], [-w / 2, h / 2, z]],
                    [[-w / 2, h / 2, z], [-w / 2, -h / 2, z]],
                ],
                dtype=np.float32,
            )
            try:
                if handle is not None and current_style == style:
                    if hasattr(handle, "visible"):
                        handle.visible = True
                    if hasattr(handle, "position"):
                        handle.position = position
                    continue
                if handle is not None:
                    handle.remove()
                self.border_handles[border_name] = self.server.scene.add_line_segments(
                    border_name,
                    points=points,
                    colors=color,
                    line_width=line_width,
                    position=position,
                    wxyz=(0.7071, 0.7071, 0.0, 0.0),
                )
                self.border_styles[border_name] = style
            except Exception as exc:
                LOG.warning("Could not update image border %s: %s", border_name, exc)

    def _set_label(self, name: str, text: str, *, row: int, col: int) -> None:
        if not self.enabled or self.server is None:
            return
        col_spacing = self.image_width * 1.22
        position = (float(col) * col_spacing - col_spacing, 1.45, 2.38 - float(row) * 0.78)
        try:
            handle = self.label_handles.get(name)
            if handle is not None and hasattr(handle, "text"):
                handle.text = text
                return
            handle = self.server.scene.add_label(
                name=name,
                text=text,
                position=position,
                wxyz=(0.7071, 0.7071, 0.0, 0.0),
            )
            self.label_handles[name] = handle
        except Exception as exc:
            LOG.warning("Could not update viser label %s: %s", name, exc)

    def update_images(
        self,
        input_frames: list[np.ndarray],
        pred_frame_or_video: np.ndarray | None,
        *,
        event_style: str | None = None,
    ) -> None:
        if not self.enabled:
            return
        self._set_label("/labels/gt", "=== GT / DATASET ===", row=0, col=-1)
        self._set_label("/labels/pred", "=== PRED / MODEL ===", row=1, col=-1)

        gt_labels = ["gt_color_0_head", "gt_color_2_left_wrist", "gt_color_3_right_wrist"]
        for idx, (label, frame) in enumerate(zip(gt_labels, input_frames)):
            self._set_scene_image(f"/images/{label}", frame, row=0, col=idx)

        if pred_frame_or_video is None or len(pred_frame_or_video) == 0:
            self._set_image_borders(event_style)
            return
        pred_arr = np.asarray(pred_frame_or_video, dtype=np.uint8)
        pred = pred_arr[0] if pred_arr.ndim == 4 else pred_arr
        for label, frame in _split_predicted_composite(pred).items():
            self._set_scene_image(
                f"/images/pred_{label}",
                frame,
                row=1,
                col={"head": 0, "right_wrist": 1, "left_wrist": 2}[label],
            )
        self._set_image_borders(event_style)

    def update(self, action: np.ndarray) -> None:
        if not self.enabled or self.robot is None:
            return
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        try:
            root_xyz, root_wxyz, body22, hand_cmd = split_action_for_viser(action)
        except ValueError as exc:
            LOG.warning("%s", exc)
            return
        if self.root_frame is not None:
            self.root_frame.position = root_xyz
            self.root_frame.wxyz = root_wxyz
        cfg = action_to_urdf_cfg(body22, hand_cmd)
        try:
            self.robot.update_cfg(cfg)
        except Exception as exc:
            LOG.warning("Could not update URDF cfg with shape %s: %s", cfg.shape, exc)


def _normalize_quat_wxyz(wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(wxyz, dtype=np.float32).reshape(4)
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm < 1e-6:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return quat / norm


def _robot_q_dim_from_action_size(action_size: int) -> int:
    full_control_dim = FULL_ROBOT_Q_DIM + HAND_CMD_DIM + 12
    legacy_control_dim = LEGACY_ROBOT_Q_DIM + HAND_CMD_DIM + 12
    if action_size >= full_control_dim:
        return FULL_ROBOT_Q_DIM
    if action_size >= legacy_control_dim:
        return LEGACY_ROBOT_Q_DIM
    raise ValueError(
        f"Expected action dim >= {legacy_control_dim} "
        f"for xyz+wxyz+22DOF+hand_cmd, got {action_size}"
    )


def split_action_for_viser(action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    robot_q_dim = _robot_q_dim_from_action_size(action.size)
    robot_q = action[:robot_q_dim]
    hand_cmd = action[robot_q_dim : robot_q_dim + HAND_CMD_DIM]
    root_xyz = robot_q[:3]
    root_wxyz = _normalize_quat_wxyz(robot_q[3:7])
    body22 = robot_q[ROOT_POSE_DIM : ROOT_POSE_DIM + PREDICTED_BODY_DOF]
    return root_xyz, root_wxyz, body22, hand_cmd


def action_to_urdf_cfg(body22: np.ndarray, hand_cmd12: np.ndarray) -> np.ndarray:
    body22 = np.asarray(body22, dtype=np.float32).reshape(-1)
    hand_cmd12 = np.asarray(hand_cmd12, dtype=np.float32).reshape(-1)
    if body22.size < PREDICTED_BODY_DOF:
        raise ValueError(f"body joint command must have {PREDICTED_BODY_DOF} dims, got {body22.size}")
    if hand_cmd12.size < HAND_CMD_DIM:
        raise ValueError(f"hand_cmd must have {HAND_CMD_DIM} dims, got {hand_cmd12.size}")

    cfg = np.zeros(URDF_CFG_DIM, dtype=np.float32)
    cfg[np.asarray(BODY_Q_TO_URDF_INDICES, dtype=int)] = body22[:PREDICTED_BODY_DOF]

    left_cmd = hand_cmd12[:6]
    right_cmd = hand_cmd12[6:12]
    left_indices = np.asarray([LEFT_HAND_URDF_BASE + idx for idx in BRAINCO_HAND_CMD_ACTUATED_INDICES], dtype=int)
    right_indices = np.asarray([RIGHT_HAND_URDF_BASE + idx for idx in BRAINCO_HAND_CMD_ACTUATED_INDICES], dtype=int)
    cfg[left_indices] = left_cmd
    cfg[right_indices] = right_cmd
    return cfg


def _split_predicted_composite(frame: np.ndarray) -> dict[str, np.ndarray]:
    frame = np.asarray(frame, dtype=np.uint8)
    h, w = frame.shape[:2]
    cell_h, cell_w = h // 2, w // 2
    if cell_h <= 0 or cell_w <= 0:
        return {"head": frame, "right_wrist": frame, "left_wrist": frame}
    return {
        "head": frame[:cell_h, :cell_w],
        "right_wrist": frame[:cell_h, cell_w : 2 * cell_w],
        "left_wrist": frame[cell_h : 2 * cell_h, :cell_w],
    }


def _sample_video_frames(video: np.ndarray | None, count: int) -> list[np.ndarray | None]:
    if video is None:
        return [None] * count
    arr = np.asarray(video)
    if arr.ndim == 3:
        arr = arr[None]
    if arr.ndim != 4 or len(arr) == 0:
        return [None] * count
    indices = np.linspace(0, len(arr) - 1, num=count).round().astype(int)
    return [np.asarray(arr[idx], dtype=np.uint8) for idx in indices]


def _collect_gt_playback_frames(
    dataset: LeRobotSingleDataset,
    *,
    episode_id: int,
    start_step: int,
    episode_length: int,
    count: int,
    frame_interval: int,
) -> list[list[np.ndarray]]:
    frames: list[list[np.ndarray]] = []
    for idx in range(count):
        step = min(start_step + idx * frame_interval, episode_length - 1)
        data = _get_step(dataset, episode_id, step)
        frames.append([_first_frame(data[key]) for key in VIDEO_KEYS])
    return frames


class TimelineCache:
    def __init__(self) -> None:
        self.actions: dict[int, np.ndarray] = {}
        self.gt_frames: dict[int, list[np.ndarray]] = {}
        self.pred_frames: dict[int, np.ndarray | None] = {}
        self.event_markers: dict[int, str] = {}

    @property
    def cached_until(self) -> int:
        if not self.actions:
            return -1
        return max(self.actions)

    def has_action(self, frame: int) -> bool:
        return int(frame) in self.actions

    def add_chunk(
        self,
        *,
        start_step: int,
        action: np.ndarray,
        gt_frames: list[list[np.ndarray]],
        pred_frames: list[np.ndarray | None],
        video_interval: int,
        episode_length: int,
        event_style: str | None,
    ) -> None:
        action = np.asarray(action, dtype=np.float32)
        for idx in range(action.shape[0]):
            frame = int(start_step + idx)
            if frame >= episode_length:
                break
            self.actions[frame] = action[idx]
        for idx, gt in enumerate(gt_frames):
            frame = min(int(start_step + idx * video_interval), int(episode_length) - 1)
            self.gt_frames[frame] = gt
            self.pred_frames[frame] = pred_frames[idx] if idx < len(pred_frames) else None
            if idx == 0 and event_style is not None:
                self.event_markers[frame] = event_style

    def nearest_image_frame(self, frame: int) -> int | None:
        if not self.gt_frames:
            return None
        frame = int(frame)
        candidates = [key for key in self.gt_frames if key <= frame]
        if candidates:
            return max(candidates)
        return min(self.gt_frames)

    def render(self, robot: ViserRobot, frame: int, *, force_images: bool = False) -> None:
        frame = int(frame)
        action = self.actions.get(frame)
        if action is not None:
            robot.update(action)
        image_frame = self.nearest_image_frame(frame)
        if image_frame is not None and (force_images or image_frame == frame):
            robot.update_images(
                self.gt_frames[image_frame],
                self.pred_frames.get(image_frame),
                event_style=self.event_markers.get(image_frame),
            )


class ModelCacheEventTracker:
    def __init__(self, *, local_attn_size: int, num_frame_per_block: int) -> None:
        self.local_attn_size = int(local_attn_size)
        self.num_frame_per_block = int(num_frame_per_block)
        self.current_start_frame = 0

    def next_event_style(self) -> str:
        if self.local_attn_size > 0 and self.current_start_frame >= self.local_attn_size:
            self.current_start_frame = 0
            return "cache_reset"
        return "chunk"

    def mark_inference_done(self) -> None:
        if self.current_start_frame == 0:
            self.current_start_frame += 1
        self.current_start_frame += self.num_frame_per_block


def _infer_chunk_to_cache(
    *,
    client: WebsocketClientPolicy,
    dataset: LeRobotSingleDataset,
    episode_id: int,
    episode_length: int,
    step: int,
    prompt: str,
    robot: ViserRobot,
    video_fps: float,
    action_hz: float,
    control_dim: int,
    cache: TimelineCache,
    event_style: str | None,
) -> int:
    data = _get_step(dataset, episode_id, step)
    frames = [_first_frame(data[key]) for key in VIDEO_KEYS]
    state_arr = np.asarray(data[STATE_KEY], dtype=np.float32).reshape(-1)
    if state_arr.size < control_dim:
        raise ValueError(f"Dataset state has dim {state_arr.size}, expected at least {control_dim}")
    state = state_arr[:control_dim]
    request = {
        "color_0": frames[0],
        "color_2": frames[1],
        "color_3": frames[2],
        "state": state,
        "prompt": prompt,
    }

    robot.set_status(f"inference step {step}")
    t0 = time.perf_counter()
    response = client.infer(request)
    dt = time.perf_counter() - t0
    action = np.asarray(response["action"], dtype=np.float32)
    LOG.info("step=%d latency=%.3fs action_shape=%s", step, dt, action.shape)

    video_count = max(1, int(round(action.shape[0] / max(action_hz / max(video_fps, 1e-6), 1e-6))))
    frame_interval = max(1, int(round(action_hz / max(video_fps, 1e-6))))
    gt_frames = _collect_gt_playback_frames(
        dataset,
        episode_id=episode_id,
        start_step=step,
        episode_length=episode_length,
        count=video_count,
        frame_interval=frame_interval,
    )
    pred_video = response.get("pred_video")
    pred_video_arr = np.asarray(pred_video) if pred_video is not None else None
    pred_frames = _sample_video_frames(pred_video_arr, video_count)
    cache.add_chunk(
        start_step=step,
        action=action,
        gt_frames=gt_frames,
        pred_frames=pred_frames,
        video_interval=frame_interval,
        episode_length=episode_length,
        event_style=event_style,
    )
    return int(action.shape[0])


def _start_terminal_key_listener(
    *,
    start_event: threading.Event,
    pause_event: threading.Event,
    stop_event: threading.Event,
) -> threading.Thread | None:
    if not sys.stdin.isatty():
        return None

    def _loop() -> None:
        import select
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not stop_event.is_set():
                readable, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not readable:
                    continue
                char = sys.stdin.read(1)
                if char.lower() == "s":
                    pause_event.clear()
                    start_event.set()
                elif char == " ":
                    pause_event.set()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    thread = threading.Thread(target=_loop, name="unitree-client-keyboard", daemon=True)
    thread.start()
    return thread


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--dataset-path",
        default="/mnt/unitree_cpfs/ruixuan/datasets/gear_format/data_sweep_floor_lerobot_gear_100eps",
    )
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--stride", type=int, default=48, help="Replay step interval between inference calls.")
    parser.add_argument("--action-playback-hz", type=float, default=30.0)
    parser.add_argument("--video-playback-fps", type=float, default=5.0)
    parser.add_argument(
        "--local-attn-size",
        type=int,
        default=9,
        help="Mirror of model.local_attn_size for cache-reset timeline markers. Default 4*2+1.",
    )
    parser.add_argument(
        "--num-frame-per-block",
        type=int,
        default=2,
        help="Mirror of model num_frame_per_block for cache-reset timeline markers.",
    )
    parser.add_argument("--control-dim", type=int, default=0, help="0 means use server metadata action_dim.")
    parser.add_argument(
        "--image-width",
        type=float,
        default=DEFAULT_IMAGE_WIDTH,
        help="Scene render width for each GT/PRED image plane. Increase this to enlarge videos.",
    )
    parser.add_argument("--auto-play", action="store_true", help="Run all chunks without waiting for viser buttons.")
    parser.add_argument("--no-viser", action="store_true")
    parser.add_argument(
        "--urdf-path",
        default=str(DREAMZERO_ROOT / "scripts/inference/assets/g1_29dof_mode_15_brainco_hand.urdf"),
    )
    parser.add_argument("--viser-port", type=int, default=8081)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args()

    client = WebsocketClientPolicy(host=args.host, port=args.port)
    metadata = client.get_server_metadata()
    LOG.info("server metadata: %s", metadata)
    control_dim = int(args.control_dim or metadata.get("action_dim") or DEFAULT_CONTROL_DIM)
    LOG.info("Using control_dim=%d for request state/action layout", control_dim)
    client.reset({})

    dataset = _load_dataset(args.dataset_path)
    episode_length = _episode_length(dataset, args.episode_id)
    prompt = args.prompt or _read_episode_prompt(Path(args.dataset_path), args.episode_id, "full-body task")
    robot = ViserRobot(
        enabled=not args.no_viser,
        urdf_path=Path(args.urdf_path),
        port=args.viser_port,
        image_width=args.image_width,
    )

    stride = max(1, int(args.stride))
    action_period = 1.0 / max(float(args.action_playback_hz), 1e-6)
    cache = TimelineCache()
    model_cache_tracker = ModelCacheEventTracker(
        local_attn_size=args.local_attn_size,
        num_frame_per_block=args.num_frame_per_block,
    )
    current_frame = 0
    next_infer_step = 0
    running_event = threading.Event()
    pause_event = threading.Event()
    stop_event = threading.Event()
    seek_lock = threading.Lock()
    seek_frame: list[int | None] = [None]

    def _start() -> None:
        pause_event.clear()
        running_event.set()

    def _pause() -> None:
        running_event.clear()
        pause_event.set()
        robot.set_status(f"paused frame {current_frame}")

    def _seek(frame: int) -> None:
        running_event.clear()
        pause_event.set()
        with seek_lock:
            seek_frame[0] = int(frame)

    if not args.no_viser:
        robot.add_controls(
            on_start=_start,
            on_pause=_pause,
            on_seek=_seek,
            max_frame=max(0, episode_length - 1),
        )
        _start_terminal_key_listener(
            start_event=running_event,
            pause_event=pause_event,
            stop_event=stop_event,
        )
    elif not args.auto_play:
        raise ValueError("Use --auto-play when running without viser.")

    if args.auto_play:
        running_event.set()
    robot.set_status("ready: press Start / terminal s")
    LOG.info("Press 's' or Start to run sequential inference/playback; press space or Pause to pause.")

    try:
        while True:
            requested_seek = None
            with seek_lock:
                if seek_frame[0] is not None:
                    requested_seek = seek_frame[0]
                    seek_frame[0] = None

            if requested_seek is not None:
                if cache.cached_until >= 0:
                    current_frame = int(np.clip(requested_seek, 0, cache.cached_until))
                    cache.render(robot, current_frame, force_images=True)
                    robot.set_status(f"paused replay frame {current_frame}")
                else:
                    current_frame = 0
                    robot.set_status("timeline empty: press Start first")
                robot.set_timeline(frame=current_frame, cached_until=cache.cached_until)

            if pause_event.is_set():
                running_event.clear()
                pause_event.clear()

            if not running_event.is_set():
                robot.set_timeline(frame=current_frame, cached_until=cache.cached_until)
                time.sleep(0.05)
                continue

            if current_frame >= episode_length:
                running_event.clear()
                robot.set_status("complete: full cached timeline is replayable")
                robot.set_timeline(frame=episode_length - 1, cached_until=cache.cached_until)
                LOG.info("episode inference complete; viser remains open for timeline replay")
                if args.no_viser:
                    break
                time.sleep(0.1)
                continue

            if not cache.has_action(current_frame):
                if next_infer_step >= episode_length:
                    running_event.clear()
                    robot.set_status("complete: full cached timeline is replayable")
                    if args.no_viser:
                        break
                    continue
                chunk_start = next_infer_step
                event_style = model_cache_tracker.next_event_style()
                _infer_chunk_to_cache(
                    client=client,
                    dataset=dataset,
                    episode_id=args.episode_id,
                    episode_length=episode_length,
                    step=chunk_start,
                    prompt=prompt,
                    robot=robot,
                    video_fps=args.video_playback_fps,
                    action_hz=args.action_playback_hz,
                    control_dim=control_dim,
                    cache=cache,
                    event_style=event_style,
                )
                model_cache_tracker.mark_inference_done()
                next_infer_step += stride
                robot.set_timeline(frame=current_frame, cached_until=cache.cached_until)
                continue

            t0 = time.perf_counter()
            cache.render(robot, current_frame)
            robot.set_status(f"playing frame {current_frame} / cached 0..{cache.cached_until}")
            robot.set_timeline(frame=current_frame, cached_until=cache.cached_until)
            current_frame += 1
            sleep_time = action_period - (time.perf_counter() - t0)
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        stop_event.set()


if __name__ == "__main__":
    print("Initiating Client")
    main()
