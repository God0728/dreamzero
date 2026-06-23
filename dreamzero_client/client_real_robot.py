#!/usr/bin/env python3
"""Real-robot inference client for Unitree G1 full-body DreamZero checkpoints.

This client talks to ``server.py`` over the same msgpack-numpy websocket protocol
as ``client_viser.py``, but instead of replaying a dataset it streams live
observations from the robot and executes the predicted action chunk.

Realtime design (lowest-latency / smoothest control)
----------------------------------------------------
DreamZero inference is slow (hundreds of ms per chunk) while real-robot control
must stay smooth.  We decouple the two with a double-buffered, prefetching
architecture over a single persistent websocket connection:

    [Inference thread]  capture obs -> server.infer() -> next 48-step chunk
                              |                                    |
                              v                                    v
                       (prefetched while the current chunk plays)  |
    [Control thread]   emit waypoint @ control-hz  <---------------+
                       (never blocks on inference; holds last cmd if late)

* The control thread plays the active chunk at a fixed rate and is never blocked
  by network/inference latency.
* When the active chunk is ``replan-stride`` steps in, the inference thread is
  asked to compute the next chunk.  The freshly predicted chunk is swapped in at
  the next stride boundary, so control is continuous.
* If the next chunk is not ready in time, the control thread simply holds the
  last commanded setpoint until it arrives (safe degradation, no jumps).

60-D control contract (see dataset/meta/modality.json, datasetinfo.md)
----------------------------------------------------------------------
state  (sent to server) = robot_q_current[:36] + hand_state[:12] + ee_state[:12]
action (returned)       = robot_q_desired[:36] + hand_cmd[:12]   + ee_state[:12]
    robot_q[0:36]  = base pose(7) + 29 body joints  (ALL fed to the WBC)
    hand[36:48]    = left hand(6) + right hand(6)  (executed)
    ee_state[48:60] = informational only (NOT executed)

The full 36-D ``robot_q_desired`` (base7 + joints29) is fed to the WBC
``SecureMotionInferencer`` as its target so the floating base is valid, exactly
like teleop feeds GMR ``raw_qpos``; ``hand_cmd`` (12) goes to the hand controller.

Integrating a real Unitree G1
-----------------------------
:class:`UnitreeG1RobotInterface` is wired to the teleop data-collection stack in
``/home/unitree/jimmy/wbc_pico_record`` (the same code that recorded the 60-D
training data) and drives the robot over ``unitree_sdk2py`` DDS:

* cameras   : ZMQ ``teleop.image_server.image_client.ImageClient`` ->
              color_0 (head left) / color_2 (left wrist) / color_3 (right wrist)
* state     : ``rt/lowstate`` LowState (29 joints + IMU) + hand states + FK ee
* joints out: predicted ``robot_q_desired[7:36]`` -> WBC ``SecureMotionInferencer``
              -> ``rt/fsm/teleop/cmd`` (FSM 504)
* hands  out: ``hand_cmd`` (12) -> hand controller ``set_hand_targets``

The 29-joint order follows ``datasetinfo.md`` robot_q[7:36]: legs(12) + waist(3)
+ left arm(7) + right arm(7).

A :class:`DummyRobotInterface` is provided so the full pipeline can be validated
against a running server without any hardware.
"""

from __future__ import annotations

import abc
import argparse
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Any

import numpy as np

CLIENT_DIR = Path(__file__).resolve().parent
if str(CLIENT_DIR) not in sys.path:
    sys.path.insert(0, str(CLIENT_DIR))

try:
    from policy_client import WebsocketClientPolicy  # noqa: E402
except ImportError:
    DREAMZERO_ROOT = Path(__file__).resolve().parents[1]
    if str(DREAMZERO_ROOT) not in sys.path:
        sys.path.insert(0, str(DREAMZERO_ROOT))
    from eval_utils.policy_client import WebsocketClientPolicy  # noqa: E402

LOG = logging.getLogger("unitree_full_body_real_robot")

# ----------------------------------------------------------------------------- #
# 60-D full-body control layout
# ----------------------------------------------------------------------------- #
ROBOT_Q_DIM = 36          # base_pose(7) + robot joints(29)
BASE_DIM = 7              # floating base pose, not executable
JOINT_DIM = ROBOT_Q_DIM - BASE_DIM  # 29 actuated body joints
HAND_DIM = 12             # left hand(6) + right hand(6)
EE_DIM = 12               # left/right xyzrpy, informational only
STATE_DIM = ROBOT_Q_DIM + HAND_DIM + EE_DIM  # 60
EXEC_DIM = ROBOT_Q_DIM + HAND_DIM            # 48 executed dims per waypoint (full robot_q + hand)

# Slices inside the executed 48-D waypoint.
WP_ROBOT_Q = slice(0, ROBOT_Q_DIM)           # full 36-D robot_q = base(7) + joints(29)
WP_HAND = slice(ROBOT_Q_DIM, ROBOT_Q_DIM + HAND_DIM)


def _default_wbc_repo() -> str:
    """Prefer the parent repo when this folder is copied under wbc_pico_record."""
    env = os.environ.get("WBC_PICO_REPO")
    if env:
        return env
    parent = Path(__file__).resolve().parents[1]
    if (parent / "utils" / "inference.py").exists() and (parent / "assets" / "g1").exists():
        return str(parent)
    return "/home/unitree/jimmy/wbc_pico_record"


@dataclass
class RobotObservation:
    """A single synchronized observation from the robot."""

    color_0: np.ndarray  # head camera, HWC uint8 RGB
    color_2: np.ndarray  # left wrist, HWC uint8 RGB
    color_3: np.ndarray  # right wrist, HWC uint8 RGB
    state: np.ndarray    # (60,) float32: robot_q[:36] + hand[:12] + ee[:12]


@dataclass
class SafetyConfig:
    """Command guards for first real-robot runs and chunk-by-chunk debugging."""

    enabled: bool = True
    max_initial_joint_delta: float = 0.75
    max_initial_hand_delta: float = 1.0
    max_initial_base_xyz_delta: float = 0.30
    max_initial_base_angle_delta: float = 1.0
    max_step_joint_delta: float = 0.12
    max_step_hand_delta: float = 0.25
    max_step_base_xyz_delta: float = 0.03
    max_step_base_angle_delta: float = 0.25


class SafetyViolation(RuntimeError):
    """Raised when a predicted command is too far from the safe envelope."""


# ----------------------------------------------------------------------------- #
# Robot interface boundary
# ----------------------------------------------------------------------------- #
class RobotInterface(abc.ABC):
    """Hardware abstraction. Implement these three methods for your robot."""

    @abc.abstractmethod
    def read_observation(self) -> RobotObservation:
        """Return the latest synchronized cameras + 60-D state."""

    @abc.abstractmethod
    def command(self, robot_q_36: np.ndarray, hand_cmd_12: np.ndarray) -> None:
        """Push one setpoint: full 36-D robot_q (base7 + joints29) + 12-D hand."""

    def reset(self) -> None:
        """Optional: move to a safe start pose / clear internal buffers."""

    def close(self) -> None:
        """Optional: release hardware resources."""

    def emergency_stop(self) -> None:
        """Optional: immediately leave the active control mode."""
        self.close()


class DummyRobotInterface(RobotInterface):
    """No-hardware stand-in for validating the pipeline against a live server."""

    def __init__(self, *, image_hw: tuple[int, int] = (480, 640)) -> None:
        self._h, self._w = image_hw
        self._rng = np.random.default_rng(0)
        self._state = np.zeros(STATE_DIM, dtype=np.float32)
        self._last_cmd = np.zeros(EXEC_DIM, dtype=np.float32)
        self._lock = threading.Lock()

    def _frame(self) -> np.ndarray:
        return self._rng.integers(0, 256, size=(self._h, self._w, 3), dtype=np.uint8)

    def read_observation(self) -> RobotObservation:
        with self._lock:
            # Reflect the last command into robot_q/hand so relative actions stay sane.
            state = self._state.copy()
            state[0:ROBOT_Q_DIM] = self._last_cmd[WP_ROBOT_Q]
            state[ROBOT_Q_DIM:ROBOT_Q_DIM + HAND_DIM] = self._last_cmd[WP_HAND]
        return RobotObservation(self._frame(), self._frame(), self._frame(), state)

    def command(self, robot_q_36: np.ndarray, hand_cmd_12: np.ndarray) -> None:
        with self._lock:
            self._last_cmd[WP_ROBOT_Q] = np.asarray(robot_q_36, dtype=np.float32).reshape(-1)[:ROBOT_Q_DIM]
            self._last_cmd[WP_HAND] = np.asarray(hand_cmd_12, dtype=np.float32).reshape(-1)[:HAND_DIM]

    def reset(self) -> None:
        with self._lock:
            self._state[:] = 0.0
            self._last_cmd[:] = 0.0

    def emergency_stop(self) -> None:
        with self._lock:
            self._last_cmd[:] = 0.0


@dataclass
class UnitreeG1Config:
    """Wiring/config for the real Unitree G1 interface.

    Defaults mirror the teleoperation data-collection stack in
    ``/home/unitree/jimmy/wbc_pico_record`` so the deployed observation matches
    the 60-D training distribution produced by that recorder.
    """

    wbc_repo: str = _default_wbc_repo()
    net_interface: str = "enp5s0"
    eef: str = "brainco"                       # brainco | inspire | dex1
    image_server_address: str = "192.168.123.164"
    urdf_rel: str = "assets/g1/g1_body29_hand14.urdf"
    wbc_model_rel: str = "models/model.enc"
    wbc_rate_hz: float = 60.0                  # WBC inference/publish rate
    teleop_fsm_id: int = 504                   # 504 = teleop/WBC tracking mode
    walk_fsm_id: int = 801                     # 801 = safe walking/idle mode
    image_bgr_to_rgb: bool = True              # ImageClient frames are BGR
    # Initial standing base for the WBC target before the first prediction; once
    # the policy runs, the full predicted robot_q_desired[0:7] base is used.
    base_xyz_wxyz: tuple = (0.0, 0.0, 0.74, 1.0, 0.0, 0.0, 0.0)


class UnitreeG1RobotInterface(RobotInterface):
    """Real Unitree G1 interface built on top of the wbc_pico_record stack.

    Reuses the validated teleop components (cameras via ZMQ ``ImageClient``,
    hand controller, FK, and the WBC ``SecureMotionInferencer``) and talks to the
    robot over ``unitree_sdk2py`` DDS, exactly mirroring how the dataset was
    recorded. The 29-joint order follows ``datasetinfo.md`` robot_q[7:36]:
    legs(12) + waist(3) + left arm(7) + right arm(7).

    Observation (60-D state) assembly mirrors
    ``wbc_pico_record/src/runtime/recording_runtime.add_data``::

        robot_q_current[:36] = base_xyz(3) + imu_quat_wxyz(4, first-frame-relative)
                               + joint_pos(29)
        hand_state[36:48]    = left_hand(6) + right_hand(6)
        ee_state[48:60]      = FK(current arm joints) -> left/right xyzrpy

    Action execution: the full predicted ``robot_q_desired`` (36-D = base7 +
    joints29) is fed to the WBC ``SecureMotionInferencer`` as the target, exactly
    like teleop feeds GMR ``raw_qpos``; ``hand_cmd`` (12) goes to the hand
    controller. The WBC thread runs at ``wbc_rate_hz``, applies the same
    zero-rotation base calibration as the teleop loop, and publishes
    ``concat(motion_vq, root_pose, cmd_wrist)`` to ``rt/fsm/teleop/cmd``.
    """

    def __init__(self, config: UnitreeG1Config | None = None) -> None:
        from scipy.spatial.transform import Rotation as R  # local import

        self.cfg = config or UnitreeG1Config()
        self._R = R

        repo = Path(self.cfg.wbc_repo)
        if not repo.exists():
            raise FileNotFoundError(f"wbc_pico_record repo not found: {repo}")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        self._repo = repo

        # --- DDS init -------------------------------------------------------- #
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize

        try:
            ChannelFactoryInitialize(0, self.cfg.net_interface)
        except Exception as exc:  # already initialized is fine
            LOG.info("ChannelFactoryInitialize: %s (may already be initialized)", exc)

        # --- LowState subscriber (joints + IMU) ------------------------------ #
        self._joint_lock = threading.Lock()
        self._joint_pos = np.zeros(JOINT_DIM, dtype=np.float32)
        self._imu_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._last_state_t = 0.0
        self._init_lowstate_subscriber()

        # --- FSM / locomotion client ---------------------------------------- #
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

        self._loco = LocoClient()
        self._loco.SetTimeout(10.0)
        self._loco.Init()

        # --- Hand controller ------------------------------------------------- #
        self._hand = self._make_hand(self.cfg.eef)

        # --- Cameras (ZMQ ImageClient into shared memory) -------------------- #
        self._init_cameras()

        # --- FK calculator --------------------------------------------------- #
        from teleop.robot_fk import ArmFKCalculator

        urdf_path = str(repo / self.cfg.urdf_rel)
        self._fk = ArmFKCalculator(
            urdf_path=urdf_path,
            package_dir=str(repo / "assets/g1/"),
            visualization=False,
        )

        # --- WBC inferencer + command publisher ------------------------------ #
        from unitree_sdk2py.core.channel import ChannelPublisher
        from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

        try:
            from utils.inference import SecureMotionInferencer
        except ImportError as exc:
            raise ImportError(
                f"Could not import SecureMotionInferencer from {repo}/utils/inference: {exc}"
            ) from exc

        self._inferencer = SecureMotionInferencer(urdf_path, str(repo / self.cfg.wbc_model_rel))
        self._cmd_pub = ChannelPublisher("rt/fsm/teleop/cmd", String_)
        self._cmd_pub.Init()
        self._String_ = String_

        # 36-D WBC target = base(7) + joints(29); start from a safe standing pose.
        self._wbc_target = np.array(self.cfg.base_xyz_wxyz, dtype=np.float64)
        self._wbc_target = np.concatenate([self._wbc_target, np.zeros(JOINT_DIM)])
        self._wbc_lock = threading.Lock()
        self._wbc_stop = threading.Event()
        self._wbc_thread = threading.Thread(target=self._wbc_loop, name="wbc", daemon=True)

        # First-frame-relative IMU reference (matches recorder transform).
        self._first_imu_rot = None
        # Zero-rotation base calibration for the WBC root_pose (matches teleop).
        self._zero_rot = None

    # -- setup helpers ---------------------------------------------------- #
    def _init_lowstate_subscriber(self) -> None:
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
        import unitree_sdk2py.idl.default as default

        self._low_state = default.unitree_hg_msg_dds__LowState_()
        self._low_state_sub = ChannelSubscriber("rt/lowstate", LowStateHG)
        self._low_state_sub.Init(self._on_low_state, 10)

    def _on_low_state(self, msg: Any) -> None:
        self._low_state = msg
        self._last_state_t = time.time()
        if len(msg.motor_state) >= JOINT_DIM:
            with self._joint_lock:
                for i in range(JOINT_DIM):
                    self._joint_pos[i] = msg.motor_state[i].q
                q = msg.imu_state.quaternion
                self._imu_quat_wxyz = np.array([q[0], q[1], q[2], q[3]], dtype=np.float32)

    def _make_hand(self, eef: str) -> Any:
        if eef == "brainco":
            from eef.brainco.brainco import Brainco

            hand = Brainco()
        elif eef == "inspire":
            from eef.inspire.ftp_hand import InspireFTPHandController

            hand = InspireFTPHandController()
        elif eef == "dex1":
            from eef.dex1.dex1 import Dex1

            hand = Dex1()
        else:
            raise ValueError(f"unknown eef {eef!r}")
        hand.set_gripper_ratios(0.0, 0.0)
        return hand

    def _init_cameras(self) -> None:
        from multiprocessing import shared_memory

        self._tv_shape = (480, 1280, 3)
        self._wrist_shape = (480, 1280, 3)
        tv_size = int(np.prod(self._tv_shape))
        wrist_size = int(np.prod(self._wrist_shape))
        self._tv_shm = shared_memory.SharedMemory(create=True, size=tv_size)
        self._wrist_shm = shared_memory.SharedMemory(create=True, size=wrist_size)
        self._tv_img = np.ndarray(self._tv_shape, dtype=np.uint8, buffer=self._tv_shm.buf)
        self._wrist_img = np.ndarray(self._wrist_shape, dtype=np.uint8, buffer=self._wrist_shm.buf)

        from teleop.image_server.image_client import ImageClient

        self._img_client = ImageClient(
            tv_img_shape=self._tv_shape,
            tv_img_shm_name=self._tv_shm.name,
            wrist_img_shape=self._wrist_shape,
            wrist_img_shm_name=self._wrist_shm.name,
            image_show=False,
            server_address=self.cfg.image_server_address,
        )
        self._img_thread = threading.Thread(target=self._img_client.receive_process, daemon=True)
        self._img_thread.start()

    # -- WBC publish thread ----------------------------------------------- #
    def _wbc_loop(self) -> None:
        import json

        dt_target = 1.0 / float(self.cfg.wbc_rate_hz)
        last = time.time()
        while not self._wbc_stop.is_set():
            start = time.time()
            with self._wbc_lock:
                target = self._wbc_target.copy()
            now = time.time()
            dt = now - last
            last = now
            try:
                motion_vq, root_pose, cmd_wrist, _ = self._inferencer.process(target, dt)
                if motion_vq is not None:
                    # Zero-rotation base calibration on the WBC root_pose, mirroring
                    # the teleop process_frame loop (rt/fsm/teleop/cmd contract).
                    rq = root_pose[3:7]
                    raw_rot = self._R.from_quat([rq[1], rq[2], rq[3], rq[0]])
                    if self._zero_rot is None:
                        self._zero_rot = raw_rot
                    delta = (self._zero_rot.inv() * raw_rot).as_quat()  # xyzw
                    root_pose[3:7] = np.array([delta[3], delta[0], delta[1], delta[2]])
                    full = np.concatenate([motion_vq, root_pose, cmd_wrist], axis=-1)
                    msg = self._String_(data=json.dumps({"name": "default", "frame": full.tolist()}, separators=(",", ":")))
                    self._cmd_pub.Write(msg)
            except Exception:
                LOG.exception("WBC process/publish failed")
            sleep = dt_target - (time.time() - start)
            if sleep > 0:
                time.sleep(sleep)

    # -- RobotInterface API ----------------------------------------------- #
    def _split_camera(self, full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mid = full.shape[1] // 2
        left = np.ascontiguousarray(full[:, :mid])
        right = np.ascontiguousarray(full[:, mid:])
        if self.cfg.image_bgr_to_rgb:
            left = left[..., ::-1]
            right = right[..., ::-1]
        return np.ascontiguousarray(left), np.ascontiguousarray(right)

    def read_observation(self) -> RobotObservation:
        color_0, _color_1 = self._split_camera(self._tv_img.copy())     # head left / right(discard)
        color_2, color_3 = self._split_camera(self._wrist_img.copy())   # left wrist / right wrist

        with self._joint_lock:
            joints = self._joint_pos.copy()
            imu_wxyz = self._imu_quat_wxyz.copy()

        # First-frame-relative IMU transform (mirrors recorder).
        curr_rot = self._R.from_quat([imu_wxyz[1], imu_wxyz[2], imu_wxyz[3], imu_wxyz[0]])
        if self._first_imu_rot is None:
            self._first_imu_rot = curr_rot
            base_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            rel = (self._first_imu_rot.inv() * curr_rot).as_quat()  # xyzw
            base_quat_wxyz = np.array([rel[3], rel[0], rel[1], rel[2]], dtype=np.float32)

        base_xyz = np.array(self.cfg.base_xyz_wxyz[:3], dtype=np.float32)
        robot_q = np.concatenate([base_xyz, base_quat_wxyz, joints]).astype(np.float32)  # 36

        l_hand, r_hand = self._hand.get_hand_states()
        hand_state = np.concatenate([np.asarray(l_hand, np.float32).reshape(-1)[:6],
                                     np.asarray(r_hand, np.float32).reshape(-1)[:6]])  # 12

        # ee_state via FK on current arm joints (waist 3 + arms 7+7).
        waist_q = joints[12:15]
        left_arm_q = joints[15:22]
        right_arm_q = joints[22:29]
        _, _, left_xyzrpy, right_xyzrpy = self._fk.compute_fk_from_real_joints(waist_q, left_arm_q, right_arm_q)
        ee_state = np.concatenate([np.asarray(left_xyzrpy, np.float32), np.asarray(right_xyzrpy, np.float32)])  # 12

        state = np.concatenate([robot_q, hand_state, ee_state]).astype(np.float32)  # 60
        return RobotObservation(color_0, color_2, color_3, state)

    def command(self, robot_q_36: np.ndarray, hand_cmd_12: np.ndarray) -> None:
        # Feed the full predicted 36-D robot_q (base7 + joints29) to the WBC,
        # exactly like teleop feeds GMR raw_qpos. The base is NOT replaced.
        target = np.asarray(robot_q_36, dtype=np.float64).reshape(-1)
        if target.size < ROBOT_Q_DIM:
            target = np.concatenate([target, np.zeros(ROBOT_Q_DIM - target.size)])
        target = target[:ROBOT_Q_DIM]
        with self._wbc_lock:
            self._wbc_target[:] = target
        self._hand.set_hand_targets(np.asarray(hand_cmd_12, dtype=np.float64).reshape(-1)[:HAND_DIM])

    def reset(self) -> None:
        self._first_imu_rot = None
        self._zero_rot = None
        # Enter teleop/WBC tracking FSM and start the WBC publish thread.
        try:
            self._loco.SetTimeout(5.0)
            self._loco.SetFsmId(self.cfg.teleop_fsm_id)
            self._loco.SetTimeout(0.01)
        except Exception:
            LOG.exception("failed to switch to teleop FSM %d", self.cfg.teleop_fsm_id)
        if not self._wbc_thread.is_alive():
            self._wbc_thread.start()

    def close(self) -> None:
        self._wbc_stop.set()
        if self._wbc_thread.is_alive():
            self._wbc_thread.join(timeout=2.0)
        # Return to a safe FSM and open hands.
        try:
            self._loco.SetTimeout(5.0)
            self._loco.SetFsmId(self.cfg.walk_fsm_id)
        except Exception:
            LOG.exception("failed to switch to safe FSM %d", self.cfg.walk_fsm_id)
        try:
            self._hand.set_gripper_ratios(0.0, 0.0)
        except Exception:
            pass
        for shm in (getattr(self, "_tv_shm", None), getattr(self, "_wrist_shm", None)):
            try:
                if shm is not None:
                    shm.close()
                    shm.unlink()
            except Exception:
                pass

    def emergency_stop(self) -> None:
        """Software stop: stop WBC publishing and switch back to the safe FSM."""
        LOG.error("EMERGENCY STOP: stopping WBC thread and switching FSM to %d", self.cfg.walk_fsm_id)
        self._wbc_stop.set()
        if self._wbc_thread.is_alive():
            self._wbc_thread.join(timeout=1.0)
        try:
            self._loco.SetTimeout(2.0)
            self._loco.SetFsmId(self.cfg.walk_fsm_id)
        except Exception:
            LOG.exception("failed to switch to safe FSM during emergency stop")


# ----------------------------------------------------------------------------- #
# Realtime controller: prefetching double buffer
# ----------------------------------------------------------------------------- #
class RealtimeController:
    def __init__(
        self,
        *,
        policy: WebsocketClientPolicy,
        robot: RobotInterface,
        prompt: str,
        control_hz: float,
        action_horizon: int,
        replan_stride: int,
        max_steps: int | None,
        manual_step: bool,
        dry_run: bool,
        keyboard_controls: bool,
        safety: SafetyConfig,
    ) -> None:
        self.policy = policy
        self.robot = robot
        self.prompt = prompt
        self.control_dt = 1.0 / float(control_hz)
        self.action_horizon = int(action_horizon)
        self.replan_stride = max(1, min(int(replan_stride), int(action_horizon)))
        self.max_steps = max_steps
        self.manual_step = bool(manual_step)
        self.dry_run = bool(dry_run)
        self.keyboard_controls = bool(keyboard_controls)
        self.safety = safety

        self._lock = threading.Lock()
        self._active: np.ndarray | None = None   # [H, 48]
        self._cursor = 0
        self._pending: np.ndarray | None = None   # [H, 48] next chunk
        self._last_cmd = np.zeros(EXEC_DIM, dtype=np.float32)
        self._replan_event = threading.Event()
        self._step_event = threading.Event()
        self._emergency = threading.Event()
        self._stop = threading.Event()
        self._infer_calls = 0

    @staticmethod
    def _state_to_exec(state: np.ndarray) -> np.ndarray:
        return np.concatenate([state[:ROBOT_Q_DIM], state[ROBOT_Q_DIM:ROBOT_Q_DIM + HAND_DIM]]).astype(np.float32)

    @staticmethod
    def _normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float32).reshape(4)
        norm = float(np.linalg.norm(q))
        if norm < 1e-6:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return q / norm

    @classmethod
    def _quat_angle(cls, q0: np.ndarray, q1: np.ndarray) -> float:
        a = cls._normalize_quat_wxyz(q0)
        b = cls._normalize_quat_wxyz(q1)
        dot = float(np.clip(abs(np.dot(a, b)), -1.0, 1.0))
        return float(2.0 * np.arccos(dot))

    @staticmethod
    def _clip_delta(target: np.ndarray, prev: np.ndarray, limit: float) -> tuple[np.ndarray, int]:
        if limit <= 0:
            return target, 0
        delta = np.asarray(target) - np.asarray(prev)
        clipped = np.clip(delta, -limit, limit)
        return np.asarray(prev) + clipped, int(np.count_nonzero(np.abs(delta) > limit))

    def _check_initial_waypoint(self, waypoint: np.ndarray, current: np.ndarray) -> None:
        if not self.safety.enabled:
            return
        joint_delta = np.max(np.abs(waypoint[BASE_DIM:ROBOT_Q_DIM] - current[BASE_DIM:ROBOT_Q_DIM]))
        hand_delta = np.max(np.abs(waypoint[ROBOT_Q_DIM:EXEC_DIM] - current[ROBOT_Q_DIM:EXEC_DIM]))
        base_xyz_delta = float(np.linalg.norm(waypoint[0:3] - current[0:3]))
        base_angle_delta = self._quat_angle(waypoint[3:7], current[3:7])
        problems = []
        if joint_delta > self.safety.max_initial_joint_delta:
            problems.append(f"joint max delta {joint_delta:.3f} > {self.safety.max_initial_joint_delta:.3f} rad")
        if hand_delta > self.safety.max_initial_hand_delta:
            problems.append(f"hand max delta {hand_delta:.3f} > {self.safety.max_initial_hand_delta:.3f}")
        if base_xyz_delta > self.safety.max_initial_base_xyz_delta:
            problems.append(f"base xyz delta {base_xyz_delta:.3f} > {self.safety.max_initial_base_xyz_delta:.3f} m")
        if base_angle_delta > self.safety.max_initial_base_angle_delta:
            problems.append(f"base angle delta {base_angle_delta:.3f} > {self.safety.max_initial_base_angle_delta:.3f} rad")
        if problems:
            raise SafetyViolation("; ".join(problems))

    def _rate_limit_chunk(self, chunk: np.ndarray, start: np.ndarray) -> np.ndarray:
        if not self.safety.enabled:
            return chunk
        safe = np.asarray(chunk, dtype=np.float32).copy()
        prev = np.asarray(start, dtype=np.float32).copy()
        clipped_counts = {"base_xyz": 0, "base_quat": 0, "joint": 0, "hand": 0}
        for i in range(safe.shape[0]):
            row = safe[i]
            row[0:3], n = self._clip_delta(row[0:3], prev[0:3], self.safety.max_step_base_xyz_delta)
            clipped_counts["base_xyz"] += n
            if self._quat_angle(row[3:7], prev[3:7]) > self.safety.max_step_base_angle_delta:
                row[3:7] = prev[3:7]
                clipped_counts["base_quat"] += 1
            row[BASE_DIM:ROBOT_Q_DIM], n = self._clip_delta(
                row[BASE_DIM:ROBOT_Q_DIM],
                prev[BASE_DIM:ROBOT_Q_DIM],
                self.safety.max_step_joint_delta,
            )
            clipped_counts["joint"] += n
            row[ROBOT_Q_DIM:EXEC_DIM], n = self._clip_delta(
                row[ROBOT_Q_DIM:EXEC_DIM],
                prev[ROBOT_Q_DIM:EXEC_DIM],
                self.safety.max_step_hand_delta,
            )
            clipped_counts["hand"] += n
            safe[i] = row
            prev = row
        if any(clipped_counts.values()):
            LOG.warning("rate-limited predicted chunk: %s", clipped_counts)
        return safe

    def _prepare_safe_chunk(self, chunk: np.ndarray, obs_state: np.ndarray) -> np.ndarray:
        current = self._state_to_exec(obs_state)
        self._check_initial_waypoint(chunk[0], current)
        return self._rate_limit_chunk(chunk, current)

    def _trigger_emergency_stop(self, reason: str) -> None:
        if self._emergency.is_set():
            return
        LOG.error("EMERGENCY STOP requested: %s", reason)
        self._emergency.set()
        self._stop.set()
        self._replan_event.set()
        self._step_event.set()
        try:
            self.robot.emergency_stop()
        except Exception:
            LOG.exception("robot emergency_stop failed")

    def _keyboard_loop(self) -> None:
        if self.manual_step:
            LOG.warning("manual-step controls: Enter/n = infer+play one chunk, e/x = emergency stop, q = quit")
        else:
            LOG.warning("keyboard controls: e/x = emergency stop, q = quit")
        while not self._stop.is_set():
            try:
                line = sys.stdin.readline()
            except Exception:
                LOG.exception("keyboard input failed; disabling keyboard controls")
                return
            if line == "":
                return
            cmd = line.strip().lower()
            if self.manual_step and cmd in {"", "n", "next", "step"}:
                self._step_event.set()
            elif cmd in {"e", "x", "stop", "estop", "emergency"}:
                self._trigger_emergency_stop("keyboard")
                return
            elif cmd in {"q", "quit", "exit"}:
                LOG.info("quit requested from keyboard")
                self.request_stop()
                return
            elif cmd:
                LOG.info("unknown command %r", cmd)

    # -- inference -------------------------------------------------------- #
    def _infer_chunk(self) -> np.ndarray:
        obs = self.robot.read_observation()
        state = np.asarray(obs.state, dtype=np.float32).reshape(-1)
        if state.size < STATE_DIM:
            raise ValueError(f"robot state must be {STATE_DIM}-D, got {state.size}")
        request = {
            "color_0": np.ascontiguousarray(obs.color_0),
            "color_2": np.ascontiguousarray(obs.color_2),
            "color_3": np.ascontiguousarray(obs.color_3),
            "state": state[:STATE_DIM],
            "prompt": self.prompt,
        }
        t0 = time.perf_counter()
        result = self.policy.infer(request)
        dt = time.perf_counter() - t0
        self._infer_calls += 1
        chunk = self._parse_action(result)
        chunk = self._prepare_safe_chunk(chunk, state[:STATE_DIM])
        LOG.info("infer #%d -> chunk %s in %.3fs", self._infer_calls, chunk.shape, dt)
        return chunk

    @staticmethod
    def _parse_action(result: dict[str, Any]) -> np.ndarray:
        """Build the [H, 48] executed-waypoint chunk (robot_q36 + hand12)."""
        if not isinstance(result, dict):
            raise RuntimeError(f"Unexpected server response type: {type(result)}")
        if "robot_q_desired" in result and "hand_cmd" in result:
            robot_q = np.asarray(result["robot_q_desired"], dtype=np.float32)
            hand = np.asarray(result["hand_cmd"], dtype=np.float32)
        else:
            action = np.asarray(result.get("action"), dtype=np.float32)
            if action.ndim != 2 or action.shape[-1] < STATE_DIM:
                raise RuntimeError(f"Expected action [H,>=60], got shape {getattr(action, 'shape', None)}")
            robot_q = action[:, :ROBOT_Q_DIM]
            hand = action[:, ROBOT_Q_DIM:ROBOT_Q_DIM + HAND_DIM]
        robot_q = robot_q.reshape(robot_q.shape[0], -1)[:, :ROBOT_Q_DIM]
        hand = hand.reshape(hand.shape[0], -1)[:, :HAND_DIM]
        if robot_q.shape[0] != hand.shape[0]:
            n = min(robot_q.shape[0], hand.shape[0])
            robot_q, hand = robot_q[:n], hand[:n]
        chunk = np.concatenate([robot_q, hand], axis=1).astype(np.float32)  # [H, 48]
        if not np.isfinite(chunk).all():
            raise RuntimeError("server action chunk contains non-finite values")
        return chunk

    def _inference_loop(self) -> None:
        while not self._stop.is_set():
            if not self._replan_event.wait(timeout=0.1):
                continue
            self._replan_event.clear()
            if self._stop.is_set():
                break
            try:
                chunk = self._infer_chunk()
            except SafetyViolation as exc:
                self._trigger_emergency_stop(f"unsafe inference chunk: {exc}")
                continue
            except Exception:
                LOG.exception("inference failed; holding last command")
                continue
            with self._lock:
                self._pending = chunk

    # -- control ---------------------------------------------------------- #
    def _control_loop(self) -> None:
        next_tick = time.perf_counter()
        global_step = 0
        while not self._stop.is_set():
            with self._lock:
                # Swap in a freshly prefetched chunk at a stride boundary.
                if self._pending is not None and (self._active is None or self._cursor >= self.replan_stride):
                    self._active = self._pending
                    self._pending = None
                    self._cursor = 0

                waypoint: np.ndarray | None = None
                if self._active is not None and self._cursor < self._active.shape[0]:
                    waypoint = self._active[self._cursor]
                    self._cursor += 1
                    self._last_cmd = waypoint
                    # Ask for the next chunk once we cross the replan point.
                    if self._cursor == self.replan_stride:
                        self._replan_event.set()
                else:
                    waypoint = self._last_cmd  # hold last setpoint if chunk exhausted

            if waypoint is not None:
                try:
                    if not self.dry_run:
                        self.robot.command(waypoint[WP_ROBOT_Q], waypoint[WP_HAND])
                except Exception as exc:
                    self._trigger_emergency_stop(f"robot.command failed: {exc}")
                    break

            global_step += 1
            if self.max_steps is not None and global_step >= self.max_steps:
                LOG.info("reached max-steps=%d, stopping", self.max_steps)
                self._stop.set()
                break

            next_tick += self.control_dt
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # Control loop fell behind; resync the clock.
                next_tick = time.perf_counter()

    def _play_chunk_blocking(self, chunk: np.ndarray) -> None:
        for i, waypoint in enumerate(chunk):
            if self._stop.is_set() or self._emergency.is_set():
                return
            try:
                if not self.dry_run:
                    self.robot.command(waypoint[WP_ROBOT_Q], waypoint[WP_HAND])
            except Exception as exc:
                self._trigger_emergency_stop(f"robot.command failed at chunk step {i}: {exc}")
                return
            with self._lock:
                self._last_cmd = waypoint
            time.sleep(self.control_dt)

    def _run_manual_step(self) -> None:
        LOG.warning("manual-step mode enabled; no background prefetch will run")
        if not self.keyboard_controls:
            self.keyboard_controls = True
        if self.keyboard_controls:
            threading.Thread(target=self._keyboard_loop, name="keyboard", daemon=True).start()
        try:
            while not self._stop.is_set():
                LOG.warning("waiting for next chunk command: press Enter/n, e for emergency stop, q to quit")
                self._step_event.wait()
                self._step_event.clear()
                if self._stop.is_set():
                    break
                try:
                    chunk = self._infer_chunk()
                except SafetyViolation as exc:
                    self._trigger_emergency_stop(f"unsafe manual-step chunk: {exc}")
                    break
                except Exception:
                    LOG.exception("manual-step inference failed")
                    continue
                LOG.warning("playing one chunk (%d waypoints)%s", chunk.shape[0], " [dry-run]" if self.dry_run else "")
                if self.dry_run:
                    with self._lock:
                        self._last_cmd = chunk[-1]
                else:
                    self._play_chunk_blocking(chunk)
        except KeyboardInterrupt:
            LOG.info("interrupted; stopping")

    # -- lifecycle -------------------------------------------------------- #
    def run(self) -> None:
        if self.dry_run:
            LOG.warning("dry-run enabled: not switching robot FSM and not starting WBC publishing")
        else:
            self.robot.reset()
        try:
            self.policy.reset({})
        except Exception:
            LOG.warning("policy.reset failed (continuing)", exc_info=True)

        if self.manual_step:
            try:
                self._run_manual_step()
            finally:
                if not self._emergency.is_set() and not self.dry_run:
                    self.robot.close()
            return

        if self.keyboard_controls:
            threading.Thread(target=self._keyboard_loop, name="keyboard", daemon=True).start()

        # Bootstrap: block on the first chunk so control starts with real data.
        LOG.info("bootstrapping first action chunk ...")
        try:
            first = self._infer_chunk()
        except SafetyViolation as exc:
            self._trigger_emergency_stop(f"unsafe bootstrap chunk: {exc}")
            return
        with self._lock:
            self._active = first
            self._cursor = 0
            self._last_cmd = first[0]

        infer_thread = threading.Thread(target=self._inference_loop, name="infer", daemon=True)
        control_thread = threading.Thread(target=self._control_loop, name="control", daemon=True)
        infer_thread.start()
        control_thread.start()

        try:
            while not self._stop.is_set():
                time.sleep(0.2)
        except KeyboardInterrupt:
            LOG.info("interrupted; stopping")
        finally:
            self._stop.set()
            self._replan_event.set()
            control_thread.join(timeout=2.0)
            infer_thread.join(timeout=2.0)
            if not self._emergency.is_set() and not self.dry_run:
                self.robot.close()

    def request_stop(self) -> None:
        self._stop.set()
        self._replan_event.set()
        self._step_event.set()


# ----------------------------------------------------------------------------- #
# CLI
# ----------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", required=True, help="Task instruction sent with every inference request.")
    parser.add_argument("--robot", choices=["dummy", "unitree_g1"], default="dummy")
    parser.add_argument("--control-hz", type=float, default=30.0, help="Waypoint emission rate to the robot.")
    parser.add_argument("--action-horizon", type=int, default=48, help="Must match the server --action-horizon.")
    parser.add_argument(
        "--replan-stride",
        type=int,
        default=24,
        help="Steps to execute before swapping to a freshly prefetched chunk (<= action-horizon).",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after N control steps (default: run until Ctrl-C).")
    parser.add_argument("--manual-step", action="store_true", help="Wait for Enter/n before each infer+play chunk.")
    parser.add_argument("--dry-run", action="store_true", help="Run inference and safety checks but do not command WBC/hands.")
    parser.add_argument(
        "--no-keyboard-controls",
        action="store_true",
        help="Disable terminal controls. By default, e/x is software emergency stop when stdin is a TTY.",
    )
    parser.add_argument("--disable-safety", action="store_true", help="Disable command safety checks and rate limiting.")
    parser.add_argument("--max-initial-joint-delta", type=float, default=0.75, help="Abort if first waypoint differs from current joints by more than this radian value.")
    parser.add_argument("--max-initial-hand-delta", type=float, default=1.0, help="Abort if first waypoint differs from current hand state by more than this value.")
    parser.add_argument("--max-initial-base-xyz-delta", type=float, default=0.30, help="Abort if first waypoint base xyz differs from current state by more than this meter value.")
    parser.add_argument("--max-initial-base-angle-delta", type=float, default=1.0, help="Abort if first waypoint base quat differs from current state by more than this radian value.")
    parser.add_argument("--max-step-joint-delta", type=float, default=0.12, help="Per-control-step joint delta clamp in radians.")
    parser.add_argument("--max-step-hand-delta", type=float, default=0.25, help="Per-control-step hand command delta clamp.")
    parser.add_argument("--max-step-base-xyz-delta", type=float, default=0.03, help="Per-control-step base xyz delta clamp in meters.")
    parser.add_argument("--max-step-base-angle-delta", type=float, default=0.25, help="Hold base rotation if one step exceeds this radian value.")
    # --- real Unitree G1 wiring (ignored for --robot dummy) ----------------- #
    parser.add_argument("--wbc-repo", default=_default_wbc_repo(), help="Path to the teleop/WBC repo to reuse.")
    parser.add_argument("--net-interface", default="enp5s0", help="DDS network interface for unitree_sdk2py.")
    parser.add_argument("--eef", choices=["brainco", "inspire", "dex1"], default="brainco", help="Hand/end-effector type.")
    parser.add_argument("--image-server-address", default="192.168.123.164", help="ZMQ image server address.")
    parser.add_argument("--wbc-rate-hz", type=float, default=60.0, help="WBC inference/publish rate.")
    parser.add_argument("--log-level", default="INFO")
    return parser


def _make_robot(name: str, args: argparse.Namespace) -> RobotInterface:
    if name == "":
        return DummyRobotInterface()
    if name == "unitree_g1":
        cfg = UnitreeG1Config(
            wbc_repo=args.wbc_repo,
            net_interface=args.net_interface,
            eef=args.eef,
            image_server_address=args.image_server_address,
            wbc_rate_hz=args.wbc_rate_hz,
        )
        return UnitreeG1RobotInterface(cfg)
    raise ValueError(f"unknown robot interface {name!r}")


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    robot = _make_robot(args.robot, args)
    policy = WebsocketClientPolicy(host=args.host, port=args.port)
    LOG.info("server metadata: %s", policy.get_server_metadata())

    controller = RealtimeController(
        policy=policy,
        robot=robot,
        prompt=args.prompt,
        control_hz=args.control_hz,
        action_horizon=args.action_horizon,
        replan_stride=args.replan_stride,
        max_steps=args.max_steps,
        manual_step=args.manual_step,
        dry_run=args.dry_run,
        keyboard_controls=(not args.no_keyboard_controls) and (args.manual_step or sys.stdin.isatty()),
        safety=SafetyConfig(
            enabled=not args.disable_safety,
            max_initial_joint_delta=args.max_initial_joint_delta,
            max_initial_hand_delta=args.max_initial_hand_delta,
            max_initial_base_xyz_delta=args.max_initial_base_xyz_delta,
            max_initial_base_angle_delta=args.max_initial_base_angle_delta,
            max_step_joint_delta=args.max_step_joint_delta,
            max_step_hand_delta=args.max_step_hand_delta,
            max_step_base_xyz_delta=args.max_step_base_xyz_delta,
            max_step_base_angle_delta=args.max_step_base_angle_delta,
        ),
    )

    signal.signal(signal.SIGTERM, lambda *_: controller.request_stop())
    controller.run()


if __name__ == "__main__":
    main()
