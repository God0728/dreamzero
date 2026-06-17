from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any


VIDEO_KEYS = ("video.image", "video.wrist_image")
STATE_KEYS = ("state.ee_pos", "state.ee_ori", "state.gripper_pos")
ACTION_KEYS = ("action.ee_delta_pose", "action.gripper")
DEFAULT_RAW_IMAGE_RESOLUTION = (256, 256)
DEFAULT_ACTION_HORIZON = 24
DEFAULT_PROFILE_NAME = "wan22_libero"
BENCHMARK_ALIASES = {
    "libero_all": ("libero_spatial", "libero_object", "libero_goal", "libero_10"),
    "libero_long": ("libero_10",),
}
DEFAULT_MAX_STEPS_BY_BENCHMARK = {
    "libero_spatial": 400,
    "libero_object": 400,
    "libero_goal": 400,
    "libero_10": 700,
    "libero_90": 700,
}


@dataclass(frozen=True)
class ObsWindowPolicy:
    initial_frames: int = 1
    subsequent_frames: int = 9
    include_current_boundary: bool = True

    def __post_init__(self) -> None:
        if self.initial_frames <= 0 or self.subsequent_frames <= 0:
            raise ValueError("obs window frame counts must be positive")

    @classmethod
    def parse(cls, value: str | ObsWindowPolicy) -> "ObsWindowPolicy":
        if isinstance(value, ObsWindowPolicy):
            return value
        left, sep, right = str(value).partition("->")
        if not sep:
            raise ValueError(f"obs_window_policy must look like '1->9', got {value!r}")
        return cls(int(left), int(right))

    def label(self) -> str:
        return f"{self.initial_frames}->{self.subsequent_frames}"


@dataclass(frozen=True)
class LiberoWan22Profile:
    name: str = DEFAULT_PROFILE_NAME
    raw_image_resolution: tuple[int, int] = DEFAULT_RAW_IMAGE_RESOLUTION
    action_horizon: int = DEFAULT_ACTION_HORIZON
    replan_steps: int = DEFAULT_ACTION_HORIZON
    wait_steps: int = 30
    max_steps: int = 400
    max_steps_by_benchmark: dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_MAX_STEPS_BY_BENCHMARK)
    )
    obs_window_policy: ObsWindowPolicy = field(default_factory=ObsWindowPolicy)
    save_rollout_videos: bool = True
    save_predicted_videos: bool = True
    pre_rotate_images_for_policy: bool = True
    rotate_rollout_frames_for_video: bool = True
    predicted_video_fps: int = 7
    rollout_video_fps: int = 20
    view_keys: tuple[str, str] = VIDEO_KEYS
    state_keys: tuple[str, str, str] = STATE_KEYS
    action_keys: tuple[str, str] = ACTION_KEYS

    def __post_init__(self) -> None:
        if tuple(self.view_keys) != VIDEO_KEYS:
            raise ValueError(f"LIBERO WAN22 uses fixed video keys {VIDEO_KEYS}, got {self.view_keys}")
        if self.action_horizon <= 0 or self.replan_steps <= 0:
            raise ValueError("action_horizon and replan_steps must be positive")
        if self.replan_steps > self.action_horizon:
            raise ValueError("replan_steps cannot exceed action_horizon")
        h, w = self.raw_image_resolution
        if h <= 0 or w <= 0:
            raise ValueError("raw_image_resolution must be positive")

    def max_steps_for_benchmark(self, benchmark_name: str | None) -> int:
        if benchmark_name and benchmark_name in self.max_steps_by_benchmark:
            return int(self.max_steps_by_benchmark[benchmark_name])
        return int(self.max_steps)

    def metadata(self) -> dict[str, Any]:
        data = asdict(self)
        data["raw_image_resolution"] = list(self.raw_image_resolution)
        data["view_keys"] = list(self.view_keys)
        data["state_keys"] = list(self.state_keys)
        data["action_keys"] = list(self.action_keys)
        data["obs_window_policy"] = asdict(self.obs_window_policy)
        data["obs_window_policy_name"] = self.obs_window_policy.label()
        return data


DEFAULT_PROFILE = LiberoWan22Profile()


def _tuple2(value: Any, fallback: tuple[int, int]) -> tuple[int, int]:
    if value is None:
        return fallback
    if isinstance(value, str):
        parts = value.replace("x", ",").split(",")
        value = [part for part in parts if part]
    if len(value) != 2:
        raise ValueError(f"expected two integers, got {value!r}")
    return (int(value[0]), int(value[1]))


def profile_with_overrides(
    base: LiberoWan22Profile = DEFAULT_PROFILE,
    *,
    name: str | None = None,
    raw_image_resolution: Any = None,
    action_horizon: int | None = None,
    replan_steps: int | None = None,
    wait_steps: int | None = None,
    max_steps: int | None = None,
    obs_window_policy: str | ObsWindowPolicy | None = None,
    include_current_boundary: bool | None = None,
    save_rollout_videos: bool | None = None,
    save_predicted_videos: bool | None = None,
    pre_rotate_images_for_policy: bool | None = None,
    rotate_rollout_frames_for_video: bool | None = None,
    predicted_video_fps: int | None = None,
    rollout_video_fps: int | None = None,
) -> LiberoWan22Profile:
    resolved_obs_window_policy = (
        ObsWindowPolicy.parse(obs_window_policy)
        if obs_window_policy is not None
        else base.obs_window_policy
    )
    if include_current_boundary is not None:
        resolved_obs_window_policy = replace(
            resolved_obs_window_policy,
            include_current_boundary=bool(include_current_boundary),
        )
    return replace(
        base,
        name=name or base.name,
        raw_image_resolution=_tuple2(raw_image_resolution, base.raw_image_resolution),
        action_horizon=int(action_horizon if action_horizon is not None else base.action_horizon),
        replan_steps=int(replan_steps if replan_steps is not None else base.replan_steps),
        wait_steps=int(wait_steps if wait_steps is not None else base.wait_steps),
        max_steps=int(max_steps if max_steps is not None else base.max_steps),
        obs_window_policy=resolved_obs_window_policy,
        save_rollout_videos=(
            base.save_rollout_videos if save_rollout_videos is None else bool(save_rollout_videos)
        ),
        save_predicted_videos=(
            base.save_predicted_videos if save_predicted_videos is None else bool(save_predicted_videos)
        ),
        pre_rotate_images_for_policy=(
            base.pre_rotate_images_for_policy
            if pre_rotate_images_for_policy is None
            else bool(pre_rotate_images_for_policy)
        ),
        rotate_rollout_frames_for_video=(
            base.rotate_rollout_frames_for_video
            if rotate_rollout_frames_for_video is None
            else bool(rotate_rollout_frames_for_video)
        ),
        predicted_video_fps=int(
            predicted_video_fps if predicted_video_fps is not None else base.predicted_video_fps
        ),
        rollout_video_fps=int(rollout_video_fps if rollout_video_fps is not None else base.rollout_video_fps),
    )


def profile_from_metadata(metadata: dict[str, Any], *, save_rollout_videos: bool | None = None) -> LiberoWan22Profile:
    profile_data = dict(metadata.get("profile_metadata") or {})
    obs_policy = profile_data.get("obs_window_policy") or {}
    include_current_boundary = obs_policy.get("include_current_boundary", metadata.get("include_current_boundary"))
    if include_current_boundary is None:
        # Older LIBERO servers did not publish this field and used the previous
        # completed-block/exclude-boundary schedule.  New servers include the
        # field in profile_metadata/top-level config, so absence means legacy.
        include_current_boundary = False
    profile = profile_with_overrides(
        DEFAULT_PROFILE,
        name=profile_data.get("name") or metadata.get("default_profile") or DEFAULT_PROFILE.name,
        raw_image_resolution=profile_data.get("raw_image_resolution") or metadata.get("image_resolution"),
        action_horizon=profile_data.get("action_horizon") or metadata.get("action_horizon"),
        replan_steps=profile_data.get("replan_steps") or metadata.get("replan_steps"),
        wait_steps=profile_data.get("wait_steps"),
        max_steps=profile_data.get("max_steps"),
        obs_window_policy=ObsWindowPolicy(
            int(obs_policy.get("initial_frames", metadata.get("initial_frames", 1))),
            int(obs_policy.get("subsequent_frames", metadata.get("subsequent_frames", 9))),
            bool(include_current_boundary),
        ),
        save_rollout_videos=profile_data.get("save_rollout_videos"),
        save_predicted_videos=profile_data.get("save_predicted_videos"),
        pre_rotate_images_for_policy=profile_data.get("pre_rotate_images_for_policy"),
        rotate_rollout_frames_for_video=profile_data.get("rotate_rollout_frames_for_video"),
        predicted_video_fps=profile_data.get("predicted_video_fps"),
        rollout_video_fps=profile_data.get("rollout_video_fps"),
    )
    if save_rollout_videos is not None:
        profile = replace(profile, save_rollout_videos=bool(save_rollout_videos))
    merged_steps = dict(DEFAULT_MAX_STEPS_BY_BENCHMARK)
    merged_steps.update(profile_data.get("max_steps_by_benchmark") or {})
    return replace(profile, max_steps_by_benchmark={k: int(v) for k, v in merged_steps.items()})


def resolve_benchmark_names(name: str) -> tuple[str, ...]:
    return BENCHMARK_ALIASES.get(name, (name,))
