#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate

SCRIPT_PATH = Path(__file__).resolve()
DREAMZERO_ROOT = SCRIPT_PATH.parents[2]
if str(DREAMZERO_ROOT) not in sys.path:
    sys.path.insert(0, str(DREAMZERO_ROOT))


def _resolve_libero_root() -> Path | None:
    candidates = []
    env_root = os.environ.get("LIBERO_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.extend(
        [
            DREAMZERO_ROOT.parent / "LIBERO",
            DREAMZERO_ROOT.parent.parent / "LIBERO",
            Path("/DATA/disk2/yuhangwang/code/WorldModel/dz/source/LIBERO"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


LIBERO_ROOT = _resolve_libero_root()
if LIBERO_ROOT is not None and str(LIBERO_ROOT) not in sys.path:
    sys.path.insert(0, str(LIBERO_ROOT))

DEFAULT_DATA_ROOT = "/DATA/disk2/yuhangwang/data/WorldModel/libero-spatial-dreamzero"
DEFAULT_CHECKPOINT = "/DATA/disk2/yuhangwang/data/WorldModel/dreamzero/checkpoints/dreamzero_libero_spatial_droid_step20000_initfix_8gpu/checkpoint-20000"
DEFAULT_OUTPUT_ROOT = "/DATA/disk2/yuhangwang/data/WorldModel/dump/libero_spatial_task0_concat_inputs"
DEFAULT_WAN_CKPT_DIR = "/DATA/disk2/yuhangwang/model/WorldModel/dreamzero/Wan2.1-I2V-14B-480P"
DEFAULT_TOKENIZER_DIR = "/DATA/disk2/yuhangwang/model/WorldModel/dreamzero/umt5-xxl"
DEFAULT_PRETRAINED_MODEL_PATH = "/DATA/disk2/yuhangwang/model/WorldModel/dreamzero/DreamZero-DROID"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump LIBERO-Spatial task0 concatenated pre-model images for train/eval")
    parser.add_argument("--mode", choices=["train", "eval", "both"], default="both")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--train-data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--camera-height", type=int, default=128)
    parser.add_argument("--camera-width", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--num-zero-steps", type=int, default=5)
    parser.add_argument("--n-eval", type=int, default=3)
    parser.add_argument("--max-train-attempts", type=int, default=5000)
    parser.add_argument("--clear-output", action="store_true")
    return parser.parse_args()


def load_task_prompt(data_root: Path, task_id: int) -> str:
    tasks_path = data_root / "meta" / "tasks.jsonl"
    with tasks_path.open() as f:
        for line in f:
            entry = json.loads(line)
            if int(entry["task_index"]) == task_id:
                return str(entry["task"])
    raise ValueError(f"Task id {task_id} not found in {tasks_path}")


def ensure_clean_dir(path: Path, clear: bool) -> None:
    if clear and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def to_uint8_image(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape {array.shape}")
    if array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def write_contact_sheet(frames: list[np.ndarray], output_path: Path, columns: int = 5) -> None:
    if not frames:
        return
    images = [to_uint8_image(frame) for frame in frames]
    max_h = max(image.shape[0] for image in images)
    max_w = max(image.shape[1] for image in images)
    rows = math.ceil(len(images) / columns)
    sheet = np.zeros((rows * max_h, columns * max_w, 3), dtype=np.uint8)
    for idx, image in enumerate(images):
        row = idx // columns
        col = idx % columns
        h, w = image.shape[:2]
        sheet[row * max_h : row * max_h + h, col * max_w : col * max_w + w] = image
    imageio.imwrite(output_path, sheet)


def build_train_overrides(train_data_root: str) -> list[str]:
    return [
        "data=dreamzero/libero_spatial_relative",
        "model=dreamzero/vla",
        "model/dreamzero/action_head=wan_flow_matching_action_tf",
        "model/dreamzero/transform=dreamzero_cotrain",
        "train_architecture=lora",
        "num_frames=33",
        "action_horizon=24",
        "num_views=2",
        "num_frame_per_block=2",
        "num_action_per_block=24",
        "num_state_per_block=1",
        "seed=42",
        "report_to=tensorboard",
        "wandb_project=dump_libero_spatial_concat_inputs",
        "output_dir=/tmp/dump_libero_spatial_concat_inputs",
        "bf16=true",
        "tf32=true",
        "eval_bf16=true",
        "dataloader_pin_memory=false",
        "dataloader_num_workers=0",
        "image_resolution_width=320",
        "image_resolution_height=176",
        "max_chunk_size=4",
        "frame_seqlen=880",
        f"libero_spatial_data_root={train_data_root}",
        f"dit_version={DEFAULT_WAN_CKPT_DIR}",
        f"text_encoder_pretrained_path={DEFAULT_WAN_CKPT_DIR}/models_t5_umt5-xxl-enc-bf16.pth",
        f"image_encoder_pretrained_path={DEFAULT_WAN_CKPT_DIR}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        f"vae_pretrained_path={DEFAULT_WAN_CKPT_DIR}/Wan2.1_VAE.pth",
        f"tokenizer_path={DEFAULT_TOKENIZER_DIR}",
        f"pretrained_model_path={DEFAULT_PRETRAINED_MODEL_PATH}",
        "++action_head_cfg.config.skip_component_loading=true",
        "++action_head_cfg.config.defer_lora_injection=true",
    ]


def sample_train_frames(args: argparse.Namespace, task_prompt: str, output_dir: Path) -> dict[str, Any]:
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    config_dir = DREAMZERO_ROOT / "groot" / "vla" / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="conf", overrides=build_train_overrides(args.train_data_root))
    train_dataset = instantiate(cfg.train_dataset)

    rng = random.Random(args.seed)
    saved_frames: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    dataset_length = len(train_dataset)
    base_dataset = train_dataset.datasets[0] if hasattr(train_dataset, "datasets") else train_dataset
    trajectory_to_shard = {
        trajectory_id: shard_index
        for shard_index, trajectory_ids in enumerate(base_dataset.sharded_trajectories)
        for trajectory_id in trajectory_ids
    }

    episodes_path = Path(args.train_data_root) / "meta" / "episodes.jsonl"
    task_episode_ids: list[int] = []
    with episodes_path.open() as f:
        for line in f:
            entry = json.loads(line)
            if any(int(task["task_index"]) == args.task_id for task in entry.get("tasks", [])):
                task_episode_ids.append(int(entry["episode_index"]))
    if not task_episode_ids:
        raise RuntimeError(f"No training episodes found for task {args.task_id}")

    for attempt in range(1, args.max_train_attempts + 1):
        trajectory_id = rng.choice(task_episode_ids)
        trajectory_index = base_dataset.get_trajectory_index(trajectory_id)
        trajectory_length = int(base_dataset.trajectory_lengths[trajectory_index])
        allowed_length = trajectory_length - int(base_dataset.max_delta_index)
        allowed_indices = base_dataset.step_filter[trajectory_id]
        allowed_indices = allowed_indices[allowed_indices <= allowed_length]
        if len(allowed_indices) == 0:
            continue
        step_index = int(rng.choice(allowed_indices))
        shard_index = trajectory_to_shard[trajectory_id]
        if getattr(base_dataset, "shard_start_indices", None) is None or trajectory_id not in base_dataset.shard_start_indices:
            if getattr(base_dataset, "cached_df", None) is not None:
                base_dataset.delete_cached_shard()
            base_dataset.start_cache_shard(shard_index)
            base_dataset.finish_cache_shard()
        indices = {key: delta_indices + step_index for key, delta_indices in base_dataset.delta_indices.items()}
        sample = base_dataset.transforms(base_dataset.get_step_data(trajectory_id, indices))
        text = str(sample["text"]).strip()
        if text != task_prompt:
            continue
        images = np.asarray(sample["images"])
        if images.ndim != 4:
            raise ValueError(f"Expected train sample images to have shape [T,H,W,C], got {images.shape}")
        frame_index = rng.randrange(images.shape[0])
        frame = to_uint8_image(images[frame_index])
        slot = len(saved_frames)
        image_path = output_dir / f"train_{slot:02d}.png"
        imageio.imwrite(image_path, frame)
        saved_frames.append(frame)
        metadata.append(
            {
                "slot": slot,
                "sample_attempt": attempt,
                "trajectory_id": trajectory_id,
                "step_index": step_index,
                "frame_index": frame_index,
                "num_frames_in_sample": int(images.shape[0]),
                "text": text,
                "image_path": str(image_path),
            }
        )
        if len(saved_frames) >= args.num_frames:
            break

    if getattr(base_dataset, "cached_df", None) is not None:
        base_dataset.delete_cached_shard()

    if len(saved_frames) < args.num_frames:
        raise RuntimeError(
            f"Only collected {len(saved_frames)} training frames for task {args.task_id} within {args.max_train_attempts} sampled steps"
        )

    write_contact_sheet(saved_frames, output_dir / "train_contact_sheet.png")
    summary = {
        "mode": "train",
        "task_id": args.task_id,
        "task_prompt": task_prompt,
        "num_frames": len(saved_frames),
        "dataset_length": dataset_length,
        "num_task_episodes": len(task_episode_ids),
        "metadata": metadata,
    }
    (output_dir / "train_metadata.json").write_text(json.dumps(summary, indent=2))
    return summary


def center_crop_and_resize(image: np.ndarray, scale: float, output_height: int, output_width: int) -> np.ndarray:
    from PIL import Image

    image = to_uint8_image(image)
    height, width = image.shape[:2]
    crop_height = max(1, int(round(height * scale)))
    crop_width = max(1, int(round(width * scale)))
    top = max(0, (height - crop_height) // 2)
    left = max(0, (width - crop_width) // 2)
    cropped = image[top : top + crop_height, left : left + crop_width]
    resized = Image.fromarray(cropped).resize((output_width, output_height), resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def rotate_image_180(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(to_uint8_image(image)[::-1, ::-1])


def build_eval_concat_frame(agentview_image: np.ndarray, eye_in_hand_image: np.ndarray) -> np.ndarray:
    agentview = center_crop_and_resize(rotate_image_180(agentview_image), scale=0.95, output_height=176, output_width=320)
    eye_in_hand = center_crop_and_resize(rotate_image_180(eye_in_hand_image), scale=0.95, output_height=176, output_width=320)
    return np.concatenate([np.repeat(agentview, 2, axis=0), np.repeat(eye_in_hand, 2, axis=0)], axis=1)


def sample_eval_frames(args: argparse.Namespace, task_prompt: str, output_dir: Path) -> dict[str, Any]:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    rng = random.Random(args.seed)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_spatial"]()
    task = task_suite.get_task(args.task_id)
    init_states = task_suite.get_task_init_states(args.task_id)
    env = OffScreenRenderEnv(
        bddl_file_name=os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
        camera_heights=args.camera_height,
        camera_widths=args.camera_width,
    )
    env.seed(args.seed)

    candidate_frames: list[np.ndarray] = []
    candidate_metadata: list[dict[str, Any]] = []
    target_candidates = max(args.num_frames, 20)

    try:
        for episode_idx in range(args.n_eval):
            env.reset()
            env.set_init_state(init_states[episode_idx % len(init_states)])
            obs = None
            for _ in range(args.num_zero_steps):
                obs, _, _, _ = env.step([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
            if obs is None:
                raise RuntimeError("failed to initialize LIBERO observation")

            for step_idx in range(args.max_steps):
                frame = build_eval_concat_frame(
                    obs["agentview_image"],
                    obs["robot0_eye_in_hand_image"],
                )
                candidate_frames.append(frame)
                candidate_metadata.append(
                    {
                        "episode_index": episode_idx,
                        "step_index": step_idx,
                        "text": task.language,
                    }
                )
                obs, _, done, _ = env.step([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
                if bool(done):
                    break
                if len(candidate_frames) >= target_candidates:
                    break
            if len(candidate_frames) >= target_candidates:
                break
    finally:
        env.close()

    if len(candidate_frames) < args.num_frames:
        raise RuntimeError(f"Only collected {len(candidate_frames)} eval frames for task {args.task_id}")

    chosen_indices = list(range(len(candidate_frames)))
    rng.shuffle(chosen_indices)
    chosen_indices = sorted(chosen_indices[: args.num_frames])

    final_frames: list[np.ndarray] = []
    final_metadata: list[dict[str, Any]] = []
    for slot, source_idx in enumerate(chosen_indices):
        frame = candidate_frames[source_idx]
        image_path = output_dir / f"eval_{slot:02d}.png"
        imageio.imwrite(image_path, frame)
        final_frames.append(frame)
        item = dict(candidate_metadata[source_idx])
        item.update({"slot": slot, "source_index": source_idx, "image_path": str(image_path)})
        final_metadata.append(item)

    write_contact_sheet(final_frames, output_dir / "eval_contact_sheet.png")
    summary = {
        "mode": "eval",
        "task_id": args.task_id,
        "task_prompt": task_prompt,
        "num_frames": len(final_frames),
        "num_candidates_seen": len(candidate_frames),
        "metadata": final_metadata,
    }
    (output_dir / "eval_metadata.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    task_prompt = load_task_prompt(Path(args.train_data_root), args.task_id)

    summaries: dict[str, Any] = {"task_id": args.task_id, "task_prompt": task_prompt}

    if args.mode in {"train", "both"}:
        train_dir = output_root / "train"
        ensure_clean_dir(train_dir, clear=args.clear_output)
        summaries["train"] = sample_train_frames(args, task_prompt, train_dir)

    if args.mode in {"eval", "both"}:
        eval_dir = output_root / "eval"
        ensure_clean_dir(eval_dir, clear=args.clear_output)
        summaries["eval"] = sample_eval_frames(args, task_prompt, eval_dir)

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
