#!/usr/bin/env python3
"""Patch a converted LeRobot/GEAR dataset to use one task prompt.

This updates:
  - data/chunk-*/episode_*.parquet: annotation.task and task_index
  - meta/tasks.jsonl
  - meta/episodes.jsonl tasks
  - meta/info.json total_tasks
  - meta/task_prompt.json with optional extra prompt metadata
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm


DEFAULT_PROMPT_EN = (
    "A bimanual robotic task in a home living room environment. The robot looks down from a "
    "head-mounted camera at a light wood-grain desk with several objects: a green building block "
    "and an orange-yellow building block. A light-brown woven basket rests on a white side table "
    "to the right. The robot first turns to the side table, grasps the woven basket with the right "
    "hand, and lifts it. It then holds the basket suspended over the main desk with the right hand "
    "alone. The left hand reaches down, pinches the orange-yellow building block, lifts it, and "
    "drops it into the basket. The left hand then picks up the green building block and drops it "
    "into the basket as well. Finally, both hands carry the basket back to the white side table on "
    "the right and gently place it down. The motions are slow, deliberate, and highly smooth, with "
    "precise finger coordination and seamless weight transfer between hands."
)

DEFAULT_PROMPT_ZH = (
    "一项在家庭客厅环境中进行的双臂机器人操作任务。机器人从头部摄像头俯视一张浅色木纹桌面，"
    "桌上有多个物品：一块绿色积木、一块橙黄色积木。右侧的白色边桌上放着一个浅棕色编织篮。"
    "机器人首先转向边桌，右手抓住编织篮并将其提起。随后将篮子的重量转移到右手单独悬持在"
    "主桌上方，左手向下伸出，精确夹取橙黄色积木并投入篮中。紧接着左手再次伸出，夹取绿色"
    "积木投入篮中。最后双手协同将装有积木的篮子搬回白色边桌上轻轻放下。整个动作缓慢、"
    "从容且高度流畅，手指协调精准，双手间的重量转移衔接自然。"
)


def _jsonl_load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _jsonl_dump(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def patch_dataset(dataset_root: Path, args: argparse.Namespace) -> None:
    meta_dir = dataset_root / "meta"
    data_dir = dataset_root / "data"
    if not meta_dir.is_dir():
        raise FileNotFoundError(f"missing meta dir: {meta_dir}")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"missing data dir: {data_dir}")

    parquet_paths = sorted(data_dir.glob("chunk-*/episode_*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"no parquet files under {data_dir}/chunk-*")

    for path in tqdm(parquet_paths, desc=f"Patching {dataset_root.name} parquet"):
        df = pd.read_parquet(path)
        df["task_index"] = int(args.task_index)
        df["annotation.task"] = args.prompt_en
        df.to_parquet(path, index=False)

    _jsonl_dump(
        meta_dir / "tasks.jsonl",
        [{"task_index": int(args.task_index), "task": args.prompt_en}],
    )

    episodes = _jsonl_load(meta_dir / "episodes.jsonl")
    for episode in episodes:
        episode["tasks"] = [args.prompt_en]
    _jsonl_dump(meta_dir / "episodes.jsonl", episodes)

    info_path = meta_dir / "info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        info["total_tasks"] = 1
        info_path.write_text(json.dumps(info, indent=4, ensure_ascii=False))

    prompt_metadata = {
        "prompt_en": args.prompt_en,
        "prompt_zh": args.prompt_zh,
        "image_name": args.image_name,
        "reference_video": args.reference_video,
        "task_name": args.task_name,
    }
    (meta_dir / "task_prompt.json").write_text(
        json.dumps(prompt_metadata, indent=4, ensure_ascii=False)
    )
    print(f"Patched {len(parquet_paths)} parquet episodes under {dataset_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "dataset_roots",
        nargs="+",
        help="One or more converted GEAR roots, e.g. .../train .../val.",
    )
    parser.add_argument("--prompt-en", default=DEFAULT_PROMPT_EN)
    parser.add_argument("--prompt-zh", default=DEFAULT_PROMPT_ZH)
    parser.add_argument("--image-name", default="/home/ubuntu/first_frame.png")
    parser.add_argument(
        "--reference-video",
        default="/home/ubuntu/upload/episode_000000_observation.images.color_0.mp4",
    )
    parser.add_argument("--task-name", default="collect_blocks_into_basket")
    parser.add_argument("--task-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for root in args.dataset_roots:
        patch_dataset(Path(root).expanduser().resolve(), args)


if __name__ == "__main__":
    main()
