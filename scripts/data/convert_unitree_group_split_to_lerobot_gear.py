#!/usr/bin/env python3
"""Convert Unitree mcaps to train/val LeRobot-v2/GEAR datasets.

This is a thin wrapper around ``convert_stack_blocks_to_lerobot_gear.py`` for
raw roots that contain multiple collection groups, for example:

  raw_root/
    2026_06_15_robot_7297/
      episode_0000.mcap
      ...
    2026_06_18_robot_7297/
      episode_0000.mcap
      ...

By default the sorted last episode is converted to ``output_root/val`` and all
earlier episodes are converted to ``output_root/train``.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from types import SimpleNamespace

from convert_stack_blocks_to_lerobot_gear import _episode_sort_key, convert


DEFAULT_COLLECT_BLOCKS_PROMPT_EN = (
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


def _episode_group_dirs(raw_root: Path) -> list[Path]:
    groups = []
    for path in sorted(raw_root.iterdir()):
        if not path.is_dir():
            continue
        if path.name == "ossutil_output":
            continue
        if list(path.glob("episode_*.mcap")):
            groups.append(path)
    return groups


def _episode_mcaps(raw_root: Path) -> list[Path]:
    return sorted(raw_root.glob("*/episode_*.mcap"), key=_episode_sort_key)


def _symlink_episodes(episodes: list[Path], tmp_root: Path) -> None:
    for episode in episodes:
        group_dir = tmp_root / episode.parent.name
        group_dir.mkdir(parents=True, exist_ok=True)
        (group_dir / episode.name).symlink_to(episode)


def _run_convert(
    *,
    raw_root: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> None:
    ns = SimpleNamespace(
        raw_root=str(raw_root),
        output_root=str(output_root),
        max_episodes=args.max_episodes,
        fps=args.fps,
        action_horizon=args.action_horizon,
        width=args.width,
        height=args.height,
        control_key=args.control_key,
        default_task=args.default_task,
        force=args.force,
    )
    convert(ns)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--raw-root",
        required=True,
        help="Raw Unitree root containing one or more group directories with episode_*.mcap files.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Output root. The wrapper writes train/ and val/ underneath it.",
    )
    parser.add_argument(
        "--val-last-n-episodes",
        type=int,
        default=1,
        help="Use the last N sorted episodes as validation.",
    )
    parser.add_argument("--max-episodes", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--action-horizon", type=int, default=48)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--control-key", default="sweep_floor_control")
    parser.add_argument(
        "--default-task",
        default=DEFAULT_COLLECT_BLOCKS_PROMPT_EN,
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    groups = _episode_group_dirs(raw_root)
    if not groups:
        raise RuntimeError(f"{raw_root} 下未找到包含 episode_*.mcap 的组目录")
    episodes = _episode_mcaps(raw_root)
    if not episodes:
        raise RuntimeError(f"{raw_root} 下未找到 */episode_*.mcap")
    if args.val_last_n_episodes <= 0:
        raise ValueError("--val-last-n-episodes must be >= 1")
    if args.val_last_n_episodes >= len(episodes):
        raise ValueError(
            f"--val-last-n-episodes={args.val_last_n_episodes} 会覆盖全部 {len(episodes)} 条 episode，"
            "需要至少保留一条 train episode"
        )

    train_episodes = episodes[: -args.val_last_n_episodes]
    val_episodes = episodes[-args.val_last_n_episodes :]
    print(f"Total episodes: {len(episodes)}")
    print(f"Train episodes: {len(train_episodes)}")
    print(f"Val episodes: {len(val_episodes)}")
    print("Val episode(s):")
    for episode in val_episodes:
        print(f"  {episode.parent.name}/{episode.name}")

    with tempfile.TemporaryDirectory(prefix="unitree_group_split_") as tmp:
        tmp_path = Path(tmp)
        train_raw = tmp_path / "train_raw"
        val_raw = tmp_path / "val_raw"
        _symlink_episodes(train_episodes, train_raw)
        _symlink_episodes(val_episodes, val_raw)
        _run_convert(raw_root=train_raw, output_root=output_root / "train", args=args)
        _run_convert(raw_root=val_raw, output_root=output_root / "val", args=args)

    print(f"完成 split 转换 -> {output_root}")
    print(f"  train: {output_root / 'train'}")
    print(f"  val:   {output_root / 'val'}")


if __name__ == "__main__":
    main()
