#!/usr/bin/env python
"""Precompute DreamZero T5 prompt embeddings for LeRobot/GEAR datasets.

The training transform builds a final embodiment-specific prompt before T5
encoding. This script scans the configured mixture datasets, builds the same
final prompts, encodes each unique prompt once, and writes cache files consumed
by DreamTransform via `text_embedding_cache_dir`.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.distributed as dist
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf import ensure_file
from groot.vla.model.dreamzero.transform.dreamzero_cotrain import (
    HuggingfaceTokenizer,
    format_dreamzero_prompt,
    split_language_variants,
    text_embedding_cache_path,
)


DEFAULT_BATCH_SIZE = 16


def _init_distributed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, 0

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    return True, dist.get_rank(), dist.get_world_size(), local_rank


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse bool value: {value!r}")


def _iter_dataset_paths(cfg: DictConfig) -> list[tuple[str, Path]]:
    mixture_spec = OmegaConf.to_container(cfg.train_dataset.mixture_spec, resolve=True)
    paths: list[tuple[str, Path]] = []
    for mixture_item in mixture_spec:
        dataset_path = mixture_item.get("dataset_path", {})
        for embodiment_name, embodiment_paths in dataset_path.items():
            for path in embodiment_paths:
                paths.append((str(embodiment_name), Path(to_absolute_path(str(path)))))
    return paths


def _task_jsonl_candidates(dataset_dir: Path) -> list[Path]:
    return [
        dataset_dir / "meta" / "tasks.jsonl",
        dataset_dir / "tasks.jsonl",
    ]


def _read_tasks(dataset_dir: Path) -> list[str]:
    for tasks_path in _task_jsonl_candidates(dataset_dir):
        if not tasks_path.exists():
            continue
        tasks: list[str] = []
        with tasks_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                task = record.get("task")
                if task is None:
                    task = record.get("language") or record.get("instruction")
                if task is None:
                    continue
                tasks.append(str(task))
        return tasks
    raise FileNotFoundError(
        f"Could not find tasks.jsonl under {dataset_dir}. Checked: "
        + ", ".join(str(path) for path in _task_jsonl_candidates(dataset_dir))
    )


def _collect_prompts(cfg: DictConfig) -> list[str]:
    mapping = OmegaConf.to_container(cfg.embodiment_tag_to_projector_index, resolve=True)
    num_views = int(cfg.get("num_views", 1))
    prompts: set[str] = set()

    for embodiment_name, dataset_dir in _iter_dataset_paths(cfg):
        if embodiment_name not in mapping:
            raise KeyError(f"Embodiment {embodiment_name!r} missing from embodiment_tag_to_projector_index")
        embodiment_id = int(mapping[embodiment_name])
        tasks = _read_tasks(dataset_dir)
        for task in tasks:
            for variant in split_language_variants(task):
                prompts.add(
                    format_dreamzero_prompt(
                        variant,
                        embodiment_id=embodiment_id,
                        num_views=num_views,
                        embodiment_tag_mapping=mapping,
                        embodiment_tag=embodiment_name,
                    )
                )
    return sorted(prompts)


def _load_text_encoder(cfg: DictConfig, device: str) -> torch.nn.Module:
    text_encoder = instantiate(cfg.action_head_cfg.config.text_encoder_cfg)
    text_enc_path = ensure_file(
        text_encoder.text_encoder_pretrained_path,
        "models_t5_umt5-xxl-enc-bf16.pth",
    )
    text_encoder.load_state_dict(torch.load(text_enc_path, map_location="cpu"))
    return text_encoder.to(device=device, dtype=torch.bfloat16).eval()


def _cache_exists(cache_dirs: list[Path], prompt: str, *, max_length: int, cache_tag: str) -> bool:
    return all(
        Path(
            text_embedding_cache_path(str(cache_dir), prompt, max_length=max_length, cache_tag=cache_tag)
        ).exists()
        for cache_dir in cache_dirs
    )


@hydra.main(config_path="../../groot/vla/configs", config_name="conf", version_base=None)
def main(cfg: DictConfig) -> None:
    is_distributed, rank, world_size, local_rank = _init_distributed()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    cache_dir_value = cfg.get("text_embedding_cache_dir")
    if cache_dir_value is None:
        raise ValueError(
            "text_embedding_cache_dir is required. Example: "
            "text_embedding_cache_dir=/path/to/dreamzero_t5_cache"
        )
    cache_dirs = [Path(to_absolute_path(str(cache_dir_value)))]
    for cache_dir in cache_dirs:
        cache_dir.mkdir(parents=True, exist_ok=True)

    max_length = int(cfg.get("max_length", 512))
    cache_tag = str(cfg.get("text_embedding_cache_tag", "dreamzero_wan_t5"))
    batch_size = int(cfg.get("precompute_text_embedding_batch_size", DEFAULT_BATCH_SIZE))
    overwrite = _as_bool(cfg.get("precompute_text_embedding_overwrite"), default=False)

    prompts = _collect_prompts(cfg)
    if rank == 0:
        print(
            f"Collected {len(prompts)} unique final DreamZero prompts from "
            f"{len(_iter_dataset_paths(cfg))} dataset entries. cache_dir={cache_dirs[0]}"
        )

    prompts = prompts[rank::world_size] if is_distributed else prompts
    if not overwrite:
        prompts = [
            prompt
            for prompt in prompts
            if not _cache_exists(cache_dirs, prompt, max_length=max_length, cache_tag=cache_tag)
        ]

    if is_distributed:
        count = torch.tensor([len(prompts)], device=device, dtype=torch.long)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        if rank == 0:
            print(f"Prompts to encode after cache check: {int(count.item())}")
    else:
        print(f"Prompts to encode after cache check: {len(prompts)}")

    if not prompts:
        if is_distributed:
            dist.barrier()
            dist.destroy_process_group()
        return

    tokenizer = HuggingfaceTokenizer(name=str(cfg.tokenizer_path), seq_len=max_length, clean="whitespace")
    text_encoder = _load_text_encoder(cfg, device)

    progress = tqdm(
        total=len(prompts),
        desc=f"Encoding T5 prompts rank {rank}/{world_size}" if is_distributed else "Encoding T5 prompts",
        unit="prompt",
        disable=is_distributed and rank != 0,
        dynamic_ncols=True,
    )
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            ids, mask = tokenizer(batch_prompts, return_mask=True, add_special_tokens=True)
            ids = ids.to(device)
            mask = mask.to(device=device, dtype=torch.bool)
            context = text_encoder(ids, mask).clone().to(dtype=torch.bfloat16)
            seq_lens = mask.gt(0).sum(dim=1).long()
            for i, seq_len in enumerate(seq_lens):
                context[i, seq_len:] = 0

            for i, prompt in enumerate(batch_prompts):
                payload = {
                    "context": context[i].detach().cpu().contiguous(),
                    "mask": mask[i].detach().cpu().contiguous(),
                    "prompt": prompt,
                    "max_length": max_length,
                    "cache_tag": cache_tag,
                }
                for cache_dir in cache_dirs:
                    cache_path = Path(
                        text_embedding_cache_path(
                            str(cache_dir),
                            prompt,
                            max_length=max_length,
                            cache_tag=cache_tag,
                        )
                    )
                    if cache_path.exists() and not overwrite:
                        continue
                    tmp_path = cache_path.with_suffix(cache_path.suffix + f".rank{rank}.tmp")
                    torch.save(payload, tmp_path)
                    os.replace(tmp_path, cache_path)
            progress.update(len(batch_prompts))
    progress.close()

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
