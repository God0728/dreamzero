from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from groot.vla.data.dataset.lerobot_sharded import (  # noqa: E402
    ShardedLeRobotPaddedLangActionChunkDatasetDROID,
    ShardedLeRobotSubLangSingleActionChunkDatasetDROID,
)
from groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf import (  # noqa: E402
    _masked_mean_per_sample,
)


def _fake_padded_dataset(
    *,
    max_chunks: int = 4,
) -> ShardedLeRobotPaddedLangActionChunkDatasetDROID:
    dataset = object.__new__(ShardedLeRobotPaddedLangActionChunkDatasetDROID)
    dataset.max_chunk_size = max_chunks
    dataset._current_padded_plan = None
    dataset._modality_keys = {
        "state": ["state.foo"],
        "video": ["video.agentview"],
        "action": ["action.foo"],
    }
    dataset.curr_traj_data = pd.DataFrame({"language": ["task"]})
    dataset.get_trajectory_data = types.MethodType(
        lambda self, trajectory_id: self.curr_traj_data,
        dataset,
    )
    return dataset


def _raw_data(valid_chunks: int) -> dict[str, np.ndarray]:
    video_len = 8 * valid_chunks + 1
    state_len = valid_chunks
    action_len = 24 * valid_chunks
    return {
        "video.agentview": np.arange(video_len, dtype=np.float32).reshape(video_len, 1),
        "state.foo": np.arange(state_len, dtype=np.float32).reshape(state_len, 1),
        "action.foo": np.arange(action_len, dtype=np.float32).reshape(action_len, 1),
    }


def _key_modalities() -> dict[str, str]:
    return {
        "video.agentview": "video",
        "state.foo": "state",
        "action.foo": "action",
    }


def test_short_segment_with_zero_valid_chunks_is_skipped():
    dataset = _fake_padded_dataset()
    raw = {
        "video.agentview": np.zeros((1, 1), dtype=np.float32),
        "state.foo": np.zeros((0, 1), dtype=np.float32),
        "action.foo": np.zeros((0, 1), dtype=np.float32),
    }

    assert dataset._pad_raw_step_data(raw, _key_modalities()) is None


def test_get_step_data_preserves_parent_no_pad_prefix_and_pads_tail():
    dataset = _fake_padded_dataset(max_chunks=4)
    raw = _raw_data(valid_chunks=4)
    calls: list[tuple[str, str]] = []
    original = ShardedLeRobotSubLangSingleActionChunkDatasetDROID.get_data_by_modality

    def fake_get_data_by_modality(self, trajectory_id, modality, key, step_indices):
        calls.append((modality, key))
        return raw[key]

    ShardedLeRobotSubLangSingleActionChunkDatasetDROID.get_data_by_modality = (
        fake_get_data_by_modality
    )
    try:
        padded = dataset.get_step_data(
            0,
            {
                "video.agentview": np.array([0]),
                "state.foo": np.array([0]),
                "action.foo": np.array([0]),
            },
        )
    finally:
        ShardedLeRobotSubLangSingleActionChunkDatasetDROID.get_data_by_modality = original

    assert padded is not None
    assert calls[0][0] == "video"
    np.testing.assert_array_equal(padded["video.agentview"][:33], raw["video.agentview"])
    np.testing.assert_array_equal(padded["state.foo"][:4], raw["state.foo"])
    np.testing.assert_array_equal(padded["action.foo"][:96], raw["action.foo"])


def test_temporal_masks_encode_valid_chunk_counts():
    cases = [1, 4]
    for expected_chunks in cases:
        dataset = _fake_padded_dataset(max_chunks=4)
        padded = dataset._pad_raw_step_data(_raw_data(expected_chunks), _key_modalities())

        assert padded is not None
        assert "valid_chunks" not in padded
        assert dataset._current_padded_plan["valid_chunks"] == expected_chunks
        assert padded["video_temporal_mask"].sum() == 8 * expected_chunks + 1
        assert padded["state_temporal_mask"].sum() == expected_chunks
        assert padded["action_temporal_mask"].sum() == 24 * expected_chunks
        assert padded["chunk_temporal_mask"].sum() == expected_chunks


def test_padding_repeats_last_valid_value_after_prefix():
    dataset = _fake_padded_dataset(max_chunks=4)
    padded = dataset._pad_raw_step_data(_raw_data(valid_chunks=1), _key_modalities())

    assert padded is not None
    assert padded["video.agentview"].shape[0] == 33
    assert padded["state.foo"].shape[0] == 4
    assert padded["action.foo"].shape[0] == 96
    assert np.all(padded["video.agentview"][9:] == padded["video.agentview"][8])
    assert np.all(padded["state.foo"][1:] == padded["state.foo"][0])
    assert np.all(padded["action.foo"][24:] == padded["action.foo"][23])


def test_per_sample_masked_mean_weights_k1_and_k4_samples_equally():
    values = torch.tensor(
        [
            [10.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    mask = torch.tensor(
        [
            [True, False, False, False],
            [True, True, True, True],
        ]
    )

    loss = _masked_mean_per_sample(values, mask)

    assert torch.isclose(loss, torch.tensor(5.5))


def test_per_sample_masked_mean_ignores_invalid_action_dims():
    values = torch.tensor(
        [
            [[10.0, 100.0], [0.0, 0.0]],
            [[1.0, 1.0], [1.0, 1.0]],
        ]
    )
    mask = torch.tensor(
        [
            [[True, False], [False, False]],
            [[True, True], [True, True]],
        ]
    )

    loss = _masked_mean_per_sample(values, mask)

    assert torch.isclose(loss, torch.tensor(5.5))


if __name__ == "__main__":
    test_short_segment_with_zero_valid_chunks_is_skipped()
    test_get_step_data_preserves_parent_no_pad_prefix_and_pads_tail()
    test_temporal_masks_encode_valid_chunk_counts()
    test_padding_repeats_last_valid_value_after_prefix()
    test_per_sample_masked_mean_weights_k1_and_k4_samples_equally()
    test_per_sample_masked_mean_ignores_invalid_action_dims()
