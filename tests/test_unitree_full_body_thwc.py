from collections import deque
from types import SimpleNamespace

import numpy as np

from scripts.inference.unitree_full_body.server import (
    ACTION_DIM,
    VIDEO_KEYS,
    UnitreeFullBodyPolicyServer,
    _as_rgb_frame,
)


def test_as_rgb_frame_preserves_thwc_and_converts_bgr() -> None:
    bgr = np.zeros((9, 4, 5, 3), dtype=np.uint8)
    bgr[..., 0] = 11
    bgr[..., 1] = 22
    bgr[..., 2] = 33

    rgb = _as_rgb_frame(bgr, key="camera", color_order="bgr")

    assert rgb.shape == (9, 4, 5, 3)
    assert rgb.flags.c_contiguous
    np.testing.assert_array_equal(rgb[0, 0, 0], [33, 22, 11])


def test_as_rgb_frame_decodes_jpeg_window_to_rgb() -> None:
    import cv2

    encoded_frames = []
    for red in (40, 120, 200):
        bgr = np.zeros((12, 16, 3), dtype=np.uint8)
        bgr[..., 2] = red
        ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        assert ok
        encoded_frames.append(encoded.tobytes())

    rgb = _as_rgb_frame(
        {"encoding": "jpeg", "frames": encoded_frames},
        key="camera",
        color_order="bgr",  # JPEG decoding must not depend on the raw-array setting.
    )

    assert rgb.shape == (3, 12, 16, 3)
    assert rgb.flags.c_contiguous
    assert rgb[0, ..., 0].mean() > 35
    assert rgb[0, ..., 2].mean() < 5


def test_build_model_obs_passes_presampled_thwc_through() -> None:
    server = UnitreeFullBodyPolicyServer.__new__(UnitreeFullBodyPolicyServer)
    server.args = SimpleNamespace(
        client_image_color_order="rgb",
        prompt="",
        action_horizon=48,
        video_stride=6,
    )
    server.histories = {key: deque(maxlen=49) for key in VIDEO_KEYS}

    windows = {
        "color_0": np.full((9, 4, 5, 3), 10, dtype=np.uint8),
        "color_2": np.full((9, 4, 5, 3), 20, dtype=np.uint8),
        "color_3": np.full((9, 4, 5, 3), 30, dtype=np.uint8),
    }
    model_obs = server._build_model_obs(
        {**windows, "state": np.zeros(ACTION_DIM, dtype=np.float32), "prompt": "stack blocks"}
    )

    for model_key, client_key in zip(VIDEO_KEYS, ("color_0", "color_2", "color_3")):
        np.testing.assert_array_equal(model_obs[model_key], windows[client_key])
        assert server.histories[model_key][-1].shape == (4, 5, 3)
