from pathlib import Path
import unittest

import numpy as np

from eval_utils import serve_unitree_dreamzero_eef as server


class UnitreeTransportOptimizationTest(unittest.TestCase):
    def _policy_stub(self):
        policy = object.__new__(server.UnitreeDreamZeroEEFPolicy)
        policy.action_horizon = 48
        policy.obs_chunk_size = 49
        policy.initial_frames = 1
        policy.subsequent_frames = 8
        policy.include_current_boundary = False
        policy.image_resize = None
        return policy

    def test_phase1_selected_frames_match_legacy_server_sampling(self):
        policy = self._policy_stub()
        frames = [np.full((4, 5, 3), i, dtype=np.uint8) for i in range(49)]

        policy.is_first_request = False
        legacy = policy._select_video_window(frames, source_key="camera")
        selected_indices = policy._client_subsequent_image_indices()
        optimized = policy._select_video_window([frames[i] for i in selected_indices], source_key="camera")

        self.assertEqual(selected_indices, [0, 6, 12, 18, 24, 30, 36, 42])
        np.testing.assert_array_equal(optimized, legacy)

    def test_phase1_initial_frame_matches_legacy_latest_frame(self):
        policy = self._policy_stub()
        frames = [np.full((4, 5, 3), i, dtype=np.uint8) for i in range(49)]

        policy.is_first_request = True
        legacy = policy._select_video_window(frames, source_key="camera")
        optimized = policy._select_video_window([frames[-1]], source_key="camera")

        np.testing.assert_array_equal(optimized, legacy)

    def test_phase3_jpeg_payload_decodes_to_declared_shape(self):
        try:
            import cv2
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"cv2 unavailable: {exc}")

        frames = np.zeros((2, 8, 10, 3), dtype=np.uint8)
        frames[0, :, :, 0] = 50
        frames[1, :, :, 1] = 150
        encoded = []
        for frame in frames:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            self.assertTrue(ok)
            encoded.append(buf.tobytes())
        payload = {
            server.IMAGE_TRANSPORT_SENTINEL: "jpeg",
            "shape": list(frames.shape),
            "frames": encoded,
        }

        decoded = server._as_frames(payload, key="camera")
        self.assertEqual(len(decoded), 2)
        self.assertEqual(decoded[0].shape, (8, 10, 3))
        self.assertEqual(decoded[1].shape, (8, 10, 3))

    def test_skip_img_transform_resize_keeps_metadata_width_height_order(self):
        sim_policy_path = (
            Path(__file__).resolve().parents[1]
            / "groot"
            / "vla"
            / "model"
            / "n1_5"
            / "sim_policy.py"
        )
        source = sim_policy_path.read_text()

        self.assertIn("height, width = t.height, t.width", source)
        self.assertIn("metadata.modalities.video[key].resolution = (width, height)", source)
        self.assertNotIn("metadata.modalities.video[key].resolution = (height, width)", source)


if __name__ == "__main__":
    unittest.main()
