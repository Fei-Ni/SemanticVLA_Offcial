"""Device selection tests: mocked host detection, dispatch, and one XPU parity check.

Run from the repository root:
    PYTHONPATH=. python -m pytest -q tests
"""

import importlib.util
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import torch
from omegaconf import OmegaConf

from semanticvla import device as dev

HAS_SERVER_DEPS = all(importlib.util.find_spec(m) for m in ("websockets", "msgpack"))
HAS_XPU = hasattr(torch, "xpu") and torch.xpu.is_available()


@contextmanager
def fake_host(xpu=False, cuda=False):
    with mock.patch.object(dev, "is_available", side_effect=lambda t: {"xpu": xpu, "cuda": cuda}.get(t, t == "cpu")):
        yield


class DeviceModuleTest(unittest.TestCase):
    def test_default_prefers_xpu_then_cuda_then_cpu(self):
        with fake_host(xpu=True, cuda=True):
            self.assertEqual(dev.default_device_type(), "xpu")
        with fake_host(cuda=True):
            self.assertEqual(dev.default_device_type(), "cuda")
        with fake_host():
            self.assertEqual(dev.default_device_type(), "cpu")

    def test_explicit_device_wins(self):
        with fake_host(xpu=True):
            self.assertEqual(dev.resolve_device("cuda:1"), torch.device("cuda", 1))
            self.assertEqual(dev.resolve_device("cpu"), torch.device("cpu"))
            self.assertEqual(dev.resolve_device("cuda"), torch.device("cuda", 0))
            self.assertEqual(dev.resolve_device(None), torch.device("xpu", 0))

    def test_autocast_is_disabled_on_cpu_and_for_float32_on_xpu(self):
        with dev.autocast("cpu", torch.bfloat16):
            self.assertFalse(torch.is_autocast_enabled("cpu"))
        with dev.autocast("xpu", torch.float32):
            self.assertFalse(torch.is_autocast_enabled("cpu"))

    def test_flash_attention_falls_back_off_cuda(self):
        self.assertEqual(dev.attn_implementation("flash_attention_2", "xpu"), "sdpa")
        self.assertEqual(dev.attn_implementation("flash_attention_2", "cpu"), "sdpa")
        self.assertEqual(dev.attn_implementation("flash_attention_2", "cuda"), "flash_attention_2")
        self.assertEqual(dev.attn_implementation("eager", "xpu"), "eager")
        with fake_host(xpu=True):
            self.assertEqual(dev.attn_implementation("flash_attention_2", {"": 0}), "sdpa")


class QwenInterfaceTest(unittest.TestCase):
    def _build(self, qwenvl, xpu=False, cuda=False):
        from semanticvla.model.modules.vlm import QWen3

        config = OmegaConf.create({"framework": {"qwenvl": qwenvl}})
        fake_model = mock.MagicMock()
        vlm = QWen3.Qwen3VLForConditionalGeneration
        with fake_host(xpu=xpu, cuda=cuda), \
                mock.patch.object(vlm, "from_pretrained", return_value=fake_model) as load, \
                mock.patch.object(QWen3.AutoProcessor, "from_pretrained"):
            QWen3._QWen3_VL_Interface(config)
        return load.call_args.kwargs

    def test_default_device_map_follows_host(self):
        self.assertEqual(self._build({"attn_implementation": "flash_attention_2"}, xpu=True)["device_map"], "xpu")
        self.assertEqual(self._build({"attn_implementation": "flash_attention_2"}, cuda=True)["device_map"], "cuda")

    def test_cuda_host_keeps_flash_attention(self):
        kwargs = self._build({"attn_implementation": "flash_attention_2"}, cuda=True)
        self.assertEqual(kwargs["attn_implementation"], "flash_attention_2")

    def test_xpu_host_uses_sdpa_and_explicit_device_map_wins(self):
        kwargs = self._build({"attn_implementation": "flash_attention_2", "device_map": "cpu"}, xpu=True)
        self.assertEqual(kwargs["attn_implementation"], "sdpa")
        self.assertEqual(kwargs["device_map"], "cpu")


class DeploymentEntryPointsTest(unittest.TestCase):
    @unittest.skipUnless(HAS_SERVER_DEPS, "needs websockets and msgpack")
    def test_server_device_defaults_to_detection(self):
        from deployment.model_server.server_policy import build_argparser

        self.assertIsNone(build_argparser().parse_args([]).device)
        self.assertEqual(build_argparser().parse_args(["--device", "xpu"]).device, "xpu")

    def test_offline_loader_builds_on_the_resolved_device(self):
        from deployment.model_server import predict_action_offline as offline

        cfg = OmegaConf.create({"trainer": {}, "framework": {"qwenvl": {}}})
        model = mock.MagicMock()
        model.load_state_dict.return_value = ([], [])
        model.to.return_value = model
        built = {}

        def build(config):
            built["device_map"] = config.framework.qwenvl.device_map
            return model

        run_dir = mock.MagicMock()
        with fake_host(xpu=True), \
                mock.patch.object(offline.OmegaConf, "load", return_value=cfg), \
                mock.patch.object(offline, "build_framework", side_effect=build), \
                mock.patch.object(offline.torch, "load", return_value={}):
            offline.load_model(run_dir, Path("model.pt"))
            self.assertEqual(built["device_map"], "xpu:0")
            offline.load_model(run_dir, Path("model.pt"), device="cpu")
            self.assertEqual(built["device_map"], "cpu")


@contextmanager
def cpu_noise():
    """Draw random noise on the CPU and move it, so both devices see the same values."""
    originals = {n: getattr(torch, n) for n in ("randn", "randn_like")}

    def randn(*size, device=None, generator=None, **kw):
        out = originals["randn"](*size, generator=torch.Generator().manual_seed(0), **kw)
        return out.to(device) if device is not None else out

    with mock.patch.object(torch, "randn", randn):
        yield


@unittest.skipUnless(HAS_XPU, "needs an Intel XPU")
class XpuParityTest(unittest.TestCase):
    def test_action_head_matches_cpu(self):
        from semanticvla.model.modules.action_model.SemanticVLA_ActionHeader import SemanticVLA_ActionHead

        hidden = 768
        cfg = OmegaConf.create({"framework": {"action_model": {
            "action_model_type": "DiT-B", "hidden_size": hidden, "action_dim": 7, "state_dim": 8,
            "future_action_window_size": 7, "action_horizon": 8, "past_action_window_size": 0,
            "num_inference_timesteps": 2, "num_timestep_buckets": 1000, "num_target_vision_tokens": 8,
            "max_seq_len": 256, "noise_beta_alpha": 1.5, "noise_beta_beta": 1.0, "noise_s": 0.999,
            "clamp_sample_time": False, "add_pos_embed": True,
            "diffusion_model_cfg": {"num_layers": 2, "cross_attention_dim": 64, "output_dim": hidden, "trace_dim": 0},
            "trace": {"injection_mode": "none"},
        }}})
        torch.manual_seed(0)
        head = SemanticVLA_ActionHead(full_config=cfg).eval()
        with torch.no_grad():  # no zero-initialised layer may hide a difference
            for p in head.parameters():
                p.copy_(torch.randn_like(p) * 0.05)
        vl = torch.randn(1, 12, 64)
        state = torch.randn(1, 1, 8)

        with cpu_noise(), torch.no_grad():
            ref = head.predict_action(vl, state, do_sample=False, sample_seed=0)
            xpu_head = head.to("xpu")
            out = xpu_head.predict_action(vl.to("xpu"), state.to("xpu"), do_sample=False, sample_seed=0).cpu()
        self.assertGreater(ref.std().item(), 0.0)
        torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-3)


if __name__ == "__main__":
    unittest.main()
