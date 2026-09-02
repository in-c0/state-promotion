"""Guard against an accelerator that silently returns wrong numbers.

Motivated by a reproducible Apple M1 Max failure: the Qwen2 forward pass on MPS
zero-fills whole layer outputs after a spurious Metal out-of-memory report and
returns a wrong loss with no Python exception. Every downstream metric would be
silently invalid, so the loader has to refuse the device rather than trust it.
"""
import pytest
import torch

from state_promotion.lm import DeviceNumericsError, verify_device_numerics


class ProbeTokenizer:
    def __call__(self, text, return_tensors=None):
        return {"input_ids": torch.tensor([[1, 2, 3, 4]])}


class FakeBase(torch.nn.Module):
    """Returns a fixed CPU loss and a scripted sequence of accelerator losses."""

    def __init__(self, cpu_loss, device_losses):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.cpu_loss = cpu_loss
        self.device_losses = list(device_losses)
        self.calls = 0

    def forward(self, input_ids=None, labels=None):
        on_cpu = input_ids.device.type == "cpu"
        if on_cpu and self.calls == 0:
            value = self.cpu_loss
        else:
            value = self.device_losses[min(self.calls - 1, len(self.device_losses) - 1)]
        self.calls += 1
        return type("Out", (), {"loss": torch.tensor(value)})()


def verify(cpu_loss, device_losses):
    return verify_device_numerics(
        FakeBase(cpu_loss, device_losses), ProbeTokenizer(), "cpu", tolerance=0.05,
    )


def test_accepts_accelerator_matching_cpu():
    report = verify(6.75374, [6.75374, 6.75375, 6.75373])
    assert report["verified"] is True
    assert report["matches_cpu"] is True
    assert report["deterministic"] is True


def test_rejects_accelerator_that_disagrees_with_cpu():
    """The observed MPS failure: stable, plausible-looking, and wrong."""
    with pytest.raises(DeviceNumericsError) as err:
        verify(6.75374, [13.84513, 13.84513, 13.84513])
    assert "failed the numerical self-check" in str(err.value)


def test_rejects_nondeterministic_accelerator():
    with pytest.raises(DeviceNumericsError):
        verify(6.75374, [6.75374, 11.60714, 0.0])


def test_rejects_non_finite_accelerator():
    with pytest.raises(DeviceNumericsError):
        verify(6.75374, [float("nan"), float("nan"), float("nan")])
