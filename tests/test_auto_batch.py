"""`train.batch_size: -1` probes VRAM for the largest batch that fits the budget."""

import pytest
import torch
from omegaconf import OmegaConf

from img_clf.dl.utils import auto_batch_size


def _cfg():
    return OmegaConf.create(
        {
            "model_name": "efficientnet_b0",
            "train": {
                "img_size": [256, 256],
                "amp_enabled": True,
                "amp_dtype": "bfloat16",
                "layers_to_train": -1,
                "use_ema": False,
                "augs": {"multiscale_prob": 0.0},
            },
        }
    )


def test_non_cuda_device_falls_back_without_probing():
    assert auto_batch_size(_cfg(), 3, "cpu", default=7) == 7


@pytest.mark.slow
@pytest.mark.gpu
def test_budget_bounds_the_batch_and_rng_is_untouched():
    """Twice the budget must buy a bigger batch, and the probe must leave the torch RNG
    where it found it - the real model inits right after, and "same seed, same run" holds
    only if a run with `-1` draws the same numbers as one with an explicit batch size."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    torch.manual_seed(0)
    before = torch.get_rng_state()
    small = auto_batch_size(_cfg(), 3, "cuda", target_fraction=0.1)
    large = auto_batch_size(_cfg(), 3, "cuda", target_fraction=0.2)
    assert 1 <= small < large
    assert torch.equal(before, torch.get_rng_state())
