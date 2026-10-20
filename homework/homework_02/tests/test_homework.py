"""Проверки функций на маленьких массивах. Веса и GPU не нужны."""

import numpy as np
import pytest
import torch

from homework import aggregate_logits, transform_clip, predict_logits, sample_indices


@pytest.mark.parametrize("fps, expected", [(15, np.arange(16)), (30, np.arange(16) * 2)])
def test_sampling_uses_seconds(fps, expected):
    actual = sample_indices(100, fps, 0)
    np.testing.assert_array_equal(actual, expected)
    assert np.issubdtype(actual.dtype, np.integer)


def test_sampling_offset_and_last_frame():
    np.testing.assert_array_equal(sample_indices(46, 30, 0.5), np.arange(15, 46, 2))


def test_sampling_outside_video():
    with pytest.raises(ValueError):
        sample_indices(15, 15, 0)


def test_inference_disables_training_and_gradients():
    class TinyModel(torch.nn.Module):
        def forward(self, batch):
            assert not self.training
            assert not torch.is_grad_enabled()
            assert batch.shape == (2, 3, 16, 4, 4)
            return torch.tensor([[2., -1.], [-3., 4.]])

    result = predict_logits(TinyModel(), torch.zeros(2, 3, 16, 4, 4))
    torch.testing.assert_close(result, torch.tensor([[2., -1.], [-3., 4.]]))


def test_aggregation_averages_logits_before_softmax():
    logits = torch.tensor([[8., 0.], [0., 2.], [0., 2.]])
    probabilities = aggregate_logits(logits)
    torch.testing.assert_close(probabilities, torch.tensor([4/3, 0.]).softmax(0))
    assert probabilities.shape == (2,)
    assert torch.isclose(probabilities.sum(), torch.tensor(1.))


def test_aggregation_is_numerically_stable():
    result = aggregate_logits(torch.tensor([[10000., 10001., 9999.]]))
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, torch.tensor([0., 1., -1.]).softmax(0))


def test_freezing_preserves_shape_pixels_and_input():
    clip = torch.arange(4*3*2*2).reshape(4, 3, 2, 2)
    original = clip.clone()
    result = transform_clip(clip, "frozen")
    assert result.shape == clip.shape and result.dtype == clip.dtype
    for frame in result:
        torch.testing.assert_close(frame, original[2])
    torch.testing.assert_close(clip, original)


def test_shuffle_preserves_every_frame_and_input():
    clip = torch.arange(16*3*2*2).reshape(16, 3, 2, 2)
    original = clip.clone()
    result = transform_clip(clip, "shuffled", seed=42)
    assert result.shape == clip.shape and result.dtype == clip.dtype
    expected = torch.randperm(16, generator=torch.Generator().manual_seed(42))
    torch.testing.assert_close(result, original[expected])
    assert not torch.equal(result, original)
    torch.testing.assert_close(clip, original)


def test_shuffle_reproducibility_and_local_random_state():
    clip = torch.arange(16).reshape(16, 1, 1, 1)
    state = torch.random.get_rng_state().clone()
    first = transform_clip(clip, "shuffled", seed=7)
    torch.testing.assert_close(first, transform_clip(clip, "shuffled", seed=7))
    assert not torch.equal(first, transform_clip(clip, "shuffled", seed=8))
    torch.testing.assert_close(torch.random.get_rng_state(), state)


def test_unknown_transform_mode():
    with pytest.raises(ValueError):
        transform_clip(torch.zeros(16, 3, 2, 2), "unknown")
