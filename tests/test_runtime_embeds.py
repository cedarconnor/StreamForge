import torch

from streamforge.diffusion.runtime_eager import effective_embeds


def test_tm_one_returns_prompt_unchanged():
    prompt = torch.ones(1, 4, 8)
    neutral = torch.zeros(1, 4, 8)
    out = effective_embeds(prompt, neutral, 1.0)
    assert torch.equal(out, prompt)


def test_tm_zero_returns_neutral():
    prompt = torch.ones(1, 4, 8)
    neutral = torch.zeros(1, 4, 8)
    out = effective_embeds(prompt, neutral, 0.0)
    assert torch.allclose(out, neutral)


def test_tm_half_is_midpoint():
    prompt = torch.full((1, 4, 8), 2.0)
    neutral = torch.zeros(1, 4, 8)
    out = effective_embeds(prompt, neutral, 0.5)
    assert torch.allclose(out, torch.full((1, 4, 8), 1.0))


def test_none_prompt_returns_none():
    assert effective_embeds(None, torch.zeros(1, 4, 8), 0.5) is None


def test_shape_mismatch_falls_back_to_prompt():
    prompt = torch.ones(1, 4, 8)
    neutral = torch.zeros(1, 6, 8)   # different seq len
    out = effective_embeds(prompt, neutral, 0.5)
    assert torch.equal(out, prompt)


def test_missing_neutral_falls_back_to_prompt():
    prompt = torch.ones(1, 4, 8)
    out = effective_embeds(prompt, None, 0.5)
    assert torch.equal(out, prompt)
