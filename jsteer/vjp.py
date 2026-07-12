"""Direct VJP pullback: `J^T @ w` in one backward pass, no cached Jacobian. (Claude)

VJP = vector-Jacobian product: reverse-mode autodiff contracts a COTANGENT `w`
(a direction placed at the final layer) inside the backward pass, yielding the
same per-layer `J_l^T @ w` that `Jacobian.pullback` reads off the cached matrix
-- by linearity `mean_p(J_p^T w) = mean_p(J_p)^T w`. Cost is ONE backward per
prompt (vs ceil(d_model/dim_batch) for the full fit). Use it for a single
concept when you don't need the reusable cache, and as the parity reference for
the cached path (cos > 0.999 per layer expected; fp16 cache storage is the only
gap).

Estimator conventions are jlens's exactly (this is the code path that produced
the verified j-steer-dev result): cotangent placed at every valid target
position of the final layer, gradient read at every valid source position and
meaned, positions before skip_first=16 excluded (attention sinks), final
position excluded (no next-token target). Right-padded batches: the valid mask
excludes pads, so batching does not change the estimate.
"""
from __future__ import annotations

import torch
from jlens.fitting import SKIP_FIRST_N_POSITIONS
from jlens.hf import from_hf
from jlens.hooks import ActivationRecorder
from loguru import logger
from torch import Tensor
from tqdm.auto import tqdm

from steering_lite.vector import Vector

from .applies import JacobianWordC
from .jacobian import _resolve_layers, _to_vector, _word_cotangent


def _valid_mask(attention_mask: Tensor, skip_first: int) -> Tensor:
    """Boolean [B, S]: right-padded real tokens in [skip_first : real_len-1].
    jlens valid_position_mask extended to a right-padded batch."""
    real_len = attention_mask.sum(dim=1, keepdim=True)           # [B, 1]
    pos = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    mask = (pos[None, :] >= skip_first) & (pos[None, :] < real_len - 1)
    # the & with attention_mask is redundant for right-padded batches
    # (pos < real_len-1 already excludes pads) but guards non-right-padded input.
    return mask & attention_mask.bool()


def pullback_vjp(model, tok, prompts: list[str], layers, cotangent: Tensor, *,
                 batch_size: int = 8, max_length: int = 128,
                 skip_first: int = SKIP_FIRST_N_POSITIONS) -> dict[int, Tensor]:
    """Per-layer mean over `prompts` of J_l^T @ cotangent, via one backward per
    batch (grads for every source layer come from the same backward)."""
    lm = from_hf(model, tok)   # freezes params, locates blocks; grads flow to
    target_layer = lm.n_layers - 1                     # activations only
    layers = _resolve_layers(layers, lm.n_layers)
    if max(layers) >= target_layer:
        raise ValueError(f"source layers {layers} must be < target {target_layer}")
    d = cotangent.shape[0]
    G = {l: torch.zeros(d, dtype=torch.float32, device=model.device) for l in layers}
    count = 0
    for i in tqdm(range(0, len(prompts), batch_size), desc="pullback_vjp",
                  disable=len(prompts) <= batch_size, mininterval=120, maxinterval=120):
        batch = prompts[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_length, padding_side="right").to(model.device)
        valid = _valid_mask(enc["attention_mask"], skip_first)    # [B, S] bool
        if valid.sum(dim=1).min() == 0:
            raise ValueError(f"a prompt has 0 valid positions "
                             f"(too short for skip_first={skip_first})")
        with ActivationRecorder(lm.layers, at=[*layers, target_layer],
                                start_graph_at=min(layers)) as rec, torch.enable_grad():
            model(**enc)
            h_final = rec.activations[target_layer]               # [B, S, d]
            srcs = [rec.activations[l] for l in layers]
            c = (cotangent.to(h_final.device).to(h_final.dtype).view(1, 1, d)
                 * valid.unsqueeze(-1))
            grads = torch.autograd.grad(h_final, srcs, grad_outputs=c)
        den = valid.sum(dim=1, keepdim=True).float()              # [B, 1]
        for l, g in zip(layers, grads):
            v_b = (g.float() * valid.unsqueeze(-1)).sum(dim=1) / den   # [B, d]
            G[l] += v_b.sum(0)
        count += len(batch)
    return {l: G[l] / count for l in layers}


def word_vector_vjp(model, tok, prompts: list[str], words: list[str], *,
                    layers=None, batch_size: int = 8, max_length: int = 128,
                    skip_first: int = SKIP_FIRST_N_POSITIONS) -> Vector:
    """The verified j-steer-dev word extraction, self-contained: the word
    cotangent pulled back over `prompts` (the prompts J is linearized on). Same
    vector as Jacobian.fit(model, tok, prompts).word_vector(...) when the
    prompts, layers, skip_first and max length match."""
    cot = _word_cotangent(model, tok, words)
    G = pullback_vjp(model, tok, prompts, layers, cot, batch_size=batch_size,
                     max_length=max_length, skip_first=skip_first)
    logger.info("word_vector_vjp per-layer |v| (pre-norm): " +
                " ".join(f"{l}:{v.norm():.3g}" for l, v in G.items()))
    return _to_vector(JacobianWordC(layers=tuple(G)), G)
