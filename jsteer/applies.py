"""steering-lite method registration + modular delivery of jacobian vectors.

(Claude)

jsteer vectors are plain `steering_lite.Vector` objects: one unit direction v
per layer in `stacked["v"]` with a leading k=1 dim `[1, d]` (byte-identical
layout to steering-lite's mean_diff), so attach / calibrate / save / `with
v(model, C=...)` all work unchanged.

Extraction never goes through steering-lite's `train()` (it needs gradients
that train's no_grad path can't give) -- it lives in `jacobian.py` (the cached
pullback, `J^T @ w` read off the stored Jacobian) and `vjp.py` (the same vector
via one backward, a vector-Jacobian product). The `extract` entries here are
stubs that say so.

DELIVERY of v to the residual stream is decoupled from extraction: the same v
can be added everywhere, clamped to a target component, gated to the last
positions, or overwrite a span. `cfg.apply_mode` selects the mode; adding a
mode = one function + one APPLY_REGISTRY entry.

Protocol (steering-lite config.py Method.apply):
    apply(mod, x, y, shared, stacked, cfg) -> y_new   # same shape [b, s, d]

Position semantics: generation uses LEFT padding, so the last real token is
position -1; `cfg.apply_span` selects how many trailing positions to target.
Note that during incremental generation (KV cache) every decode step has s=1,
so add_last touches each generated token but only the tail of the prefill.

Sign conventions (set by the cotangent in jacobian.py):
    jacobian_word          +C raises the words' output logits
    jacobian_persona       +C moves toward the POS persona
    jacobian_persona_topk  +C moves toward the POS persona's evoked vocabulary
    jacobian_persona_soft  +C moves toward the POS persona's next-token dist
    jacobian_persona_pinv  +C moves toward the POS persona (tangent transport)
    random                 norm-matched control, no meaning
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from steering_lite.config import SteeringConfig, register, register_config

ε = 1e-8


# --- configs -----------------------------------------------------------------
# One config per method name so saved vectors deserialize with provenance.
# All share the same knobs: apply_mode (delivery) + apply_span (tail width).

@register_config
@dataclass
class JacobianWordC(SteeringConfig):
    method: str = "jacobian_word"
    normalize: bool = True
    apply_mode: str = "add"
    apply_span: int = 1


@register_config
@dataclass
class JacobianPersonaC(SteeringConfig):
    method: str = "jacobian_persona"
    normalize: bool = True
    apply_mode: str = "add"
    apply_span: int = 1


@register_config
@dataclass
class JacobianPersonaTopkC(SteeringConfig):
    method: str = "jacobian_persona_topk"
    normalize: bool = True
    apply_mode: str = "add"
    apply_span: int = 1


@register_config
@dataclass
class JacobianPersonaSoftC(SteeringConfig):
    method: str = "jacobian_persona_soft"
    normalize: bool = True
    apply_mode: str = "add"
    apply_span: int = 1


@register_config
@dataclass
class JacobianPersonaPinvC(SteeringConfig):
    method: str = "jacobian_persona_pinv"
    normalize: bool = True
    apply_mode: str = "add"
    apply_span: int = 1


@register_config
@dataclass
class RandomC(SteeringConfig):
    method: str = "random"
    normalize: bool = True
    apply_mode: str = "add"
    apply_span: int = 1


# --- delivery modes ----------------------------------------------------------

def _v_sum(stacked: dict[str, Tensor], y: Tensor) -> Tensor:
    """The per-layer direction v, summed over the k-stack, on y's device/dtype."""
    return stacked["v"].to(y).sum(dim=0)  # [d]


def apply_add(mod, x, y, shared, stacked, cfg) -> Tensor:
    """y += coeff * v at ALL positions (the verified default; same delivery as
    steering-lite's mean_diff, so calibrated coeffs are comparable)."""
    v = _v_sum(stacked, y)
    return y + cfg.coeff * v


def apply_add_last(mod, x, y, shared, stacked, cfg) -> Tensor:
    """y[:, -k:] += coeff * v -- nudge only the last k positions (decision
    region). k = cfg.apply_span; k >= s degenerates to apply_add."""
    v = _v_sum(stacked, y)
    k = cfg.apply_span
    head = y[:, :-k, :]
    tail = y[:, -k:, :] + cfg.coeff * v
    return torch.cat([head, tail], dim=1)


def apply_clamp(mod, x, y, shared, stacked, cfg) -> Tensor:
    """Set y's component along v_hat to coeff at ALL positions:
    y += (coeff - <y, v_hat>) * v_hat. Unlike apply_add the perturbation stays
    bounded however long generation runs: each decode step re-targets the same
    component value instead of pushing again on top of the previous push (the
    compounding that degenerates high-|C| adds via the KV cache). coeff=0 is
    directional ablation (Arditi et al. 2024); -C reverses the component."""
    v = _v_sum(stacked, y)
    v_hat = v / (v.norm() + ε)
    comp = torch.einsum("bsd,d->bs", y, v_hat).unsqueeze(-1)   # [b, s, 1]
    return y + (cfg.coeff - comp) * v_hat


def apply_replace_last(mod, x, y, shared, stacked, cfg) -> Tensor:
    """Overwrite the last k positions with the concept direction at each
    position's original magnitude: energy from y, direction from v, strength
    from coeff. A "virtual token" injection that keeps the [b, s, d] shape
    (true sequence insertion would break RoPE / attention mask / KV cache)."""
    v = _v_sum(stacked, y)
    v_unit = v / (v.norm() + ε)
    k = cfg.apply_span
    tail = y[:, -k:, :]                           # [b, k, d]
    energy = tail.norm(dim=-1, keepdim=True)      # [b, k, 1]
    new_tail = energy * (cfg.coeff * v_unit)
    head = y[:, :-k, :]
    return torch.cat([head, new_tail], dim=1)


APPLY_REGISTRY: dict[str, Callable[..., Tensor]] = {
    "add": apply_add,
    "clamp": apply_clamp,
    "add_last": apply_add_last,
    "replace_last": apply_replace_last,
}


def apply_dispatch(mod, x, y, shared, stacked, cfg) -> Tensor:
    fn = APPLY_REGISTRY.get(cfg.apply_mode)
    if fn is None:
        raise KeyError(
            f"unknown apply_mode={cfg.apply_mode!r}; registered: {list(APPLY_REGISTRY)}")
    return fn(mod, x, y, shared, stacked, cfg)


# --- method registration -----------------------------------------------------

def _extract_stub(pos_acts, neg_acts, cfg):
    raise NotImplementedError(
        "jsteer methods are extracted via Jacobian.{word,persona,persona_topk}_vector "
        "(jacobian.py) or vjp.py, not steering_lite.train -- they need gradients "
        "that train's no_grad path can't provide.")


def _register_method(method_name: str) -> None:
    @register
    class _M:  # noqa: N801 -- registry keys on .name, class name irrelevant
        name = method_name
        extract = staticmethod(_extract_stub)
        apply = staticmethod(apply_dispatch)


for _name in ("jacobian_word", "jacobian_persona", "jacobian_persona_topk",
              "jacobian_persona_soft", "jacobian_persona_pinv", "random"):
    _register_method(_name)
