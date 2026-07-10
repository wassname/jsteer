"""Shared demo display: steer, generate through the chat template, show the
lens readout + the raw generation per strength C. (Claude)

Used by all the notebooks so they render steering the same way. The chat
template matters: these models are trained on user/assistant turns (and
run-524's verified vectors were extracted on that format), and enable_thinking
opens Qwen3's <think> block. We print the generation RAW (skip_special_tokens
=False): the model's own <think>/</think> and <|im_end|> are visible so the
output is debuggable and nothing is parsed or reconstructed.
"""
from __future__ import annotations

import dataclasses

import torch
from loguru import logger
from steering_lite import Vector

from .jacobian import Jacobian


def chat_input(tok, user_msg: str, *, enable_thinking: bool = True) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        add_generation_prompt=True, tokenize=False, enable_thinking=enable_thinking)


def _cthulhu_say(text: str) -> str:
    """The j-space readout in a mini cowsay bubble -- Cthulhu speaks the tokens
    the steered residual points to. Cosmetic; the tokens are the payload."""
    n = len(text) + 2
    return ("\n".join([" " + "_" * n, f"< {text} >", " " + "-" * n,
                       "   \\", "    ^(;,;)^"]))


@torch.no_grad()
def show_steer(jac: Jacobian, model, tok, vec, user_msg: str, *,
               Cs=(-6, 0, 6), layer: int | None = None, k: int = 6,
               max_new_tokens: int = 512, seed: int = 0,
               apply_mode: str | None = None, apply_span: int = 1) -> None:
    """One block per C: lens readout at `layer`, then the raw generation, all
    under steering. Uses the model's own generation_config sampling; `seed`
    fixes it so the C blocks are comparable. `layer` defaults to the top fitted
    layer. max_new_tokens defaults to 512 so Qwen3's <think> block can close;
    256 truncates mid-reasoning.

    Extraction is decoupled from DELIVERY (see applies.py): pass `apply_mode`
    (add | clamp | add_last | replace_last) to swap how v hits the residual
    without re-extracting; `apply_span` is the trailing-position width for the
    last/replace modes. Coefficient units differ by mode (clamp sets a component
    VALUE, add scales a direction), so each mode wants its own Cs."""
    if apply_mode is not None:
        vec = Vector(dataclasses.replace(vec.cfg, apply_mode=apply_mode,
                                         apply_span=apply_span), vec.shared, vec.stacked)
    layer = jac.layers[-1] if layer is None else layer
    prompt = chat_input(tok, user_msg)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    name = getattr(model.config, "name_or_path", "model").split("/")[-1]
    # header carries name/method/delivery/prompt once; per-C blocks only vary in C.
    # steering-lite's own configs (e.g. MeanDiffC baseline) have no apply_mode, so
    # only jsteer vectors show a delivery tag.
    delivery = getattr(vec.cfg, "apply_mode", None)
    tag = f" · delivery={delivery}" if delivery else ""
    rule = "=" * 72
    logger.info(f"\n\n{rule}\n{name} · method={vec.cfg.method}{tag}"
                f"\nprompt: {user_msg!r}\n{rule}")
    # SHOULD: C=0 is the baseline; +C tilts the lens tokens and tone toward the
    # concept, -C away; all stay coherent (gibberish = coeff too large).
    for C in Cs:
        torch.manual_seed(seed)
        with vec(model, C=C):
            jtop = jac.lens_topk(model, tok, prompt, layer=layer, k=k)
            out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 pad_token_id=tok.eos_token_id)
        # raw decode WITH special tokens: real <think>/</think>, <|im_end|> visible,
        # nothing parsed or re-wrapped -- debuggable exactly as the model emitted it
        gen = tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=False)
        readout = " · ".join(t.strip() for t, _ in jtop)
        block = [f"\n--- C={C:+g} " + "-" * 60,
                 f"  lens @L{layer}:", _cthulhu_say(readout), gen]
        logger.info("\n".join(block) + "\n")
