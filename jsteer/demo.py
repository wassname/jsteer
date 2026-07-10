"""Shared demo display: steer, generate through the chat template, show the
lens readout + <think> trace + answer per strength C. (Claude)

Used by all the notebooks so they render steering the same way. The chat
template matters: these models are trained on user/assistant turns (and
run-524's verified vectors were extracted on that format), and enable_thinking
opens Qwen3's <think> block.
"""
from __future__ import annotations

import torch
from loguru import logger

from .jacobian import Jacobian


def chat_input(tok, user_msg: str, *, enable_thinking: bool = True) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        add_generation_prompt=True, tokenize=False, enable_thinking=enable_thinking)


def split_think(text: str) -> tuple[str, str]:
    """Qwen3 emits `<think>reasoning</think>answer` -> (thoughts, answer)."""
    if "</think>" in text:
        thoughts, _, answer = text.partition("</think>")
        return thoughts.replace("<think>", "").strip(), answer.strip()
    return "", text.strip()


@torch.no_grad()
def show_steer(jac: Jacobian, model, tok, vec, user_msg: str, *,
               Cs=(-6, 0, 6), layer: int | None = None, k: int = 6,
               max_new_tokens: int = 256, seed: int = 0) -> None:
    """One block per C: lens readout at `layer`, the <think> trace, the answer,
    all under steering. Uses the model's own generation_config sampling; `seed`
    fixes it so the C blocks are comparable. `layer` defaults to the top fitted
    layer."""
    layer = jac.layers[-1] if layer is None else layer
    prompt = chat_input(tok, user_msg)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    name = getattr(model.config, "name_or_path", "model").split("/")[-1]
    logger.info(f"{name} · {vec.cfg.method} · {user_msg!r}")
    # SHOULD: C=0 is the baseline; +C tilts the lens tokens and tone toward the
    # concept, -C away; all stay coherent (gibberish = coeff too large).
    for C in Cs:
        torch.manual_seed(seed)
        with vec(model, C=C):
            jtop = jac.lens_topk(model, tok, prompt, layer=layer, k=k)
            out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 pad_token_id=tok.eos_token_id)
        thoughts, answer = split_think(
            tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True))
        block = [f"{name} · steer→{vec.cfg.method} · C={C:+g}",
                 f"  lens @L{layer}:  " + " · ".join(t.strip() for t, _ in jtop)]
        if thoughts:
            block.append(f"  <think> {thoughts} </think>")
        block.append(f"  → {answer}")
        logger.info("\n".join(block) + "\n")
