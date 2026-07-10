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


def split_think(text: str) -> tuple[str, str, bool]:
    """Qwen3 emits `<think>reasoning</think>answer`. Returns (thoughts, answer,
    closed). closed=False means generation hit the token limit still inside
    <think>: `answer` is empty and `thoughts` holds the truncated reasoning.
    Without this flag, unclosed reasoning silently masquerades as the answer."""
    body = text.replace("<think>", "").strip()
    if "</think>" in body:
        thoughts, _, answer = body.partition("</think>")
        return thoughts.strip(), answer.strip(), True
    return body, "", False


@torch.no_grad()
def show_steer(jac: Jacobian, model, tok, vec, user_msg: str, *,
               Cs=(-6, 0, 6), layer: int | None = None, k: int = 6,
               max_new_tokens: int = 512, seed: int = 0) -> None:
    """One block per C: lens readout at `layer`, the <think> trace, the answer,
    all under steering. Uses the model's own generation_config sampling; `seed`
    fixes it so the C blocks are comparable. `layer` defaults to the top fitted
    layer. max_new_tokens defaults to 512 so Qwen3's <think> block can close;
    256 truncates mid-reasoning, leaving no answer."""
    layer = jac.layers[-1] if layer is None else layer
    prompt = chat_input(tok, user_msg)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    name = getattr(model.config, "name_or_path", "model").split("/")[-1]
    # header carries name/method/prompt once; per-C blocks below only vary in C
    rule = "=" * 72
    logger.info(f"\n\n{rule}\n{name} · method={vec.cfg.method}\nprompt: {user_msg!r}\n{rule}")
    # SHOULD: C=0 is the baseline; +C tilts the lens tokens and tone toward the
    # concept, -C away; all stay coherent (gibberish = coeff too large).
    for C in Cs:
        torch.manual_seed(seed)
        with vec(model, C=C):
            jtop = jac.lens_topk(model, tok, prompt, layer=layer, k=k)
            out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 pad_token_id=tok.eos_token_id)
        thoughts, answer, closed = split_think(
            tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True))
        block = [f"\n--- C={C:+g} " + "-" * 60,
                 f"  lens @L{layer}:  " + " · ".join(t.strip() for t, _ in jtop)]
        if thoughts:
            tag = "<think>" if closed else "<think> (UNCLOSED: hit max_new_tokens)"
            block.append(f"  {tag}\n  {thoughts}\n  </think>")
        if answer:
            block.append(f"  answer: {answer}")
        elif not closed:
            block.append("  answer: (none -- reasoning truncated; raise max_new_tokens)")
        logger.info("\n".join(block) + "\n")
