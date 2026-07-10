"""Shared demo display for the notebooks: steer, generate, show j-space + trace.

(Claude)

Why this exists once (not per-notebook): every demo answers the same question --
"what does strength C do to the model?" -- so they should show it the same way.

Three things matter for a faithful demo, and the raw-`tok(prompt)` path missed
the first two:

1. Chat template. These models are trained on `<|im_start|>user ... assistant`
   turns, and run-524's VERIFIED vectors were extracted on exactly that format
   (see artifacts/u4_prompts.json). A raw completion string is off-distribution.
   `apply_chat_template(..., enable_thinking=True)` also opens Qwen3's `<think>`
   block, which is what lets us show the reasoning separately from the answer.

2. The model's own sampling. Qwen3 ships `generation_config` with
   `do_sample=True, temperature=0.6, top_p=0.95, top_k=20`; forcing greedy
   (`do_sample=False`) is off-recipe and, in thinking mode, loops. So we DON'T
   override sampling -- `generate` reads the shipped config -- and we log what it
   resolved to.

3. j-space readout. `Jacobian.lens_topk` transports the residual at a layer into
   the final (vocab) basis and decodes it: literally "what the model is thinking
   at layer L". Run it UNDER steering and it shows how C bends that thought.

Layout is Tufte small-multiples: one identical block per C so the eye compares
straight down the column, C=0 as the "compared to what?" baseline.
"""
from __future__ import annotations

import torch
from loguru import logger

from .jacobian import Jacobian


def chat_input(tok, user_msg: str, *, enable_thinking: bool = True) -> str:
    """The user turn formatted as the model expects, ending at the point where
    the assistant (and, for Qwen3, its <think> block) begins."""
    return tok.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        add_generation_prompt=True, tokenize=False, enable_thinking=enable_thinking)


def split_think(text: str) -> tuple[str, str]:
    """Qwen3 emits `<think>reasoning</think>answer`. Returns (thoughts, answer);
    thoughts is "" for a non-thinking reply."""
    if "</think>" in text:
        thoughts, _, answer = text.partition("</think>")
        return thoughts.replace("<think>", "").strip(), answer.strip()
    return "", text.strip()


@torch.no_grad()
def show_steer(jac: Jacobian, model, tok, vec, user_msg: str, *,
               Cs=(-6, 0, 6), layer: int | None = None, k: int = 6,
               max_new_tokens: int = 256) -> None:
    """Print one block per C: j-space top-k at `layer`, the <think> trace, the
    answer -- all under steering at that C. `layer` defaults to the top fitted
    layer (closest to the readout). Uses the model's own generation_config
    sampling (no greedy override)."""
    layer = jac.layers[-1] if layer is None else layer
    prompt = chat_input(tok, user_msg)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    gc = model.generation_config
    name = getattr(model.config, "name_or_path", "model").split("/")[-1]

    logger.info(f"steer demo: {name} · {vec.cfg.method} · prompt={user_msg!r}")
    logger.info(f"sampling (from generation_config): do_sample={gc.do_sample} "
                f"temp={gc.temperature} top_p={gc.top_p} top_k={gc.top_k}")
    # SHOULD: C=0 block reads as a normal, coherent assistant answer (baseline).
    # +C should tilt the j-space tokens and the tone toward the concept, -C away.
    # ELSE steering is unwired or the sign is flipped. Any block turning to
    # gibberish means the coeff is too large for this vector, not a steering win.
    logger.info(f"SHOULD: C=0 is the baseline; +C tilts j-space@L{layer} + tone "
                f"toward the concept, -C away; all stay coherent.\n")

    for C in Cs:
        with vec(model, C=C):
            jtop = jac.lens_topk(model, tok, prompt, layer=layer, k=k)
            out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
        thoughts, answer = split_think(text)

        toks = " · ".join(t.strip() for t, _ in jtop)
        block = [f"{name} · steer→{vec.cfg.method} · C={C:+d}",
                 f"  j-space @L{layer}:  {toks}"]
        if thoughts:
            block.append(f"  <think> {thoughts} </think>")
        block.append(f"  → {answer}")
        logger.info("\n".join(block) + "\n")
