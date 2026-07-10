"""Smoke test: fit a tiny Jacobian on Qwen3-0.6B, steer on the word "happy/joy".

(authored by Claude)

Touches the whole jsteer path end-to-end on a real model, cheaply:
  fit -> save -> load round-trip -> word_vector -> generate at C in {-8, 0, 8}.

Run:
    uv run python scripts/smoke.py 2>&1 | tee /tmp/claude-1000/jsteer_smoke.log

Read the prints: the FULL first fit prompt (with special tokens), the FULL
generation prompt as the model sees it, and each generation verbatim, with a
SHOULD line so a deviation is legible.
"""
from __future__ import annotations

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from jsteer import Jacobian

MODEL = "Qwen/Qwen3-0.6B"
CACHE = "artifacts/qwen3-0.6b-smoke.jac"
DEVICE = "cuda"
DTYPE = torch.bfloat16

# Claude: 8 english web-text-ish prompts, each padded past 17 tokens so jlens
# (skip_first=16, drop final) has >=1 valid source position per prompt.
PROMPTS = [
    "The weather this morning was cold and grey, so I made a large pot of coffee and sat by the window watching the rain fall.",
    "Scientists have long argued about whether the early universe expanded smoothly or in sudden bursts that left traces we can still measure today.",
    "My grandmother used to tell stories about growing up on a small farm, where every season brought a different kind of hard and honest work.",
    "The city council voted last night to repair the old stone bridge downtown, a project residents have been requesting for well over a decade.",
    "After months of training, the runners lined up at dawn, breath fogging in the cold air, waiting nervously for the starting gun to fire.",
    "Learning to cook well takes patience more than talent, a willingness to taste often, to fail a few times, and to pay attention to detail.",
    "The library on the corner smells of old paper and dust, and its quiet reading room has been my favourite place to think for many years.",
    "Software written in a hurry tends to accumulate small mistakes that hide quietly until, one ordinary afternoon, they surface all at once together.",
]

GEN_PROMPT = "I went to the park today and"


def _show_tokens(tok, text: str, label: str) -> None:
    ids = tok(text, add_special_tokens=True).input_ids
    logger.info(f"{label}: {len(ids)} tokens (add_special_tokens=True)")
    logger.info(f"{label} decoded-with-special:\n{tok.decode(ids)!r}")


def main() -> None:
    logger.info(f"loading {MODEL} ({DTYPE}) on {DEVICE}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(DEVICE).eval()

    # SHOULD: fit prompt shows a BOS-like sink token then english web text.
    # ELSE tokenizer/template drift (jlens force_bos should add an attention sink).
    _show_tokens(tok, PROMPTS[0], "FIT PROMPT[0]")

    logger.info("fitting Jacobian (layers=0.3..0.9 band, dim_batch=8)")
    jac = Jacobian.fit(model, tok, PROMPTS, layers=(0.3, 0.9),
                       dim_batch=8, max_seq_len=128)
    logger.info(f"fitted: {jac!r}  layers={jac.layers}")

    jac.save(CACHE)
    jac2 = Jacobian.load(CACHE)
    logger.info(f"load round-trip: {jac2!r}  layers={jac2.layers}")
    # SHOULD: reloaded layers identical to fitted. ELSE save/load wiring bug.
    assert jac2.layers == jac.layers, (jac2.layers, jac.layers)

    v = jac2.word_vector(model, tok, ["happy", "joy"])
    logger.info(f"word_vector layers={sorted(v.stacked)}")

    # SHOULD: generation prompt shows a sink token then "I went to the park today and".
    _show_tokens(tok, GEN_PROMPT, "GEN PROMPT")
    enc = tok(GEN_PROMPT, return_tensors="pt").to(DEVICE)

    for C in (-8, 0, 8):
        with v(model, C=C):
            out = model.generate(**enc, max_new_tokens=40, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
        logger.info(f"=== C={C:+d} generation ===\n{text!r}")

    logger.info(
        "SHOULD: C=+8 mentions happiness/joy more than C=0; C=-8 less or "
        "negative tone. ELSE steering wiring or sign issue. All three SHOULD "
        "stay coherent english; gibberish means the coeff is too large or the "
        "vector is malformed.")


if __name__ == "__main__":
    main()
