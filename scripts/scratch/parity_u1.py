"""U1 parity gate: cached-Jacobian pullback == direct VJP, per layer.

(authored by Claude)

The two paths are linear-identical:  mean_p(J_p)^T w == mean_p(J_p^T w).
Path A pulls the word cotangent through the CACHED pooled Jacobian.
Path B contracts the same cotangent inside per-prompt backward passes.
The only expected gap is fp16 storage in the cache, so per-layer cosine must
exceed 0.999. A failure is a WIRING bug (layer index, position mask, pooling),
not a threshold to tune.

Run:
    uv run python scripts/parity_u1.py 2>&1 | tee /tmp/claude-1000/jsteer_parity.log
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from loguru import logger
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer

from jsteer import Jacobian, word_vector_vjp

# Claude: repo root on path so `scripts.smoke` imports whether run as a file or -m.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.smoke import CACHE, DEVICE, DTYPE, MODEL, PROMPTS  # same inputs  # noqa: E402

WORDS = ["happy", "joy"]
OUT = "artifacts/parity_u1.txt"


def _unit_dir(vec, layer: int) -> torch.Tensor:
    """Pull the per-layer unit direction out of a steering_lite Vector."""
    return vec.stacked[layer]["v"].squeeze(0).float().cpu()  # [d]


def main() -> None:
    logger.info(f"loading {MODEL} ({DTYPE}) on {DEVICE}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(DEVICE).eval()

    jac = Jacobian.load(CACHE)
    layers = jac.layers  # exact int layers fitted by the smoke
    logger.info(f"cached layers={layers}")

    # Path A: cached pooled Jacobian pullback.
    vA = jac.word_vector(model, tok, WORDS)
    # Path B: direct per-prompt VJP over the SAME prompts / layers / skip_first / max_length.
    vB = word_vector_vjp(model, tok, PROMPTS, WORDS, layers=layers, max_length=128)

    rows = []
    for l in layers:
        a, b = _unit_dir(vA, l), _unit_dir(vB, l)
        cos = float(torch.dot(a, b) / (a.norm() * b.norm()))
        rows.append((l, cos, float(a.norm()), float(b.norm())))

    table = tabulate(rows, headers=["layer", "cos", "|vA|", "|vB|"],
                     tablefmt="pipe", floatfmt="+.6f")
    min_cos = min(r[1] for r in rows)
    gate = "PASS" if min_cos > 0.999 else "FAIL"
    report = f"{table}\n\nmin cos = {min_cos:+.6f}   GATE (>0.999): {gate}\n"

    logger.info("U1 parity table:\n" + report)
    with open(OUT, "w") as f:
        f.write("U1 parity: cached-Jacobian pullback vs direct VJP (word=happy/joy)\n")
        f.write(f"model={MODEL}  prompts={len(PROMPTS)}  layers={layers}\n\n")
        f.write(report)
    logger.info(f"wrote {OUT}")
    if gate == "FAIL":
        raise SystemExit(f"U1 parity FAILED: min cos={min_cos:.6f} <= 0.999 (wiring bug)")


if __name__ == "__main__":
    main()
