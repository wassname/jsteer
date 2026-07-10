"""Fit and cache any HF causal LM's Jacobian for the notebooks and README. (Claude)

Pass `--model`; the cache lands at `config.cache_path(model)` (e.g.
`artifacts/qwen3-0.6b.jac`). Prompts come from jlens's own WikiText-103 corpus
(`load_wikitext_prompts`), not a hand-rolled set, so the fitted lens is
comparable to a jlens fit rather than a forked substrate. jlens guidance: ~100
prompts is usable, the paper uses 1000; 128 is a cheap default. Idempotent:
re-running loads the existing cache instead of refitting (Jacobian.fit_cached).

    uv run python scripts/fit.py --model Qwen/Qwen3-0.6B
    uv run python scripts/fit.py --model Qwen/Qwen3-4B --dim-batch 16
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from jlens.examples import load_wikitext_prompts
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root for config
import config  # noqa: E402
from jsteer import Jacobian  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--n-prompts", type=int, default=128)
    p.add_argument("--dim-batch", type=int, default=8, help="d_model dims per backward batch (memory knob)")
    p.add_argument("--layers", type=float, nargs=2, default=(0.3, 0.9),
                   metavar=("LO", "HI"), help="fractional layer band to fit")
    p.add_argument("--max-seq-len", type=int, default=128)
    args = p.parse_args()

    out = config.cache_path(args.model)
    logger.info(f"loading {args.model} ({config.DTYPE}) on {config.DEVICE}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=config.DTYPE).to(config.DEVICE).eval()

    logger.info(f"fit-or-load {out} (layers={tuple(args.layers)}, "
                f"dim_batch={args.dim_batch}, n_prompts={args.n_prompts} WikiText)")
    t0 = time.monotonic()
    jac = Jacobian.fit_cached(model, tok, lambda: load_wikitext_prompts(args.n_prompts), out,
                              layers=tuple(args.layers), dim_batch=args.dim_batch,
                              max_seq_len=args.max_seq_len,
                              checkpoint_path=str(config.cache_path(args.model, "ckpt")))
    logger.info(f"{jac!r} -> {out}  ({(time.monotonic() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
