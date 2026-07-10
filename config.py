"""Repo-local paths and slug/cache conventions shared by scripts/ and notebooks.

NOT imported by the jsteer library (which stays path-agnostic so `pip install
jsteer` never needs a repo root). Fitting prompts are deliberately NOT hand-
rolled here: fit.py draws them from jlens's own corpus
(`jlens.examples.load_wikitext_prompts`) so the fitted lens is comparable to a
jlens fit rather than a forked substrate.
"""
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
DEVICE = "cuda"
DTYPE = torch.bfloat16


def slug(model_name: str) -> str:
    """'Qwen/Qwen3-0.6B' -> 'qwen3-0.6b': a filesystem-safe cache stem."""
    return model_name.split("/")[-1].lower()


def cache_path(model_name: str, suffix: str = "jac") -> Path:
    """Where the fitted Jacobian for `model_name` is cached."""
    return ART / f"{slug(model_name)}.{suffix}"
