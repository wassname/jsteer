"""U4 step 1/3: regenerate run-524's verified jacobian_word vector.

(Claude) Run this under j-steer-dev's venv, NOT jsteer's:

    cd ../j-steer-dev && uv run python ../jsteer/scripts/u4_step1_ref524.py

There `import jsteer` resolves to the OLD experiment package (j-steer-dev/src),
whose extract_word_pullback produced the verified 3/5 result. Run 524 never
persisted its vector tensors (only eval JSONs), but the extraction is
deterministic (seed-0 prompts, greedy, no sampling), so re-running it IS the
reference. Also dumps the 512 substrate prompts so steps 2/3 consume this one
artifact instead of regenerating them (no drift axis).

Exact run-524 parameters: Qwen/Qwen3-4B, persona=authority, n_pairs=256,
seed=0, layers "mid" (7..27 of 36), words authority/obey/command/hierarchy,
batch_size=4, max_length=384, cotangent_scope=source_scope=all_valid (defaults).
"""
import json
from pathlib import Path

import torch
from loguru import logger
from steering_lite.data import PERSONA_REGISTRY, make_persona_pairs
from transformers import AutoModelForCausalLM, AutoTokenizer

from jsteer.pullback import extract_word_pullback  # OLD package (j-steer-dev/src)

ART = Path(__file__).resolve().parent.parent / "artifacts"
MODEL = "Qwen/Qwen3-4B"
WORDS = ["authority", "obey", "command", "hierarchy"]

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16).to("cuda").eval()

n = model.config.num_hidden_layers
assert n == 36, f"expected Qwen3-4B with 36 layers, got {n}"
layers = tuple(range(max(2, int(n * 0.2)), min(n - 2, int(n * 0.8))))  # run_sweep "mid" -> 7..27

persona_pairs, template = PERSONA_REGISTRY["authority"]
pos, neg = make_persona_pairs(tok, n_pairs=256, thinking=True,
                              persona_pairs=persona_pairs, template=template, seed=0)
prompts = pos + neg  # run_sweep feeds pos+neg as the linearization substrate
(ART / "u4_prompts.json").write_text(json.dumps(
    {"model": MODEL, "layers": list(layers), "words": WORDS, "prompts": prompts}))
logger.info(f"dumped {len(prompts)} prompts, layers={layers}")
logger.info("SHOULD: chat-templated authority-persona prompt with <think>. ELSE template drift.\n"
            f"--- PROMPT[0] (full, special tokens) ---\n{prompts[0]}")

vec = extract_word_pullback(model, tok, prompts, layers, WORDS,
                            batch_size=4, max_length=384)["jacobian_word"]
ref = {str(l): vec.stacked[l]["v"].squeeze(0).float().cpu() for l in layers}
torch.save(ref, ART / "u4_ref_524.pt")
logger.info(f"saved {ART / 'u4_ref_524.pt'}  "
            f"norms={[round(ref[str(l)].norm().item(), 3) for l in layers][:5]}... "
            "SHOULD: all 1.0 (unit vectors). ELSE _to_vector changed.")
