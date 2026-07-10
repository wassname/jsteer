"""U4 step 2/3: port check -- jsteer's word_vector_vjp vs the run-524 reference.

(Claude) Runs in jsteer's venv on the step-1 artifacts. Both sides are fp32
direct-VJP extractions of the same math on the same 512 prompts; the only
difference is code lineage (old experiment package vs this library) plus
batch-order fp noise. GATE: cos > 0.999 per layer. A failure is a PORT BUG
(position mask, pooling, layer indexing) -- debug, do not tune.
"""
import json
from pathlib import Path

import torch
from loguru import logger
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer

from jsteer import word_vector_vjp

ART = Path(__file__).resolve().parent.parent.parent / "artifacts"  # scripts/scratch/ -> repo root
meta = json.loads((ART / "u4_prompts.json").read_text())
ref = torch.load(ART / "u4_ref_524.pt")

tok = AutoTokenizer.from_pretrained(meta["model"])
model = AutoModelForCausalLM.from_pretrained(
    meta["model"], torch_dtype=torch.bfloat16).to("cuda").eval()

v = word_vector_vjp(model, tok, meta["prompts"], meta["words"],
                    layers=meta["layers"], batch_size=4, max_length=384)
torch.save({str(l): v.stacked[l]["v"].squeeze(0).float().cpu() for l in meta["layers"]},
           ART / "u4_vjp.pt")

rows = []
for l in meta["layers"]:
    a = v.stacked[l]["v"].squeeze(0).float()
    b = ref[str(l)].float()
    rows.append((l, torch.nn.functional.cosine_similarity(a, b, dim=0).item()))
table = tabulate(rows, headers=["layer", "cos(jsteer_vjp, ref524)"], floatfmt="+.6f")
min_cos = min(c for _, c in rows)
verdict = "PASS" if min_cos > 0.999 else "FAIL"
out = (f"U4 step 2: jsteer word_vector_vjp vs regenerated run-524 vector\n"
       f"model={meta['model']} prompts={len(meta['prompts'])} words={meta['words']}\n\n"
       f"{table}\n\nmin cos = {min_cos:+.6f}   GATE (>0.999): {verdict}\n")
(ART / "u4_step2_vjp_parity.txt").write_text(out)
print(out)
if verdict == "FAIL":
    raise SystemExit("U4 step 2 FAILED: port bug, do not run step 3 until root-caused")
