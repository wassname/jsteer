"""U4 step 3/3: full 4B Jacobian fit on run-524's substrate + cache loop-close.

(Claude) The expensive one: 512 prompts x ceil(2560/dim_batch) backwards.
checkpoint_path makes it resumable, so a kill/OOM loses at most one prompt.
After fitting, the cached word vector must match BOTH the step-2 jsteer VJP
vector and the step-1 run-524 reference (linearity: mean_p(J_p)^T w =
mean_p(J_p^T w); fp16 cache storage is the only gap). GATE: cos > 0.999.

This closes the loop on the verified 3/5 moral-foundations result: the library
artifact (artifacts/qwen3-4b-authority.jac) provably contains the verified
steering vector.
"""
import json
import time
from pathlib import Path

import torch
from loguru import logger
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer

from jsteer import Jacobian

ART = Path(__file__).resolve().parent.parent / "artifacts"
meta = json.loads((ART / "u4_prompts.json").read_text())
ref524 = torch.load(ART / "u4_ref_524.pt")
vjp = torch.load(ART / "u4_vjp.pt")

tok = AutoTokenizer.from_pretrained(meta["model"])
model = AutoModelForCausalLM.from_pretrained(
    meta["model"], torch_dtype=torch.bfloat16).to("cuda").eval()

t0 = time.time()
jac = Jacobian.fit(model, tok, meta["prompts"], layers=meta["layers"],
                   dim_batch=16, max_seq_len=384,
                   checkpoint_path=str(ART / "qwen3-4b-authority.ckpt"))
logger.info(f"fit wall-time: {(time.time() - t0) / 3600:.2f} h")
jac.save(str(ART / "qwen3-4b-authority.jac"))
logger.info(f"saved cache: {(ART / 'qwen3-4b-authority.jac').stat().st_size / 1e9:.2f} GB")

v = jac.word_vector(model, tok, meta["words"], layers=meta["layers"])
rows = []
for l in meta["layers"]:
    a = v.stacked[l]["v"].squeeze(0).float()
    rows.append((l,
                 torch.nn.functional.cosine_similarity(a, vjp[str(l)].float(), dim=0).item(),
                 torch.nn.functional.cosine_similarity(a, ref524[str(l)].float(), dim=0).item()))
table = tabulate(rows, headers=["layer", "cos(cache, jsteer_vjp)", "cos(cache, ref524)"],
                 floatfmt="+.6f")
min_cos = min(min(r[1], r[2]) for r in rows)
verdict = "PASS" if min_cos > 0.999 else "FAIL"
out = (f"U4 step 3: cached-4B word vector vs step-2 VJP and run-524 reference\n"
       f"model={meta['model']} prompts={len(meta['prompts'])} words={meta['words']} "
       f"dim_batch=16 fp16-cache\n\n{table}\n\n"
       f"min cos = {min_cos:+.6f}   GATE (>0.999): {verdict}\n")
(ART / "u4_loopclose.txt").write_text(out)
print(out)
if verdict == "FAIL":
    raise SystemExit("U4 step 3 FAILED: cache wiring bug, root-cause before shipping the artifact")
