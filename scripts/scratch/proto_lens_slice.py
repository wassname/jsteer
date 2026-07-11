"""Prototype: use the REFERENCE jlens.vis machinery (compute_slice) for the
j-lens readout instead of our hand-picked lens_rank. (Claude)

compute_slice does the "complex logic over layers and tokens" wassname remembered:
auto-selects tracked tokens by a frequency-weighted 1/(rank+1) score over the whole
top-N grid, sweeps every fitted layer, appends the final layer as the J=I model row,
and returns full-vocab rank tensors. We pin the answer token, window to the last
position, and pull rank_tensor into a table + a rank-vs-layer plot -- reference logic,
our presentation, no fork.

    uv run python scripts/scratch/proto_lens_slice.py

SHOULD: ' Paris' rank collapses toward 0 (top) at the deepest layers and the final
(J=I / model) row; generic ' city' is low-rank mid-depth then climbs; the auto-tracked
set contains Paris/city without us hand-listing every token. ELSE compute_slice wiring
or token pinning is wrong.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import config  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402
from tabulate import tabulate  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from jlens.hf import from_hf  # noqa: E402
from jlens.vis import build_page, compute_slice, notebook_iframe  # noqa: E402
from jsteer import Jacobian  # noqa: E402

MODEL = "Qwen/Qwen3.5-4B"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
jac = Jacobian.from_pretrained(config.LENS_REPO, filename=config.hub_lens_file(MODEL),
                               revision=config.LENS_REVISION)

PROMPT = "The Eiffel Tower is located in the city of"
lm = from_hf(model, tok)
pin = {tok(" Paris", add_special_tokens=False).input_ids[0],
       tok(" city", add_special_tokens=False).input_ids[0]}
sd = compute_slice(lm, jac.lens, PROMPT, top_n=10, max_tracked=6,
                   pinned_token_ids=pin, last_n_tokens=1, mask_display=True)

logger.info(f"layers={sd.layers}")
logger.info(f"tracked={[sd.vocab_fragment[t] for t in sd.tracked_token_ids]}")

# rank of each tracked token across layers at the (single) last position
ranks = sd.rank_tensor[-1]  # [n_layers, n_tracked]
labels = [sd.vocab_fragment[t] for t in sd.tracked_token_ids]
# table: show a few layers to stay narrow (first, mid, last-fitted, model row = last)
show = [0, len(sd.layers) // 2, len(sd.layers) - 1]
rows = [[labels[j]] + [int(ranks[i, j]) for i in show] for j in range(len(labels))]
hdr = ["token"] + [f"L{sd.layers[i]}" + ("(model)" if i == len(sd.layers) - 1 else "")
                   for i in show]
logger.info("\n" + tabulate(rows, headers=hdr, tablefmt="github"))

# plot: rank+1 (log, inverted so top-of-plot = rank 0 = the model's next token)
fig, ax = plt.subplots(figsize=(6, 3.2))
for j, lab in enumerate(labels):
    ax.plot(sd.layers, ranks[:, j] + 1, marker="o", ms=3, label=repr(lab))
ax.set_yscale("log")
ax.invert_yaxis()
ax.axvline(sd.layers[-1], color="0.85", lw=0.8, zorder=0)  # J=I model row
ax.set_xlabel("layer (rightmost = final, J=I = model)")
ax.set_ylabel("rank+1 (log; 1 = top token)")
ax.set_title("lens rank of tracked tokens vs depth")
ax.legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig("/tmp/claude-1000/proto_lens_slice.png", dpi=110)
logger.info("wrote /tmp/claude-1000/proto_lens_slice.png")

# and confirm the reference's own HTML page builds without error (heatmap view)
page, w, h = build_page(sd, PROMPT, title="lens slice", description="proto")
_ = notebook_iframe(page)
logger.info(f"build_page OK: {w}x{h} grid, page {len(page)} chars; notebook_iframe OK")
