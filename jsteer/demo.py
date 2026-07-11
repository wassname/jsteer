"""Shared demo display: steer, generate through the chat template, show the
lens readout + the raw generation per strength C. (Claude)

Used by all the notebooks so they render steering the same way. The chat
template matters: these models are trained on user/assistant turns (and
run-524's verified vectors were extracted on that format), and enable_thinking
opens Qwen3's <think> block. We print the generation RAW (skip_special_tokens
=False): the model's own <think>/</think> and <|im_end|> are visible so the
output is debuggable and nothing is parsed or reconstructed.
"""
from __future__ import annotations

import dataclasses

import torch
from jlens.vis import _meaningful_token_mask
from loguru import logger
from steering_lite import Vector

from .jacobian import Jacobian


def chat_input(tok, user_msg: str, *, enable_thinking: bool = True) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        add_generation_prompt=True, tokenize=False, enable_thinking=enable_thinking)


def _cthulhu_say(text: str) -> str:
    """A mini cowsay bubble -- Cthulhu speaks the tokens. Cosmetic; the tokens are
    the payload."""
    n = len(text) + 2
    return ("\n".join([" " + "_" * n, f"< {text} >", " " + "-" * n,
                       "   \\", "    ^(;,;)^"]))


# think-then-answer rubric read: the demo's one-number sanity signal that steering
# moved the target axis. Same mechanism as moral-maps guided.py (let the model think,
# then read the logprobs at a forced answer slot), reduced to a single scalar. (Claude)

# Coherence = repetition of the think trace, NOT a forced-object gate. Every steer
# breakdown we observed is a REPETITION loop ("happy and happy...", wedding-jewelry,
# "favorite books..."), so 1 - distinct-3 is the natural, simple coherence signal (and
# it catches long-generation degeneration that a short forced JSON object survives).
# Threshold from the empirical gap in scripts/scratch/rep_metric_check.py over 40+ real
# generations: coherent reasoning scores rep3 < ~0.3, degenerate loops > ~0.6.
REP_COHERENT_MAX = 0.35


def _rep_frac(text: str, n: int = 3) -> float:
    """1 - distinct-n over whitespace tokens: 0 = all n-grams unique (fluent), ->1 as
    the text collapses into a repeated loop (steer degeneration)."""
    toks = text.split()
    if len(toks) < n + 1:
        return 0.0
    ngrams = list(zip(*[toks[i:] for i in range(n)]))
    return 1.0 - len(set(ngrams)) / len(ngrams)


# a readout = (format suffix appended to the question, forced slot after </think>, the
# answer tokens to read logprobs over, and the scalar value each maps to). DIGIT is the
# 0-9 rubric; YESNO reads P(YES) for a binary dilemma (a real decision, not a self-rating
# the model refuses to give). expected = sum_i value_i * softmax(logit over answer tokens).
DIGIT = dict(fmt=' Think it over, then answer with JSON {"ans": N} where N is a single'
                 ' digit from 0 (least) to 9 (most).',
             prefix='</think>\n{"ans": ',
             tokens=[str(d) for d in range(10)], values=list(range(10)))
YESNO = dict(fmt=' Think it over, then give your final answer as one word, YES or NO.',
             prefix='</think>\nFinal answer: ',
             tokens=[' NO', ' YES'], values=[0.0, 1.0])   # expected = P(YES)


@torch.no_grad()
def rubric_score(model, tok, rubric: str, *, max_new_tokens: int, seed: int,
                 do_sample: bool = False, temperature: float = 0.7,
                 readout: dict = DIGIT) -> tuple[float, float]:
    """Ask `rubric`, let the model think, then force `readout['prefix']` and read the
    logprob-weighted answer. Returns (expected, rep) where:

    expected = sum_i value_i * softmax(logit_i over readout['tokens']) at the forced slot
    -- a continuous scalar from single-token logprobs. DIGIT -> expected 0-9 rubric digit;
    YESNO -> P(YES) for a binary dilemma.
    rep = 1 - distinct-3 of the think trace -- the coherence signal. Low (~0.05) while the
    model reasons fluently, ->1 when steering degenerates it into a repeat loop. We measure
    coherence on the long think trace (which degenerates under steering), not on the short
    forced answer (which stays scorable well past the breakdown)."""
    prompt = chat_input(tok, rubric + readout["fmt"])
    enc = tok(prompt, return_tensors="pt").to(model.device)
    torch.manual_seed(seed)
    # this model ships no generation_config, so generate() is greedy by default: seeds
    # only matter (distinct think traces) when do_sample=True -- which is what makes the
    # coherence_sweep's multi-seed BMA average over anything.
    gen_kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id,
                  do_sample=do_sample)
    if do_sample:
        gen_kw["temperature"] = temperature
    out = model.generate(**enc, **gen_kw)
    think = tok.decode(out[0][enc.input_ids.shape[1]:],
                       skip_special_tokens=False).split("</think>")[0]
    forced = prompt + think + readout["prefix"]              # our own deterministic slot
    fenc = tok(forced, return_tensors="pt").to(model.device)
    logits = model(**fenc).logits[0, -1].float()
    ids = torch.tensor([tok(t, add_special_tokens=False).input_ids[0]
                        for t in readout["tokens"]], device=logits.device)
    vals = torch.tensor(readout["values"], device=logits.device, dtype=torch.float)
    expected = float((logits[ids].softmax(0) * vals).sum())
    return expected, _rep_frac(think)


@torch.no_grad()
def coherence_sweep(model, tok, vec, rubric: str, *, step: float = 0.1,
                    max_steps: int = 15, n_samples: int = 3, readout: dict = DIGIT,
                    temperature: float = 0.7, max_new_tokens: int = 512) -> list[dict]:
    """Walk C outward from 0 in +/- directions, scoring the rubric each step, and STOP a
    direction the step AFTER the think trace degenerates (mean rep >= REP_COHERENT_MAX,
    i.e. it collapses into a repeat loop). Maps the coherent dose-response of the steered
    axis without hand-picking Cs. Returns rows sorted by C:
    {"C","ans","ans_std","rep","coherent"}. Each C is averaged over `n_samples` think
    traces (seeds 0..n-1) to tame single-sample noise -- a lightweight stand-in for
    guided.py's Bayesian model averaging; ans_std is the spread. rep = 1 - distinct-3 of
    the think trace catches the actual failure mode (repetition), unlike a short forced
    object that stays scorable past the breakdown."""
    def score(C):
        with vec(model, C=C):
            pairs = [rubric_score(model, tok, rubric, max_new_tokens=max_new_tokens, seed=s,
                                  do_sample=n_samples > 1, temperature=temperature,
                                  readout=readout)
                     for s in range(n_samples)]
        anss = torch.tensor([e for e, _ in pairs])
        rep = float(torch.tensor([r for _, r in pairs]).mean())
        return {"C": round(float(C), 3), "ans": float(anss.mean()),
                "ans_std": float(anss.std(unbiased=False)), "rep": rep,
                "coherent": rep < REP_COHERENT_MAX}
    rows = [score(0.0)]
    for d in (step, -step):                      # outward each way; keep the 1st incoherent point
        C = d
        for _ in range(max_steps):
            r = score(C)
            rows.append(r)
            if not r["coherent"]:
                break
            C += d
    rows.sort(key=lambda r: r["C"])
    return rows


def plot_sweep(rows: list[dict], *, title: str = "rubric ans vs C"):
    """ans vs C, points colored by think-trace repetition (rep = 1 - distinct-3);
    degenerate points (rep >= REP_COHERENT_MAX) also get a red edge. Low rep = fluent
    reasoning (bright), high rep = the steer has collapsed the trace into a repeat loop
    (dark + red edge). The ans curve carries the dose-response; rep marks where to stop
    trusting it. Colorbar is inverted (viridis_r) so brighter = more coherent."""
    import matplotlib.pyplot as plt
    Cs = [r["C"] for r in rows]
    ans = [r["ans"] for r in rows]
    rep = [r["rep"] for r in rows]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(Cs, ans, "-", color="0.8", lw=1, zorder=1)
    if all("ans_std" in r for r in rows):
        ax.errorbar(Cs, ans, yerr=[r["ans_std"] for r in rows], fmt="none",
                    ecolor="0.6", capsize=2, lw=1, zorder=1)
    sc = ax.scatter(Cs, ans, c=rep, cmap="viridis_r", vmin=0.0, vmax=1.0,
                    zorder=2, edgecolor=["0.2" if r["coherent"] else "red" for r in rows],
                    linewidth=1.2)
    ax.axvline(0, color="0.85", lw=0.8, zorder=0)
    ax.set_xlabel("steering coefficient C")
    ax.set_ylabel("rubric ans (0-9)")
    ax.set_ylim(-0.3, 9.3)
    ax.set_title(title)
    cbar = fig.colorbar(sc, ax=ax, label="think-trace repetition (1 - distinct-3)")
    cbar.ax.axhline(REP_COHERENT_MAX, color="red", lw=1)   # the degeneration cutoff
    fig.tight_layout()
    return fig


def lens_slice_ranks(slice_data):
    """From a jlens `compute_slice` SliceData: (labels, layers, ranks) where
    ranks[layer_idx, token_idx] is that tracked token's full-vocab rank at the last
    slice position (0 = the model's next token). The final layer is the model's own
    output (J=I), so its column is the ground-truth ranking the lens approximates."""
    labels = [slice_data.vocab_fragment[t] for t in slice_data.tracked_token_ids]
    return labels, slice_data.layers, slice_data.rank_tensor[-1]  # [n_layers, n_tracked]


def plot_lens_slice(slice_data, *, title: str = "lens rank vs depth"):
    """Rank of each AUTO-tracked token across every fitted layer, from jlens's
    `compute_slice` (the reference's frequency-weighted token selection + full-vocab
    ranks -- we render its output, not a reimplementation). Log y, inverted so the
    top of the plot is rank 0 (the model's next token): a token that resolves late
    (e.g. the answer) dives toward the top near the final layers; a generic token
    peaks mid-depth then falls away. The rightmost x is the final layer (J=I = the
    model), where the lines meet the model's true ranks."""
    import matplotlib.pyplot as plt
    # Noto CJK first (it also has Latin glyphs) so multilingual tokens like 巴黎 render
    # in the legend instead of tofu -- matplotlib picks ONE font from this list, not a
    # per-glyph fallback chain, so DejaVu-first would still tofu the CJK. (Claude)
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans", "sans-serif"]
    labels, layers, ranks = lens_slice_ranks(slice_data)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for j, lab in enumerate(labels):
        ax.plot(layers, ranks[:, j] + 1, marker="o", ms=3, label=repr(lab))
    ax.set_yscale("log")
    ax.invert_yaxis()                                    # rank 0 (top token) at the top
    ax.axvline(layers[-1], color="0.85", lw=0.8, zorder=0)   # final layer = model (J=I)
    ax.set_xlabel("layer  (rightmost = final layer, J=I = the model)")
    ax.set_ylabel("rank+1  (log; 1 = the model's next token)")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


@torch.no_grad()
def show_steer(jac: Jacobian, model, tok, vec, user_msg: str, *,
               Cs=(-6, 0, 6), max_new_tokens: int = 512, seed: int = 0,
               apply_mode: str | None = None, apply_span: int = 1,
               rubric: str | None = None) -> None:
    """Per C: the steer-promoted tokens in a cowsay bubble, then the raw generation,
    all under steering. Uses the model's own generation_config sampling; `seed` fixes
    it so the C blocks are comparable. max_new_tokens defaults to 512 so Qwen3's <think>
    block can close; 256 truncates mid-reasoning. The cowsay speaks the top of
    (steered - unsteered) next-token logits -- what THIS C pushes up, with the shared
    think-opener prior subtracted out (the old lens_topk-at-last-position surfaced only
    Okay/Here/The for every C; the calibrated cross-layer lens readout is compute_slice
    on a completion prompt). `jac` is unused here, kept for call-site stability.

    Extraction is decoupled from DELIVERY (see applies.py): pass `apply_mode`
    (add | clamp | add_last | replace_last) to swap how v hits the residual
    without re-extracting; `apply_span` is the trailing-position width for the
    last/replace modes. Coefficient units differ by mode (clamp sets a component
    VALUE, add scales a direction), so each mode wants its own Cs.

    Pass `rubric` (a 0-9 rating question about the steered axis) to add the
    quantitative readout: per C, the model thinks then answers a JSON object and we
    report the logprob-weighted expected digit plus a coherence gate (valid object,
    2+2==4). ans SHOULD rise with +C and fall with -C; flat means the steer isn't
    moving that axis; json=False means the steer broke the model (see rubric_score)."""
    if apply_mode is not None:
        vec = Vector(dataclasses.replace(vec.cfg, apply_mode=apply_mode,
                                         apply_span=apply_span), vec.shared, vec.stacked)
    prompt = chat_input(tok, user_msg)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    name = getattr(model.config, "name_or_path", "model").split("/")[-1]
    # header carries name/method/delivery/prompt once; per-C blocks only vary in C.
    # steering-lite's own configs (e.g. MeanDiffC baseline) have no apply_mode, so
    # only jsteer vectors show a delivery tag.
    delivery = getattr(vec.cfg, "apply_mode", None)
    tag = f" · delivery={delivery}" if delivery else ""
    rule = "=" * 72
    logger.info(f"\n\n{rule}\n{name} · method={vec.cfg.method}{tag}"
                f"\nprompt: {user_msg!r}\n{rule}")
    base = model(**enc).logits[0, -1].float()   # unsteered next-token logits; the steer's
                                                 # effect reads as the top of (steered-base)
    # SHOULD: C=0 is the baseline; +C tilts the promoted tokens and tone toward the
    # concept, -C away; all stay coherent (gibberish = coeff too large).
    for C in Cs:
        torch.manual_seed(seed)
        with vec(model, C=C):
            steered = model(**enc).logits[0, -1].float()
            out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 pad_token_id=tok.eos_token_id)
            ans = (rubric_score(model, tok, rubric, max_new_tokens=max_new_tokens,
                                seed=seed) if rubric is not None else None)
        # steer-promoted tokens: top of (steered - base), word-like only. The subtraction
        # cancels the shared "about to open <think>" prior (Okay/Here/The) so what the
        # cowsay speaks is what THIS C actually pushes up, not the reasoning boilerplate
        # the old lens_topk-at-last-position surfaced for every C. (Claude)
        if C == 0:
            readout = "(baseline, no steer)"
        else:
            wl = _meaningful_token_mask(tok, steered.shape[-1], steered.device)
            promoted = (steered - base).masked_fill(~wl, float("-inf")).topk(6)
            readout = " · ".join(tok.decode([i]).strip() for i in promoted.indices.tolist())
        # raw decode WITH special tokens: real <think>/</think>, <|im_end|> visible,
        # nothing parsed or re-wrapped -- debuggable exactly as the model emitted it
        gen = tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=False)
        block = [f"\n--- C={C:+g} " + "-" * 60, "  steer promotes:",
                 _cthulhu_say(readout), gen]
        if ans is not None:
            # SHOULD rise with +C, fall with -C; flat => steer not moving this axis.
            # rep>=0.35 => the think trace degenerated into a loop, distrust the number.
            e, rep = ans
            block.append(f"  rubric ans≈{e:.2f}/9  (rep={rep:.2f}"
                         f"{' DEGENERATE' if rep >= REP_COHERENT_MAX else ''})")
        logger.info("\n".join(block) + "\n")
