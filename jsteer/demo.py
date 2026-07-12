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
from steering_lite.eval.edge import (
    ANSWER_MASS_FRACTION,
    DIGIT,
    REP_LIMIT,
    YESNO,
    chat_prompt,
    five_coefficients,
    measure_readout,
    repetition_fraction,
    search_edge,
    summarize_anchors,
)
from tabulate import tabulate

from .jacobian import Jacobian


def chat_input(tok, user_msg: str, *, enable_thinking: bool = True) -> str:
    assert enable_thinking is True
    return chat_prompt(tok, user_msg)


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
REP_COHERENT_MAX = REP_LIMIT


# The calibrated edge is where EITHER off-target budget is first spent -- a dual gate,
# min(rep-margin, ans_mass-margin). Two distinct failure modes, whichever binds first:
#   rep       = the think trace degenerates into a repeat loop (fluency breakdown).
#   ans_mass  = the forced slot stops committing to an answer token (observed 'imers',
#               '信任', '(') so the P(YES) readout goes meaningless (readout-validity).
# For a YES/NO VERDICT the answer dies BEFORE repetition breaks (task 42: rep-only
# over-steers, ans_mass collapses to 0.0-0.4 at the rep edge, base 0.56, so the swings are
# read off dead answers). So ans_mass binds first for verdicts, rep first for a forced-format
# DIGIT -- the min auto-selects the right limiter per readout. ans_mass is base-anchored: the
# readout is valid iff ans_mass >= ANS_MASS_FRAC of the C=0 baseline (per prompt/model).
ANS_MASS_FRAC = ANSWER_MASS_FRACTION


def _rep_frac(text: str, n: int = 3) -> float:
    assert n == 3
    return repetition_fraction(text)


# a readout = (format suffix appended to the question, forced slot after </think>, the
# answer tokens to read logprobs over, and the scalar value each maps to). DIGIT is the
# 0-9 rubric; YESNO reads P(YES) for a binary dilemma (a real decision, not a self-rating
# the model refuses to give). expected = sum_i value_i * softmax(logit over answer tokens).
@torch.no_grad()
def rubric_score(model, tok, rubric: str, *, max_new_tokens: int, seed: int,
                 do_sample: bool = False, temperature: float = 0.7,
                 readout: dict = DIGIT) -> tuple[float, float, float]:
    """Ask `rubric`, let the model think, then force `readout['prefix']` and read the
    logprob-weighted answer. Returns (expected, rep, ans_mass) where:

    expected = sum_i value_i * softmax(logit_i over readout['tokens']) at the forced slot
    -- a continuous scalar from single-token logprobs. DIGIT -> expected 0-9 rubric digit;
    YESNO -> P(YES) for a binary dilemma.
    rep = 1 - distinct-3 of the think trace -- fluency: ~0 while the model reasons, ->1
    when steering degenerates it into a repeat loop (or a too-short stub).
    ans_mass = full-vocab softmax mass on the answer tokens -- did the model actually
    COMMIT to an answer? Under hard steering the forced slot's top token is often not an
    answer token ('imers', '信任', '('), so expected is meaningless; low ans_mass flags it.
    Coherence needs BOTH rep low and ans_mass high (see coherence_sweep)."""
    measurement = measure_readout(
        model, tok, rubric, max_new_tokens=max_new_tokens, readout=readout,
        seed=seed, do_sample=do_sample, temperature=temperature)
    return (
        measurement["answer"],
        measurement["repetition"],
        measurement["answer_mass"],
    )


@torch.no_grad()
def coherent_edge(model, tok, vec, probe: str, *, readout: dict = DIGIT, sign: int = 1,
                  max_C: float = 1e5, budget: int = 6, seed: int = 0,
                  max_new_tokens: int = 200) -> float:
    # max_C is a runaway safety bound only -- the edge should be set by an off-target budget,
    # never by this cap. If a method never breaks below max_C the real limiter is `budget`
    # (too few step-outs), which shows up as max_rep << REP_COHERENT_MAX (at_budget=False).
    """Find the strongest usable |C| in the `sign` direction via the Illinois method
    (bracket a good/broken pair, then modified false-position). The edge is a DUAL gate:
    margin m(C) = min(REP_COHERENT_MAX - rep, ans_mass - ANS_MASS_FRAC*base_am) > 0 while the
    trace is BOTH fluent (rep low) AND still committing to an answer (ans_mass held vs the C=0
    baseline); the edge is where either crosses 0 (whichever budget binds first). base_am is
    measured once at C=0. ~`budget` generations instead of a coarse fixed-step sweep. Returns
    the largest usable C (0.0 if the first step breaks)."""
    assert seed == 0
    result = search_edge(
        model, tok, vec, probe, readout=readout, sign=sign,
        max_new_tokens=max_new_tokens, budget=budget,
        max_coefficient=max_C)
    return result["coefficient"]


def steer_anchors(model, tok, vec, probe: str, *, readout: dict = DIGIT, budget: int = 6,
                  half: bool = True, **kw) -> list[float]:
    """Search both directions for the strongest coherent steer and return the anchor Cs to
    demo: [-C*, (-C*/2), 0, (+C*/2), +C*] -- the max coherent steer each way, baseline, and
    (if half) the half-strength midpoints for the dose-response. Every demo calls this
    instead of hand-picking Cs, so it always shows the strongest COHERENT effect."""
    cp = coherent_edge(model, tok, vec, probe, readout=readout, sign=1, budget=budget, **kw)
    cn = coherent_edge(model, tok, vec, probe, readout=readout, sign=-1, budget=budget, **kw)
    if half:
        return five_coefficients(cn, cp)
    return [cn, 0.0, cp]


@torch.no_grad()
def coherence_sweep(model, tok, vec, rubric: str, *, step: float = 0.1,
                    max_steps: int = 15, n_samples: int = 3, readout: dict = DIGIT,
                    temperature: float = 0.7, max_new_tokens: int = 512) -> list[dict]:
    """Walk C outward from 0 in +/- directions, scoring the rubric each step, and STOP a
    direction the step AFTER the point goes INCOHERENT. Coherent = the think trace is fluent
    (mean rep < REP_COHERENT_MAX) -- repetition only, the off-target budget we calibrate on.
    ans_mass is reported as a separate readout-validity flag (`readout_valid`: mean ans_mass
    >= ANS_MASS_FRAC of the C=0 baseline), NOT a coherence gate. Maps the coherent
    dose-response without hand-picking Cs. Returns rows sorted by C: {"C","ans","ans_std",
    "rep","ans_mass","coherent","readout_valid"}. Each C is averaged over `n_samples` think
    traces (seeds 0..n-1); ans_std is the spread."""
    def score(C):
        with vec(model, C=C):
            triples = [rubric_score(model, tok, rubric, max_new_tokens=max_new_tokens, seed=s,
                                    do_sample=n_samples > 1, temperature=temperature,
                                    readout=readout)
                       for s in range(n_samples)]
        anss = torch.tensor([e for e, _, _ in triples])
        rep = float(torch.tensor([r for _, r, _ in triples]).mean())
        ans_mass = float(torch.tensor([m for _, _, m in triples]).mean())
        return {"C": round(float(C), 3), "ans": float(anss.mean()),
                "ans_std": float(anss.std(unbiased=False)), "rep": rep, "ans_mass": ans_mass,
                "coherent": rep < REP_COHERENT_MAX}
    rows = [score(0.0)]
    base_am = rows[0]["ans_mass"]                 # C=0 baseline for the readout-validity flag
    for d in (step, -step):                      # outward each way; keep the 1st incoherent point
        C = d
        for _ in range(max_steps):
            r = score(C)
            rows.append(r)
            if not r["coherent"]:
                break
            C += d
    for r in rows:                                # readout trustworthy iff ans_mass held >= 90% of base
        r["readout_valid"] = r["ans_mass"] >= ANS_MASS_FRAC * base_am
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
               Cs=None, max_new_tokens: int = 512, seed: int = 0,
               apply_mode: str | None = None, apply_span: int = 1,
               rubric: str | None = None, readout: dict = DIGIT, budget: int = 6) -> list[dict]:
    """Per C: the steer-promoted tokens in a cowsay bubble, then the raw generation, all
    under steering. When `Cs` is None (the default), SEARCH for the strongest coherent steer
    each way (Illinois edge-find, ~`budget` evals/side, coherence probed on `rubric`) and
    demo the anchors [-C*, -C*/2, 0, +C*/2, +C*] -- the max coherent steer both ways, plus
    half-strength and baseline. So every demo shows the strongest COHERENT effect instead of
    hand-picked Cs that either do nothing or degenerate. Pass an explicit `Cs` to override.

    Uses the model's own generation_config sampling; `seed` fixes it so the C blocks are
    comparable. The cowsay speaks the top of (steered - unsteered) next-token logits -- what
    THIS C pushes up, with the shared think-opener prior subtracted out. `jac` is unused,
    kept for call-site stability.

    Extraction is decoupled from DELIVERY (see applies.py): pass `apply_mode`
    (add | clamp | add_last | replace_last) to swap how v hits the residual without
    re-extracting; `apply_span` is the trailing width for the last/replace modes.

    Pass `rubric` + `readout` (DIGIT for a 0-9 rating, YESNO for a binary dilemma) to add
    the quantitative readout AND to drive the edge search: per C the model thinks then
    answers, and we report the readout value + the coherence signals (rep, ans_mass)."""
    if apply_mode is not None:
        vec = Vector(dataclasses.replace(vec.cfg, apply_mode=apply_mode,
                                         apply_span=apply_span), vec.shared, vec.stacked)
    if Cs is None:                    # auto-search the coherent range (needs a probe)
        probe = rubric if rubric is not None else user_msg
        # search at the SAME generation length as the demo -- coherence (repetition) is
        # length-sensitive, so a shorter search probe overshoots the real edge.
        Cs = steer_anchors(model, tok, vec, probe, readout=readout, budget=budget,
                           max_new_tokens=max_new_tokens)
        logger.info(f"searched coherent anchors: C = {Cs}")
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
    anchors = []
    for C in Cs:
        torch.manual_seed(seed)
        with vec(model, C=C):
            steered = model(**enc).logits[0, -1].float()
            out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 pad_token_id=tok.eos_token_id)
            ans = (rubric_score(model, tok, rubric, max_new_tokens=max_new_tokens,
                                seed=seed, readout=readout) if rubric is not None else None)
        # steer-promoted tokens: top of (steered - base), word-like only. The subtraction
        # cancels the shared "about to open <think>" prior (Okay/Here/The) so what the
        # cowsay speaks is what THIS C actually pushes up, not the reasoning boilerplate
        # the old lens_topk-at-last-position surfaced for every C. (Claude)
        if C == 0:
            promoted_txt = "(baseline, no steer)"
        else:
            wl = _meaningful_token_mask(tok, steered.shape[-1], steered.device)
            promoted = (steered - base).masked_fill(~wl, float("-inf")).topk(6)
            promoted_txt = " · ".join(tok.decode([i]).strip() for i in promoted.indices.tolist())
        # raw decode WITH special tokens: real <think>/</think>, <|im_end|> visible,
        # nothing parsed or re-wrapped -- debuggable exactly as the model emitted it
        gen = tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=False)
        block = [f"\n--- C={C:+g} " + "-" * 60, "  steer promotes:",
                 _cthulhu_say(promoted_txt), gen]
        row = {"C": C, "promotes": promoted_txt, "gen": gen}
        if ans is not None:
            # SHOULD rise with +C, fall with -C; flat => steer not moving this axis.
            # rep>=REP_COHERENT_MAX => repetition breakdown (the off-target budget is spent).
            e, rep, am = ans
            degenerate = rep >= REP_COHERENT_MAX
            block.append(f"  rubric ans≈{e:.2f}  (rep={rep:.2f} ans_mass={am:.2f}"
                         f"{' DEGENERATE' if degenerate else ''})")
            row.update(ans=e, rep=rep, ans_mass=am, coherent=not degenerate)
        anchors.append(row)
        logger.info("\n".join(block) + "\n")
    # readout-validity flag, base-anchored: P(YES) is trustworthy only where the model kept
    # committing to an answer token, i.e. ans_mass >= ANS_MASS_FRAC of the C=0 baseline. This
    # is separate from coherence (rep) and never gates the edge search.
    q = [a for a in anchors if "ans_mass" in a]
    if q:
        base_am = min(q, key=lambda a: abs(a["C"]))["ans_mass"]
        for a in q:
            a["readout_valid"] = a["ans_mass"] >= ANS_MASS_FRAC * base_am
    return anchors


@torch.no_grad()
def demo_steer(jac: Jacobian, model, tok, vecs: dict, user_msg: str, *,
               rubric: str | None = None, readout: dict = DIGIT, budget: int = 6,
               max_new_tokens: int = 256, seed: int = 0) -> dict:
    """THE steering demo, one call for every notebook. For each named vector: a clear
    heading, a search for the strongest coherent steer each way, the raw generations at
    [-C*, -C*/2, 0, +C*/2, +C*] (comparable qualitative outputs, same prompt), then ONE
    table at the end comparing methods on the quant readout at the coherent edges.

    `vecs` maps a display name (state its config there, e.g. 'persona_pinv k=8') to a
    steering-lite Vector. `rubric`+`readout` add the quantitative answer and drive the
    edge search (DIGIT = 0-9 rating, YESNO = P(YES) on a binary dilemma).

    Returns {"summary": [one row per method], "detail": {name: [anchor rows incl. 'gen']}}
    so a UI (e.g. marimo) can render the comparison table AND the per-method generations."""
    summary, detail = [], {}
    for name, vec in vecs.items():
        logger.info(f"\n\n{'#' * 72}\n#  {name}\n{'#' * 72}")
        anchors = show_steer(jac, model, tok, vec, user_msg, Cs=None, rubric=rubric,
                             readout=readout, budget=budget, max_new_tokens=max_new_tokens,
                             seed=seed)
        detail[name] = anchors
        q = [a for a in anchors if "ans" in a]
        if not q:
            continue
        canonical = [
            {
                "coefficient": anchor["C"],
                "answer": anchor["ans"],
                "repetition": anchor["rep"],
                "answer_mass": anchor["ans_mass"],
                "display_generation": anchor["gen"],
            }
            for anchor in q
        ]
        summary.append(summarize_anchors(name, canonical))
    if summary:
        unit = "P(YES)" if readout is YESNO else "ans(0-9)"
        logger.info(f"\n\n{'=' * 72}\nCOMPARISON: {unit} at the dual-gated edges (edge = first of "
                    f"rep>={REP_COHERENT_MAX} or ans_mass<{ANS_MASS_FRAC}*base to bind); "
                    f"swing = ans@+ - ans@-; score = swing weighted by readout validity"
                    f"\nprompt={user_msg[:50]!r}\n{'=' * 72}\n"
                    + tabulate(summary, headers="keys", tablefmt="github", floatfmt=".3g"))
    return {"summary": summary, "detail": detail}
