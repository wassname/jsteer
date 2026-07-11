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
import json

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
# moved the target axis. Same mechanism as moral-maps guided.py (let the model
# think, then read the logprobs at a JSON answer slot), reduced to a single scalar.
# The object carries a trivial-arithmetic canary ("2+2") and a free-text field so a
# steer-degraded model has ROOM to break the object -- coherence is measured on the
# free-generated object, not on the forced digit slot (which is ~always a digit
# because the `{"ans": ` prefix makes one obvious). (Claude)
_ANS_FMT = (' Think it over, then answer with ONE line of JSON and nothing after it:'
            ' {"ans": N, "why": "<=3 words", "2+2": M} where N is a single digit from'
            ' 0 (least) to 9 (most) and M is the value of 2+2.')


@torch.no_grad()
def rubric_score(model, tok, rubric: str, *, max_new_tokens: int, seed: int,
                 do_sample: bool = False, temperature: float = 0.7) -> tuple[float, dict]:
    """Ask `rubric`, let the model think, force the slot `{"ans": ` for a clean scalar
    read, then FREE-GENERATE the rest of the JSON object as a coherence probe.
    Returns (expected, coh).

    expected = sum_d d * softmax(logit_d over the 10 digit tokens) at the forced slot
    -- a continuous scalar from single-token logprobs (cleaner than parsing a float).

    coh = {"valid", "chk_ok", "span_pmass"} measured on the free-generated object:
      valid      -- the object parses as JSON (a steer-fried model fails to close it),
      chk_ok     -- its "2+2" field == 4 (trivial-arithmetic canary),
      span_pmass -- mean top-1 softmax prob over the generated span. It degrades in the
                    COHERENT regime (~0.95 -> 0.81 as the steer bites) but is NOT a
                    coherence measure on its own: a steer-fried model collapses into a
                    confident degenerate loop, so span_pmass climbs back toward ~1 on
                    repeated garbage (observed valid=False, span_pmass=0.97 at C=3.0).
                    Coherence therefore GATES on (valid and chk_ok); span_pmass is only
                    a within-coherent confidence read, trustworthy where valid is True.
    The rigorous K-way, position-debiased version is moral-maps guided.py; this is
    the demo's cheap readout, scored under whatever steering is active."""
    prompt = chat_input(tok, rubric + _ANS_FMT)
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
    forced = prompt + think + '</think>\n{"ans": '            # our own deterministic slot
    fenc = tok(forced, return_tensors="pt").to(model.device)
    logits = model(**fenc).logits[0, -1].float()
    ids = torch.tensor([tok(str(d), add_special_tokens=False).input_ids[0]
                        for d in range(10)], device=logits.device)
    expected = float((logits[ids].softmax(0) * torch.arange(10., device=ids.device)).sum())

    # free-generate the rest of the object; short cap so incoherence shows fast
    gob = model.generate(**fenc, max_new_tokens=20, do_sample=False,
                         pad_token_id=tok.eos_token_id,
                         output_scores=True, return_dict_in_generate=True)
    span_pmass = float(torch.stack([s[0].float().softmax(-1).max()
                                    for s in gob.scores]).mean())
    body = '{"ans": ' + tok.decode(gob.sequences[0][fenc.input_ids.shape[1]:],
                                   skip_special_tokens=True)
    try:                          # invalid JSON IS the signal (fried model can't close it)
        # raw_decode parses the first object and ignores trailing tokens, so an early
        # `}` inside a string value doesn't truncate a valid object (json.loads would).
        obj, _ = json.JSONDecoder().raw_decode(body)
        valid, chk_ok = True, obj.get("2+2") == 4
    except json.JSONDecodeError:
        valid, chk_ok = False, False
    return expected, {"valid": valid, "chk_ok": chk_ok, "span_pmass": span_pmass}


@torch.no_grad()
def coherence_sweep(model, tok, vec, rubric: str, *, step: float = 0.1,
                    max_steps: int = 15, n_samples: int = 3,
                    temperature: float = 0.7, max_new_tokens: int = 512) -> list[dict]:
    """Walk C outward from 0 in +/- directions, scoring the rubric each step, and STOP a
    direction the step AFTER the model can no longer emit a valid answer object (the
    majority of seeds fail JSON-parse or the "2+2" canary). Maps the coherent dose-
    response of the steered axis without hand-picking Cs. Returns rows sorted by C:
    {"C","ans","ans_std","span_pmass","valid_frac","coherent"}. Each C is averaged over
    `n_samples` think traces (seeds 0..n-1) to tame single-sample answer noise -- a
    lightweight stand-in for guided.py's Bayesian model averaging; ans_std is the spread.
    Coherence = the model still free-generates a well-formed object AND gets 2+2 right;
    span_pmass grades its confidence. This breaks well before free-form fluency does at
    large |C|, so read the qualitative show_steer for the long-generation frailty."""
    def score(C):
        with vec(model, C=C):
            pairs = [rubric_score(model, tok, rubric, max_new_tokens=max_new_tokens, seed=s,
                                  do_sample=n_samples > 1, temperature=temperature)
                     for s in range(n_samples)]
        anss = torch.tensor([e for e, _ in pairs])
        span = float(torch.tensor([c["span_pmass"] for _, c in pairs]).mean())
        valid_frac = sum(c["valid"] and c["chk_ok"] for _, c in pairs) / len(pairs)
        return {"C": round(float(C), 3), "ans": float(anss.mean()),
                "ans_std": float(anss.std(unbiased=False)), "span_pmass": span,
                "valid_frac": valid_frac, "coherent": valid_frac >= 0.5}
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
    """ans vs C, points colored by coherence = valid_frac (fraction of seeds that emit
    a well-formed {"ans",...,"2+2"} object with 2+2==4); points below majority also get
    a red edge. Coherence is NEAR-BINARY (flat while the model holds, a cliff when it
    breaks), so the color reads as a gate and the ans curve carries the dose-response.
    We deliberately do NOT color by span_pmass: a steer-fried model collapses into a
    confident degenerate loop (span_pmass climbs back toward 1 on repeated garbage), so
    peakiness is not coherence -- valid_frac is what can't be fooled by confident junk."""
    import matplotlib.pyplot as plt
    Cs = [r["C"] for r in rows]
    ans = [r["ans"] for r in rows]
    coh = [r["valid_frac"] for r in rows]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(Cs, ans, "-", color="0.8", lw=1, zorder=1)
    if all("ans_std" in r for r in rows):
        ax.errorbar(Cs, ans, yerr=[r["ans_std"] for r in rows], fmt="none",
                    ecolor="0.6", capsize=2, lw=1, zorder=1)
    sc = ax.scatter(Cs, ans, c=coh, cmap="viridis", vmin=0.0, vmax=1.0,
                    zorder=2, edgecolor=["0.2" if r["coherent"] else "red" for r in rows],
                    linewidth=1.2)
    ax.axvline(0, color="0.85", lw=0.8, zorder=0)
    ax.set_xlabel("steering coefficient C")
    ax.set_ylabel("rubric ans (0-9)")
    ax.set_ylim(-0.3, 9.3)
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, label="coherence (valid-object fraction)")
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
            # json=False or 2+2!=4 => the steer broke the model, distrust the number.
            e, c = ans
            block.append(f"  rubric ans≈{e:.2f}/9  (json={c['valid']} 2+2ok={c['chk_ok']}"
                         f" conf={c['span_pmass']:.2f})")
        logger.info("\n".join(block) + "\n")
