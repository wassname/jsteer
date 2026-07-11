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
from loguru import logger
from steering_lite import Vector

from .jacobian import Jacobian


def chat_input(tok, user_msg: str, *, enable_thinking: bool = True) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        add_generation_prompt=True, tokenize=False, enable_thinking=enable_thinking)


def _cthulhu_say(text: str) -> str:
    """The j-space readout in a mini cowsay bubble -- Cthulhu speaks the tokens
    the steered residual points to. Cosmetic; the tokens are the payload."""
    n = len(text) + 2
    return ("\n".join([" " + "_" * n, f"< {text} >", " " + "-" * n,
                       "   \\", "    ^(;,;)^"]))


# think-then-answer rubric read: the demo's one-number sanity signal that steering
# moved the target axis. Same mechanism as moral-maps guided.py (let the model
# think, then read the logprobs at a JSON answer slot), reduced to a single scalar.
_ANS_FMT = (' Think it over, then answer with JSON {"ans": N} where N is a single'
            ' digit from 0 (least) to 9 (most).')


@torch.no_grad()
def rubric_score(model, tok, rubric: str, *, max_new_tokens: int, seed: int,
                 do_sample: bool = False, temperature: float = 0.7) -> tuple[float, float]:
    """Ask `rubric`, let the model think, then FORCE the answer slot `{"ans": ` and
    read the logprob-weighted expected digit 0-9 there. Returns (expected, pmass).

    expected = sum_d d * softmax(logit_d over the 10 digit tokens) -- a continuous
    scalar from single-token logprobs (cleaner than parsing a multi-token float).
    pmass = full-vocab softmax mass on the 10 digit tokens: a coherence guard, ~0
    means the slot isn't a digit (prefix/tokenizer mismatch), so distrust expected.
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
    pmass = float(logits.softmax(0)[ids].sum())
    return expected, pmass


@torch.no_grad()
def coherence_sweep(model, tok, vec, rubric: str, *, step: float = 0.1,
                    pmass_floor: float = 0.9, max_steps: int = 15, n_samples: int = 3,
                    temperature: float = 0.7, max_new_tokens: int = 512) -> list[dict]:
    """Walk C outward from 0 in +/- directions, scoring the rubric each step, and STOP
    a direction the step AFTER the answer slot goes incoherent (pmass<pmass_floor). Maps
    the coherent dose-response of the steered axis without hand-picking Cs. Returns rows
    sorted by C: {"C","ans","ans_std","pmass","coherent"}. Each C is averaged over
    `n_samples` think traces (seeds 0..n-1) to tame single-sample answer noise -- a
    lightweight stand-in for guided.py's Bayesian model averaging; ans_std is the spread.
    Coherence here is the ANSWER slot's pmass (is the forced digit well-defined), NOT
    free-form fluency -- the short forced answer survives steering that already frays long
    generation, so the coherent C-window is wider than the fluent-text window (read the
    qualitative show_steer for the latter)."""
    def score(C):
        with vec(model, C=C):
            pairs = [rubric_score(model, tok, rubric, max_new_tokens=max_new_tokens, seed=s,
                                  do_sample=n_samples > 1, temperature=temperature)
                     for s in range(n_samples)]
        anss = torch.tensor([a for a, _ in pairs])
        pmass = float(torch.tensor([p for _, p in pairs]).mean())
        return {"C": round(float(C), 3), "ans": float(anss.mean()),
                "ans_std": float(anss.std(unbiased=False)), "pmass": pmass,
                "coherent": pmass >= pmass_floor}
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


def plot_sweep(rows: list[dict], *, title: str = "rubric ans vs C",
               pmass_floor: float = 0.9):
    """ans vs C, points colored by answer coherence (pmass); incoherent points
    (pmass<floor) get a red edge so the coherent dose-response reads at a glance.
    The colorbar spans [floor-0.15, 1] (not 0-1): pmass barely varies while the
    answer stays coherent, so anchoring the ramp to the floor is what makes the
    slot fraying toward the cutoff visible instead of a flat wash of one color."""
    import matplotlib.pyplot as plt
    Cs = [r["C"] for r in rows]
    ans = [r["ans"] for r in rows]
    pm = [r["pmass"] for r in rows]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(Cs, ans, "-", color="0.8", lw=1, zorder=1)
    if all("ans_std" in r for r in rows):
        ax.errorbar(Cs, ans, yerr=[r["ans_std"] for r in rows], fmt="none",
                    ecolor="0.6", capsize=2, lw=1, zorder=1)
    sc = ax.scatter(Cs, ans, c=pm, cmap="viridis", vmin=pmass_floor - 0.15, vmax=1.0,
                    zorder=2, edgecolor=["0.2" if r["coherent"] else "red" for r in rows],
                    linewidth=1.2)
    ax.axvline(0, color="0.85", lw=0.8, zorder=0)
    ax.set_xlabel("steering coefficient C")
    ax.set_ylabel("rubric ans (0-9)")
    ax.set_ylim(-0.3, 9.3)
    ax.set_title(title)
    cbar = fig.colorbar(sc, ax=ax, label="answer coherence (pmass)")
    cbar.ax.axhline(pmass_floor, color="red", lw=1)   # the incoherent cutoff
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
               Cs=(-6, 0, 6), layer: int | None = None, k: int = 6,
               max_new_tokens: int = 512, seed: int = 0,
               apply_mode: str | None = None, apply_span: int = 1,
               rubric: str | None = None) -> None:
    """One block per C: lens readout at `layer`, then the raw generation, all
    under steering. Uses the model's own generation_config sampling; `seed`
    fixes it so the C blocks are comparable. `layer` defaults to the top fitted
    layer. max_new_tokens defaults to 512 so Qwen3's <think> block can close;
    256 truncates mid-reasoning.

    Extraction is decoupled from DELIVERY (see applies.py): pass `apply_mode`
    (add | clamp | add_last | replace_last) to swap how v hits the residual
    without re-extracting; `apply_span` is the trailing-position width for the
    last/replace modes. Coefficient units differ by mode (clamp sets a component
    VALUE, add scales a direction), so each mode wants its own Cs.

    Pass `rubric` (a 0-9 rating question about the steered axis) to add the
    quantitative readout: per C, the model thinks then answers `{"ans": N}` and we
    report the logprob-weighted expected digit. It SHOULD rise with +C and fall
    with -C; flat means the steer isn't moving that axis (see rubric_score)."""
    if apply_mode is not None:
        vec = Vector(dataclasses.replace(vec.cfg, apply_mode=apply_mode,
                                         apply_span=apply_span), vec.shared, vec.stacked)
    layer = jac.layers[-1] if layer is None else layer
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
    # SHOULD: C=0 is the baseline; +C tilts the lens tokens and tone toward the
    # concept, -C away; all stay coherent (gibberish = coeff too large).
    for C in Cs:
        torch.manual_seed(seed)
        with vec(model, C=C):
            jtop = jac.lens_topk(model, tok, prompt, layer=layer, k=k)
            out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 pad_token_id=tok.eos_token_id)
            ans = (rubric_score(model, tok, rubric, max_new_tokens=max_new_tokens,
                                seed=seed) if rubric is not None else None)
        # raw decode WITH special tokens: real <think>/</think>, <|im_end|> visible,
        # nothing parsed or re-wrapped -- debuggable exactly as the model emitted it
        gen = tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=False)
        readout = " · ".join(t.strip() for t, _ in jtop)
        block = [f"\n--- C={C:+g} " + "-" * 60,
                 f"  lens @L{layer}:", _cthulhu_say(readout), gen]
        if ans is not None:
            # SHOULD rise with +C, fall with -C; flat => steer not moving this axis.
            # pmass<~0.5 => answer slot isn't a digit, distrust the number.
            block.append(f"  rubric ans≈{ans[0]:.2f}/9  (pmass={ans[1]:.2f})")
        logger.info("\n".join(block) + "\n")
