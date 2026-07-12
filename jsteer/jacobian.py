"""Fit a model's Jacobian once, then steer any concept by pulling a direction
back through it. (Claude)

The method (verified in j-steer-dev: word steering beat a norm-matched random
control on 3/5 moral foundations, Qwen3-4B, n=3 seeds):

    v_l = unit( J_l^T @ w )

`J_l = E_prompts[ d h_final / d h_l ]` is the Jacobian of the final-layer
residual with respect to layer `l`, averaged over prompts and token positions
(jlens's verified estimator, via `jlens.fitting.fit`, never reimplemented). `w`
is a COTANGENT: a direction placed at the OUTPUT (final-layer basis) naming the
concept -- for words, the unembedding row that raises those tokens' logits.
`J_l^T @ w` sends that output-space target back to a residual direction at layer
`l`: the PULLBACK of `w` (the standard autodiff / differential-geometry name for
J-transpose applied to a cotangent; the reverse-mode-autodiff way to compute the
same vector is the vector-Jacobian product, VJP -- see vjp.py). Three ways to
build `w`:

    word_vector           w = mean unembedding row of the words     VERIFIED
    persona_vector        w = h_bar(pos) - h_bar(neg)               EXPERIMENTAL*
    persona_topk_vector   w = top-k tokens of the pos-neg logit      EXPERIMENTAL
                          contrast (differ before top-k, else null)
    persona_soft_vector   w = W_U^T (softmax contrast) -- topk's     EXPERIMENTAL
                          full-vocab limit; a genuine cotangent
    persona_topk/soft mask non-word-like tokens (emoji/specials) out of the
    contrast: they are degenerate emit-targets (the emoji-spam failure mode).

    persona_pinv_vector is NOT a pullback: h_bar(pos)-h_bar(neg) is a TANGENT
    (an activation displacement), and J^T only transports cotangents
    (gradients). It solves J_l delta = h_diff instead -- see its docstring.

    * persona-contrast vectors FAILED specificity controls in j-steer-dev
      (moved the target axis no more than an unrelated persona did). Shipped
      for experimentation, not as a recommendation.

Why cache the full J: by linearity `mean_p(J_p)^T w = mean_p(J_p^T w)`, so a
vector pulled back through the cached averaged Jacobian is numerically identical
to the direct per-prompt VJP (vjp.py; parity-tested). Fitting is the expensive
step (one forward + ceil(d_model/dim_batch) backwards per prompt); afterwards
every concept vector is a CPU matvec.

Vectors come out as `steering_lite.Vector` (unit direction per layer in
stacked["v"], k=1 leading dim -- mean_diff's exact layout), so steering is:

    with v(model, C=8):
        model.generate(**inputs)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from jlens.fitting import fit as _jlens_fit
from jlens.hf import HFLensModel, from_hf
from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens
from jlens.vis import _meaningful_token_mask  # jlens's word-like vocab mask (cached)
from loguru import logger
from torch import Tensor
from tqdm.auto import tqdm

from steering_lite.config import SteeringConfig
from steering_lite.vector import Vector

from .applies import (
    JacobianPersonaC,
    JacobianPersonaPinvC,
    JacobianPersonaSoftC,
    JacobianPersonaTopkC,
    JacobianWordC,
    RandomC,
    ε,
)


# --- small helpers ------------------------------------------------------------

def _unit(v: Tensor) -> Tensor:
    return v / (v.norm() + ε)


def _resolve_layers(layers, n_layers: int) -> list[int]:
    """None -> all layers below final; (lo, hi) floats -> fraction band;
    iterable of ints -> as-is (negatives count from the end)."""
    if layers is None:
        return list(range(n_layers - 1))
    layers = tuple(layers)
    if len(layers) == 2 and all(isinstance(x, float) for x in layers):
        lo, hi = (int(layers[0] * n_layers), int(layers[1] * n_layers))
        return list(range(lo, min(hi, n_layers - 1)))
    return sorted({l + n_layers if l < 0 else l for l in layers})


def _to_vector(cfg: SteeringConfig, per_layer: dict[int, Tensor]) -> Vector:
    """Wrap unit directions as a steering-lite Vector (mean_diff's layout:
    stacked["v"] with leading k=1 dim, shared empty). Always CPU fp32 --
    Jacobian.pullback is CPU-native but the VJP path accumulates on cuda;
    without the .cpu() the two paths return device-inconsistent Vectors."""
    shared = {l: {} for l in per_layer}
    stacked = {l: {"v": _unit(v.float().cpu()).unsqueeze(0)} for l, v in per_layer.items()}
    return Vector(cfg, shared, stacked)


def _word_cotangent(model, tok, words: list[str]) -> Tensor:
    """Mean unembedding row over `words` (first sub-token of each): the
    final-basis direction that most raises those tokens' output logits.
    Pulling THIS back through J^T is the pure concept->residual map -- no
    persona pairs, so no persona-bundle confound. +C enhances the concept."""
    W_U = model.lm_head.weight                                   # [vocab, d]
    ids = [tok(w, add_special_tokens=False).input_ids[0] for w in words]
    rows = W_U[torch.tensor(ids, device=W_U.device)].float()     # [n, d]
    cot = rows.mean(0)
    logger.info(f"word cotangent: {words} -> first-subtoken ids={ids} |w|={cot.norm():.3f}")
    return cot


@torch.no_grad()
def _h_bar_final(model, tok, prompts: list[str], *, batch_size: int = 8,
                 max_length: int = 384, label: str = "") -> Tensor:
    """Mean last-token activation at the FINAL layer over `prompts`
    (right-padded batches; last real token located via the attention mask)."""
    lm = from_hf(model, tok)                                     # layout detection only
    target_layer = lm.n_layers - 1
    acc, n = None, 0
    # single-batch persona lists (<= batch_size) make a useless 1/1 bar that still prints
    # a completion line; disable those. Both intervals equal for the multi-batch case.
    for i in tqdm(range(0, len(prompts), batch_size), desc=f"h_bar {label}",
                  disable=len(prompts) <= batch_size, mininterval=120, maxinterval=120):
        batch = prompts[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_length, padding_side="right").to(model.device)
        with ActivationRecorder(lm.layers, at=[target_layer]) as rec:
            model(**enc)
        act = rec.activations[target_layer]                      # [B, S, d]
        last_idx = enc["attention_mask"].sum(dim=1) - 1          # [B]
        last = act[torch.arange(act.shape[0]), last_idx].float()
        acc = last.sum(0) if acc is None else acc + last.sum(0)
        n += last.shape[0]
    return acc / n


# --- the core object ----------------------------------------------------------

@dataclass
class Jacobian:
    """A fitted, cached, model-specific bundle of per-layer Jacobians
    `{layer: J_l [d, d]}` (a `jlens.JacobianLens`) plus the concept->vector
    methods. Fit once (expensive), derive steering vectors forever (matvec)."""

    lens: JacobianLens

    # -- fit / persist (all delegated to the researchers' jlens code) ----------

    @classmethod
    def fit(cls, model, tok, prompts: list[str], *, layers=None, dim_batch: int = 8,
            max_seq_len: int = 128, checkpoint_path: str | None = None,
            compile: bool = False) -> "Jacobian":
        """Fit `J_l` on `prompts` (generic text; jlens guidance is ~100+ for
        lens-quality pooling). Cost: 1 forward + ceil(d_model/dim_batch)
        backwards per prompt -- ALL source layers come from the same backwards,
        so fitting more layers costs memory, not compute. `checkpoint_path`
        makes the fit resumable (atomic writes)."""
        lm = from_hf(model, tok, compile=compile)
        source_layers = _resolve_layers(layers, lm.n_layers)

        # Full trace of the first fit prompt as jlens sees it (special tokens on).
        # SHOULD: a chat fit opens with the template's <|im_start|>user and ends at
        # the assistant/<think> start; plain text means the template was skipped.
        ids0 = tok(prompts[0], add_special_tokens=True).input_ids
        logger.info(f"fit on {len(prompts)} prompts, layers={source_layers} "
                    f"(dim_batch={dim_batch}, max_seq_len={max_seq_len})")
        logger.info(f"FIT PROMPT[0] ({len(ids0)} tok): {tok.decode(ids0)!r}")

        # jlens.fit has no progress bar; wrap prompts so we get one (fit consumes
        # them only via enumerate/len, so this is safe). Both tqdm intervals set
        # (token-efficient-logging) to avoid CR-spam in non-tty logs.
        bar = tqdm(prompts, desc="fit J", mininterval=30, maxinterval=30)
        lens = _jlens_fit(lm, bar, source_layers=source_layers,
                          dim_batch=dim_batch, max_seq_len=max_seq_len,
                          checkpoint_path=checkpoint_path)
        jac = cls(lens=lens)
        logger.info(f"fit done: {jac!r}")
        return jac

    def save(self, path: str) -> None:
        self.lens.save(path)      # fp16 by default; jlens-compatible file

    @classmethod
    def load(cls, path: str) -> "Jacobian":
        return cls(lens=JacobianLens.load(path))

    @classmethod
    def from_pretrained(cls, name_or_path: str, **kw) -> "Jacobian":
        """Local file/dir or HuggingFace Hub repo_id (see JacobianLens)."""
        return cls(lens=JacobianLens.from_pretrained(name_or_path, **kw))

    @classmethod
    def fit_cached(cls, model, tok, prompts, path, **fit_kw) -> "Jacobian":
        """Load `path` if it exists, else fit and save it there. `prompts` may be
        a list or a zero-arg callable returning one -- the callable runs only on
        a cache MISS, so a cache hit never pays to build the corpus (e.g. stream
        WikiText). Path is caller-supplied so the library never needs the repo
        layout; scripts and notebooks derive it from the model name
        (config.cache_path), which is what lets one line fit-or-load any model."""
        path = str(path)
        if Path(path).exists():
            logger.info(f"loading cached Jacobian: {path}")
            return cls.load(path)
        logger.info(f"no cache at {path}; fitting (the expensive step)")
        jac = cls.fit(model, tok, prompts() if callable(prompts) else prompts, **fit_kw)
        jac.save(path)
        return jac

    @property
    def layers(self) -> list[int]:
        return self.lens.source_layers

    def __repr__(self) -> str:
        return f"Jacobian({self.lens!r})"

    # -- the one pullback -------------------------------------------------------

    def pullback(self, cotangent: Tensor, cfg: SteeringConfig) -> Vector:
        """v_l = unit(J_l^T @ w) for every layer in cfg.layers.

        J_l rows are output dims (each row is the gradient of one final-basis
        dim over the source layer), so the pullback is `w @ J_l`. Computed on
        CPU fp32 against the cached matrices -- no model, no backward."""
        w = cotangent.detach().float().cpu()
        if w.shape != (self.lens.d_model,):
            raise ValueError(f"cotangent shape {tuple(w.shape)} != ({self.lens.d_model},)")
        missing = set(cfg.layers) - set(self.lens.source_layers)
        if missing:
            raise ValueError(f"layers {sorted(missing)} not fitted; have {self.layers}")
        per_layer = {l: w @ self.lens.jacobians[l] for l in cfg.layers}
        # per-layer pullback norms before unit-normalizing: a fit-health trace
        # (flat/near-zero everywhere => the lens didn't pick up this cotangent).
        logger.debug(f"{cfg.method} per-layer |J^T w| (pre-norm): " +
                     " ".join(f"{l}:{per_layer[l].norm():.3g}" for l in cfg.layers))
        return _to_vector(cfg, per_layer)

    def steer_band(self, model, *, lo: float = 0.3, hi: float = 0.9) -> tuple[int, ...]:
        """Fitted layers within the [lo, hi] fraction of model depth. The authors'
        pre-fitted lenses span EVERY layer; steering all of them at once over-drives
        the residual, so restrict to the mid-depth band run-524 used."""
        n = model.config.num_hidden_layers
        return tuple(l for l in self.lens.source_layers if lo <= l / n <= hi)

    def _steer_layers(self, layers) -> tuple[int, ...]:
        """None -> all fitted layers; else explicit int indices (float bands are
        a fit-time concept -- the lens doesn't know n_layers to resolve them)."""
        if layers is None:
            return tuple(self.lens.source_layers)
        if any(isinstance(l, float) for l in layers):
            # int() would silently truncate (0.5, 0.8) -> layer 0 and steer the
            # wrong layer; float bands only exist at fit time.
            raise ValueError(f"float layer bands are fit-time only; got {layers}, "
                             f"pass explicit ints from .layers={self.layers}")
        return tuple(sorted(int(l) for l in layers))

    # -- concept -> vector -------------------------------------------------------

    def word_vector(self, model, tok, words: list[str], *, layers=None) -> Vector:
        """VERIFIED method: pull the words' unembedding direction back through
        the Jacobian. +C enhances the concept, -C suppresses it."""
        cfg = JacobianWordC(layers=self._steer_layers(layers))
        return self.pullback(_word_cotangent(model, tok, words), cfg)

    def persona_vector(self, model, tok, pos_prompts: list[str],
                       neg_prompts: list[str], *, layers=None,
                       batch_size: int = 8) -> Vector:
        """EXPERIMENTAL: pull the persona activation contrast back through the
        Jacobian. Persona-contrast pullbacks failed specificity controls in
        j-steer-dev -- prefer word_vector for targeted steering."""
        h_pos = _h_bar_final(model, tok, pos_prompts, batch_size=batch_size, label="pos")
        h_neg = _h_bar_final(model, tok, neg_prompts, batch_size=batch_size, label="neg")
        logger.info(f"h_bar_diff |pos|={h_pos.norm():.3f} |neg|={h_neg.norm():.3f} "
                    f"|diff|={ (h_pos - h_neg).norm():.3f}")
        cfg = JacobianPersonaC(layers=self._steer_layers(layers))
        return self.pullback(h_pos - h_neg, cfg)

    def persona_topk_vector(self, model, tok, pos_prompts: list[str],
                            neg_prompts: list[str], *, k: int = 8, layers=None,
                            batch_size: int = 8) -> Vector:
        """EXPERIMENTAL: persona CONTRAST -> vocabulary bottleneck -> word pullback.
        Unembed each persona's final-layer mean, take the DIFFERENCE of the two
        logit vectors, and read the top-k tokens pos evokes more than neg (and
        vice versa). Contrasting BEFORE top-k is essential: both persona means
        unembed to the same generic high-frequency tokens (\\n, ' I', ' The'),
        so top-k of each separately gives a near-null contrast -- the persona
        signal is only in the difference. Then contrast those two token sets'
        unembedding rows and pull that back. Composes the persona signal with
        the verified word mechanism; untested for specificity."""
        lm = from_hf(model, tok)
        h_pos = _h_bar_final(model, tok, pos_prompts, batch_size=batch_size, label="pos")
        h_neg = _h_bar_final(model, tok, neg_prompts, batch_size=batch_size, label="neg")
        W_U = model.lm_head.weight                                # [vocab, d]
        # token SELECTION goes through the full logit pipeline (final norm);
        # the cotangent below uses raw W_U rows to match _word_cotangent.
        logits_pos = lm.unembed(h_pos.to(model.device).to(model.dtype)).float()
        logits_neg = lm.unembed(h_neg.to(model.device).to(model.dtype)).float()
        diff = logits_pos - logits_neg                           # [vocab]
        # word-like tokens only: emoji/special/punct tokens are degenerate
        # emit-targets -- steering toward them collapses generation into
        # repeating them (the emoji-spam failure mode at higher C). (Claude)
        wordlike = _meaningful_token_mask(tok, diff.shape[-1], diff.device)
        top_pos = diff.masked_fill(~wordlike, -torch.inf).topk(k)     # pos evokes > neg
        top_neg = (-diff).masked_fill(~wordlike, -torch.inf).topk(k)  # neg evokes > pos
        toks_pos = [tok.decode([i]) for i in top_pos.indices.tolist()]
        toks_neg = [tok.decode([i]) for i in top_neg.indices.tolist()]
        # SHOULD be persona-specific words (positive vs negative affect here), not
        # generic \n/the; generic on both sides means the contrast is null.
        logger.info(f"j-thoughts (content of mental workspace, top-{k})\n"
                    f"    positive: {toks_pos}\n"
                    f"    negative: {toks_neg}")
        w = (W_U[top_pos.indices].float().mean(0)
             - W_U[top_neg.indices].float().mean(0)).cpu()
        cfg = JacobianPersonaTopkC(layers=self._steer_layers(layers))
        return self.pullback(w, cfg)

    def persona_soft_vector(self, model, tok, pos_prompts: list[str],
                            neg_prompts: list[str], *, temperature: float = 1.0,
                            layers=None, batch_size: int = 8) -> Vector:
        """EXPERIMENTAL: persona_topk's full-vocab limit, and a genuine cotangent.

            w = W_U^T (softmax(u_pos/T) - softmax(u_neg/T))

        where u = unembedded persona mean. This is exactly the gradient wrt the
        final residual of E_{t~p_pos}[log p(t|h)] - E_{t~p_neg}[log p(t|h)]
        (each term's softmax-baseline E_p[W_U] cancels in the difference), so
        unlike persona_vector's activation diff, J^T transports it legitimately.
        Vs topk: no hard k=8 compression to the personas' most extreme tokens
        (the over-literal "emit :-)" failure); every tone-correlated token
        contributes, weighted by how much the personas disagree on it.
        `temperature` subsumes k: low T -> topk-like sparsity, high T -> broad
        support. Non-word-like tokens are masked out before the softmax, same
        rationale as topk. Untested for specificity. (Claude)"""
        lm = from_hf(model, tok)
        h_pos = _h_bar_final(model, tok, pos_prompts, batch_size=batch_size, label="pos")
        h_neg = _h_bar_final(model, tok, neg_prompts, batch_size=batch_size, label="neg")
        W_U = model.lm_head.weight                                # [vocab, d]
        u_pos = lm.unembed(h_pos.to(model.device).to(model.dtype)).float()
        u_neg = lm.unembed(h_neg.to(model.device).to(model.dtype)).float()
        wordlike = _meaningful_token_mask(tok, u_pos.shape[-1], u_pos.device)
        p_pos = u_pos.masked_fill(~wordlike, -torch.inf).div(temperature).softmax(-1)
        p_neg = u_neg.masked_fill(~wordlike, -torch.inf).div(temperature).softmax(-1)
        Δp = p_pos - p_neg                                        # [vocab], sums to 0
        # read your data: TV distance = how much the personas disagree about the
        # next token at all. SHOULD be clearly > 0 (~0 => null contrast, same
        # failure mode as topk's identical token sets); top tokens SHOULD be
        # persona-specific words, not generic sentence-starters.
        tv = 0.5 * float(Δp.abs().sum())
        top_pos, top_neg = Δp.topk(8), (-Δp).topk(8)
        logger.info(
            f"j-thoughts (soft, T={temperature}) TV(p_pos, p_neg)={tv:.3f}\n"
            f"    positive: {[tok.decode([i]) for i in top_pos.indices.tolist()]}\n"
            f"    negative: {[tok.decode([i]) for i in top_neg.indices.tolist()]}")
        w = (Δp @ W_U.float()).cpu()          # raw W_U rows, matching _word_cotangent
        cfg = JacobianPersonaSoftC(layers=self._steer_layers(layers))
        return self.pullback(w, cfg)

    def persona_pinv_vector(self, model, tok, pos_prompts: list[str],
                            neg_prompts: list[str], *, ridge: float = 1e-3,
                            layers=None, batch_size: int = 8) -> Vector:
        """EXPERIMENTAL: transport the persona contrast as a TANGENT, which it is.

        h_diff = h_bar(pos) - h_bar(neg) is an activation DISPLACEMENT at the
        final layer, not a gradient -- persona_vector's J^T h_diff pulls it back
        as if it were a cotangent, a type error (only correct if J were
        orthogonal). The right transport asks: which layer-l perturbation
        delta pushes forward to h_diff?

            delta_l = argmin |J_l delta - h_diff|^2 + lam |delta|^2
                    = (J_l^T J_l + lam I)^{-1} J_l^T h_diff,  lam = ridge * mean diag(J^T J)

        Ridge because the position-averaged J is ill-conditioned. If THIS still
        fails specificity, the failure is the averaged Jacobian itself (it can't
        carry contextual features), not the algebra. CPU fp32 against the cached
        matrices, ~seconds per layer, no backward. (Claude)"""
        h_pos = _h_bar_final(model, tok, pos_prompts, batch_size=batch_size, label="pos")
        h_neg = _h_bar_final(model, tok, neg_prompts, batch_size=batch_size, label="neg")
        h_diff = (h_pos - h_neg).float().cpu()
        logger.info(f"h_bar_diff |pos|={h_pos.norm():.3f} |neg|={h_neg.norm():.3f} "
                    f"|diff|={h_diff.norm():.3f}")
        cfg = JacobianPersonaPinvC(layers=self._steer_layers(layers))
        per_layer, residuals = {}, {}
        eye = torch.eye(self.lens.d_model)
        for l in cfg.layers:
            J = self.lens.jacobians[l]                            # [d_out, d_in] fp32 cpu
            JtJ = J.T @ J
            lam = ridge * JtJ.diagonal().mean()
            δ = torch.linalg.solve(JtJ + lam * eye, J.T @ h_diff)
            per_layer[l] = δ
            residuals[l] = float((J @ δ - h_diff).norm() / h_diff.norm())
        # SHOULD be well below 1.0 at most layers: 1.0 means J can't realize
        # h_diff at all (h_diff orthogonal to J's row space) and the vector is
        # ridge-noise; small residual means the transport is faithful.
        logger.info("pinv relative residual |J d - h|/|h| per layer: " +
                    " ".join(f"{l}:{residuals[l]:.2f}" for l in cfg.layers))
        return _to_vector(cfg, per_layer)

    def random_vector(self, *, seed: int = 0, layers=None) -> Vector:
        """Norm-matched control: unit random direction per layer. Any honest
        demo/eval should show the concept vector beating THIS at the same C."""
        gen = torch.Generator().manual_seed(seed)
        cfg = RandomC(layers=self._steer_layers(layers), seed=seed)
        per_layer = {l: torch.randn(self.lens.d_model, generator=gen)
                     for l in cfg.layers}
        return _to_vector(cfg, per_layer)

    # -- bonus: the lens's native forward readout --------------------------------

    def lens_topk(self, model, tok, prompt: str, layer: int, *, k: int = 10,
                  position: int = -1, mask_wordlike: bool = True) -> list[tuple[str, float]]:
        """Lens readout at `layer`: transport the residual to the final basis
        with J_l and decode to tokens (a linear approximation, not the literal
        computation). jlens's native use, handy in demos.

        `mask_wordlike` reuses jlens's own word-like vocab mask so the readout
        hides punctuation/single-char/special tokens (which, per the walkthrough,
        trail the interesting word tokens on Qwen); ranks are unaffected."""
        lm = from_hf(model, tok)
        lens_logits, _, _ = self.lens.apply(lm, prompt, layers=[layer],
                                            positions=[position])
        logits = lens_logits[layer][0]
        if mask_wordlike:
            wl = _meaningful_token_mask(tok, logits.shape[-1], logits.device)
            logits = logits.masked_fill(~wl, float("-inf"))
        top = logits.topk(k)
        return [(tok.decode([i]), float(v)) for i, v in
                zip(top.indices.tolist(), top.values.tolist())]
