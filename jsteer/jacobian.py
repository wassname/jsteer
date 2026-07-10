"""Full-Jacobian extraction for steering: fit once, steer any concept.

(drafted by Claude, ported from the verified j-steer-dev experiment code)

The method (verified in j-steer-dev: word steering beat a norm-matched random
control on 3/5 moral foundations, Qwen3-4B, n=3 seeds):

    v_l = unit( J_l^T @ w )

where `J_l = E_prompts[ d h_final / d h_l ]` is the jlens position-averaged
Jacobian -- the researchers' verified estimator, reused via `jlens.fitting.fit`
(never reimplemented) -- and `w` is a cotangent in the FINAL-layer basis naming
the concept to steer:

    word_vector           w = mean unembedding row of the words     VERIFIED
    persona_vector        w = h_bar(pos) - h_bar(neg)               EXPERIMENTAL*
    persona_topk_vector   w = contrast of the top-k tokens each     EXPERIMENTAL
                          persona evokes at the final layer

    * persona-contrast pullbacks FAILED specificity controls in j-steer-dev
      (moved the target axis no more than an unrelated persona did). Shipped
      for experimentation, not as a recommendation.

Why cache the full J: by linearity `mean_p(J_p)^T w = mean_p(J_p^T w)`, so a
vector pulled back through the cached pooled Jacobian is numerically the same
vector the direct per-prompt VJP produces (see vjp.py; parity-tested). Fitting
is the expensive step (one forward + ceil(d_model/dim_batch) backwards per
prompt); afterwards every concept vector is a CPU matvec.

The Jacobian is always fit against the FINAL layer basis (jlens default), so
cotangents are measured there: unembedding rows live there natively, persona
means are recorded there.

Vectors come out as `steering_lite.Vector` (unit direction per layer in
stacked["v"], k=1 leading dim -- mean_diff's exact layout), so steering is:

    with v(model, C=8):
        model.generate(**inputs)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from jlens.fitting import fit as _jlens_fit
from jlens.hf import HFLensModel, from_hf
from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens
from loguru import logger
from torch import Tensor
from tqdm.auto import tqdm

from steering_lite.config import SteeringConfig
from steering_lite.vector import Vector

from .applies import (
    JacobianPersonaC,
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
    without the .cpu() the two paths return device-inconsistent Vectors
    (Claude: found by U4 step-2 crash, pueue 550)."""
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
    for i in tqdm(range(0, len(prompts), batch_size), desc=f"h_bar {label}",
                  mininterval=30, maxinterval=60):
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
        lens = _jlens_fit(lm, prompts, source_layers=source_layers,
                          dim_batch=dim_batch, max_seq_len=max_seq_len,
                          checkpoint_path=checkpoint_path)
        return cls(lens=lens)

    def save(self, path: str) -> None:
        self.lens.save(path)      # fp16 by default; jlens-compatible file

    @classmethod
    def load(cls, path: str) -> "Jacobian":
        return cls(lens=JacobianLens.load(path))

    @classmethod
    def from_pretrained(cls, name_or_path: str, **kw) -> "Jacobian":
        """Local file/dir or HuggingFace Hub repo_id (see JacobianLens)."""
        return cls(lens=JacobianLens.from_pretrained(name_or_path, **kw))

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
        logger.info(f"{cfg.method} per-layer |J^T w| (pre-norm): " +
                    " ".join(f"{l}:{per_layer[l].norm():.3g}" for l in cfg.layers))
        return _to_vector(cfg, per_layer)

    def _steer_layers(self, layers) -> tuple[int, ...]:
        """None -> all fitted layers; else explicit int indices (float bands are
        a fit-time concept -- the lens doesn't know n_layers to resolve them)."""
        if layers is None:
            return tuple(self.lens.source_layers)
        if any(isinstance(l, float) for l in layers):
            # Claude: int() would silently truncate (0.5, 0.8) -> layer 0 and steer
            # the wrong layer; float bands only exist at fit time (external review).
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
        """EXPERIMENTAL: persona -> vocabulary bottleneck -> word pullback.
        Read each persona's final-layer mean through the unembedding, take the
        top-k tokens it most evokes, contrast the two token sets' unembedding
        rows, pull that back. Composes the persona signal with the verified
        word mechanism; untested for specificity."""
        lm = from_hf(model, tok)
        h_pos = _h_bar_final(model, tok, pos_prompts, batch_size=batch_size, label="pos")
        h_neg = _h_bar_final(model, tok, neg_prompts, batch_size=batch_size, label="neg")
        W_U = model.lm_head.weight                                # [vocab, d]
        cots = {}
        for name, h in (("pos", h_pos), ("neg", h_neg)):
            logits = lm.unembed(h.to(model.device).to(model.dtype)).float()
            top = logits.topk(k)
            toks = [tok.decode([i]) for i in top.indices.tolist()]
            logger.info(f"persona_topk {name} top-{k}: {toks}")   # read your data:
            # gibberish/punctuation here means the persona mean is off-manifold
            # Claude: asymmetry is intentional -- token SELECTION goes through the
            # full logit pipeline (final norm) above, but the cotangent uses raw
            # W_U rows to match _word_cotangent's verified convention.
            cots[name] = W_U[top.indices].float().mean(0).cpu()
        cfg = JacobianPersonaTopkC(layers=self._steer_layers(layers))
        return self.pullback(cots["pos"] - cots["neg"], cfg)

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
                  position: int = -1) -> list[tuple[str, float]]:
        """What the model 'thinks' at `layer`: transport the residual to the
        final basis with J_l and decode. jlens's native use, handy in demos."""
        lm = from_hf(model, tok)
        lens_logits, _, _ = self.lens.apply(lm, prompt, layers=[layer],
                                            positions=[position])
        top = lens_logits[layer][0].topk(k)
        return [(tok.decode([i]), float(v)) for i, v in
                zip(top.indices.tolist(), top.values.tolist())]
