# jsteer -- agent notes

Plan of record: /home/wassname/.claude/plans/review-specs-00-minimal-experiment-md-an-peppy-sky.md
Evidence base: ../j-steer-dev/docs/RESEARCH_JOURNAL.md (verified 3/5 word-steering result).

## What this is

repeng-style UX for Jacobian pullback steering. Core algo files are WRITTEN
(by the main agent, ported from verified j-steer-dev code -- do not rewrite
the math, it is parity-gated against the verified experiment):

- `jsteer/jacobian.py` -- Jacobian.fit/save/load (wraps jlens, the
  researchers' verified primary code) + word/persona/persona_topk/random
  vectors -> steering_lite.Vector.
- `jsteer/applies.py` -- steering-lite method registration + delivery modes.
- `jsteer/vjp.py` -- direct VJP path; the parity reference for the cache.

Runtime is steering-lite: `with v(model, C=8): model.generate(...)`.

## Status (shipped + verified)

- Core API built: `Jacobian.fit/save/load/from_pretrained` + word / persona /
  persona_topk / random vectors, one shared pullback path (`jacobian.py`);
  delivery modes (add / add_last / replace_last) in `applies.py`.
- Smoke (`scripts/smoke.py`) and any-model fit (`scripts/fit.py --model ...`,
  prompts from jlens's WikiText corpus) green; `config.py` holds slug/paths.
- U1 parity gate PASS -- cache pullback == direct VJP, cos > 0.999:
  `docs/evidence/parity_u1.txt`.
- U4 port check PASS -- jsteer VJP == run-524 reference vector, cos +1.0:
  `docs/evidence/u4_step2_vjp_parity.txt`.
- Notebooks: `word_steering` (verified), `persona_steering` (experimental --
  failed specificity controls in j-steer-dev, framing kept honest).
- README at classic-repeng length with an honest evidence section.

## Open

- U4 loop-close (`scripts/u4_step3_fit4b.py`): full 4B fit -> cached word
  vector must match the VJP and run-524 vectors (cos > 0.999). Resumable from
  `artifacts/qwen3-4b-authority.ckpt`; writes `artifacts/u4_loopclose.txt`.
- One-off validation scripts live in `scripts/scratch/` (u4_step1/2, parity_u1).
- TODO eval notebook: steer -authority, tinymfv fast (N=16, tokens=16, mfq-2)
  vs unsteered baseline.
- OPEN: steering-demo calibration is not method-comparable yet. To compare
  methods you want the same OFF-TARGET budget, then read the on-target effect.
  The current search finds each method's own "max coherent" edge, which does NOT
  equalize off-target -- but the CAUSE (checked in artifacts/steering_demo_results
  .json) is NOT that `rep` is insensitive. It is that the dual gate
  `rep<0.35 AND ans_mass>0.5` let `ans_mass` pre-empt: 11 of 14 edges stopped on
  `ans_mass<0.5` with rep still 0.00-0.02, nowhere near its gate. rep never got
  to fire, so we can't conclude it fails as a calibration axis.
  `ans_mass` is answer-commitment (confidence, same family as the rejected
  `pmass`), a READOUT-VALIDITY concern, not off-target coherence; folding it into
  the search is what broke comparability. (Interesting: ans_mass drops before rep
  rises -> the steer makes the model hedge/refuse the answer BEFORE its reasoning
  goes incoherent. Real effect, keep it as a per-row flag.)
  FIX (simpler than a graded measure): calibrate on `rep` ALONE (iso-rep budget,
  each method to rep~=0.35), demote `ans_mass` to a per-row "is P(YES) valid"
  flag out of the search, then compare on-target at iso-rep. If the readout is
  invalid at the rep-budget for a method, that is itself a finding.
  0.35 is anchored to the empirical coherent/degenerate gap (rep<~0.3 vs >~0.6,
  rep_metric_check.py), not to a base degradation; base C=0 rep=0.00, gap is wide
  so anything ~[0.35,0.55] gives the same edge.
  Caveat: the "rep non-monotone in C" anomaly (word -0.35 rep=1.0 vs -0.70
  rep=0.34) is UNCHECKED -- read the traces qualitatively; likely a short-trace/
  seed artifact, not real.

## Style

Fail fast, no defensive programming, loguru, no LLM-tell prose in README.
Comments marked as Claude-authored where opinionated.
