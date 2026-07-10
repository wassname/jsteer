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

## Style

Fail fast, no defensive programming, loguru, no LLM-tell prose in README.
Comments marked as Claude-authored where opinionated.
