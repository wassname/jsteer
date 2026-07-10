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

## Remaining work (task list has details; U-numbers from the plan)

1. uv scaffold: `uv sync` (torch cu121+ index if needed), fix any import errors
   fail-fast (no defensive fallbacks). LOCAL DEV: you may switch
   [tool.uv.sources] to path deps (../j-steer-dev/docs/vendor/jacobian-lens
   and ../../lite/steering-lite, editable) if the git fetches are slow --
   leave a comment saying which is active and why.
2. Smoke on Qwen/Qwen3-0.6B: tiny fit (8 short web-text prompts, mid layers,
   dim_batch 8), word_vector(["happy","joy"]), generate at C in {-8, 0, 8},
   print FULL first prompt + generations (token-efficient-logging skill).
3. U1 parity gate BEFORE demos: cos(Jacobian-cache pullback, word_vector_vjp)
   per layer > 0.999, same prompts/max_length/skip_first. If it fails, that is
   a bug in the wiring (the math is linear-identical), debug do not tune.
4. 0.6B real fit (~64 prompts) cached to artifacts/; 4B via pueue (label
   why:/resolve:).
5. notebooks/word_steering.ipynb (hello-world), persona_steering.ipynb
   (persona variants are EXPERIMENTAL -- they failed specificity controls in
   j-steer-dev; keep that framing), lens_readout.ipynb optional.
6. README: classic-repeng length, honest evidence section.

## Style

Fail fast, no defensive programming, loguru, no LLM-tell prose in README.
Comments marked as Claude-authored where opinionated.
