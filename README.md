# jsteer

Steer a language model by pulling concept directions back through its [Jacobian](https://github.com/anthropics/jacobian-lens).

Fit the model's full per-layer Jacobian once (expensive, cached to disk); after
that every steering vector is a CPU matvec. Name the words you want more or
less of, get a steering vector, and generate inside a `with` block:

```
v_l = unit( J_l^T @ w )
```

where `J_l = E_prompts[ d h_final / d h_l ]` is the Jacobian averaged over
prompts and positions (from [jlens](../j-steer-dev/docs/vendor/jacobian-lens))
and `w` is a cotangent: a direction in the final-layer basis naming the concept
(for words, the mean unembedding row). `J_l^T @ w` is the pullback of `w`, the
standard autodiff name for J-transpose applied to a cotangent. By linearity the
cached pullback equals the direct per-prompt VJP (vector-Jacobian product, the
same map computed in one backward): `mean_p(J_p)^T w = mean_p(J_p^T w)`,
parity-tested in [`docs/evidence/parity_u1.txt`](docs/evidence/parity_u1.txt),
so caching costs nothing but fp16 rounding.

## Install

```sh
uv sync
```

Note: `[tool.uv.sources]` points at local editable checkouts (see
pyproject.toml for the paths). steering-lite is public on GitHub. jlens is
NOT publicly fetchable at the time of writing; this repo depends on the copy
vendored in the j-steer-dev experiment repo, so without that checkout you
cannot install jsteer yet.

## Hello world

First build the Jacobian cache (any HF model; prompts are jlens's WikiText
wrapped in the model's chat template, closer to the distribution you steer in
than raw documents):

```sh
uv run python scripts/fit.py --model Qwen/Qwen3.5-4B
```

Then, from the repo root:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jsteer import Jacobian, show_steer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B", dtype=torch.bfloat16).to("cuda").eval()

jac = Jacobian.load("artifacts/qwen3.5-4b.jac")
v = jac.word_vector(model, tok, ["happy", "joy"])

# generate through the chat template with thinking on; print, per strength C,
# the j-space readout + the <think> trace + the answer.
show_steer(jac, model, tok, v, "Describe how your week has been going.", Cs=(-6, 0, 6))
```

The coefficient is model-dependent, so sweep it: a moderate +C moves the tone
while the text and reasoning stay fluent, and too large a |C| degenerates into
token spam. `nbs/word_steering.ipynb` shows the full sweep with the j-space and
`<think>` views.

## API

| call | status | what it does |
| --- | --- | --- |
| `Jacobian.fit(model, tok, prompts, layers=(0.3, 0.9))` | — | fit per-layer `J_l` (jlens; 1 forward + ~d_model/8 backwards per prompt, resumable) |
| `Jacobian.fit_cached(model, tok, prompts, path)` | — | load `path` if present, else fit and save it (idempotent build-or-load) |
| `jac.save(path)` / `Jacobian.load(path)` | — | fp16 cache on disk, jlens-compatible |
| `jac.word_vector(model, tok, words)` | verified | pull the words' unembedding direction back; +C says them more |
| `jac.persona_vector(model, tok, pos, neg)` | experimental | pull back the personas' final-layer activation contrast |
| `jac.persona_topk_vector(model, tok, pos, neg, k=8)` | experimental | persona → top-k evoked tokens → word pullback |
| `jac.random_vector(seed=0)` | control | norm-matched random direction, the baseline a concept vector has to beat |
| `jac.lens_topk(model, tok, prompt, layer)` | bonus | decode what the model "thinks" at a layer (full-J only) |

Vectors are plain `steering_lite.Vector` objects: `v.save(path)` /
`Vector.load(path)` (safetensors), `v.calibrate(...)` for iso-KL coefficient
calibration, `with v(model, C=...)` to steer.

## Evidence

Word-concept pullback is verified on exactly one setting: it beat a
norm-matched random control on 3 of 5 moral foundations (authority and
loyalty cleanly, fairness by mean) on Qwen3-4B with one eval harness, n=3
seeds. See the [j-steer-dev research
journal](../j-steer-dev/docs/RESEARCH_JOURNAL.md) for the runs. That is the
whole evidence base; treat other models and concepts as untested.

The persona variants failed specificity controls in the same experiments:
they steer generations, but no more selectively than an unrelated persona's
vector. They are shipped for experimentation only
(`nbs/persona_steering.ipynb` keeps this framing and includes a
mean_diff baseline).

## Credits

- [jlens](../j-steer-dev/docs/vendor/jacobian-lens): the Jacobian estimator
  and cache format, by the jacobian-lens authors (wrapped, never
  reimplemented). [Antropics works](https://github.com/anthropics/jacobian-lens)
  - Earlier work:
     - How to recover the latent process using Jacobians (Identifiability of nonlinear ICA): https://arxiv.org/pdf/2206.07751
     - How to handle dependent latents and assumption violations (again, through Jacobians): https://arxiv.org/pdf/2311.00866
     - For general latent variable models, what remains recoverable with guarantees, and why Jacobians are universally helpful? (We could generalize SAEs to the general nonlinear case, with Jacobians!): https://arxiv.org/pdf/2604.17568
- [steering-lite](https://github.com/wassname/steering-lite): the runtime
  (`Vector`, attach/detach hooks, calibration).
- Shape of the library inspired by [repeng](https://github.com/vgel/repeng).
