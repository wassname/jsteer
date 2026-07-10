# jsteer

Steer a language model by pulling concept directions back through its [Jacobian](https://github.com/anthropics/jacobian-lens).

Load the model's full per-layer Jacobian once (the authors publish n=1000 lenses
on the Hub, or fit your own); after that every steering vector is a CPU matvec.
Name the words you want more or less of, get a steering vector, and generate
inside a `with` block:

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

Load the authors' pre-fitted n=1000 lens from the Hub (raw Salesforce-wikitext,
zero local compute). From the repo root:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import config
from jsteer import Jacobian, show_steer

MODEL = "Qwen/Qwen3.5-4B"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()

jac = Jacobian.from_pretrained(config.LENS_REPO, filename=config.hub_lens_file(MODEL),
                               revision=config.LENS_REVISION)
band = jac.steer_band(model)                       # steer the mid-depth 0.3-0.9 band
v = jac.word_vector(model, tok, ["happy", "joy"], layers=band)

# generate through the chat template with thinking on; print, per strength C,
# the j-space readout + the <think> trace + the answer.
show_steer(jac, model, tok, v, "Describe how your week has been going.", Cs=(0, 0.5, 1.5))
```

For a model the authors do not publish, fit your own (expensive, resumable):

```sh
uv run python scripts/fit.py --model <hf/model>
```

The coefficient is lens-dependent, so sweep it. The pre-fitted lens gives a clean,
concentrated direction, so its knee is steep: C~0.5 moves the tone while the text
and reasoning stay fluent, and by C~1 it degenerates into token spam.
`nbs/word_steering.ipynb` shows the full sweep with the j-space and `<think>` views.

## API

| call | status | what it does |
| --- | --- | --- |
| `Jacobian.fit(model, tok, prompts, layers=(0.3, 0.9))` | — | fit per-layer `J_l` (jlens; 1 forward + ~d_model/8 backwards per prompt, resumable) |
| `Jacobian.fit_cached(model, tok, prompts, path)` | — | load `path` if present, else fit and save it (idempotent build-or-load) |
| `jac.save(path)` / `Jacobian.load(path)` | — | fp16 cache on disk, jlens-compatible |
| `Jacobian.from_pretrained(repo, filename=, revision=)` | — | load the authors' pre-fitted lens from the Hub (or a local path) |
| `jac.steer_band(model, lo=0.3, hi=0.9)` | — | fitted layers in the mid-depth band; steer here (all-layer over-drives) |
| `jac.word_vector(model, tok, words, layers=band)` | verified | pull the words' unembedding direction back; +C says them more |
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
