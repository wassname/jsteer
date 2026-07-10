# jsteer

Steer a language model by pulling concept directions back through its Jacobian.

Fit the model's full per-layer Jacobian once (expensive, cached to disk); after
that every steering vector is a CPU matvec. Name the words you want more or
less of, get a steering vector, and generate inside a `with` block:

```
v_l = unit( J_l^T @ w )
```

where `J_l = E_prompts[ d h_final / d h_l ]` is the position-averaged Jacobian
from [jlens](../j-steer-dev/docs/vendor/jacobian-lens) and `w` is a direction
in the final-layer basis naming the concept (for words: the mean unembedding
row). By linearity the cached pullback equals the direct per-prompt VJP
(`mean_p(J_p)^T w = mean_p(J_p^T w)`, parity-tested in
`artifacts/parity_u1.txt`), so caching costs nothing but fp16 rounding.

## Install

```sh
uv sync
```

Note: `[tool.uv.sources]` currently points at local editable checkouts of
jlens and steering-lite (see pyproject.toml for the paths); adjust if your
checkouts live elsewhere.

## Hello world

First build the Jacobian cache (a few minutes on a consumer GPU):

```sh
uv run python scripts/fit_qwen06b.py
```

Then, from the repo root:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jsteer import Jacobian

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16).to("cuda").eval()

jac = Jacobian.load("artifacts/qwen3-0.6b.jac")
v = jac.word_vector(model, tok, ["happy", "joy"])

enc = tok("I went to the park today and", return_tensors="pt").to("cuda")
for C in (-1, 0, 1):
    with v(model, C=C):
        out = model.generate(**enc, max_new_tokens=40, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    print(f"C={C:+d}:", tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True))
```

The coefficient is model-dependent: on this 0.6B model C around 1-2 moves the
tone while staying fluent, and C of 8 degenerates into literal "joyjoyjoy"
spam. `notebooks/word_steering.ipynb` shows the sweep.

## API

| call | status | what it does |
| --- | --- | --- |
| `Jacobian.fit(model, tok, prompts, layers=(0.3, 0.9))` | — | fit per-layer `J_l` (jlens; 1 forward + ~d_model/8 backwards per prompt, resumable) |
| `jac.save(path)` / `Jacobian.load(path)` | — | fp16 cache on disk, jlens-compatible |
| `jac.word_vector(model, tok, words)` | verified | pull the words' unembedding direction back; +C says them more |
| `jac.persona_vector(model, tok, pos, neg)` | experimental | pull back the personas' final-layer activation contrast |
| `jac.persona_topk_vector(model, tok, pos, neg, k=8)` | experimental | persona → top-k evoked tokens → word pullback |
| `jac.random_vector(seed=0)` | control | norm-matched random direction; honest demos beat this |
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
vector. They are shipped for experimentation, not as a recommendation
(`notebooks/persona_steering.ipynb` keeps this framing and includes a
mean_diff baseline).

## Credits

- [jlens](../j-steer-dev/docs/vendor/jacobian-lens): the Jacobian estimator
  and cache format (the researchers' verified code, wrapped, never
  reimplemented).
- [steering-lite](https://github.com/wassname/steering-lite): the runtime
  (`Vector`, attach/detach hooks, calibration).
- Shape of the library inspired by [repeng](https://github.com/vgel/repeng).
