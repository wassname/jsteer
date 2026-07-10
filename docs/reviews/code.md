Now let me verify one more file — checking if there are tests or any additional files:Here is the full review.

---

## Code Review: jsteer — Jacobian pullback steering library

### Summary
This is a thin, well-structured wrapper around `jlens` (Jacobian fitting/caching) and `steering-lite` (hook runtime). Fit once, derive steering vectors nearly for free. The math is parity-gated (cos > 0.999 vs direct VJP), so the review focuses on wiring, device/dtype handling, silent-failure anti-patterns, and API misuse. Overall the code is clean and follows the project's "fail fast, no defensive programming" convention, but has a few sharp edges.

### Important (should fix)

- **`jsteer/vjp.py:54`** — `assert max(layers) < target_layer` is an assert, not a `ValueError`. Asserts are stripped when Python runs with `-O`, which would let through invalid layer configs and produce incorrect gradients (source layers at or past the target layer). Replace with:
  ```python
  if max(layers) >= target_layer:
      raise ValueError(f"source layers {layers} must be < {target_layer}; got max={max(layers)}")
  ```
  The convention of raising `ValueError` for this is established in `Jacobian.pullback` which does exactly that for missing layers.

- **`jsteer/jacobian.py:92-105`** (`_h_bar_final`) — No guard for empty `prompts`. If called with an empty list, `acc` stays `None` and `acc / n` raises `TypeError` (None / 0) rather than a clear error. Given that `persona_vector` and `persona_topk_vector` both call this with caller-supplied prompt lists, a well-meaning empty-pass is plausible. Add a check at the top:
  ```python
  if not prompts:
      raise ValueError("prompts must not be empty")
  ```

### Suggestions

- **`jsteer/jacobian.py:60-68`** (`_steer_layers`) — `tuple(sorted(int(l) for l in layers))` silently truncates float bands (e.g. `(0.5, 0.8)` → `(0, 0)`) rather than rejecting them. Float bands are only meaningful at fit time (`_resolve_layers` handles them), and `_steer_layers` is post-fit. A `ValueError` for float inputs would make the contract explicit and prevent a user from accidentally passing `layers=(0.5, 0.8)` and getting nonsense layers (0, 0).

- **`jsteer/jacobian.py:166-179`** (`persona_topk_vector`) — Calls `from_hf(model, tok)` on its first line, then `_h_bar_final` twice (which internally also calls `from_hf`). Three redundant `HFLensModel` constructions: three iterations over all params to freeze, three layout detections. Each `from_hf` is ~1-2 ms plus linear in param count, so for the 0.6B model it's invisible but for larger models it adds up. Extract `lm = from_hf(model, tok)` once and pass `n_layers` to `_h_bar_final` (or refactor `_h_bar_final` to accept a pre-made `lm`).

- **`jsteer/jacobian.py:96-98`** (`_h_bar_final`) — Uses `model(**enc)` (the full HF model including LM head forward) rather than `lm.forward(input_ids)` (residual stack only). The LM head computation is wasted work done for every batch. Under `@torch.no_grad()` the overhead is minor but inconsistent: `pullback_vjp` uses `model(**enc)` too (needs the full model because of hook placement), but `jlens.fit` correctly uses `lm.forward`. Would be cleaner to use `lm.forward` here since only residuals are needed.

- **`jsteer/jacobian.py:171,185-186`** (`persona_topk_vector`) — `cots[name] = W_U[top.indices].float().mean(0).cpu()` reads raw unembedding rows (no final norm), while `lm.unembed(...)` above goes through final norm to pick the top-k tokens. This is intentional (consistent with `_word_cotangent`'s raw-row convention and the docstring), but it does mean the "most evoked tokens" are selected via the full logit pipeline while the downstream cotangent uses the raw dueling basis. A single-line comment explaining the asymmetry would help future readers.

- **`jsteer/vjp.py:29-34`** (`_valid_mask`) — `mask & attention_mask.bool()` redundantly masks with the attention mask after already filtering by `pos < real_len - 1`. For standard HF right-padded batches these are equivalent, but the redundancy isn't harmful. Fine to leave, but a one-line comment that it's a belt-and-suspenders check would prevent a future reader from "simplifying" it and breaking left-padded or non-square attention mask scenarios.

- **`jsteer/jacobian.py:188`** (`random_vector`) — Generates directions on CPU without an explicit `dtype` argument. `torch.randn` defaults to `torch.float32`, which is correct. If this ever needs to match the model dtype (e.g. bf16), it would need updating.

- **`jsteer/applies.py`** — The `_extract_stub` and registration loop are clean but the docstring in `_extract_stub` could mention that `steering_lite.train` is the entry point being blocked. Currently the error message explains what to do, but a developer seeing "NotImplementedError: jsteer methods are extracted via Jacobian..." from inside `steering_lite.train()` might not immediately connect the dots. Minor.

### Positive

- **`jsteer/jacobian.py:109-117`** (`pullback`) — Pre-validates cotangent shape and layer membership with clear `ValueError` messages before touching tensors. Exactly the right fail-fast pattern.

- **`jsteer/vjp.py:62-64`** — The zero-valid-positions check catches short prompts early with a clear error, preventing silent zeros downstream.

- **`jsteer/applies.py`** — `apply_add_last` correctly degrades to `apply_add` when span ≥ sequence length (the slicing `y[:, :-k, :]` yields empty, `cat` reconstructs the full sequence). Documented and correct.

- **Sign convention consistency** — All three concept-method docstrings explicitly state what `+C` does, and the pullback computation (`w @ J_l`, i.e., `J_l^T @ w`) is consistent: `+C` enhances the named concept.

- **`_to_vector` layout** — The `stacked["v"].unsqueeze(0)` with `k=1` leading dim matches `steering_lite`'s `mean_diff` layout byte-for-byte, so calibration and serialization reuse the upstream code unchanged. This is the correct integration pattern.

### Verdict
**APPROVE** with minor fixes.

The `assert` → `ValueError` in `vjp.py:54` and the empty-prompts guard in `_h_bar_final` are the two changes worth making before shipping. Everything else is suggestions. The wiring is correct, the sign conventions are consistent, the jlens API is used properly (no reinvention of the estimator), and the parity gate confirms numerical equivalence.