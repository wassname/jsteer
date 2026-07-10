## Code Review: demo/chat-template steering updates

### Summary
Adds shared demo rendering, chat-templated fitting, and refreshed notebooks/README. The pullback/VJP terminology is basically correct, but several comments/docs overclaim what the demo proves, and there are a couple of concrete runtime/API bugs.

### Important (should fix)
- `jsteer/demo.py:80` `tok.decode(..., skip_special_tokens=True)` likely removes Qwen3’s `</think>` token before `split_think()` runs, so the demo will not actually separate the `<think>` trace from the answer. Split on generated token IDs first, or decode with `skip_special_tokens=False` for parsing and strip special tokens afterward.

- `config.py:19-31`, `scripts/fit.py:1-8`, `README.md:33-35` overclaim “we fit J where we steer.” `jlens` averages over token positions; wrapping WikiText as a chat user message still means most fitted positions are user/document tokens, not assistant `<think>` generation tokens. This is a reasonable distributional move, but not an exact chat/thinking operating-point fit. Reword to “closer to the chat prompt distribution,” or fit on assistant/thinking continuations / position-select the assistant tokens if that exact claim is needed.

- `jsteer/demo.py:84` formats `C` with `+d`, so `show_steer(..., Cs=(-2.5, 0, 2.5))` crashes. Steering coefficients are naturally floats. Use `C={C:+g}` or similar.

### Suggestions
- `jsteer/demo.py:1-21`, `config.py:1-8`, `scripts/fit.py:1-9`, `nbs/word_steering.ipynb`, `nbs/persona_steering.ipynb`: comments/docstrings are much too explanatory for demo code. Trim the “why this exists,” “SHOULD/ELSE,” “Tufte,” and repeated Claude-authored rationale. Keep short usage notes and move caveats to README/docs if needed.

- `jsteer/demo.py:13-20`, `jsteer/jacobian.py:268`, notebooks: “literally what the model is thinking” is too strong. `lens_topk` is an approximate Jacobian-lens readout of the current activation under active hooks, using a fitted linear map. Under large steering it can be off-linearization. Reword to “lens readout” / “decoded linear readout,” not literal thought.

- `jsteer/demo.py:56-58`: using the model’s `generation_config` by default is defensible for Qwen demos, but the comments overgeneralize “greedy loops” as if universal. Also, stochastic sampling makes per-C comparisons noisy. Consider optional `generation_kwargs` and/or a seed for demo reproducibility.

### Positive
- `README.md` and `nbs/persona_steering.ipynb` clearly mark persona methods as experimental and mention the failed specificity controls.

### Verdict
REQUEST CHANGES  
Fix the `<think>` parsing bug and float-`C` crash, then trim/soften the overclaimed demo explanations before merging.