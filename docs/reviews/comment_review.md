Note: `CLAUDE.md` was not present at the requested path (`ENOENT`). I applied the rules quoted in the request.

Ranked findings, largest cuts first:

1. `jsteer/demo.py:275`
   `"Per C: the steer-promoted tokens in a cowsay bubble, then the raw generation, all"`
   Rule: documentation repeats operational detail, uses colon-stacking, all-caps emphasis, and spends 18 lines narrating a function whose signature and body already establish most of it.
   Rewrite:
   `"""Render generations at supplied Cs, or find coherent anchors when Cs is None.`
   `Optionally report rubric score, repetition, and answer-token mass."""`

2. `jsteer/demo.py:83`
   `"Ask `rubric`, let the model think, then force `readout['prefix']` and read the"`
   Rule: overly dense docstring; repeats formulas, threshold semantics, and examples already documented beside the constants/readout definitions.
   Rewrite:
   `"""Generate a think trace, score the forced readout slot, and return`
   `(expected, repetition, answer_token_mass)."""`

3. `jsteer/demo.py:174`
   `"Walk C outward from 0 in +/- directions, scoring the rubric each step, and STOP a"`
   Rule: long prose restates the loop and return schema; caps emphasis and “Both are needed:” explanation stacking.
   Rewrite:
   `"""Sweep outward from zero in both directions until the first incoherent point.`
   `Each row averages n_samples generated traces."""`

4. `jsteer/demo.py:355`
   `"THE steering demo, one call for every notebook. For each named vector: a clear"`
   Rule: promotional/all-caps language (“THE”, “ONE”, “AND”), colon stacking, and a full UI walkthrough rather than a contract.
   Rewrite:
   `"""Run show_steer for each named vector and return summary and per-anchor rows.`
   `When rubric is set, summarize coherent-edge readouts across methods."""`

5. `jsteer/demo.py:1`
   `"Shared demo display: steer, generate through the chat template, show the"`
   Rule: module docstring is a dense implementation memo, repeats lower-level comments, and includes gratuitous attribution.
   Rewrite:
   `"""Shared steering-demo rendering for notebooks.`
   `Uses the chat template and preserves special tokens in displayed generations."""`

6. `jsteer/demo.py:41`
   `"# Coherence = repetition of the think trace, NOT a forced-object gate. Every steer"`
   Rule: false contrast, all-caps emphasis, “we observed” anecdote, and repeated rationale across the module.
   Rewrite:
   `# Use trace repetition as the coherence signal; forced-format compliance can miss long-generation loops.`

7. `jsteer/demo.py:49`
   `"# a point is coherent iff the reasoning trace is fluent (rep < REP_COHERENT_MAX AND long"`
   Rule: definition dump repeats `coherent_edge`, `coherence_sweep`, and `rubric_score`; all-caps emphasis and excessive examples.
   Rewrite:
   `# A coherent readout needs both a fluent trace and sufficient probability mass on answer tokens.`

8. `jsteer/demo.py:68`
   `"# a readout = (format suffix appended to the question, forced slot after </think>, the"`
   Rule: colon/explanation stacking and a prose specification of a dict whose keys make the structure clear.
   Rewrite:
   `# Readout formats define the forced answer slot and token-to-value mapping.`
   Or delete the comment and give the dicts a small typed dataclass if this explanation remains necessary.

9. `jsteer/demo.py:121`
   `"Find the STRONGEST coherent |C| in the `sign` direction via the Illinois method"`
   Rule: over-detailed algorithm narration in a public docstring; all-caps emphasis; repeats the inline comments and function name.
   Rewrite:
   `"""Find the largest coherent coefficient in one direction with Illinois false position.`
   `Returns 0.0 when the baseline is incoherent."""`

10. `jsteer/demo.py:157`
    `"Search both directions for the strongest coherent steer and return the anchor Cs to"`
    Rule: repeats the returned list literal and motivation in four lines; “every demo” is repository-level policy, not function documentation.
    Rewrite:
    `"""Return coherent negative, zero, and positive steering anchors, with optional midpoints."""`

11. `jsteer/demo.py:245`
    `"Rank of each AUTO-tracked token across every fitted layer, from jlens's"`
    Rule: seven-line plot tour repeats axis labels and implementation details; caps emphasis and dash-as-aside construction.
    Rewrite:
    `"""Plot jlens token ranks by layer. The final layer shows the model's true ranks."""`

12. `jsteer/demo.py:324`
    `"# steer-promoted tokens: top of (steered - base), word-like only. The subtraction"`
    Rule: valuable intent buried in four dense lines, repeated attribution, and a verbose historical aside (“the old lens...”).
    Rewrite:
    `# Rank word-like tokens by steered-minus-baseline logits to remove the shared think-opening prior.`

13. `jsteer/demo.py:297`
    `"# search at the SAME generation length as the demo -- coherence (repetition) is"`
    Rule: all-caps emphasis and dash-as-aside. The underlying limitation is worth keeping.
    Rewrite:
    `# Match the search and display lengths because repetition is length-sensitive.`

14. `jsteer/demo.py:253`
    `"# Noto CJK first (it also has Latin glyphs) so multilingual tokens like 巴黎 render"`
    Rule: useful compatibility rationale, but three lines explain matplotlib internals more than necessary and end with irrelevant “(Claude)” attribution.
    Rewrite:
    `# Put Noto CJK first so multilingual legend tokens render instead of tofu.`

15. `jsteer/demo.py:98`
    `"# this model ships no generation_config, so generate() is greedy by default: seeds"`
    Rule: useful caveat, but three lines over-explain the consequence and use dash-as-aside.
    Rewrite:
    `# Seeds affect this call only when sampling; greedy generation is deterministic.`

16. `jsteer/demo.py:334`
    `"# raw decode WITH special tokens: real <think>/</think>, <|im_end|> visible,"`
    Rule: repeats the module docstring’s stated display policy; caps emphasis and dash-as-aside.
    Rewrite:
    `# Preserve special tokens so the displayed generation matches model output.`
    Or delete: this is already visible in `skip_special_tokens=False`.

17. `jsteer/demo.py:340`
    `"# SHOULD rise with +C, fall with -C; flat => steer not moving this axis."`
    Rule: clipped fragment, all-caps pseudo-emphasis, and test-like expectation embedded as an implementation comment. It adds no implementation rationale.
    Rewrite:
    `delete -- this is an interpretation of output, not a code-maintenance comment.`

18. `jsteer/demo.py:314`
    `"# SHOULD: C=0 is the baseline; +C tilts the promoted tokens and tone toward the"`
    Rule: false certainty about experimental behavior, clipped phrasing, and a restatement of the displayed coefficient values.
    Rewrite:
    `delete -- output expectations belong in the notebook narrative or a test, not here.`

19. `jsteer/demo.py:230`
    `cbar.ax.axhline(REP_COHERENT_MAX, color="red", lw=1)   # the degeneration cutoff`
    Rule: pure restatement; the named constant and drawn line say this already.
    Rewrite:
    `delete -- the code/name already says this.`

20. `jsteer/demo.py:107`
    `forced = prompt + think + readout["prefix"]              # our own deterministic slot`
    Rule: restates the assignment and uses vague wording (“our own”).
    Rewrite:
    `delete -- the code/name already says this.`

21. `jsteer/demo.py:137`
    `while fb > 0 and abs(b) < max_C and evals < budget - 2:   # step out to bracket the edge`
    Rule: restates the loop condition.
    Rewrite:
    `delete -- the code/name already says this.`

22. `jsteer/demo.py:142`
    `if fb > 0:                        # coherent all the way to max_C`
    Rule: restates the branch condition and return value.
    Rewrite:
    `delete -- the code/name already says this.`

23. `jsteer/demo.py:262`
    `ax.invert_yaxis()                                    # rank 0 (top token) at the top`
    Rule: repeats standard plotting semantics and the nearby axis documentation.
    Rewrite:
    `delete -- the code/name already says this.`

24. `jsteer/demo.py:39`
    `"# then read the logprobs at a forced answer slot), reduced to a single scalar. (Claude)"`
    Rule: attribution without a source, plus an unnecessarily specific cross-project comparison. “(Claude)” is an LLM tell, not documentation.
    Rewrite:
    `# Read logprobs at a forced answer slot as a one-number steering sanity check.`

High-value keepers, after tightening:

- `jsteer/demo.py:297-298`: search and displayed generation lengths must match because repetition is length-sensitive.
- `jsteer/demo.py:324-327`: subtracting baseline logits removes the shared think-opening prior. This is genuine non-obvious measurement intent.
- `jsteer/demo.py:253-255`: Noto CJK ordering avoids tofu in multilingual legends. Keep the compatibility reason, not the matplotlib mini-essay.
- `jsteer/demo.py:98-100`: seeds only change generated traces under sampling. This prevents a misleading multi-seed experiment.
- `jsteer/demo.py:49-53`: coherence requires both fluent reasoning and answer-token mass. Keep this once, near the threshold constants, in the short rewrite above.

Pure restatement or zero-information-loss deletions:

- `jsteer/demo.py:107` — “our own deterministic slot.”
- `jsteer/demo.py:114` — “did it commit to an answer token?”
- `jsteer/demo.py:137` — “step out to bracket the edge.”
- `jsteer/demo.py:142` — “coherent all the way to max_C.”
- `jsteer/demo.py:230` — “the degeneration cutoff.”
- `jsteer/demo.py:262` — “rank 0 ... at the top.”
- `jsteer/demo.py:340-341` — the `SHOULD` interpretation and threshold recap.
- Every trailing `(Claude)` attribution (`:2`, `:39`, `:255`, `:327`) — these are neither sources nor maintenance-relevant provenance.

Bottom line: comment density is too heavy for research code. The file has real non-obvious choices worth documenting, but they are repeatedly explained in module docs, function docs, constant comments, and inline comments. That repetition increases entropy and makes the important limitations harder to find. The prose also has visible LLM tells: all-caps emphasis, dash-asides, false contrasts, formula dumps, and unsupported `(Claude)` tags.

Tightened-comment budget: cut roughly 65-80 comment/docstring lines, replacing them with about 25-35 focused lines. The file would read better, not worse, for a fresh researcher: the surviving comments would surface experimental assumptions and measurement limitations instead of narrating code already visible on the page.