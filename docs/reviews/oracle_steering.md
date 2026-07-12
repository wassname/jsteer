## blunt reviewer assessment

You are measuring model breakage and calling it robustness. The verdict isn't moving because your steering vectors are, at best, a mild static offset the model can write-about / work-around during the think trace, and at worst they are injecting junk that corrupts the autoregressive process before any decision can be read. The `ans_mass` collapse is the **primary signal**, not a nuisance to edge-gate out. Here is the cold read.

---

### 1. Robustness or breakage?

**Breakage dominates.** The facts on the ground:

- Under the only gating that keeps the answer slot meaningful (`ans_mass` ≥ 0.90-ish), every method except the word vector has a swing indistinguishable from the random control (random swing = 0.027, persona_soft = 0.045, meandiff = 0.040). That’s below the noise floor of a single-seed greedy measurement. You are measuring zero effect inside the valid regime.
- The moment a method actually produces a large P(YES) swing (word, swing 0.932) the answer slot collapses to 0.14 mass, the trace abandons YES/NO altogether, and promoted tokens are generic continuations. That is the model abandoning the task, not “moving the verdict.”
- The phrase “verdict is robust to a fixed residual offset” is misleading when the same offset obliterates task coherence just a hair past the calibration threshold. If the verdict were genuinely robust you’d see a plateau where P(YES) is unmoved but the answer stays crisp. Instead you see a cliff: answer dies first, then the model drifts. That cliff is fragility, not robustness.

Self-deception flag: you calibrated on `ans_mass` to avoid reading artifacts, but then you treat the near-zero swings inside that gate as evidence for deliberation robustness. The zero swing is equally consistent with “the steering vector simply doesn’t encode this decision at any safe strength.” You cannot distinguish “robust to steering” from “steering direction is off-target” on these numbers.

---

### 2. Why word works (incoherently) but persona-contrast fails entirely

The averaged Jacobian is a **position-agnostic, prompt-agnostic first-order approximation** of the model’s computation. What survives averaging over thousands of positions and many unrelated prompts?

- A crude `d_model`-wide concept direction that projects onto near-top singular vectors of `E[J]` — i.e., **directions the model would use everywhere anyway** (frequent syntactic / stylistic / high-level topic shifts). The “lie/deceive” unembedding contrast is exactly that: a broad semantic axis that correlates with many context-invariant features. It produces a big swing because it’s essentially steering the model’s entire stylistic register toward “covert / deceptive” output, which derails the think trace into rambling advice. The same effect would happen on *any* prompt; this is not verdict-specific.
- A context-specific persona shift (deceptive-character-prompt minus honest-character-prompt) is quantified as a **subtle activation displacement** that lives in a narrow subspace of the residual stream, one that depends strongly on position and on the particular prompt. Averaging the Jacobian over positions **smears that subspace into noise**. The residual stream at layer l does not have a single fixed direction to add that will reliably push the final-layer representation by `h_diff` across all tokens and prompts. Therefore `J_l^T w` for any persona-derived `w` is mostly orthogonal to the actual local computation paths for that decision, and you get noise-level swings.

In short: the word vector works *because it’s so generic it breaks the model*, not because it’s a valid verdict intervention. The persona vectors fail because the averaged Jacobian kills their specificity.

#### 2b. persona_vector vs. soft vs. pinv — they all tie at zero. What does that say?

If the algebra mattered, **persona_pinv** should win: it correctly asks “what δ makes `J_l δ ≈ h_diff`”, and `persona_soft` is at least a legitimate cotangent. The fact that all three are within 0.01–0.05 of each other and indistinguishable from **meandiff** (which uses no Jacobian at all) tells you:

> The Jacobian is adding no information over simply adding the layer-l mean difference directly.

The `J^T` or `pinv` mappings don’t preserve or focus the persona signal at the steered layers. meandiff is essentially a static bias that the model can incorporate into its current; the Jacobian variants are just a noisier static bias. The numbers say: drop the Jacobian, you’re not gaining anything for this task.

---

### 3. What would actually move a deliberated verdict?

The chain-of-thought (`thinking`) re-derives the answer from the prompt, and each token attends heavily to the prompt and previous tokens. A fixed per-position residual offset can be **integrated out** by the subsequent reasoning steps, or it can corrupt the computation so badly that coherence breaks before the verdict flips. To move the answer while keeping coherence:

- **(c) Intervene on the think trace** — this is the only approach that can causally redirect the deliberation without breaking the autoregressive flow. A fixed offset during thinking can be amplified by the model’s own computation. But you must do it carefully: the think trace is typically in a different format (e.g., JSON / special tokens). The steering could just cause the model to self-correct (“I know you’re pushing me to lie but I think…”). Still, ranked #1.
- **(a) Steer only the answer token** — much easier, but it only affects the mapping from final hidden state to logits. If the think trace already concluded a verdict, steering the answer token might force a mismatch that the model then tries to reconcile (potentially producing the collapse you already see). Ranked #2 for cheapness, but less likely to succeed.
- **(d) Accept that a single linear direction can’t move the verdict** — more honest than your current framing, but empirically open. Ranked #3 as a null hypothesis to test.
- **(b) Steer a later layer band** — later layers are closer to the answer; you’ll just break the output sooner, as your over-steer results already demonstrate. Last place.

Missing option: **steer the attention mechanism or a specific head** — a single residual addition to all positions is a blunt instrument. The model’s decision might be localized to a few heads at specific positions. If you must use linear steering, at least restrict it to the first answer token position and the layers immediately before it.

---

### 4. Single cheapest distinguishing experiment

**Remove the think trace entirely and force a zero-shot direct answer.** Prompt: “Answer only YES or NO. Should you lie to your boss? YES or NO:” Then run the exact same steering sweep. If the swing becomes substantial *within the valid `ans_mass` budget*, the verdict isn’t robust; the CoT was buffering the static offset. If the swing remains near zero, then (a) your steering directions genuinely don’t encode the concept at the right places, or (b) the model’s decision boundary is highly non-linear in these directions. Either way, you stop blaming “deliberation robustness” for what might be a steering-construction failure.

Bonus control: in the no-think setting, check if the concept-token logits (lie/honest) actually move before the answer slot goes degenerate. If they don’t, your `w` is not reaching the relevant output dimensions even at the answer position.

---

### 5. Discarding `||J^T w||` is a catastrophic omission

Yes. By unit-normalizing, you are **injecting full-strength noise** when `||J^T w|| ≈ 0`. A near-dead pullback gets amplified to norm 1 and whacks the model regardless of whether it encodes anything. This explains:

- Why the persona directions, which likely have tiny `||J^T w||`, produce noise-level swings at best.
- Why over-steering rapidly breaks answer mass: the model is being kicked in essentially random directions per layer, with no guarantee they cohere into a concept.
- Why the random control looks non-trivially bad: injecting a unit random vector at every position is a strong perturbation; the fact that it doesn’t completely destroy the model at C=0.5 is interesting, but the swing it produces is still within your noise bounds.

Report `||J^T w||` per layer and per method. I predict the word method will show a *larger* (or at least non-negligible) raw magnitude than the persona methods, explaining its ability to derail the model. The persona methods will be near zero, confirming that the Jacobian smears them out. If `||J^T w||` is small, the “direction” you’re injecting is essentially `unit(ε)` — a random vector chosen by the numerical noise in your pseudo-inverse / pullback.

**Suggestion:** before unit-normalizing, threshold on `||J^T w||`. If it’s below some noise floor, that layer gets no injection (or a scaled-down injection). That will force the methods to stand on signal alone, not on amplified junk.

---

### summary of what you’re fooling yourselves about

- Calling the answer-mass collapse a “gating” issue rather than the main effect. The steering kills the task before it moves the answer. That is not a trade-off; it’s failure.
- Treating a null result (0.04 swing with 0.03 random control) as evidence for “robust deliberation” rather than evidence that your mapping yields noise.
- Trusting the averaged Jacobian to transport a context-dependent persona signal when every basic sanity check (meandiff tie, random-token promotions, zero `||J^T w||`) indicates it’s not working.
- Discarding the one diagnostic that would immediately show why the persona methods are floundering: the pre-normalization magnitude.

The cheapest path forward: measure `||J^T w||`, run the no-think zero-shot variant, and see if the concept tokens ever light up at the answer slot under any safe C. If they don’t, the whole apparatus is probing stretchy glue, not a decision.
---

## Triage (Claude, scout-mindset -- agree/disagree with reasons)

The oracle (deepseek-v4-pro, single seed, no repo access) is largely right and sharpens
the framing. Where I agree, disagree, and what I'll act on:

- **ADOPT: ans_mass collapse is the primary signal, not a nuisance.** Agree. Reframing the
  headline from "verdict is robust" to "steering breaks the answer before it moves it, and
  inside the valid budget the effect is noise-level." The notebook qualitative cell already
  says this; I strengthened it.
- **ADOPT (strongest): the `||J^T w||` pre-norm magnitude is the missing diagnostic.** Agree,
  and it is *already computed* -- `Jacobian.pullback` logs per-layer `|J^T w|` at DEBUG
  (jacobian.py:240). We just discard it by unit-normalizing. Cheap to surface. The promoted
  tokens being function-word/cross-lingual junk for every method is independent evidence the
  injected direction is off-target, which tilts me past the oracle's "can't distinguish"
  toward "the persona directions are near-dead and normalization amplifies noise."
- **ADOPT (cheapest decisive test): the no-think zero-shot sweep.** Agree this is the single
  experiment that separates "CoT buffers the offset" from "direction is off-target." Queued
  as the next run.
- **PARTIAL: word "works only because it's generic breakage".** Half-agree. On THIS verdict
  readout, yes -- word's swing is a dead-answer artifact. But word_vector is the one method
  verified in j-steer-dev to beat a norm-matched random control on 3/5 moral foundations with
  a *rating* readout (tinyMFV). So the failure is readout-specific: a fixed offset can move a
  0-9 rating but not a deliberated YES/NO. The oracle lacked that context (my brief
  under-stated it). This is itself a finding: verdict readouts are harder than ratings.
- **ADOPT: meandiff ties the Jacobian variants -> the Jacobian adds nothing here.** Agree for
  this task/readout. Caveat: it is not globally worthless (see word-on-ratings above); it is
  worthless for persona-contrast on a verdict.
- **DEFER (measure before changing): threshold on `||J^T w||` before normalizing.** Plausible,
  but that changes the method. First MEASURE the magnitudes across methods (predicted: word
  non-negligible, personas ~0); only then decide whether to threshold. Don't fold an untested
  gate into the extractor while the instrument is still single-seed noisy.
- **ADOPT (ordering): intervene-on-think-trace (c) > answer-token (a) > accept-null (d) >
  later-layers (b); plus steer specific heads/positions.** Agree with the ranking and the
  blunt-instrument critique of all-position addition.

### Concrete next steps (in priority order)

1. Report per-method per-layer `||J^T w||` (pre-normalization) in the demo table -- surface
   the already-logged number. Predicted: word >> personas ~ 0. (folds into task #28's
   scale-invariant reporting)
2. No-think zero-shot P(YES) sweep (drop the `<think>` trace, force a direct YES/NO), same
   dual gate. Decides CoT-buffering vs off-target-direction.
3. Multi-seed the edge measurement (task #28) -- the readout_ok flag flickers at the 0.90
   boundary on single-seed noise (persona_soft 0.895 vs 0.90); a valid rank needs it.
