# jsteer research journal

Reverse-chronological. Entries by Claude with wassname. Claims link to evidence.

## 2026-07-12 -- measuring steering effect: coherence gates + edge search + a negative result

Context: making the demos quantitative and honest. wassname repeatedly (correctly) flagged
that the measurement was wrong; each fix exposed the next artifact.

### Instrument evolution (the readout kept fooling us)
- v1 `pmass` on a forced digit slot `{"ans": N}`: ~always 1 because the JSON prefix forces
  a digit. Blind coherence guard.
- v2 JSON object `{"ans","why","2+2"}`, gate on valid-parse + `2+2==4`: caught SOME
  breakdown but on a SHORT forced object that survives long-generation degeneration, so it
  over-credited (persona_vector scored rubric 9/9 while its actual text was wedding-jewelry
  loops).
- v3 repetition `rep = 1 - distinct-3` of the think trace (wassname's idea: "it's all
  repetition breakdown"): every steer breakdown is a repeat loop; threshold 0.35 from the
  empirical gap over 40+ real generations (coherent <0.3, degenerate >0.6). Simple + right.
- v4 DUAL gate after reading the dilemma traces: under hard steering the model emits a
  NON-answer token at the forced slot ('imers', 'lie', '信任', '(') or a 1-word stub, so the
  binary readout is meaningless there. Coherent now = fluent (rep<0.35, trace>=8 words) AND
  committed (ans_mass = full-vocab mass on the answer tokens > 0.5). ans_mass is the
  PRINCIPLED version of the pmass removed in v1: blind on a format-forcing slot, load-bearing
  on an open YES/NO slot where the model can decline to answer.
  Evidence: scripts/scratch/validate_traces.py; commits 6a080db, eba1ba4, a233e3a.

### Edge search (wassname: "use the Illinois method to find the edge within ~5 steps")
Fixed-step sweeps are too coarse to locate where coherence breaks. `coherent_edge()` brackets
a coherent/incoherent pair then does modified false-position (Illinois) to find the coherence
boundary in ~6 evals/side; `steer_anchors()` returns `[-C*, -C*/2, 0, +C*/2, +C*]`; every demo
(`show_steer(Cs=None)`) now shows the STRONGEST coherent steer both ways instead of hand-picked
Cs. Search must probe at the demo generation length (coherence is length-sensitive).
Evidence: commits 781b703, 2c17603; scripts/scratch/demo_edges.py.

### Result: steering moves tone, not a deliberated moral verdict
Self-honesty dilemma (say you were sick to avoid getting fired), readout P(YES=lie), honesty
axis (deceptive vs honest personas). At the SEARCHED strongest-coherent steer both ways,
P(lie) stays flat ~0.03-0.11 for every method (baseline 0.11):

    method        coherent -C* -> P(lie)   base   coherent +C* -> P(lie)
    persona_pinv  -0.14 -> 0.05            0.11   +1.0  -> 0.10
    word(lie)     -0.70 -> 0.08            0.11   +0.12 -> 0.03
    meandiff      -0.28 -> 0.05            0.11   +0.32 -> 0.09

Reading the persona_pinv +1.0 trace: still a balanced "no clear answer... not my place to
decide", verdict unchanged. Because we steered to the coherence edge, this is NOT dismissable
as "didn't push hard enough". Steering this axis changes tone/word-choice but not the
deliberated YES/NO on a hard dilemma. (n=1-2 seeds; ans_mass>0.5 and the answer tokens are a
knob; a harder/more-tempting dilemma or an axis-matched decision is untested.)
Evidence: scripts/scratch/measure_all.py, artifacts/measure_all.jsonl, task-35 edge demo.

### Method note (secondary axis)
persona_pinv has the widest coherent window on BOTH the optimism-tone axis and the honesty
dilemma -- the gentlest/most-robust extractor. persona_topk is a clean bidirectional TONE
steer (optimism) but breaks the answer format on the dilemma. word-vector directly promotes
tokens and is strong for tone, breaks fast on the deceptive direction.

### Meta-lesson
Read the actual generations, not the metric. The metric was wrong at four successive layers
and every time the fix came from reading the text (validate_traces.py, rep_metric_check.py).
"Excitement is evidence of bullshit" -- the big P(lie)=0.8-0.97 shifts were all artifacts.
