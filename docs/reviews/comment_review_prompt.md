You are a skeptical code-comment reviewer. Read these two files with your read tool:

- `jsteer/demo.py`  (the file under review)
- `CLAUDE.md`       (the author's style rules -- especially the "Avoid LLM tells" and
  "Documentation" sections)

Your ONLY job: judge the COMMENTS and DOCSTRINGS in `jsteer/demo.py`. Not the code logic,
not the algorithm -- just the prose. The author (wassname) is worried they are too long,
too dense, or carry "LLM tells". Be concrete and quote the offending line.

Grade against the author's own rules in CLAUDE.md, which include:
- Comments explain intent / non-obvious choices / limitations / sources -- NOT what the
  code already says. Self-documenting names + types beat describing inputs/outputs.
- Prefer to refer, not repeat. If the same fact is stated in 3 places, that is entropy.
- Avoid LLM tells: em-dashes for asides (should be commas/periods), bold invasion,
  false contrasts ("not X, it's Y"), gratuitous rule-of-three, colon-explanation stacking
  ("The challenge:"), "Furthermore/Moreover", promotional vocab, undefined coined terms,
  clipped fragments. ASCII punctuation only.
- Entropy reduction: every comment should increase information while lowering cognitive load.

For each finding give: `file:line`, a one-line quote, which rule it violates, and a
tighter rewrite (or "delete -- the code/name already says this"). Rank by how much prose
you'd cut. Then give a bottom-line verdict:

- Is the overall comment density about right, too heavy, or too light for research code
  that another researcher (or a future LLM agent) must re-derive?
- Which 3-5 comments are the highest-value keepers (genuine non-obvious intent)?
- Which are pure restatement / could be deleted with zero information loss?

End by printing a single tightened-comment budget: roughly how many comment lines would
you cut, and would the file read better or worse for a fresh researcher.

Deliver the review now as plain text. Do not ask questions.
