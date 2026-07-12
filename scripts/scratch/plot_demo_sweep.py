# Claude: small-multiples dose-response for the 7-method demo, styled like
# word_steering's plot_sweep but read straight from the edge-find detail in
# steering_demo_results.json. y = P(YES), point colour = ans_mass (readout
# validity: bright = answer alive, dark = answer dying). red edge = readout
# invalid (ans_mass < 0.9*base). Single-seed edge-find, so no error bars.
import json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
d = json.load(open(ROOT / "artifacts" / "steering_demo_results.json"))
detail, summary = d["detail"], {r["method"]: r for r in d["summary"]}
BASE_PYES = 0.107  # P(YES)@C=0, shared across methods (oracle brief)

methods = list(detail)
fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharey=True, layout="constrained")
axes = axes.ravel()
for ax, m in zip(axes, methods):
    pts = sorted(detail[m], key=lambda p: p["C"])
    Cs = [p["C"] for p in pts]
    pyes = [p["ans"] for p in pts]
    am = [p["ans_mass"] for p in pts]
    edges = ["red" if not p["readout_valid"] else "0.2" for p in pts]
    ax.plot(Cs, pyes, "-", color="0.85", lw=1, zorder=1)
    ax.axvline(0, color="0.85", lw=0.8, zorder=0)
    ax.axhline(BASE_PYES, color="0.85", lw=0.8, ls="--", zorder=0)
    sc = ax.scatter(Cs, pyes, c=am, cmap="viridis", vmin=0.0, vmax=1.0,
                    edgecolor=edges, linewidth=1.4, s=70, zorder=2)
    s = summary[m]
    ax.set_title(f"{m}\nscore={s['score']:+.3f}  ok={s['readout_ok']}", fontsize=9)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("steering coefficient C", fontsize=8)
axes[0].set_ylabel("P(YES = lie)")
axes[4].set_ylabel("P(YES = lie)")
for ax in axes[len(methods):]:
    ax.set_visible(False)
cbar = fig.colorbar(sc, ax=axes.tolist(), label="ans_mass (readout validity)",
                    fraction=0.025, pad=0.01)
cbar.ax.axhline(0.90 * 0.56, color="red", lw=1)  # ~0.9*base_am floor (base_am~0.56)
fig.suptitle("Dose-response per method: P(YES) vs C, coloured by answer-mass "
             "(dark = answer dying). Dashed = baseline P(YES)=0.107. Single seed.",
             fontsize=11)
out = ROOT / "artifacts" / "steering_demo_sweep.png"
fig.savefig(out, dpi=110)
print("wrote", out)
