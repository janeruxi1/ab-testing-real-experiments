"""Draw the README's summary figure from the recorded results (no data needed).

    python scripts/make_readme_figure.py   ->  reports/figures/00_readme_summary.png
"""
import json
from pathlib import Path

# --- shared README figure style (same block in every repo) ---------------------------------
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29d"
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 12, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.facecolor": "white", "savefig.facecolor": "white",
    "xtick.major.size": 0, "ytick.major.size": 0,
})


def save(fig, path):
    fig.tight_layout(w_pad=3)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print("wrote", path.name)
# ---------------------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "reports" / "metrics"


def rec(name):
    return json.loads((M / f"{name}.json").read_text(encoding="utf-8"))


readout, target = rec("02_email_test_readout"), rec("03_email_test_targeting")
srm, win = rec("04_sample_ratio_confirmatory"), rec("05_winners_confirmatory")
spend = {e["comparison"]: e for e in readout["effects"]["spend_capped"]}
conf = target["confirmation"]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1.6, 1, 1]})

ax = axes[0]
rows = [("Men's e-mail vs none", spend["mens vs none"]), ("Women's e-mail vs none", spend["womens vs none"]),
        ("Men's vs women's", spend["mens vs womens"]),
        ("Targeting by segment vs men's for all\n(held-out customers)",
         {"diff": conf["gain"], "ci_low": conf["ci"][0], "ci_high": conf["ci"][1]})]
for i, (label, e) in enumerate(rows[::-1]):
    clear = e["ci_low"] > 0
    c = BLUE if clear else GREY
    ax.plot([e["ci_low"], e["ci_high"]], [i, i], color=c, lw=2.5, solid_capstyle="round")
    ax.plot(e["diff"], i, "o", ms=9, color=c, mec="white", mew=2)
    ax.text(e["ci_high"] + 0.04, i, f"{'+' if e['diff'] >= 0 else '−'}${abs(e['diff']):.2f}", va="center",
            color=INK, fontsize=10.5)
ax.axvline(0, color=MUTED, lw=1)
ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]])
ax.set_xlabel("extra spend per customer over two weeks ($, 95% interval)")
ax.set_xlim(-0.4, 1.3)
ax.grid(axis="y", visible=False)
ax.set_title(f"1 · Send the men's e-mail to everyone:\n+${spend['mens vs none']['diff']:.2f} per customer; targeting adds nothing")

ax = axes[1]
vals = [srm["flag_rate_inside"], srm["flag_rate_outside"]]
bars = ax.bar(["Tests in the\nbroken months", "All other\ntests"], [100 * v for v in vals], color=[BLUE, GREY],
              width=0.55)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, 100 * v + 2, f"{v:.1%}" if v < 0.1 else f"{v:.0%}", ha="center",
            fontsize=11)
ax.set_ylim(0, 100)
ax.set_ylabel("% of tests flagged by the check")
ax.grid(axis="x", visible=False)
ax.set_title("2 · A sample-ratio check finds\nthe broken randomization")

ax = axes[2]
w = win["winners"]
vals = [w["lift_a_mean"], w["lift_b_mean"]]
bars = ax.bar(["On the traffic\nthat picked it", "On fresh traffic"], [100 * v for v in vals], color=[GREY, BLUE],
              width=0.55)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, 100 * v + 1, f"+{v:.0%}", ha="center", fontsize=11)
ax.set_ylim(0, 100 * max(vals) * 1.25)
ax.set_ylabel("winning headline's lift in click-through")
ax.grid(axis="x", visible=False)
ax.set_title(f"3 · A winning headline keeps\n{w['survival']:.0%} of its lift on fresh traffic")

save(fig, ROOT / "reports" / "figures" / "00_readme_summary.png")


# ---- hero: is the e-mail worth sending, at any margin? --------------------------------------
cost = readout["cost_per_email_assumed"]
margin0 = readout["margin_assumed"]
fig, ax = plt.subplots(figsize=(11, 4.4))
ms = [m / 100 for m in range(0, 81)]
for key, lab, c in (("mens", "Men's e-mail", BLUE), ("womens", "Women's e-mail", GREY)):
    v = readout["value"][key]
    lo, hi = v["ci"]

    def val(m, lift):
        return 1000 * (m * lift - cost)

    ax.fill_between([100 * m for m in ms], [val(m, lo) for m in ms], [val(m, hi) for m in ms], color=c, alpha=0.15,
                    lw=0)
    ax.plot([100 * m for m in ms], [val(m, v["lift"]) for m in ms], color=c, lw=2.5)
    ax.plot(100 * v["breakeven_margin"], 0, "o", ms=9, color=c, mec="white", mew=2)
    ax.annotate(f"{lab}: breaks even at a {v['breakeven_margin']:.0%} margin", (100 * v["breakeven_margin"], 0),
                (2, 160) if key == "mens" else (27, -75),
                fontsize=10.5, arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
mv = readout["value"]["mens"]
ax.axvline(100 * margin0, color=MUTED, lw=1, ls=(0, (3, 3)))
ax.plot(100 * margin0, mv["value_per_1000"], "o", ms=11, color=BLUE, mec="white", mew=2)
ax.annotate(f"at the assumed {margin0:.0%} margin: \\${mv['value_per_1000']:,.0f} per 1,000 e-mailed\n"
            f"(95% interval \\${mv['value_per_1000_ci'][0]:,.0f} to \\${mv['value_per_1000_ci'][1]:,.0f})",
            (100 * margin0, mv["value_per_1000"]), (100 * margin0 + 4, mv["value_per_1000"] + 90), fontsize=10.5,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
ax.axhline(0, color=MUTED, lw=1)
ax.set_xlim(0, 80)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}".replace("$-", "−$")))
ax.set_xlabel("retailer's margin on the extra spend")
ax.set_ylabel("profit per 1,000 e-mailed, two weeks")
ax.set_title(f"Send the men's e-mail to everyone: it pays at any margin above {mv['breakeven_margin']:.0%} "
             "and beats the women's\ne-mail at every margin (shaded: 95% intervals; \\$0.10 per e-mail sent)")
save(fig, ROOT / "reports" / "figures" / "00_readme_hero.png")
