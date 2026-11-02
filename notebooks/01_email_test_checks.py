# %% [markdown]
# # 01 · The e-mail test: what could it detect, and can it be trusted?
#
# | | |
# |---|---|
# | **Question** | Before looking at any results: was the test big enough to answer the question, and did the randomization work? |
# | **Data** | Kevin Hillstrom's e-mail test (MineThatData, 2008): 64,000 customers who bought in the past 12 months, split at random into three equal groups: a men's merchandise e-mail, a women's merchandise e-mail, or no e-mail. Visits, purchases and spend were recorded for two weeks. |
# | **Answer** | The test can be trusted: each group got a third of the customers, the groups look alike on everything known before the send, and no row is impossible. It is big enough to catch a clearly profitable e-mail, but could miss one that only just pays for itself. |
# | **Next** | [Notebook 02](02_email_test_readout.ipynb) reads out what each e-mail did. |
#
# **Why this comes first.** A test whose groups were not split the way they should have been
# gives a wrong answer however carefully it is analysed. So these checks run before any effect is
# looked at, and the analysis plan
# ([`reports/hillstrom_analysis_plan.md`](../reports/hillstrom_analysis_plan.md)) fixes them in
# advance: a failure here stops the readout.
#
# **Sections**
#
# 0. **Design:** the smallest effect this test could detect, and the size it needed.
# 1. **Sample ratio:** did each group get a third of the customers?
# 2. **Balance:** do the groups look alike on everything known before the send?
# 3. **Data:** missing values and impossible combinations (spend without a purchase, a purchase
#    without a visit).
# 4. **A first look** at the outcomes, for orientation only (the readout is notebook 02).
#
# **Key terms**
#
# - **Arm:** one of the three randomized groups (`mens`, `womens`, `none`).
# - **Sample ratio mismatch:** groups that are not the size the randomization intended; the most
#   common sign of a broken test.
# - **Standardized mean difference (SMD):** the gap between two groups' averages, measured in
#   standard deviations. Under 0.05 is a negligible gap.
# - **Minimum detectable effect (MDE):** the smallest true effect the test would detect 80% of
#   the time.
# - **Break-even lift:** the extra spend per customer at which an e-mail pays for itself.

# %%
import sys
from pathlib import Path


def _root() -> Path:
    here = Path.cwd().resolve()
    for p in (here, *here.parents):
        if (p / "src" / "data" / "hillstrom.py").exists():
            return p
    raise FileNotFoundError("run from inside the project folder")


ROOT = _root()
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from src import records
from src.analysis import readout
from src.data import hillstrom

df, SOURCE = hillstrom.load()
raw = hillstrom.find_raw()
print(f"source: {SOURCE}; {len(df):,} customers"
      + (f"; file {raw.name}, sha256 {hillstrom.sha256(raw)[:16]}..." if SOURCE == "real" and raw else ""))
df.head()

# %% [markdown]
# ## 0. Design: what could this test detect?
#
# A test should be sized before it runs, for the smallest effect worth acting on. This one was
# run in 2008, so this is a design review after the fact: baselines and spreads come from the
# no-e-mail arm. (Two-sided 5% test, 80% power, equal arms.)
#
# The effect worth acting on comes from the decision rule in notebook 02: with a 40% margin and
# $0.10 to send an e-mail, an e-mail pays for itself if it lifts spend by at least
# $0.10 / 0.40 = **$0.25 per customer**.

# %%
MARGIN, COST = 0.40, 0.10                              # the plan's assumptions (used again in notebook 02)
ctrl = df[df["arm"] == "none"]
n_arm = int(df["arm"].value_counts().min())
worth = {"visit": 0.01, "conversion": 0.002, "spend": COST / MARGIN}   # +1 pt, +0.2 pt, break-even spend
design = []
for m, w in worth.items():
    sd = float(ctrl[m].std())
    design.append({"metric": m, "baseline": float(ctrl[m].mean()), "sd": sd,
                   "mde_at_this_size": readout.mde(sd, n_arm), "effect_worth_detecting": w,
                   "n_per_arm_needed": readout.n_per_arm(sd, w)})
design = pd.DataFrame(design).set_index("metric")
d = design.loc["spend"]
design["n_per_arm_needed"] = design["n_per_arm_needed"].astype(int)
d = design.loc["spend"]
print(f"{n_arm:,} customers per arm: the test reliably detects a spend lift of ${d['mde_at_this_size']:.2f} per "
      f"customer; the break-even lift of ${d['effect_worth_detecting']:.2f} needs {int(d['n_per_arm_needed']):,} per arm. "
      + ("An e-mail that only just pays for itself could be missed; a clearly profitable one would not."
         if d["mde_at_this_size"] > d["effect_worth_detecting"] else "The test is big enough for the break-even lift."))
design.round(4)

# %% [markdown]
# **How to read the table.** One row per metric. `mde_at_this_size` is what this test can
# reliably detect with the customers it had; `effect_worth_detecting` is the smallest effect
# worth acting on; `n_per_arm_needed` is how many customers per group it would take to detect
# that. When the first is bigger than the second, the test could miss an effect that matters.
#
# Spend is the hard metric: a few large orders make it noisy, so it needs far more customers than
# visits do for an effect worth the same money. Notebook 02 checks how much regression adjustment
# on pre-send variables narrows the interval (the CUPED idea).

# %% [markdown]
# ## 1. Sample ratio
#
# The randomization put a third of the customers in each group. If the counts are further from
# equal thirds than chance allows (a chi-square test, flagged at p < 0.001), something went
# wrong between assignment and the data: a bug, a filter, lost records. A high p-value is what a
# healthy test looks like.

# %%
counts = {a: int((df["arm"] == a).sum()) for a in hillstrom.ARM_ORDER}
srm = readout.srm(counts)
print(f"arm counts {counts}; chi-square {srm['chi2']:.3f}, p = {srm['p_value']:.3f} -> "
      + ("passes" if srm["passed"] else "FAILS (p < 0.001): stop and investigate"))

# %% [markdown]
# **What this means.** The three counts are within about a hundred customers of each other, which is
# what chance alone produces; the check passes.

# %% [markdown]
# ## 2. Balance on what was known before the send
#
# The largest standardized mean difference between any two arms, for every pre-send variable
# (categories expanded to one indicator per level). The plan's threshold is 0.05; anything above
# it would be added to the regression adjustment (the planned adjustment already includes every
# variable below except `history_segment`, which is `history` cut into bands).

# %%
balance = readout.smd(df, ["recency", "history", "mens", "womens", "newbie", "zip_code", "channel",
                           "history_segment"])
worst = balance.iloc[0]
print(f"largest |SMD|: {worst.name} {worst['max_abs_smd']:.4f}; "
      f"{int((balance['max_abs_smd'] > 0.05).sum())} variable(s) above 0.05")
balance.head(8).round(4)

# %% [markdown]
# **How to read the table.** Each row is a variable known before the send; `max_abs_smd` is the
# largest gap between any two groups. Every value is far below 0.05, so the groups were alike
# before the e-mails went out, and any later difference can be put down to the e-mails.

# %% [markdown]
# ## 3. Data checks

# %%
checks = {
    "missing values": int(df[hillstrom.COLUMNS[:-4] + hillstrom.OUTCOMES].isna().sum().sum()),
    "negative spend": int((df["spend"] < 0).sum()),
    "spend without a purchase": int(((df["spend"] > 0) & (df["conversion"] == 0)).sum()),
    "purchase without spend": int(((df["spend"] == 0) & (df["conversion"] == 1)).sum()),
    "purchase without a visit": int(((df["conversion"] == 1) & (df["visit"] == 0)).sum()),
}
pd.Series(checks, name="rows")

# %% [markdown]
# **What this means.** Each line counts rows that would be impossible if the data were recorded
# correctly. All zeros: the outcomes are internally consistent.

# %% [markdown]
# ## 4. What the outcomes look like
#
# Shown for orientation only, by arm. Spend is mostly zeros with a long tail, which is why the
# plan caps it at the 99th percentile of positive spend, computed on all arms pooled.

# %%
cap = readout.spend_cap(df["spend"])
pos = df.loc[df["spend"] > 0, "spend"]
profile = {"buyers": int((df["spend"] > 0).sum()), "share_buying": float((df["spend"] > 0).mean()),
           "spend_per_buyer_median": float(pos.median()), "spend_per_buyer_max": float(pos.max()),
           "cap_p99_positive": cap, "buyers_above_cap": int((df["spend"] > cap).sum())}
print({k: round(v, 4) if isinstance(v, float) else v for k, v in profile.items()})

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
axes[0].hist(pos.clip(upper=pos.quantile(0.995)), bins=60, color="#5B8FF9")
axes[0].axvline(cap, color="#2B2B2B", ls="--", lw=1)
axes[0].set(title="Spend among buyers", xlabel="dollars in the two weeks", ylabel="buyers")
axes[0].annotate(f"cap ${cap:,.0f}", (cap, axes[0].get_ylim()[1] * 0.9), xytext=(-6, 0),
                 textcoords="offset points", ha="right", fontsize=9)
top = balance.head(10)[::-1]
axes[1].barh(top.index, top["max_abs_smd"], color="#5AD8A6")
axes[1].axvline(0.05, color="#2B2B2B", ls="--", lw=1)
axes[1].set(title="Balance: largest |SMD| between arms", xlabel="standardized mean difference")
for ax in axes:
    ax.grid(alpha=0.25, axis="x")
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
records.save_figure(fig, "01_email_test_checks.png", SOURCE)
plt.show()

# %% [markdown]
# **How to read the figure.** Left: most buyers spend under a few hundred dollars, with a long
# tail; the dashed line is the cap, so a handful of very large orders cannot swing the result.
# Right: the balance check from section 2 as bars; every bar is far left of the dashed 0.05 line.
#
# **Raw outcomes by group.** The averages below are only a first look: they ignore the spend
# cap, the regression adjustment and the uncertainty, all of which notebook 02 handles.

# %%
raw_by_arm = df.groupby("arm", observed=True)[["visit", "conversion", "spend"]].mean().reindex(hillstrom.ARM_ORDER)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
for ax, col, title, fmt in zip(axes, raw_by_arm.columns, ["Visit rate", "Purchase rate", "Spend per customer"],
                               ["{:.1%}", "{:.2%}", "${:.2f}"]):
    bars = ax.bar(["No e-mail", "Men's", "Women's"], raw_by_arm[col], color=["#BFBFBF", "#5B8FF9", "#5AD8A6"])
    ax.bar_label(bars, labels=[fmt.format(v) for v in raw_by_arm[col]], fontsize=9, padding=2)
    ax.set(title=title)
    ax.margins(y=0.18)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
fig.suptitle("Raw averages by group, two weeks after the send (no adjustment, no intervals)",
             x=0.01, ha="left", fontsize=11)
fig.tight_layout()
plt.show()
raw_by_arm.round(4)

# %%
record = {"n": len(df), "design": design.reset_index().to_dict(orient="records"), "srm": srm, "balance_max_abs_smd": balance["max_abs_smd"].round(6).to_dict(),
          "balance_above_0_05": int((balance["max_abs_smd"] > 0.05).sum()), "data_checks": checks,
          "spend_profile": profile}
print("wrote", records.write("01_email_test_checks", record, SOURCE).name)

failed = [k for k, v in checks.items() if v and k != "missing values"]
verdict = (f"Sample ratio p = {srm['p_value']:.3f} ({'passes' if srm['passed'] else 'FAILS'}); "
           f"largest imbalance |SMD| {worst['max_abs_smd']:.3f} ({worst.name}); "
           + ("no impossible rows." if not failed else f"problems: {', '.join(failed)}."))
print(verdict)

# %% [markdown]
# ## Takeaways
#
# - **The test can be trusted.** Group sizes, balance and data checks all pass, so the readout
#   can go ahead as planned.
# - **It is only just big enough.** The test can detect a clearly profitable e-mail, but an
#   e-mail that only just pays for itself could be missed. Keep that in mind if an effect comes
#   out near the break-even line.
# - **Spend needs care.** It is mostly zeros with a long tail, so it is capped before analysis.
#
# **Next:** [notebook 02](02_email_test_readout.ipynb), the readout.
