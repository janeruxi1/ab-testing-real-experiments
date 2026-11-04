# %% [markdown]
# # 02 · The e-mail test: what did each e-mail do?
#
# | | |
# |---|---|
# | **Question** | Did each e-mail raise spend, and is either worth sending to every customer? |
# | **Data** | The same 64,000 customers as [notebook 01](01_email_test_checks.ipynb), whose checks all passed. |
# | **Answer** | Both e-mails raised spend, and the men's e-mail raised it more, by a margin that holds up after correcting for several comparisons. Under the plan's rule: **send the men's e-mail to everyone**, keep a small holdback group to keep measuring it, and test a new version against it next. |
# | **Next** | [Notebook 03](03_email_test_targeting.ipynb) asks whether different customers should get different e-mails. |
#
# **How the readout works.** Every choice below was fixed in the analysis plan
# ([`reports/hillstrom_analysis_plan.md`](../reports/hillstrom_analysis_plan.md)) before this
# notebook ran: the metrics, the spend cap, the covariates for regression adjustment, the three
# comparisons and their multiple-testing correction, and the decision rule.
#
# - **Primary:** spend per customer over the two weeks (intention to treat), capped at the 99th
#   percentile of positive spend.
# - **Secondary:** conversion rate and visit rate.
# - **Comparisons:** men's vs no e-mail, women's vs no e-mail, men's vs women's; Holm at 5% within
#   each metric.
# - **Adjustment:** Lin (2013) regression adjustment on the pre-send variables, HC2 errors; the
#   decision uses the adjusted estimate, the unadjusted one is shown beside it.
#
# **Key terms**
#
# - **Intention to treat:** everyone is counted in the group they were assigned to, whether or
#   not they opened the e-mail. It measures what *sending* the e-mail does.
# - **95% confidence interval (CI):** the range of effects consistent with the data. If it
#   excludes zero, the effect is unlikely to be chance.
# - **Holm correction:** three comparisons give three chances of a false alarm; Holm raises the
#   bar for each so the chance of any false alarm stays at 5%.
# - **Regression adjustment (the CUPED idea):** use what was known about each customer before
#   the send to remove some noise, which narrows the interval without biasing the estimate.

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
import numpy as np
import pandas as pd
from scipy import stats

from src import records
from src.analysis import readout
from src.data import hillstrom

df, SOURCE = hillstrom.load()
COVARIATES = ["history", "recency", "mens", "womens", "newbie", "zip_code", "channel"]   # fixed in the plan
MARGIN, COST = 0.40, 0.10                                                                # assumptions in the plan
COMPARISONS = [("mens", "none"), ("womens", "none"), ("mens", "womens")]
CAP = readout.spend_cap(df["spend"])
df["spend_capped"] = df["spend"].clip(upper=CAP)
print(f"source: {SOURCE}; {len(df):,} customers; spend cap ${CAP:,.2f}")

# %% [markdown]
# ## 1. The effects
#
# Each e-mail against no e-mail, and the two e-mails against each other, on every metric. Spend is
# shown capped (the primary) and uncapped.

# %%
METRICS = {"spend_capped": "Spend per customer, capped ($)", "spend": "Spend per customer, uncapped ($)",
           "conversion": "Conversion rate", "visit": "Visit rate"}
rows, effects = [], {}
for m in METRICS:
    adj = [readout.lin_adjusted(df, m, t, c, COVARIATES) for t, c in COMPARISONS]
    raw = [readout.diff_in_means(df, m, t, c) for t, c in COMPARISONS]
    holm = readout.holm(adj)
    effects[m] = {"adjusted": adj, "unadjusted": raw, "holm": holm}
    for a, r, h in zip(adj, raw, holm):
        rows.append({"metric": METRICS[m], "comparison": f"{a.treatment} vs {a.control}",
                     "control mean": a.mean_c, "adjusted diff": a.diff, "95% CI low": a.ci[0],
                     "95% CI high": a.ci[1], "relative": a.rel, "p": a.p_value, "Holm 5%": h,
                     "unadjusted diff": r.diff, "CI narrower by": 1 - a.se / r.se})
table = pd.DataFrame(rows)
table.round(4)

# %% [markdown]
# **How to read the table.** One row per metric and comparison. `adjusted diff` is the effect
# (for spend, in dollars per customer over two weeks; for rates, as a share, so 0.005 is half a
# point); the CI columns give its range; `Holm 5%` says whether it survives the correction for
# three comparisons. The figure in section 4 shows the same numbers as a picture.

# %% [markdown]
# ## 2. What regression adjustment bought
#
# The pre-send variables explain part of each customer's outcome, so removing that part narrows
# the interval without moving the estimate (the arms were randomized, so the adjustment should
# change the point estimate by little). The width of the interval is reported, and the
# equivalent extra sample: an interval that is narrower by a share `s` would otherwise need
# 1/(1-s)^2 times as many customers.

# %%
gain = table.groupby("metric", sort=False)["CI narrower by"].mean()
gain_sample = 1 / (1 - gain) ** 2 - 1
pd.DataFrame({"CI narrower by": gain, "equivalent extra sample": gain_sample}).round(3)

# %% [markdown]
# **What this means.** The adjustment barely narrows the intervals here: what was known about a
# customer before the send (recency, past spend, channel) says little about their spend in the
# next two weeks. It is still the planned estimate, and it costs nothing. The StreamFlix appendix
# shows a case where it helps a lot.

# %% [markdown]
# ## 3. The decision rule, applied
#
# From the plan:
#
# 1. An e-mail is worth sending to everyone only if its adjusted, capped spend lift over no
#    e-mail passes Holm and the lift times the margin (40%) exceeds the cost of sending ($0.10 an
#    e-mail).
# 2. If both pass, the higher one is chosen only if men's vs women's also passes Holm; otherwise
#    the memo says they cannot be told apart.
# 3. Targeting by customer is decided in notebook 03.

# %%
prim = {f"{e.treatment} vs {e.control}": (e, h) for e, h in zip(effects["spend_capped"]["adjusted"],
                                                                  effects["spend_capped"]["holm"])}
value = {}
for arm in ("mens", "womens"):
    e, h = prim[f"{arm} vs none"]
    value[arm] = {"lift": e.diff, "ci": list(e.ci), "holm_pass": bool(h),
                  "value_per_1000": readout.value_per_1000(e.diff, MARGIN, COST),
                  "value_per_1000_ci": [readout.value_per_1000(x, MARGIN, COST) for x in e.ci],
                  "breakeven_margin": readout.breakeven_margin(e.diff, COST),
                  "passes": bool(h and e.diff * MARGIN > COST)}
head, head_pass = prim["mens vs womens"]
passing = [a for a in ("mens", "womens") if value[a]["passes"]]
if not passing:
    decision = "none"
elif len(passing) == 1:
    decision = passing[0]
else:
    better = "mens" if head.diff > 0 else "womens"
    decision = better if head_pass else "either"
print({k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in value.items()})
print(f"men's vs women's: {head.diff:+.3f} [{head.ci[0]:+.3f}, {head.ci[1]:+.3f}], Holm pass {bool(head_pass)}")
print("decision under the plan's rule:", decision)

# %% [markdown]
# **The same decision as a picture.** The net value of e-mailing 1,000 customers (extra spend
# times the 40% margin, less $0.10 per e-mail), with its 95% interval. Above zero, the e-mail
# pays for itself. The men's e-mail clearly does; the women's e-mail probably does, but its
# interval reaches below zero.

# %%
fig, ax = plt.subplots(figsize=(7.5, 2.6))
for i, (arm, label, colour) in enumerate([("mens", "Men's e-mail", "#5B8FF9"), ("womens", "Women's e-mail", "#5AD8A6")]):
    v = value[arm]
    ax.plot(v["value_per_1000_ci"], [i, i], color=colour, lw=4, solid_capstyle="round")
    ax.plot(v["value_per_1000"], i, "o", color=colour, ms=9, mec="white", mew=1.5)
    ax.annotate(f"${v['value_per_1000']:,.0f}", (v["value_per_1000"], i), xytext=(0, 9), textcoords="offset points",
                ha="center", fontsize=9)
ax.axvline(0, color="#2B2B2B", lw=0.8)
ax.set_yticks([0, 1], ["Men's e-mail", "Women's e-mail"])
ax.set_ylim(-0.6, 1.6)
ax.invert_yaxis()
ax.set(xlabel="net value per 1,000 customers e-mailed ($, 95% CI)", title="Is each e-mail worth sending?")
ax.grid(alpha=0.25, axis="x")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Figure

# %%
LABEL = {"mens vs none": "Men's vs no e-mail", "womens vs none": "Women's vs no e-mail",
         "mens vs womens": "Men's vs women's"}
fig, axes = plt.subplots(1, 3, figsize=(14, 3.4), sharey=True)
for ax, m, unit in zip(axes, ["spend_capped", "conversion", "visit"], ["$ per customer", "points", "points"]):
    scale = 1 if m.startswith("spend") else 100
    for i, (a, r) in enumerate(zip(effects[m]["adjusted"], effects[m]["unadjusted"])):
        key = f"{a.treatment} vs {a.control}"
        ax.plot([a.ci[0] * scale, a.ci[1] * scale], [i, i], color="#5B8FF9", lw=3)
        ax.plot(a.diff * scale, i, "o", color="#5B8FF9", ms=7, mec="white", mew=1.5)
        ax.plot([r.ci[0] * scale, r.ci[1] * scale], [i + 0.22, i + 0.22], color="#BFBFBF", lw=2)
    ax.axvline(0, color="#2B2B2B", lw=0.8)
    ax.set_yticks(range(3), [LABEL[f"{t} vs {c}"] for t, c in COMPARISONS])
    ax.invert_yaxis()
    ax.set(title=METRICS[m].split(",")[0], xlabel=f"difference ({unit})")
    ax.grid(alpha=0.25, axis="x")
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("The e-mail test: effect of each e-mail over two weeks", x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
fig.text(0.99, 0.97, "blue: regression-adjusted 95% CI · grey: unadjusted", ha="right", va="top",
         fontsize=9, color="#555555")
records.save_figure(fig, "02_email_test_effects.png", SOURCE)
plt.show()

# %% [markdown]
# **How to read the figure.** Each panel is one metric; each row one comparison. The dot is the
# estimated effect and the blue bar its 95% interval; the thin grey bar above it is the interval
# without regression adjustment. A bar that does not cross the vertical zero line is an effect
# unlikely to be chance. Both e-mails clearly raise visits, purchases and spend; men's against
# women's (bottom row) is the closest call.

# %%
record = {
    "cap": CAP, "covariates": COVARIATES, "margin_assumed": MARGIN, "cost_per_email_assumed": COST,
    "effects": {m: [{**a.as_dict(), "holm_pass": bool(h), "unadjusted_diff": r.diff,
                     "unadjusted_ci": list(r.ci), "ci_narrower_by": 1 - a.se / r.se}
                    for a, r, h in zip(v["adjusted"], v["unadjusted"], v["holm"])] for m, v in effects.items()},
    "value": value, "head_to_head": {"diff": head.diff, "ci": list(head.ci), "holm_pass": bool(head_pass)},
    "decision": decision,
}
print("wrote", records.write("02_email_test_readout", record, SOURCE).name)

# %% [markdown]
# ## Verdict
#
# Rendered from the record written above.

# %%
DECISION_TEXT = {"none": "send nothing", "mens": "send the men's e-mail to everyone",
                 "womens": "send the women's e-mail to everyone",
                 "either": "send either e-mail; they cannot be told apart"}


def fmt(v):
    return f"{v['lift']:+.2f} per customer [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}]"


lines = [
    f"Men's e-mail: spend {fmt(value['mens'])}; worth ${value['mens']['value_per_1000']:,.0f} per 1,000 "
    f"customers e-mailed at a 40% margin (break-even margin {value['mens']['breakeven_margin']:.0%}).",
    f"Women's e-mail: spend {fmt(value['womens'])}; ${value['womens']['value_per_1000']:,.0f} per 1,000 "
    f"(break-even margin {value['womens']['breakeven_margin']:.0%}).",
    f"Men's against women's: {head.diff:+.2f} [{head.ci[0]:+.2f}, {head.ci[1]:+.2f}]; "
    + ("the difference passes Holm." if head_pass else "the two cannot be told apart."),
    f"Regression adjustment narrowed the spend intervals by {gain[METRICS['spend_capped']]:.0%} on average, "
    f"worth {gain_sample[METRICS['spend_capped']]:.0%} more customers.",
    f"Decision under the plan's rule: {DECISION_TEXT[decision]}.",
]
print("\n".join(f"{i}. {t}" for i, t in enumerate(lines, 1)))

# %% [markdown]
# ## 5. Rollout and follow-up
#
# The decision is to e-mail everyone, but the e-mail's value has to keep being measured: once
# everyone gets it, there is no comparison group left. The usual answer is a **permanent
# holdback**: a small random share of customers who never get the campaign e-mail.
#
# How small can it be? With 64,000 customers per send, the table shows the smallest spend lift
# each holdback size detects in one send, and how many sends must be pooled before it can detect
# **half** the lift measured here (an early warning that the e-mail is wearing out).

# %%
LIST = len(df)
ctrl_sd = float(df.loc[df["arm"] == "none", "spend_capped"].std())
z = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)
ARM = decision if decision in ("mens", "womens") else "mens"   # the e-mail being rolled out
target = value[ARM]["lift"] / 2
holdback = []
for share in (0.02, 0.05, 0.10, 0.20):
    n_c, n_t = LIST * share, LIST * (1 - share)
    per_send = z * ctrl_sd * np.sqrt(1 / n_c + 1 / n_t)
    holdback.append({"holdback_share": share, "customers_held_back": int(n_c), "mde_one_send": per_send,
                     "sends_to_detect_half_the_lift": int(np.ceil((per_send / target) ** 2)),
                     "margin_given_up_per_send": share * LIST * value[ARM]["lift"] * MARGIN})
holdback = pd.DataFrame(holdback)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
labels = [f"{s:.0%}" for s in holdback["holdback_share"]]
for ax, col, title, fmt in [
        (axes[0], "margin_given_up_per_send", "Cost: margin given up per send", "${:,.0f}"),
        (axes[1], "sends_to_detect_half_the_lift", "Speed: sends pooled to detect half the lift", "{:,.0f}")]:
    bars = ax.bar(labels, holdback[col], color=["#BFBFBF", "#5B8FF9", "#BFBFBF", "#BFBFBF"])
    ax.bar_label(bars, labels=[fmt.format(v) for v in holdback[col]], fontsize=9, padding=2)
    ax.set(title=title, xlabel="share of customers held back")
    ax.margins(y=0.15)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
fig.suptitle("Choosing a holdback: a bigger one costs more each send but spots a fading e-mail sooner (5% chosen)",
             x=0.01, ha="left", fontsize=11)
fig.tight_layout()
plt.show()

pick = holdback[holdback["holdback_share"] == 0.05].iloc[0]
print(f"a 5% holdback ({int(pick['customers_held_back']):,} customers) gives up about "
      f"${pick['margin_given_up_per_send']:,.0f} of margin a send and detects half the measured lift "
      f"after {int(pick['sends_to_detect_half_the_lift'])} sends pooled")
holdback.round(3)

# %% [markdown]
# **Guardrails to watch during rollout.** This data has none, but a real rollout would track
# unsubscribes and spam complaints per send (an e-mail that sells this week but loses the
# address costs every later send), deliverability, and whether e-mailing men's merchandise to
# women's-line buyers shifts their spending rather than adding to it.
#
# **The next test.** Men's against women's was the closest call. The natural next test is the
# men's e-mail against a new version, for example one that features the customer's own line.
# Sized with notebook 01's calculation, to detect a difference worth the break-even amount:

# %%
mens_sd = float(df.loc[df["arm"] == "mens", "spend_capped"].std())
next_test = {"effect_to_detect": COST / MARGIN, "n_per_arm": readout.n_per_arm(mens_sd, COST / MARGIN),
             "n_per_arm_for_head_to_head_seen_here": readout.n_per_arm(mens_sd, abs(head.diff))}
next_test["sends_needed"] = 2 * next_test["n_per_arm"] / LIST
print(f"two arms of {next_test['n_per_arm']:,} customers ({next_test['sends_needed']:.1f} sends of this list) "
      f"to detect a ${next_test['effect_to_detect']:.2f} difference in spend. That is more than notebook 01's "
      f"{readout.n_per_arm(ctrl_sd, COST / MARGIN):,}, because spend varies more among e-mailed customers "
      f"(sd ${mens_sd:.2f} against ${ctrl_sd:.2f}).")
rollout = {"holdback": holdback.to_dict(orient="records"), "next_test": next_test}
record["rollout"] = rollout
print("updated", records.write("02_email_test_readout", record, SOURCE).name)

# %% [markdown]
# ## 6. Two cross-checks: heavy tails and a Bayesian reading
#
# Added after the plan's readout (dated note in the plan); neither changes the decision. Their
# results go to a separate record, `02b_email_test_robustness.json`.
#
# **Heavy tails.** Spend is mostly zeros with a long tail, so the normal-theory interval could be
# off. A bootstrap (2,000 resamples of customers within each arm) gives an interval that makes no
# such assumption, for capped and uncapped spend. If the two intervals agree, the readout's
# intervals can be trusted.

# %%
rng = np.random.default_rng(20261104)
B = 2000
arm_values = {(a, m): df.loc[df["arm"] == a, m].to_numpy(float)
              for a in hillstrom.ARM_ORDER for m in ("spend_capped", "spend")}


def boot_means(x):
    return np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)])


boot_rows = []
for m in ("spend_capped", "spend"):
    means = {a: boot_means(arm_values[(a, m)]) for a in hillstrom.ARM_ORDER}
    for t_arm, c_arm in COMPARISONS:
        diffs = means[t_arm] - means[c_arm]
        welch = readout.diff_in_means(df, m, t_arm, c_arm)
        boot_rows.append({"metric": m, "comparison": f"{t_arm} vs {c_arm}", "diff": welch.diff,
                          "welch_low": welch.ci[0], "welch_high": welch.ci[1],
                          "boot_low": float(np.percentile(diffs, 2.5)), "boot_high": float(np.percentile(diffs, 97.5))})
boot_table = pd.DataFrame(boot_rows)
boot_table["widest_gap"] = np.maximum(abs(boot_table["boot_low"] - boot_table["welch_low"]),
                                      abs(boot_table["boot_high"] - boot_table["welch_high"]))
print(f"largest gap between a bootstrap and a normal-theory interval end: ${boot_table['widest_gap'].max():.3f}")
boot_table.round(3)

# %% [markdown]
# **A Bayesian reading of the same decision.** With a flat prior and this much data, the
# posterior for each lift is close to normal around the adjusted estimate with its standard
# error. That turns the readout into the questions a manager asks: how likely is each e-mail to
# pay for itself (lift above $0.25 a customer), how likely is the men's e-mail to be the better
# one, and what is the expected cost of choosing it if it is not (expected loss)? The
# [StreamFlix appendix](streamflix/04_bayesian.ipynb) does the full Beta-Binomial version.

# %%
breakeven = COST / MARGIN
bayes = {}
for arm in ("mens", "womens"):
    e, _ = prim[f"{arm} vs none"]
    bayes[arm] = {"p_pays_for_itself": float(stats.norm.sf((breakeven - e.diff) / e.se)),
                  "p_lift_positive": float(stats.norm.sf(-e.diff / e.se)),
                  "value_per_1000_90": [readout.value_per_1000(e.diff + z * e.se, MARGIN, COST)
                                        for z in (stats.norm.ppf(0.05), stats.norm.ppf(0.95))]}
h = head
p_mens_better = float(stats.norm.sf(-h.diff / h.se))
# expected loss of choosing men's: E[max(0, women's - men's)] in spend per customer, times margin per 1,000
z = h.diff / h.se
loss_spend = float(h.se * stats.norm.pdf(z) - h.diff * stats.norm.sf(z))
bayes["mens_vs_womens"] = {"p_mens_better": p_mens_better,
                           "expected_loss_per_1000_choosing_mens": 1000 * MARGIN * loss_spend}
print(f"P(men's e-mail pays for itself) {bayes['mens']['p_pays_for_itself']:.1%}; "
      f"P(women's pays) {bayes['womens']['p_pays_for_itself']:.1%}; "
      f"P(men's better than women's) {p_mens_better:.1%}; expected loss of choosing men's "
      f"${bayes['mens_vs_womens']['expected_loss_per_1000_choosing_mens']:.2f} per 1,000 e-mailed")
robust = {"bootstrap_draws": B, "bootstrap": boot_table.to_dict(orient="records"), "bayes": bayes}
print("wrote", records.write("02b_email_test_robustness", robust, SOURCE).name)

# %% [markdown]
# **How to read it.** The bootstrap and normal-theory intervals should differ by cents: at 21,000
# customers per arm the average is close to normal even for spend. The Bayesian numbers restate
# the decision: a probability near 1 that the men's e-mail pays, and an expected loss of well under
# a dollar per 1,000 e-mailed from choosing it over the women's e-mail, against a value of about
# $200 per 1,000. The women's e-mail probably pays too, but with real doubt.

# %% [markdown]
# ## Takeaways
#
# - **Both e-mails work; the men's e-mail works better.** It raises spend more than the women's
#   e-mail, and the difference survives the correction for three comparisons.
# - **Decision: send the men's e-mail to everyone**, following the rule written before the
#   results were seen.
# - **Keep measuring.** A 5% permanent holdback costs little margin per send and shows within a
#   few sends if the e-mail is wearing out.
# - **Next test:** the men's e-mail against a new version; it needs more customers than this
#   test had, because the difference worth detecting is small.
#
# **Next:** [notebook 03](03_email_test_targeting.ipynb), whether different customers should get
# different e-mails.
