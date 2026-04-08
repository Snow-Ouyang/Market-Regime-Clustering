# Results Interpretation

This document provides a longer narrative summary of the repository after the final model selection was completed.

## Final positioning

The repository keeps two models:

- **Preferred model:** 4-state stress-aware Jump Model with `credit_spread`, penalty `p = 1.5`
- **Reference benchmark:** 3-state compact Jump Model, penalty `p = 0.6`

The benchmark remains valuable for parsimony and stability. The preferred model is the one used for final interpretation because it introduces a distinct **Macro-Financial Stress** state and aligns better with major historical episodes.

## Why the preferred model holds up

The preferred model expands the compact macro core just enough to separate stress from ordinary macro environments. The key addition is `credit_spread`, which makes the state space sensitive to funding and credit conditions without completely collapsing into a volatility-only classifier.

That produces four economically meaningful regimes:

1. **Late-Cycle / Inflationary Flat Curve**
2. **Low-Rate / Steep Curve**
3. **High-Rate / Resilient Growth**
4. **Macro-Financial Stress**

The fourth regime is the main reason the extension became preferred. It is smaller than the other three states, but it is not a trivial noise fragment. It is characterized by weak growth, wide credit spreads, high realized volatility, weak equity behavior, and strong bond performance.

## Why some variables were excluded from the preferred state space

The project tested many variables, but not all of them improved the regime structure.

- `bog_amom` helped some intermediate specifications, but did not improve the final tradeoff as much as credit spread did.
- `ur_diff` was more regime-distorting and often pushed the clustering toward less balanced solutions.
- `cp_amom`, `hs_amom`, and `realized_vol` were useful for validation, but were not attractive as core state inputs.

The final extension therefore uses a deliberately narrow macro-financial feature space rather than a large all-in panel.

## Baseline strengths vs extension strengths

The **baseline** is still strong because it is:

- parsimonious
- balanced
- relatively stable under nearby penalties and sample trims
- easy to interpret

The **extension** is preferred because it is:

- better aligned with major stress episodes
- better at separating crisis-like environments from ordinary macro states
- richer for historical narrative
- more informative for cross-asset interpretation

The choice between the two depends on the research objective. If the priority is stability and minimalism, the baseline is attractive. If the priority is historical macro-financial interpretation, the extension is better.

## Historical interpretation

The preferred model lines up more naturally with:

- the First Oil Crisis / 1973-1975 recession
- the Volcker disinflation / 1980-1982 double-dip recession
- the 1990-1991 recession / Gulf War / credit tightening
- the Global Financial Crisis
- the COVID shock

This does **not** mean that the model perfectly tracks recessions month by month. The more accurate reading is that it captures broad macro-financial stress regimes and distinguishes them from ordinary late-cycle, low-rate, or high-rate expansionary environments.

## Asset mapping

The preferred model is also more informative at the asset level:

- equities do best in the **High-Rate / Resilient Growth** regime
- bonds do best in the **Macro-Financial Stress** regime
- oil is strongest in the **Late-Cycle / Inflationary Flat Curve** regime and weakest in stress
- gold is strongest in the late-cycle inflationary regime and weaker in the high-rate resilient-growth regime

These results are descriptive. They should be read as conditional asset behavior under different macro-financial environments, not as a fully specified trading strategy.

## What is strongest about the project

The strongest contribution is not a new algorithm. It is the sequence of disciplined design choices that moved the project from:

- a broad macro panel
- to a compact stable baseline
- to a final stress-aware extension that is still interpretable and reproducible

In short, the repository now offers both:

- a **clean benchmark regime model**
- and a **preferred stress-aware final model**

That combination makes the project stronger than a one-model story.

## Next extensions

Natural next steps include:

- explicit episode-level scoring of regime alignment
- out-of-sample monitoring or rolling updates
- richer asset overlays
- transition-risk analysis around state changes
- comparing the preferred model with other persistence-aware state models
