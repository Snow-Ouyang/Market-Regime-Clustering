# Model Selection

This project started from a compact 3-state Jump Model baseline built on four macro variables: `growth_pc1`, `inflation_pc1`, `gs10`, and `term_spread_10y_1y`. That specification remains useful because it is parsimonious, balanced, and relatively stable under nearby penalties, sample trimming, and adjacent state counts.

The preferred model, however, is now the stress-aware 4-state Jump Model that adds `credit_spread`. The motivation for the extension was not to maximize stability at all costs, but to improve macro-financial narrative alignment and to separate broad crisis episodes from ordinary late-cycle or low-inflation environments.

## Why the baseline was not the final answer

The 3-state baseline works well as a benchmark, but it mixes several historically important stress episodes into broader macro states. In particular, episodes such as the 1973-1975 oil-shock recession, the Volcker disinflation double dip, the 1990-1991 credit-tightening recession, the 2008-2009 Global Financial Crisis, and the 2020 COVID shock are not cleanly isolated in the compact baseline.

That is acceptable for a benchmark model whose job is parsimony and robustness. It is less satisfactory for a final model whose goal is historical interpretation and macro-financial narrative richness.

## Why add credit spread

Among the variables tested outside the compact baseline, `credit_spread` was the most useful stress-sensitive extension. It was informative enough to help separate a distinct stress-like regime without causing the model to collapse into a purely noisy set of fragments. In contrast, other candidate variables often distorted the state structure more severely or shifted the model toward unstable monthly switching.

## Why 4 states

Moving from 3 states to 4 states is justified only in the stress-aware extension. In the compact baseline, the fourth state behaved more like a split of an existing macro environment. In the stress-aware specification, the extra state becomes economically meaningful: it isolates a smaller regime with weak growth, the widest credit spreads, the highest realized volatility, weak equity behavior, and relatively strong bond performance.

That additional state is what turns the model from a compact macro partition into a more interpretable macro-financial regime map.

## Why penalty = 1.0

The stress-aware extension was evaluated over a local penalty grid centered on moderate persistence. The preferred penalty, `p = 1.0`, was selected because it offered the most convincing middle ground between two failure modes:

- lower penalties produced more switchy and less coherent stress-state assignments
- higher penalties made the regime path too sticky and reduced narrative flexibility

The selected specification is therefore not the most stable possible model, but the one that provides the best tradeoff between persistence, stress separation, and historical narrative alignment.

## Final positioning

- **Preferred model:** 4-state stress-aware Jump Model with `credit_spread`, `p = 1.0`
- **Reference benchmark:** 3-state compact Jump Model with `p = 0.6`

The preferred model is the main interpretive result of the repository. The baseline remains in the project as a benchmark because it is cleaner, simpler, and useful for comparison.
