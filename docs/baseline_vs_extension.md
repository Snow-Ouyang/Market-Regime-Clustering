# Baseline vs Extension

This repository keeps two models on purpose.

## Reference benchmark: 3-state compact baseline

The baseline uses four variables:

- `growth_pc1`
- `inflation_pc1`
- `gs10`
- `term_spread_10y_1y`

Its main strengths are:

- parsimony
- balanced state shares
- relatively stable behavior under nearby penalties
- good interpretability for broad macro environments

For those reasons, the baseline remains a valuable benchmark specification.

## Preferred model: 4-state stress-aware extension

The extension adds one variable:

- `credit_spread`

and increases the state count to four while using a higher persistence penalty.

Its main strengths are:

- stronger crisis separation
- better historical narrative alignment
- a distinct Macro-Financial Stress regime
- richer asset-mapping interpretation

## Why the extension became preferred

The tradeoff is simple:

- the baseline is cleaner and more stable
- the extension is more informative and more historically aligned

The final project prioritizes the extension because the research objective is not just to produce a stable clustering, but to identify economically meaningful macro-financial regimes. The distinct stress regime is the key added value that the baseline does not provide clearly enough.

## Why the baseline remains

The baseline is retained because it anchors interpretation. It shows that the extension is not arbitrary: the final preferred model can be understood as a deliberate stress-aware refinement of an already sensible compact benchmark.
