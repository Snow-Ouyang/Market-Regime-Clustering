# Results Interpretation

## Why the final model holds up

The final regime detector is a 3-state Jump Model built on four variables:

- `growth_pc1`
- `inflation_pc1`
- `gs10`
- `term_spread_10y_1y`

The project started from a much broader macro panel, but repeated sensitivity checks showed that a larger feature space did not improve the state structure. Variables such as `bog_amom`, `ur_diff`, `cp_amom`, `hs_amom`, and `realized_vol` either introduced extra noise or pushed the model toward collapsed or market-driven states. The four-variable specification was the smallest space that still retained a coherent macro interpretation.

Within this reduced feature space, a 3-state Jump Model with penalty `0.6` gave the best balance between:

- state shares that are not overly concentrated
- durations that are persistent but not excessively sticky
- state profiles that remain economically interpretable

At penalty `0.6`, the state shares are roughly `37.6% / 31.6% / 30.8%`, with mean durations of about `44.8 / 37.6 / 61.0` months. That is materially healthier than the earlier specifications that produced one dominant state plus a tiny short-lived fragment.

## Why some variables were excluded

The final feature space is deliberately narrower than the full macro panel.

- `growth_pc2` and `inflation_pc2` were useful during the earlier exploratory phase, but once the project moved to a cleaner public baseline, the first PCs carried the main cross-sectional signal and the four-variable version remained comparably stable.
- `bog_amom` improved some intermediate specifications but also made the final narrative less robust. Once the penalty grid was rerun on the four core variables, the simpler specification held up well enough to become the main version.
- `ur_diff` was particularly regime-distorting in the Jump Model setting. Adding labor changes often pushed the solution back toward a nearly degenerate state.
- `credit_spread` was more acceptable than `ur_diff`, but still degraded the clean three-way split relative to the core baseline.
- `cp_amom`, `hs_amom`, and `realized_vol` are useful as validation variables, but not as primary state inputs. In particular, `realized_vol` risked turning the model into a market-stress detector rather than a macro regime classifier.

## Economic meaning of the three regimes

Using the state profiles from the selected penalty:

### 1. Moderate Growth / Flat Curve Regime

- `growth_pc1`: modestly positive
- `inflation_pc1`: near neutral
- `gs10`: moderate
- `term_spread_10y_1y`: relatively flat

This is the most equity-friendly environment in the sample. It looks like a steady expansion regime rather than a high-pressure inflationary phase.

### 2. Low Inflation / Steep Curve Regime

- `growth_pc1`: weakest among the three states
- `inflation_pc1`: clearly lowest
- `gs10`: moderate
- `term_spread_10y_1y`: steepest curve

This regime is not simply “recession.” It mixes disinflationary and policy-easing environments with steeper curves and weaker macro momentum. It performs well for bonds and reasonably well for gold, but is poor for oil.

### 3. High Inflation / High Rate Regime

- `growth_pc1`: close to neutral
- `inflation_pc1`: highest
- `gs10`: highest by far
- `term_spread_10y_1y`: flatter than the steep-curve regime

This is the clearest inflation / rate regime in the final model. It maps naturally to tightening or inflation-scare environments, though it should not be interpreted as a literal “overheating” label in every month.

## Stability analysis

### Penalty stability

The local penalty grid over `0.50, 0.55, 0.60, 0.65, 0.70` showed that all candidates were admissible, but:

- `0.50` was somewhat more switchy
- `0.65-0.70` were noticeably stickier
- `0.55-0.60` gave the cleanest balance

Penalty `0.60` was retained as the baseline because it sat inside the stable middle of the grid and preserved a balanced three-state structure.

### Time stability

Trimmed-sample checks showed partial, not perfect, stability.

- Trimming the start of the sample (`3y`, `5y`, `10y`) preserved the three-state structure well.
- Trimming the end of the sample made the third state smaller and reduced balance.
- This suggests the model is directionally stable, but it is still somewhat sensitive to the post-2020 endpoint.

### State-count stability

Holding variables and penalty fixed:

- `2-state` is too coarse and mostly merges two of the final three environments.
- `4-state` starts splitting off a smaller extra state rather than revealing a clearly new macro regime.
- `3-state` remains the most natural tradeoff between interpretability and parsimony.

## External validation

The final model was not fit on `credit_spread`, `ur_diff`, `bog_amom`, `cp_amom`, `hs_amom`, or `realized_vol`, but these variables still help assess economic meaning.

Most informative validators:

- `credit_spread`: highest in the high-inflation / high-rate regime and lowest in the moderate-growth regime
- `ur_diff`: relatively weaker in the low-inflation / steep-curve regime
- `bog_amom`: highest on average in the low-inflation / steep-curve regime, although very noisy
- `realized_vol`: differs across regimes, but less sharply than `credit_spread`

This pattern supports the interpretation that the final states capture broad macro-financial environments rather than arbitrary clustering artifacts.

## Asset mapping

The four-asset mapping uses:

- S&P 500
- Oil
- Gold
- Long-duration bond proxy (`VUSTX`)

Key regime-level results:

- **Equities** do best in the Moderate Growth / Flat Curve regime.
- **Bonds** do best in the Low Inflation / Steep Curve regime.
- **Oil** is strongest in the Moderate Growth / Flat Curve regime and weakest in the Low Inflation / Steep Curve regime.
- **Gold** performs best in the Moderate Growth / Flat Curve and Low Inflation / Steep Curve regimes, and less strongly in the High Inflation / High Rate regime.

These outcomes reinforce the interpretation that the final states are more like macro-financial policy environments than textbook business-cycle bins.

## What is strongest about the project

The strongest contribution of the current project is not a new clustering algorithm. It is the disciplined narrowing of the variable space until the regime structure became:

- balanced
- interpretable
- reasonably stable
- externally valid
- asset-relevant

In other words, the main result is a **workable macro regime definition**, not just a fitted classifier.

## Next extensions

Natural next steps include:

- event-by-event regime chronology
- out-of-sample regime monitoring
- asset allocation overlays by regime
- transition-risk analysis around regime changes
- expanding the validation set without feeding all variables back into the state model

