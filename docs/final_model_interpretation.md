# Final Model Interpretation

The final preferred specification is a 4-state Jump Model built on:

- `growth_pc1`
- `inflation_pc1`
- `gs10`
- `term_spread_10y_1y`
- `credit_spread`

with jump penalty `p = 1.5`.

## Regime definitions

### 1. Late-Cycle / Inflationary Flat Curve

This regime combines relatively flat curves with somewhat firmer inflation pressure and moderate long rates. It is not a pure crisis state; instead it looks like a late-cycle environment where macro conditions are still functioning but the curve is no longer steep and inflation pressure remains present.

### 2. Low-Rate / Steep Curve

This regime is characterized by lower long rates and the steepest term structure in the final model. It is the cleanest low-rate, recovery-like environment in the state space and is associated with relatively favorable bond performance.

### 3. High-Rate / Resilient Growth

This regime combines the highest rate levels with still-positive growth conditions. It captures restrictive-rate environments that do not immediately collapse into stress, which is why “resilient growth” is more accurate than a simple inflation or overheating label.

### 4. Macro-Financial Stress

This is the key contribution of the extension. The stress regime shows:

- the weakest growth proxy
- the widest credit spreads
- the highest realized volatility
- weak equity performance
- the strongest bond performance

It should not be interpreted as a mechanical recession classifier. Rather, it captures broad macro-financial stress episodes and funding/credit strain.

## Historical narrative alignment

The preferred model aligns more naturally with several major episodes:

- First Oil Crisis / 1973-1975 recession
- Volcker disinflation / 1980-1982 double-dip recession
- 1990-1991 recession / Gulf War / credit tightening
- Global Financial Crisis
- COVID shock

The point is not that every month of every recession falls into the stress regime. The better reading is that the model is sensitive to broad stress conditions and systemic macro-financial dislocation.

## Asset mapping

The asset mapping confirms that the regimes are not just statistical partitions:

- equities perform best in the high-rate resilient-growth and low-rate steep-curve states
- bonds perform best in the macro-financial stress regime
- oil is weakest in stress and strongest in the late-cycle inflationary regime
- gold is more mixed, but still shows regime-dependent differences

These are descriptive mappings rather than trading recommendations. They show that the preferred model carries economically meaningful cross-asset information.
