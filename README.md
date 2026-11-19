# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/EVM0x/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/EVM0x/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

For every GTA VI PS5 trophy, this entry forecasts Exophase's published earn rate at the fixed
horizon 2026-11-26T00:00Z — the share of tracked players holding the trophy — and emits it as five
quantiles (10%, 25%, 50%, 75%, 90%) in percent. Forecasts are scored by mean pinball loss computed
on the logit scale, so the quantiles are produced in logit space and mapped back rather than spread
linearly in percent: a trophy at 1% and a trophy at 80% are compared on how many log-odds they move.

## How it works

- **Refresh events.** The 10-minute read grid is collapsed: the first read is kept, then every read
  whose whole rate vector differs from the previous kept one — those are Exophase refreshes, not
  polling noise.
- **Slope.** Weighted least squares of logit(rate) on u(t) = ln(1 + hours since 2026-11-19T00:00:00Z),
  over the last 24 h with weights exp(-age / 24h); the weighted RMS residual is kept as model error.
- **Tier shrinkage.** A trophy's own slope is pulled toward the median slope of its tier (platinum,
  gold, silver, bronze) by a pseudo-count of 3, so a short or jittery series cannot dominate.
- **Extrapolation guard.** The shrunk slope is damped by exp(-0.35 · max(0, du - 1)) and the resulting
  logit step is capped at 1.5, so the ~4.5-day horizon cannot blow a forecast up.
- **Denominator nudge.** When players_tracked is growing (last count > 1.05× the first, at least 3
  points), a tanh-bounded share of its log-growth is subtracted from the step — a still-climbing rate
  is dragged down, because a bigger denominator dilutes it.
- **Spread.** sigma = sqrt(s0^2 + s0^2·du + resid^2) with s0 = 0.30 and the fit residual shrunk toward
  its tier's median and capped, then quantiles at z = (-1.645, -0.674, 0, 0.674, 1.645).

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` case from the input directory and writes one `<id>.out` per case, in case order,
one row per trophy id.

Constraints honoured: Python 3.13 standard library only, no network, no clock, no randomness, two runs
on the same case write the same bytes, and each case finishes inside 8 s and 65,536 bytes of output.
