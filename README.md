# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/EVM0x/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/EVM0x/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Five quantiles of every Grand Theft Auto VI PlayStation 5 trophy's earn rate on Exophase at
2026-11-26T00:00Z, from the reads the house recorded before the deadline: the 10-minute read grid is
collapsed to Exophase's refresh events, a weighted least-squares logit trend per trophy is shrunk toward
its tier's median slope, damped and capped at 4.5 days out, and the spread combines the horizon with the
fit's own residual — scored by mean pinball loss on the logit scale.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` case in the input directory and writes `<id>.out` beside it: one row per trophy,
`{"id": n, "q": [q05, q25, q50, q75, q95]}` in percent, non-decreasing. Standard library only, no
network, no clock, no randomness — two runs on the same case write the same bytes.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
