# 🏎️ F1 Strategy Lab

**Tyre degradation modelling and pit-stop strategy simulation, built on real Formula 1 timing data.**

![tests](https://github.com/mohabinsemaidaa/f1-strategy-lab/actions/workflows/tests.yml/badge.svg)

[![Live demo](https://img.shields.io/badge/demo-live%20on%20Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://personalf1project.streamlit.app/)

Given a circuit, a team and track conditions, this project answers the question every F1 pit wall
faces on Sunday: **when should we stop, and which tyres should we bolt on?**

A gradient-boosted model learns how lap times evolve with tyre age, compound, fuel load and
temperature from real race data (via [FastF1](https://github.com/theOehrly/Fast-F1)). A strategy
simulator then searches hundreds of 1-stop and 2-stop plans and ranks them by predicted total
race time — all explorable in an interactive Streamlit dashboard.



## 🔴 Live demo

**Try it here: [personalf1project.streamlit.app](https://personalf1project.streamlit.app/)** —
pick a Grand Prix, drag the pit-loss slider and watch the optimal strategy flip between
one-stop and two-stop.

## How it works

```
FastF1 API ──> data.py ──> raw laps (parquet)
                              │
                        features.py       clean green-flag laps,
                              │           engineer fuel/tyre/weather features
                              ▼
                         model.py         sklearn Pipeline: one-hot + HistGradientBoosting
                              │           validated with leave-races-out GroupKFold
                              ▼
                        strategy.py       enumerate legal strategies, predict every lap,
                              │           add pit loss, rank by total race time
                              ▼
                     app/dashboard.py     Streamlit: degradation curves + strategy ranking
```

**Why leave-races-out validation?** Random train/test splits would put laps from the *same race*
in both sets — the model could memorise that race's pace and report a flattering error. GroupKFold
by event scores the model only on circuits it has never seen, which is the honest number.

## Quickstart

```bash
git clone https://github.com/mohabinsemaidaa/f1-strategy-lab
cd f1-strategy-lab
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Download race data (first run takes a few minutes, then it's cached)
python scripts/build_dataset.py --year 2025 --rounds 1 2 3 4 5 6 7 8 9 10

# 2. Train + cross-validate
python scripts/train.py

# 3. Explore
streamlit run app/dashboard.py
```

Run the tests:

```bash
python -m pytest
```

## Results

| Metric | Value |
|---|---|
| MAE on unseen races (leave-races-out CV) | **4.380 s** |
| Races in training set | **10** |
| Clean racing laps after filtering | **9,371** |

This number is deliberately pessimistic: in leave-races-out CV the held-out circuit was never
seen in training, so the model cannot know that track's base lap time — which alone varies by
tens of seconds between circuits. Most of the 4.4 s is unknown-circuit baseline, not tyre
modelling error. A random train/test split would report a far lower (and misleading) number by
leaking same-race laps into training; I chose to publish the honest one.

For the strategy simulator this baseline error cancels out entirely: strategies are always
compared **within the same circuit**, so only relative effects — degradation slope per compound,
fuel burn, temperature — decide the ranking, and those are what the model transfers across races.

## Design decisions

- **Lap cleaning**: in/out-laps, safety-car laps and >107%-of-median laps are removed so the
  model learns tyre physics, not traffic and chaos.
- **Fuel proxy**: race lap number stands in for fuel load (cars burn ~1.5 kg/lap and gain
  roughly 0.03 s/lap) — a deliberate, documented simplification.
- **Fixed pit loss**: pit-lane delta is a slider, not a model, keeping the simulator
  interpretable.

## Known limitations / next steps

- No traffic or dirty-air modelling — predictions assume clear-air pace.
- No safety-car probability; a Monte Carlo layer over the simulator is the natural extension.
- Wet compounds are excluded (too few clean laps).

## Stack

Python · pandas · scikit-learn · FastF1 · Streamlit · Plotly · pytest · GitHub Actions
