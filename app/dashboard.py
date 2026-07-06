"""F1 Strategy Lab — interactive dashboard.

Run with:
    streamlit run app/dashboard.py
"""

import json
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

# Make the src/ package importable when run via "streamlit run".
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f1strategy.model import MODEL_PATH, load_model  # had to use AI for this line
from f1strategy.strategy import RaceContext, rank_strategies, stint_frame  # this too

st.set_page_config(page_title="F1 Strategy Lab", page_icon="🏎️", layout="wide")

METADATA = Path("models/metadata.json")
METRICS = Path("models/metrics.json")

@st.cache_resource
def get_model():
    return load_model()

@st.cache_data
def get_metadata() -> dict:
    return json.loads(METADATA.read_text())

if not MODEL_PATH.exists() or not METADATA.exists():
    st.error(
        "No trained model found. Run:\n\n"
        "`python scripts/build_dataset.py` then `python scripts/train.py`"
    )
    st.stop()

model = get_model()
meta = get_metadata()

st.title("🏎️ F1 Strategy Lab")
st.caption(
    "A tyre-degradation model trained on real FastF1 timing data, "
    "driving a pit-stop strategy simulator."
)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Race setup")
    event = st.selectbox("Grand Prix", meta["events"])
    team = st.selectbox("Team", meta["teams"])
    drivers = meta["drivers"]
    driver = st.selectbox("Driver", drivers)
    default_laps = meta.get("total_laps_by_event", {}).get(event, 57)
    total_laps = st.slider("Race laps", 40, 78, int(default_laps))
    track_temp = st.slider("Track temp (°C)", 15, 55, int(meta["median_track_temp"]))
    air_temp = st.slider("Air temp (°C)", 10, 40, int(meta["median_air_temp"]))
    pit_loss = st.slider("Pit loss (s)", 15.0, 30.0, 22.0, 0.5)

ctx = RaceContext(
    event=event, team=team, driver=driver, total_laps=total_laps,
    track_temp=track_temp, air_temp=air_temp, pit_loss=pit_loss,
)

tab_deg, tab_strategy, tab_model = st.tabs(
    ["Tyre degradation", "Strategy simulator", "Model card"]
)

# --------------------------------------------------------- degradation tab
with tab_deg:
    st.subheader(f"Predicted degradation — {event}")
    stint_len = st.slider("Stint length to plot", 10, 40, 25)
    frames = []
    for compound in meta["compounds"]:
        f = stint_frame(ctx, compound, start_lap=1, n_laps=stint_len)
        f = f.assign(Predicted=model.predict(f), CompoundName=compound)
        frames.append(f)
    curves = pd.concat(frames)
    fig = px.line(
        curves, x="TyreLife", y="Predicted", color="CompoundName",
        labels={"Predicted": "Predicted lap time (s)", "TyreLife": "Tyre age (laps)"},
        color_discrete_map={"SOFT": "#e10600", "MEDIUM": "#ffd12e", "HARD": "#f0f0f0"},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Each line is the model's predicted lap time for a fresh stint starting "
        "on lap 1, isolating the tyre-wear effect per compound."
    )

# ------------------------------------------------------------ strategy tab
with tab_strategy:
    st.subheader("Fastest strategies for this race")
    with st.spinner("Simulating hundreds of strategies..."):
        ranked = rank_strategies(model, ctx, top_n=10)

    rows = [
        {
            "Strategy": r.label,
            "Stops": r.n_stops,
            "Total time (s)": round(r.total_time, 1),
            "Gap to best (s)": round(r.total_time - ranked[0].total_time, 1),
        }
        for r in ranked
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    best = ranked[0]
    st.success(f"Fastest: **{best.label}** — {best.total_time:.1f}s total")
    lap_df = pd.DataFrame(
        {"Lap": range(1, len(best.lap_times) + 1), "Lap time (s)": best.lap_times}
    )
    st.plotly_chart(
        px.line(lap_df, x="Lap", y="Lap time (s)", title="Best strategy: lap-by-lap"),
        use_container_width=True,
    )

# --------------------------------------------------------------- model tab
with tab_model:
    st.subheader("Model card")
    if METRICS.exists():
        metrics = json.loads(METRICS.read_text())
        mae = metrics["cv_mae_seconds"]["mean_mae"]
        st.metric("MAE on unseen races (leave-races-out CV)", f"{mae:.3f} s")
    st.markdown(
        """
- **Model**: `HistGradientBoostingRegressor` in a sklearn `Pipeline` with one-hot encoding
- **Target**: green-flag racing lap time (seconds)
- **Features**: lap number (fuel proxy), tyre life, compound, stint, circuit, team, driver, track/air temp
- **Validation**: GroupKFold by race — the model is always scored on circuits it never trained on,
  so the MAE is honest about generalisation (random splits would leak same-race laps)
- **Known limits**: no traffic/dirty-air modelling, fixed pit loss, no safety-car probability
        """
    )