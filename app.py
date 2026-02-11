import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

from advanced_train import load_matches
from predict_match import _predict_fixture

DATA_DIR = Path(__file__).resolve().parent
ARTIFACT_PATH = DATA_DIR / "advanced_model.pkl"


@st.cache_data
def get_teams() -> list[str]:
    df = load_matches().copy()
    teams = sorted(set(df["HomeTeam"]).union(set(df["AwayTeam"])))
    return teams


@st.cache_resource
def load_artifact() -> dict:
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"{ARTIFACT_PATH.name} not found. Run `python -B advanced_train.py` first."
        )
    with ARTIFACT_PATH.open("rb") as f:
        return pickle.load(f)


def predict_one(home: str, away: str, artifact: dict) -> dict:
    return _predict_fixture(
        model=artifact["model"],
        preprocessor=artifact["preprocessor"],
        inv_label_map=artifact["inv_label_map"],
        feature_columns=artifact["feature_columns"],
        home=home,
        away=away,
        window=artifact.get("window", 5),
    )


def render_single_fixture(teams: list[str], artifact: dict) -> None:
    st.subheader("Single Fixture Prediction")

    default_home = teams.index("Arsenal") if "Arsenal" in teams else 0
    default_away = teams.index("Chelsea") if "Chelsea" in teams else min(1, len(teams) - 1)

    col1, col2 = st.columns(2)
    with col1:
        home = st.selectbox("Home Team", teams, index=default_home)
    with col2:
        away = st.selectbox("Away Team", teams, index=default_away)

    if home == away:
        st.warning("Home and away teams must be different.")
        return

    if st.button("Predict Result", type="primary"):
        pred = predict_one(home, away, artifact)

        probs = pd.DataFrame(
            {
                "Outcome": ["Home Win", "Draw", "Away Win"],
                "Probability": [pred["H"], pred["D"], pred["A"]],
            }
        )

        winner_key = max(["H", "D", "A"], key=lambda k: pred[k])
        winner_label = {"H": "Home Win", "D": "Draw", "A": "Away Win"}[winner_key]

        st.success(f"Most likely outcome: {winner_label}")
        st.dataframe(probs, use_container_width=True, hide_index=True)
        st.bar_chart(probs.set_index("Outcome"))



def render_batch_prediction(artifact: dict) -> None:
    st.subheader("Batch Prediction from CSV")
    st.caption("Upload a CSV with columns: HomeTeam, AwayTeam")

    uploaded = st.file_uploader("Fixtures CSV", type=["csv"])
    if uploaded is None:
        return

    fixtures = pd.read_csv(uploaded)
    required = {"HomeTeam", "AwayTeam"}
    if not required.issubset(set(fixtures.columns)):
        st.error("CSV must contain columns: HomeTeam, AwayTeam")
        return

    st.write("Input preview")
    st.dataframe(fixtures.head(20), use_container_width=True)

    if st.button("Run Batch Prediction"):
        rows = []
        progress = st.progress(0)
        total = len(fixtures)

        for i, (_, row) in enumerate(fixtures.iterrows(), start=1):
            rows.append(
                predict_one(
                    home=str(row["HomeTeam"]),
                    away=str(row["AwayTeam"]),
                    artifact=artifact,
                )
            )
            progress.progress(i / total)

        out = pd.DataFrame(rows)
        st.success(f"Generated predictions for {len(out)} fixtures.")
        st.dataframe(out, use_container_width=True)

        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Predictions CSV",
            data=csv_bytes,
            file_name="predictions.csv",
            mime="text/csv",
        )



def main() -> None:
    st.set_page_config(page_title="EPL Predictor", layout="wide")
    st.title("EPL Win/Draw/Loss Predictor")
    st.caption("Streamlit app for single and batch fixture predictions.")

    try:
        artifact = load_artifact()
    except Exception as exc:
        st.error(str(exc))
        st.info("Train the model first: `python -B advanced_train.py`")
        st.stop()

    with st.sidebar:
        st.subheader("Model")
        st.write(f"Artifact: `{ARTIFACT_PATH.name}`")
        st.write(f"Selected model: `{artifact.get('model_name', 'Unknown')}`")

    teams = get_teams()

    tab1, tab2 = st.tabs(["Single Fixture", "Batch CSV"])
    with tab1:
        render_single_fixture(teams, artifact)
    with tab2:
        render_batch_prediction(artifact)


if __name__ == "__main__":
    main()
