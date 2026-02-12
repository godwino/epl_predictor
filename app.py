import pickle
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from epl_features import build_advanced_features, load_matches
from predict_match import _predict_fixture

DATA_DIR = Path(__file__).resolve().parent
ARTIFACT_PATH = DATA_DIR / "advanced_model.pkl"
OUTCOME_LABELS = {"H": "Home Win", "D": "Draw", "A": "Away Win"}


def collect_startup_checks() -> dict:
    """Collect dependency/artifact/runtime checks for clear startup diagnostics."""
    checks = {
        "python": sys.executable,
        "artifact_exists": ARTIFACT_PATH.exists(),
        "required_packages": {},
    }
    for pkg in ["streamlit", "pandas", "sklearn"]:
        checks["required_packages"][pkg] = importlib.util.find_spec(pkg) is not None
    # Training/runtime model packages can be optional if fallback is used.
    for pkg in ["xgboost", "lightgbm"]:
        checks["required_packages"][pkg] = importlib.util.find_spec(pkg) is not None
    return checks


@st.cache_data
def get_teams() -> list[str]:
    df = load_matches().copy()
    teams = sorted(set(df["HomeTeam"]).union(set(df["AwayTeam"])))
    return teams


@st.cache_resource
def load_artifact() -> dict:
    if not ARTIFACT_PATH.exists():
        st.warning(
            f"{ARTIFACT_PATH.name} not found. Using runtime fallback model."
        )
        return build_fallback_artifact()
    try:
        with ARTIFACT_PATH.open("rb") as f:
            return pickle.load(f)
    except Exception as exc:
        st.warning(
            f"Could not load {ARTIFACT_PATH.name} ({exc}). Using runtime fallback model."
        )
        return build_fallback_artifact()


@st.cache_resource
def build_fallback_artifact() -> dict:
    df = load_matches().copy()
    feats = build_advanced_features(df, window=5)
    X = feats.drop(columns=["FTR", "Date"])
    y = feats["FTR"]

    cat_cols = ["HomeTeam", "AwayTeam", "SeasonKey"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        [
            ("cat", encoder, cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )
    X_processed = preprocessor.fit_transform(X)

    label_map = {"H": 0, "D": 1, "A": 2}
    y_enc = y.map(label_map).values
    try:
        model = LogisticRegression(max_iter=400, multi_class="multinomial", random_state=42)
    except TypeError:
        # Compatibility for sklearn versions where `multi_class` is not accepted.
        model = LogisticRegression(max_iter=400, random_state=42)
    model.fit(X_processed, y_enc)

    inv_label_map = {v: k for k, v in label_map.items()}
    return {
        "model_name": "LogReg (fallback)",
        "model": model,
        "preprocessor": preprocessor,
        "feature_columns": list(X.columns),
        "label_map": label_map,
        "inv_label_map": inv_label_map,
        "window": 5,
        "decision_thresholds": {"H": 0.5, "D": 0.5, "A": 0.5},
        "holdout_season": None,
    }


def predict_one(
    home: str,
    away: str,
    artifact: dict,
    home_last4: str | None = None,
    away_last4: str | None = None,
) -> dict:
    return _predict_fixture(
        model=artifact["model"],
        preprocessor=artifact["preprocessor"],
        inv_label_map=artifact["inv_label_map"],
        feature_columns=artifact["feature_columns"],
        home=home,
        away=away,
        window=artifact.get("window", 5),
        decision_thresholds=artifact.get("decision_thresholds", {"H": 0.5, "D": 0.5, "A": 0.5}),
        home_last4=home_last4,
        away_last4=away_last4,
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

    c3, c4 = st.columns(2)
    with c3:
        home_last4 = st.text_input("Home Last 4 (optional, e.g. WDLW)", value="")
    with c4:
        away_last4 = st.text_input("Away Last 4 (optional, e.g. LDWW)", value="")

    if home == away:
        st.warning("Home and away teams must be different.")
        return

    if st.button("Predict Result", type="primary"):
        pred = predict_one(
            home,
            away,
            artifact,
            home_last4=home_last4.strip() or None,
            away_last4=away_last4.strip() or None,
        )
        winner_key = pred["Prediction"]
        winner_label = OUTCOME_LABELS[winner_key]

        st.success(f"Most likely outcome: {winner_label}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Home Win", f"{pred['H'] * 100:.1f}%")
        c2.metric("Draw", f"{pred['D'] * 100:.1f}%")
        c3.metric("Away Win", f"{pred['A'] * 100:.1f}%")

        probs = pd.DataFrame(
            {
                "Outcome": [OUTCOME_LABELS["H"], OUTCOME_LABELS["D"], OUTCOME_LABELS["A"]],
                "Probability": [pred["H"], pred["D"], pred["A"]],
            }
        )
        st.bar_chart(probs.set_index("Outcome"))



def render_batch_prediction(artifact: dict) -> None:
    st.subheader("Batch Prediction from CSV")
    st.caption("Upload a CSV with columns: HomeTeam, AwayTeam. Optional: HomeLast4, AwayLast4 (W/D/L strings).")
    sample_csv = (
        "HomeTeam,AwayTeam,HomeLast4,AwayLast4\n"
        "Arsenal,Chelsea,WWDW,LDWW\n"
        "Liverpool,Man City,WDWW,WWDL\n"
    )
    st.download_button(
        "Download sample CSV",
        data=sample_csv.encode("utf-8"),
        file_name="fixtures_sample.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Fixtures CSV", type=["csv"])
    if uploaded is None:
        return

    fixtures = pd.read_csv(uploaded)
    required = {"HomeTeam", "AwayTeam"}
    if not required.issubset(set(fixtures.columns)):
        st.error("CSV must contain columns: HomeTeam, AwayTeam")
        return
    if fixtures.empty:
        st.warning("The uploaded CSV is empty.")
        return

    st.write("Input preview")
    st.dataframe(fixtures.head(20), use_container_width=True)

    if st.button("Run Batch Prediction"):
        rows = []
        errors = []
        progress = st.progress(0)
        total = len(fixtures)

        for i, (_, row) in enumerate(fixtures.iterrows(), start=1):
            home = str(row["HomeTeam"]).strip()
            away = str(row["AwayTeam"]).strip()
            try:
                rows.append(
                    predict_one(
                        home=home,
                        away=away,
                        artifact=artifact,
                        home_last4=str(row["HomeLast4"]).strip() if "HomeLast4" in fixtures.columns and pd.notna(row["HomeLast4"]) else None,
                        away_last4=str(row["AwayLast4"]).strip() if "AwayLast4" in fixtures.columns and pd.notna(row["AwayLast4"]) else None,
                    )
                )
            except Exception as exc:
                errors.append({"HomeTeam": home, "AwayTeam": away, "Error": str(exc)})
            progress.progress(i / total)

        out = pd.DataFrame(rows)
        if not out.empty:
            st.success(f"Generated predictions for {len(out)} fixtures.")
            st.dataframe(out, use_container_width=True)

            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Predictions CSV",
                data=csv_bytes,
                file_name="predictions.csv",
                mime="text/csv",
            )
        else:
            st.warning("No valid fixtures were predicted from this file.")

        if errors:
            st.error(f"{len(errors)} fixture(s) could not be predicted.")
            st.dataframe(pd.DataFrame(errors), use_container_width=True, hide_index=True)



def main() -> None:
    st.set_page_config(page_title="EPL Predictor", layout="wide")
    st.title("EPL Win/Draw/Loss Predictor")
    st.caption("Streamlit app for single and batch fixture predictions.")
    checks = collect_startup_checks()

    if not checks["artifact_exists"]:
        st.warning(
            f"`{ARTIFACT_PATH.name}` not found. The app will use fallback model. "
            "Run `python -B advanced_train.py` to build the deployable model."
        )

    if not checks["required_packages"]["xgboost"]:
        st.info(
            "Package `xgboost` is not installed in this Python environment. "
            "If your artifact uses XGBoost, fallback mode will be used. "
            "Install with `pip install -r requirements-train.txt`."
        )

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
        if artifact.get("holdout_season"):
            st.write(f"Holdout season: `{artifact.get('holdout_season')}`")
        if artifact.get("decision_thresholds"):
            st.write(f"Thresholds: `{artifact.get('decision_thresholds')}`")
        with st.expander("Startup Checks", expanded=False):
            st.write(f"Python: `{checks['python']}`")
            st.write(f"Artifact exists: `{checks['artifact_exists']}`")
            dep_rows = pd.DataFrame(
                [
                    {"Package": k, "Available": v}
                    for k, v in checks["required_packages"].items()
                ]
            )
            st.dataframe(dep_rows, use_container_width=True, hide_index=True)

    teams = get_teams()

    tab1, tab2 = st.tabs(["Single Fixture", "Batch CSV"])
    with tab1:
        render_single_fixture(teams, artifact)
    with tab2:
        render_batch_prediction(artifact)


if __name__ == "__main__":
    main()
