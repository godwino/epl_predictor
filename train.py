import re
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_DIR = Path(__file__).resolve().parent

def _season_key(path: Path) -> str:
    m = re.search(r"season-(\d{4})\.csv", path.name)
    return m.group(1) if m else ""


def load_matches() -> pd.DataFrame:
    files = sorted(DATA_DIR.glob("season-*.csv"))
    if not files:
        raise FileNotFoundError("No season-*.csv files found in the project folder.")

    selected = [f for f in files if _season_key(f)]

    frames = []
    for f in selected:
        df = pd.read_csv(f)
        df["SeasonKey"] = _season_key(f)
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    data["Date"] = pd.to_datetime(data["Date"], format="%d/%m/%y", errors="coerce")
    data = data.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])
    data = data.sort_values("Date").reset_index(drop=True)
    return data


def build_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    history = defaultdict(lambda: deque(maxlen=window))
    rows = []

    def agg_stats(team: str):
        h = history[team]
        if not h:
            return {
                "pts_avg": 0.0,
                "gf_avg": 0.0,
                "ga_avg": 0.0,
                "gd_avg": 0.0,
                "shots_avg": 0.0,
                "sot_avg": 0.0,
                "corners_avg": 0.0,
                "fouls_avg": 0.0,
                "yellows_avg": 0.0,
                "reds_avg": 0.0,
            }
        pts = [x[0] for x in h]
        gf = [x[1] for x in h]
        ga = [x[2] for x in h]
        gd = [x[1] - x[2] for x in h]
        shots = [x[3] for x in h]
        sot = [x[4] for x in h]
        corners = [x[5] for x in h]
        fouls = [x[6] for x in h]
        yellows = [x[7] for x in h]
        reds = [x[8] for x in h]
        return {
            "pts_avg": float(np.mean(pts)),
            "gf_avg": float(np.mean(gf)),
            "ga_avg": float(np.mean(ga)),
            "gd_avg": float(np.mean(gd)),
            "shots_avg": float(np.mean(shots)),
            "sot_avg": float(np.mean(sot)),
            "corners_avg": float(np.mean(corners)),
            "fouls_avg": float(np.mean(fouls)),
            "yellows_avg": float(np.mean(yellows)),
            "reds_avg": float(np.mean(reds)),
        }

    for _, r in df.iterrows():
        home = r["HomeTeam"]
        away = r["AwayTeam"]

        h_stats = agg_stats(home)
        a_stats = agg_stats(away)

        rows.append(
            {
                "Date": r["Date"],
                "SeasonKey": r["SeasonKey"],
                "HomeTeam": home,
                "AwayTeam": away,
                "home_pts_avg": h_stats["pts_avg"],
                "home_gf_avg": h_stats["gf_avg"],
                "home_ga_avg": h_stats["ga_avg"],
                "home_gd_avg": h_stats["gd_avg"],
                "home_shots_avg": h_stats["shots_avg"],
                "home_sot_avg": h_stats["sot_avg"],
                "home_corners_avg": h_stats["corners_avg"],
                "home_fouls_avg": h_stats["fouls_avg"],
                "home_yellows_avg": h_stats["yellows_avg"],
                "home_reds_avg": h_stats["reds_avg"],
                "away_pts_avg": a_stats["pts_avg"],
                "away_gf_avg": a_stats["gf_avg"],
                "away_ga_avg": a_stats["ga_avg"],
                "away_gd_avg": a_stats["gd_avg"],
                "away_shots_avg": a_stats["shots_avg"],
                "away_sot_avg": a_stats["sot_avg"],
                "away_corners_avg": a_stats["corners_avg"],
                "away_fouls_avg": a_stats["fouls_avg"],
                "away_yellows_avg": a_stats["yellows_avg"],
                "away_reds_avg": a_stats["reds_avg"],
                "FTR": r["FTR"],
            }
        )

        # Update history with actual result
        if r["FTR"] == "H":
            h_pts, a_pts = 3, 0
        elif r["FTR"] == "A":
            h_pts, a_pts = 0, 3
        else:
            h_pts, a_pts = 1, 1

        def to_int(val):
            try:
                return int(val)
            except Exception:
                return 0

        history[home].append(
            (
                h_pts,
                to_int(r["FTHG"]),
                to_int(r["FTAG"]),
                to_int(r.get("HS", 0)),
                to_int(r.get("HST", 0)),
                to_int(r.get("HC", 0)),
                to_int(r.get("HF", 0)),
                to_int(r.get("HY", 0)),
                to_int(r.get("HR", 0)),
            )
        )
        history[away].append(
            (
                a_pts,
                to_int(r["FTAG"]),
                to_int(r["FTHG"]),
                to_int(r.get("AS", 0)),
                to_int(r.get("AST", 0)),
                to_int(r.get("AC", 0)),
                to_int(r.get("AF", 0)),
                to_int(r.get("AY", 0)),
                to_int(r.get("AR", 0)),
            )
        )

    return pd.DataFrame(rows)


def train_and_eval(features: pd.DataFrame) -> None:
    X = features.drop(columns=["FTR", "Date"])
    y = features["FTR"]

    seasons = sorted(features["SeasonKey"].dropna().unique().tolist())
    holdout_season = seasons[-1]
    val_season = seasons[-2] if len(seasons) > 1 else None

    # Hold out last season for testing
    train_mask = features["SeasonKey"] != holdout_season
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[~train_mask]
    y_test = y[~train_mask]

    # Use most recent train season as validation to reduce overfitting
    val_mask = X_train["SeasonKey"] == val_season if val_season is not None else np.zeros(len(X_train), dtype=bool)
    X_val = X_train[val_mask]
    y_val = y_train[val_mask]
    X_train = X_train[~val_mask]
    y_train = y_train[~val_mask]

    cat_cols = ["HomeTeam", "AwayTeam", "SeasonKey"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )

    # Baseline: Logistic Regression
    lr = LogisticRegression(max_iter=300, n_jobs=None)
    lr_model = Pipeline([("pre", pre), ("clf", lr)])
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_preds)
    print(f"[LogReg] Test accuracy (season {holdout_season}): {lr_acc:.3f}")
    print(classification_report(y_test, lr_preds))

    # XGBoost: better non-linear model with regularization + early stopping
    label_map = {"H": 0, "D": 1, "A": 2}
    y_train_enc = y_train.map(label_map).to_numpy()
    y_val_enc = y_val.map(label_map).to_numpy() if len(y_val) else None
    y_test_enc = y_test.map(label_map).to_numpy()

    X_train_enc = pre.fit_transform(X_train)
    X_val_enc = pre.transform(X_val) if len(X_val) else None
    X_test_enc = pre.transform(X_test)

    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.0,
    )

    if X_val_enc is not None and len(y_val_enc) > 0:
        xgb.fit(
            X_train_enc,
            y_train_enc,
            eval_set=[(X_val_enc, y_val_enc)],
            verbose=False,
        )
    else:
        xgb.fit(X_train_enc, y_train_enc, verbose=False)

    xgb_preds = xgb.predict(X_test_enc)
    xgb_acc = accuracy_score(y_test_enc, xgb_preds)
    print(f"[XGBoost] Test accuracy (season {holdout_season}): {xgb_acc:.3f}")
    print(classification_report(y_test_enc, xgb_preds, target_names=["H", "D", "A"]))


def main():
    df = load_matches()
    feats = build_features(df, window=5)
    train_and_eval(feats)


if __name__ == "__main__":
    main()
