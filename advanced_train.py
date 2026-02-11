"""
Advanced EPL Predictor - Professional Data Science Implementation
Features: Advanced feature engineering, ensemble models, proper validation
"""
import re
import pickle
import warnings
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).resolve().parent
ARTIFACT_PATH = DATA_DIR / "advanced_model.pkl"

def _season_key(path: Path) -> str:
    m = re.search(r"season-(\d{4})\.csv", path.name)
    return m.group(1) if m else ""


def load_matches() -> pd.DataFrame:
    """Load all season data with proper validation."""
    files = sorted(DATA_DIR.glob("season-*.csv"))
    if not files:
        raise FileNotFoundError("No season-*.csv files found.")

    allowed = {"1415", "1516", "1617", "1718", "1819", "1920", "2021", "2122", "2223", "2324"}
    selected = [f for f in files if _season_key(f) in allowed]

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


def build_advanced_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Build advanced features with:
    - Form decay (recent matches weighted higher)
    - Head-to-head history
    - Momentum and consistency
    - Strength metrics
    """
    history = defaultdict(lambda: deque(maxlen=window))
    h2h_history = defaultdict(deque)  # Head-to-head records
    rows = []

    def to_int(val):
        try:
            return int(val)
        except:
            return 0

    def get_form_stats(team: str, decay=True):
        """Get team form with exponential decay for recent matches."""
        h = history[team]
        if not h:
            return {
                "pts_avg": 0.0, "gf_avg": 0.0, "ga_avg": 0.0, "gd_avg": 0.0,
                "shots_avg": 0.0, "sot_avg": 0.0, "sot_pct": 0.0,
                "corners_avg": 0.0, "fouls_avg": 0.0, "yellows_avg": 0.0,
                "reds_avg": 0.0, "consistency": 0.0, "momentum": 0.0,
            }
        
        matches = list(h)
        pts = [x[0] for x in matches]
        gf = [x[1] for x in matches]
        ga = [x[2] for x in matches]
        shots = [x[3] for x in matches]
        sot = [x[4] for x in matches]
        corners = [x[5] for x in matches]
        fouls = [x[6] for x in matches]
        yellows = [x[7] for x in matches]
        reds = [x[8] for x in matches]

        # Exponential decay weighting (recent matches more important)
        if decay and len(matches) > 1:
            weights = np.exp(np.linspace(-1, 0, len(matches)))
            weights /= weights.sum()
        else:
            weights = np.ones(len(matches)) / len(matches)

        pts_avg = float(np.average(pts, weights=weights))
        gf_avg = float(np.average(gf, weights=weights))
        ga_avg = float(np.average(ga, weights=weights))
        gd_avg = gf_avg - ga_avg
        
        shots_avg = float(np.average(shots, weights=weights))
        sot_avg = float(np.average(sot, weights=weights))
        sot_pct = (sot_avg / shots_avg * 100) if shots_avg > 0 else 0.0
        
        corners_avg = float(np.average(corners, weights=weights))
        fouls_avg = float(np.average(fouls, weights=weights))
        yellows_avg = float(np.average(yellows, weights=weights))
        reds_avg = float(np.average(reds, weights=weights))

        # Consistency (lower std = more consistent)
        consistency = float(1.0 / (1.0 + np.std(pts) if len(pts) > 1 else 1.0))
        
        # Momentum (recent vs older performance)
        if len(pts) > 2:
            recent_pts = np.mean(pts[-2:])
            older_pts = np.mean(pts[:-2])
            momentum = float(recent_pts - older_pts)
        else:
            momentum = float(pts_avg)

        return {
            "pts_avg": pts_avg,
            "gf_avg": gf_avg,
            "ga_avg": ga_avg,
            "gd_avg": gd_avg,
            "shots_avg": shots_avg,
            "sot_avg": sot_avg,
            "sot_pct": sot_pct,
            "corners_avg": corners_avg,
            "fouls_avg": fouls_avg,
            "yellows_avg": yellows_avg,
            "reds_avg": reds_avg,
            "consistency": consistency,
            "momentum": momentum,
        }

    def get_h2h_stats(home: str, away: str):
        """Get head-to-head statistics."""
        h2h_key = tuple(sorted([home, away]))
        h2h = h2h_history[h2h_key]
        
        if not h2h:
            return {"h2h_h_pts": 0.0, "h2h_draws": 0.0, "h2h_advantage": 0.0}
        
        h_wins = sum(1 for m in h2h if (m[0] == "H" and m[1] == home) or (m[0] == "A" and m[1] == away))
        draws = sum(1 for m in h2h if m[0] == "D")
        h2h_h_pts = h_wins * 3 + draws / 2
        
        return {
            "h2h_h_pts": float(h2h_h_pts / len(h2h)) if h2h else 0.0,
            "h2h_draws": float(draws / len(h2h)) if h2h else 0.0,
            "h2h_advantage": float((h_wins - (len(h2h) - h_wins - draws)) / len(h2h)) if h2h else 0.0,
        }

    for _, r in df.iterrows():
        home = r["HomeTeam"]
        away = r["AwayTeam"]

        h_stats = get_form_stats(home)
        a_stats = get_form_stats(away)
        h2h = get_h2h_stats(home, away)

        rows.append({
            "Date": r["Date"],
            "SeasonKey": r["SeasonKey"],
            "HomeTeam": home,
            "AwayTeam": away,
            # Home stats
            "home_pts_avg": h_stats["pts_avg"],
            "home_gf_avg": h_stats["gf_avg"],
            "home_ga_avg": h_stats["ga_avg"],
            "home_gd_avg": h_stats["gd_avg"],
            "home_shots_avg": h_stats["shots_avg"],
            "home_sot_avg": h_stats["sot_avg"],
            "home_sot_pct": h_stats["sot_pct"],
            "home_corners_avg": h_stats["corners_avg"],
            "home_fouls_avg": h_stats["fouls_avg"],
            "home_yellows_avg": h_stats["yellows_avg"],
            "home_reds_avg": h_stats["reds_avg"],
            "home_consistency": h_stats["consistency"],
            "home_momentum": h_stats["momentum"],
            # Away stats
            "away_pts_avg": a_stats["pts_avg"],
            "away_gf_avg": a_stats["gf_avg"],
            "away_ga_avg": a_stats["ga_avg"],
            "away_gd_avg": a_stats["gd_avg"],
            "away_shots_avg": a_stats["shots_avg"],
            "away_sot_avg": a_stats["sot_avg"],
            "away_sot_pct": a_stats["sot_pct"],
            "away_corners_avg": a_stats["corners_avg"],
            "away_fouls_avg": a_stats["fouls_avg"],
            "away_yellows_avg": a_stats["yellows_avg"],
            "away_reds_avg": a_stats["reds_avg"],
            "away_consistency": a_stats["consistency"],
            "away_momentum": a_stats["momentum"],
            # H2H
            "h2h_h_pts": h2h["h2h_h_pts"],
            "h2h_draws": h2h["h2h_draws"],
            "h2h_advantage": h2h["h2h_advantage"],
            # Target
            "FTR": r["FTR"],
        })

        # Update history
        if r["FTR"] == "H":
            h_pts, a_pts = 3, 0
        elif r["FTR"] == "A":
            h_pts, a_pts = 0, 3
        else:
            h_pts, a_pts = 1, 1

        history[home].append((
            h_pts, to_int(r["FTHG"]), to_int(r["FTAG"]),
            to_int(r.get("HS", 0)), to_int(r.get("HST", 0)),
            to_int(r.get("HC", 0)), to_int(r.get("HF", 0)),
            to_int(r.get("HY", 0)), to_int(r.get("HR", 0)),
        ))
        history[away].append((
            a_pts, to_int(r["FTAG"]), to_int(r["FTHG"]),
            to_int(r.get("AS", 0)), to_int(r.get("AST", 0)),
            to_int(r.get("AC", 0)), to_int(r.get("AF", 0)),
            to_int(r.get("AY", 0)), to_int(r.get("AR", 0)),
        ))

        # Update H2H
        h2h_key = tuple(sorted([home, away]))
        h2h_history[h2h_key].append((r["FTR"], home))

    return pd.DataFrame(rows)


def train_advanced_ensemble(features: pd.DataFrame) -> dict:
    """
    Train ensemble of models with proper validation and evaluation.
    Returns predictions, probabilities, and model performance.
    """
    X = features.drop(columns=["FTR", "Date"])
    y = features["FTR"]

    # Train/test split
    train_mask = features["SeasonKey"] != "2324"
    X_train = X[train_mask].copy()
    y_train = y[train_mask].copy()
    X_test = X[~train_mask].copy()
    y_test = y[~train_mask].copy()

    # Preprocessing
    cat_cols = ["HomeTeam", "AwayTeam", "SeasonKey"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", StandardScaler(), num_cols),
    ])

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Label encoding
    label_map = {"H": 0, "D": 1, "A": 2}
    y_train_enc = y_train.map(label_map).values
    y_test_enc = y_test.map(label_map).values

    print("\n" + "="*70)
    print("ENSEMBLE MODEL TRAINING")
    print("="*70)

    # Import training-only libraries lazily so inference environments
    # don't need to install them just to import this module.
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    models = {}
    predictions = {}

    # Model 1: XGBoost
    print("\n[1/3] Training XGBoost...")
    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42,
    )
    xgb.fit(X_train_processed, y_train_enc, verbose=False)
    xgb_pred = xgb.predict(X_test_processed)
    xgb_proba = xgb.predict_proba(X_test_processed)
    models["XGBoost"] = xgb
    predictions["XGBoost"] = (xgb_pred, xgb_proba)

    # Model 2: LightGBM
    print("[2/3] Training LightGBM...")
    lgb = LGBMClassifier(
        num_class=3,
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    )
    lgb.fit(X_train_processed, y_train_enc)
    lgb_pred = lgb.predict(X_test_processed)
    lgb_proba = lgb.predict_proba(X_test_processed)
    models["LightGBM"] = lgb
    predictions["LightGBM"] = (lgb_pred, lgb_proba)

    # Model 3: Logistic Regression (for baseline/calibration)
    print("[3/3] Training Logistic Regression...")
    lr = LogisticRegression(max_iter=300, multi_class="multinomial", random_state=42)
    lr.fit(X_train_processed, y_train_enc)
    lr_pred = lr.predict(X_test_processed)
    lr_proba = lr.predict_proba(X_test_processed)
    models["LogReg"] = lr
    predictions["LogReg"] = (lr_pred, lr_proba)

    # Ensemble (weighted average)
    print("\n[4/3] Creating ensemble...")
    ensemble_proba = (xgb_proba + lgb_proba + lr_proba) / 3
    ensemble_pred = np.argmax(ensemble_proba, axis=1)
    models["Ensemble"] = ensemble_proba
    predictions["Ensemble"] = (ensemble_pred, ensemble_proba)

    # Evaluate all models
    print("\n" + "="*70)
    print("MODEL PERFORMANCE (Season 2324 - Test Set)")
    print("="*70)

    results = {}
    for name, (pred, proba) in predictions.items():
        acc = accuracy_score(y_test_enc, pred)
        
        # ROC-AUC (one-vs-rest)
        try:
            auc = roc_auc_score(y_test_enc, proba, multi_class="ovr")
        except:
            auc = 0.0

        # Log loss
        ll = log_loss(y_test_enc, proba)

        results[name] = {"accuracy": acc, "auc": auc, "logloss": ll}

        print(f"\n[{name}]")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  ROC-AUC:   {auc:.4f}")
        print(f"  Log Loss:  {ll:.4f}")
        print("\n  Classification Report:")
        report = classification_report(
            y_test_enc, pred,
            target_names=["H", "D", "A"],
            digits=3
        )
        print("  " + "\n  ".join(report.split("\n")))

    # Feature importance
    print("\n" + "="*70)
    print("TOP FEATURE IMPORTANCE (XGBoost)")
    print("="*70)
    importance = pd.DataFrame({
        "feature": range(X_train_processed.shape[1]),
        "importance": xgb.feature_importances_
    }).nlargest(20, "importance")
    
    for idx, row in importance.iterrows():
        print(f"  Feature {row['feature']}: {row['importance']:.4f}")

    return {
        "models": models,
        "preprocessor": preprocessor,
        "feature_columns": list(X.columns),
        "label_map": label_map,
        "predictions": predictions,
        "results": results,
        "y_test": y_test.values,
        "test_indices": ~train_mask,
    }


def save_artifacts(training_output: dict, window: int = 5) -> Path:
    """Persist best deployable model and preprocessing objects for inference."""
    deployable = ["XGBoost", "LightGBM", "LogReg"]
    best_name = max(
        ((k, v) for k, v in training_output["results"].items() if k in deployable),
        key=lambda x: x[1]["accuracy"]
    )[0]

    label_map = training_output["label_map"]
    inv_label_map = {v: k for k, v in label_map.items()}
    payload = {
        "model_name": best_name,
        "model": training_output["models"][best_name],
        "preprocessor": training_output["preprocessor"],
        "feature_columns": training_output["feature_columns"],
        "label_map": label_map,
        "inv_label_map": inv_label_map,
        "window": window,
    }

    with ARTIFACT_PATH.open("wb") as f:
        pickle.dump(payload, f)
    return ARTIFACT_PATH


def main():
    print("\n" + "="*70)
    print("ADVANCED EPL PREDICTOR - GENIUS LEVEL DATA SCIENCE")
    print("="*70)
    
    print("\n[Step 1] Loading data...")
    df = load_matches()
    print(f"  Loaded {len(df)} matches across {len(df['SeasonKey'].unique())} seasons")

    print("\n[Step 2] Building advanced features...")
    feats = build_advanced_features(df, window=5)
    print(f"  Created {len(feats.columns) - 4} features")
    print(f"  Feature columns: {[c for c in feats.columns if c not in ['Date', 'SeasonKey', 'HomeTeam', 'AwayTeam', 'FTR']][:10]}...")

    print("\n[Step 3] Training ensemble models...")
    results = train_advanced_ensemble(feats)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    best_model = max(results["results"].items(), key=lambda x: x[1]["accuracy"])
    print(f"\nBest Model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")

    artifact_path = save_artifacts(results, window=5)
    print(f"Saved deployable artifact: {artifact_path.name}")
    print("\nAdvanced training complete!")


if __name__ == "__main__":
    main()
