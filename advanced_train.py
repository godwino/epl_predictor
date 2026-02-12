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
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, log_loss, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid

warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).resolve().parent
ARTIFACT_PATH = DATA_DIR / "advanced_model.pkl"
BASELINE_ARTIFACT_PATH = DATA_DIR / "model_baseline.pkl"
ADVANCED_ARTIFACT_PATH = DATA_DIR / "model_advanced.pkl"

def _season_key(path: Path) -> str:
    m = re.search(r"season-(\d{4})\.csv", path.name)
    return m.group(1) if m else ""


def load_matches() -> pd.DataFrame:
    """Load all season data with proper validation."""
    files = sorted(DATA_DIR.glob("season-*.csv"))
    if not files:
        raise FileNotFoundError("No season-*.csv files found.")

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


def build_advanced_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Build advanced features with:
    - Form decay (recent matches weighted higher)
    - Head-to-head history
    - Momentum and consistency
    - Strength metrics
    """
    history = defaultdict(lambda: deque(maxlen=window))
    home_history = defaultdict(lambda: deque(maxlen=window))
    away_history = defaultdict(lambda: deque(maxlen=window))
    h2h_history = defaultdict(deque)  # Head-to-head records
    last_seen_date = {}
    season_match_count = defaultdict(int)
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

    def get_rest_days(team: str, cur_date: pd.Timestamp) -> float:
        prev = last_seen_date.get(team)
        if prev is None:
            return 7.0
        days = (cur_date - prev).days
        return float(max(0, min(days, 21)))

    def get_venue_form(team: str, at_home: bool) -> dict:
        venue_hist = home_history[team] if at_home else away_history[team]
        if not venue_hist:
            return {"pts_avg": 0.0, "gd_avg": 0.0}
        pts = [x[0] for x in venue_hist]
        gf = [x[1] for x in venue_hist]
        ga = [x[2] for x in venue_hist]
        return {
            "pts_avg": float(np.mean(pts)),
            "gd_avg": float(np.mean(gf) - np.mean(ga)),
        }

    for _, r in df.iterrows():
        cur_date = r["Date"]
        season_key = r["SeasonKey"]
        home = r["HomeTeam"]
        away = r["AwayTeam"]

        h_stats = get_form_stats(home)
        a_stats = get_form_stats(away)
        h2h = get_h2h_stats(home, away)
        h_home_form = get_venue_form(home, at_home=True)
        a_away_form = get_venue_form(away, at_home=False)
        home_rest_days = get_rest_days(home, cur_date)
        away_rest_days = get_rest_days(away, cur_date)
        match_no = season_match_count[season_key]
        matchweek = int(match_no // 10) + 1
        season_progress = min(matchweek / 38.0, 1.0)
        is_midweek = 1.0 if int(cur_date.dayofweek) in {1, 2, 3} else 0.0

        rows.append({
            "Date": cur_date,
            "SeasonKey": season_key,
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
            "home_home_pts_avg": h_home_form["pts_avg"],
            "home_home_gd_avg": h_home_form["gd_avg"],
            "away_away_pts_avg": a_away_form["pts_avg"],
            "away_away_gd_avg": a_away_form["gd_avg"],
            "home_away_split_adv": h_home_form["pts_avg"] - a_away_form["pts_avg"],
            "home_rest_days": home_rest_days,
            "away_rest_days": away_rest_days,
            "rest_days_diff": home_rest_days - away_rest_days,
            "season_matchweek": float(matchweek),
            "season_progress": season_progress,
            "is_midweek": is_midweek,
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
        home_history[home].append((h_pts, to_int(r["FTHG"]), to_int(r["FTAG"])))
        away_history[away].append((a_pts, to_int(r["FTAG"]), to_int(r["FTHG"])))
        last_seen_date[home] = cur_date
        last_seen_date[away] = cur_date
        season_match_count[season_key] += 1

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

    seasons = sorted(features["SeasonKey"].dropna().unique().tolist())
    if len(seasons) < 3:
        raise ValueError("Need at least 3 seasons to run train/validation/test splits.")

    holdout_season = seasons[-1]
    val_season = seasons[-2]

    # Train/test split
    train_mask = features["SeasonKey"] != holdout_season
    X_train = X[train_mask].copy()
    y_train = y[train_mask].copy()
    X_test = X[~train_mask].copy()
    y_test = y[~train_mask].copy()

    # Use a season-aware validation split for light hyperparameter tuning.
    val_mask = train_mask & (features["SeasonKey"] == val_season)
    tune_train_mask = train_mask & ~val_mask
    if int(val_mask.sum()) == 0 or int(tune_train_mask.sum()) == 0:
        val_mask = train_mask
        tune_train_mask = train_mask

    X_tune_train = X[tune_train_mask].copy()
    y_tune_train = y[tune_train_mask].copy()
    X_val = X[val_mask].copy()
    y_val = y[val_mask].copy()

    # Preprocessing
    cat_cols = ["HomeTeam", "AwayTeam", "SeasonKey"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    tune_preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", StandardScaler(), num_cols),
    ])

    X_tune_processed = tune_preprocessor.fit_transform(X_tune_train)
    X_val_processed = tune_preprocessor.transform(X_val)

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", StandardScaler(), num_cols),
    ])
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Label encoding
    label_map = {"H": 0, "D": 1, "A": 2}
    y_tune_enc = y_tune_train.map(label_map).values
    y_val_enc = y_val.map(label_map).values
    y_train_enc = y_train.map(label_map).values
    y_test_enc = y_test.map(label_map).values

    print("\n" + "="*70)
    print("ENSEMBLE MODEL TRAINING")
    print("="*70)

    # Import training-only libraries lazily so inference environments
    # don't need to install them just to import this module.
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    def predict_with_thresholds(proba: np.ndarray, thresholds: dict) -> np.ndarray:
        scale = np.array(
            [
                max(float(thresholds.get("H", 0.5)), 1e-6),
                max(float(thresholds.get("D", 0.5)), 1e-6),
                max(float(thresholds.get("A", 0.5)), 1e-6),
            ]
        )
        return np.argmax(proba / scale, axis=1)

    def tune_decision_thresholds(y_true: np.ndarray, proba: np.ndarray) -> dict:
        best = {"H": 0.5, "D": 0.5, "A": 0.5}
        best_score = -1.0
        for h_t in [0.45, 0.50, 0.55]:
            for d_t in [0.22, 0.26, 0.30, 0.34]:
                for a_t in [0.45, 0.50, 0.55]:
                    candidate = {"H": h_t, "D": d_t, "A": a_t}
                    pred = predict_with_thresholds(proba, candidate)
                    acc = accuracy_score(y_true, pred)
                    mf1 = f1_score(y_true, pred, average="macro", zero_division=0)
                    score = 0.6 * acc + 0.4 * mf1
                    if score > best_score:
                        best_score = score
                        best = candidate
        return best

    def select_best_params(name: str, model_cls, base_params: dict, grid_params: dict) -> dict:
        best_score = -1.0
        best_params = {}
        print(f"  Tuning {name} on season {val_season} validation set...")
        for params in ParameterGrid(grid_params):
            candidate_params = dict(base_params)
            candidate_params.update(params)
            model = model_cls(**candidate_params)
            model.fit(X_tune_processed, y_tune_enc)
            pred = model.predict(X_val_processed)
            val_acc = accuracy_score(y_val_enc, pred)
            val_macro_f1 = f1_score(y_val_enc, pred, average="macro", zero_division=0)
            # Blend of accuracy and macro-F1 to slightly improve class balance.
            score = 0.7 * val_acc + 0.3 * val_macro_f1
            if score > best_score:
                best_score = score
                best_params = params
        print(f"  Best {name} params: {best_params} (blended score: {best_score:.4f})")
        return best_params

    models = {}
    predictions = {}
    decision_thresholds = {}

    # Model 1: XGBoost
    print("\n[1/3] Training XGBoost...")
    xgb_base_params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        "random_state": 42,
    }
    xgb_grid = {
        "n_estimators": [350, 500],
        "learning_rate": [0.03, 0.05],
        "max_depth": [4, 5],
        "min_child_weight": [2],
    }
    xgb_best = select_best_params("XGBoost", XGBClassifier, xgb_base_params, xgb_grid)
    xgb_tune = XGBClassifier(**{**xgb_base_params, **xgb_best})
    xgb_tune.fit(X_tune_processed, y_tune_enc, verbose=False)
    decision_thresholds["XGBoost"] = tune_decision_thresholds(y_val_enc, xgb_tune.predict_proba(X_val_processed))
    print(f"  XGBoost thresholds: {decision_thresholds['XGBoost']}")
    xgb = XGBClassifier(**{**xgb_base_params, **xgb_best})
    xgb.fit(X_train_processed, y_train_enc, verbose=False)
    xgb_pred = xgb.predict(X_test_processed)
    xgb_proba = xgb.predict_proba(X_test_processed)
    models["XGBoost"] = xgb
    predictions["XGBoost"] = (xgb_pred, xgb_proba)

    # Model 2: LightGBM
    print("[2/3] Training LightGBM...")
    lgb_base_params = {
        "num_class": 3,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        "random_state": 42,
        "verbose": -1,
    }
    lgb_grid = {
        "n_estimators": [300, 500],
        "learning_rate": [0.03, 0.05],
        "num_leaves": [31, 63],
    }
    lgb_best = select_best_params("LightGBM", LGBMClassifier, lgb_base_params, lgb_grid)
    lgb_tune = LGBMClassifier(**{**lgb_base_params, **lgb_best})
    lgb_tune.fit(X_tune_processed, y_tune_enc)
    decision_thresholds["LightGBM"] = tune_decision_thresholds(y_val_enc, lgb_tune.predict_proba(X_val_processed))
    print(f"  LightGBM thresholds: {decision_thresholds['LightGBM']}")
    lgb = LGBMClassifier(**{**lgb_base_params, **lgb_best})
    lgb.fit(X_train_processed, y_train_enc)
    lgb_pred = lgb.predict(X_test_processed)
    lgb_proba = lgb.predict_proba(X_test_processed)
    models["LightGBM"] = lgb
    predictions["LightGBM"] = (lgb_pred, lgb_proba)

    # Model 3: Logistic Regression (for baseline/calibration)
    print("[3/3] Training Logistic Regression...")
    lr_base_params = {
        "max_iter": 500,
        "multi_class": "multinomial",
        "random_state": 42,
    }
    lr_grid = {
        "C": [0.5, 1.0, 2.0],
        "class_weight": [None, "balanced"],
    }
    lr_best = select_best_params("LogReg", LogisticRegression, lr_base_params, lr_grid)
    lr_tune = LogisticRegression(**{**lr_base_params, **lr_best})
    lr_tune.fit(X_tune_processed, y_tune_enc)
    decision_thresholds["LogReg"] = tune_decision_thresholds(y_val_enc, lr_tune.predict_proba(X_val_processed))
    print(f"  LogReg thresholds: {decision_thresholds['LogReg']}")
    lr = LogisticRegression(**{**lr_base_params, **lr_best})
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
    print(f"MODEL PERFORMANCE (Season {holdout_season} - Test Set)")
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

        thresholded_acc = acc
        thresholded_macro_f1 = f1_score(y_test_enc, pred, average="macro", zero_division=0)
        if name in decision_thresholds:
            thresholded_pred = predict_with_thresholds(proba, decision_thresholds[name])
            thresholded_acc = accuracy_score(y_test_enc, thresholded_pred)
            thresholded_macro_f1 = f1_score(y_test_enc, thresholded_pred, average="macro", zero_division=0)

        results[name] = {
            "accuracy": acc,
            "auc": auc,
            "logloss": ll,
            "thresholded_accuracy": thresholded_acc,
            "thresholded_macro_f1": thresholded_macro_f1,
        }

        print(f"\n[{name}]")
        print(f"  Accuracy:  {acc:.4f}")
        if name in decision_thresholds:
            print(f"  Thresholded Accuracy: {thresholded_acc:.4f}")
            print(f"  Thresholded Macro-F1: {thresholded_macro_f1:.4f}")
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
    
    for _, row in importance.iterrows():
        print(f"  Feature {int(row['feature'])}: {row['importance']:.4f}")

    return {
        "models": models,
        "preprocessor": preprocessor,
        "feature_columns": list(X.columns),
        "label_map": label_map,
        "predictions": predictions,
        "decision_thresholds": decision_thresholds,
        "results": results,
        "y_test": y_test.values,
        "test_indices": ~train_mask,
        "holdout_season": holdout_season,
    }


def _build_payload(training_output: dict, model_name: str, window: int = 5) -> dict:
    label_map = training_output["label_map"]
    inv_label_map = {v: k for k, v in label_map.items()}
    return {
        "model_name": model_name,
        "model": training_output["models"][model_name],
        "preprocessor": training_output["preprocessor"],
        "feature_columns": training_output["feature_columns"],
        "label_map": label_map,
        "inv_label_map": inv_label_map,
        "window": window,
        "decision_thresholds": training_output.get("decision_thresholds", {}).get(
            model_name, {"H": 0.5, "D": 0.5, "A": 0.5}
        ),
        "holdout_season": training_output.get("holdout_season"),
    }


def save_artifacts(training_output: dict, window: int = 5) -> dict:
    """
    Persist:
    - Baseline model artifact (LogReg)
    - Advanced model artifact (best of XGBoost/LightGBM by holdout accuracy)
    - Deployment artifact (best of baseline vs advanced by holdout accuracy)
    """
    results = training_output["results"]
    baseline_name = "LogReg"
    advanced_candidates = ["XGBoost", "LightGBM"]
    advanced_name = max(advanced_candidates, key=lambda k: results[k]["accuracy"])

    baseline_payload = _build_payload(training_output, baseline_name, window=window)
    advanced_payload = _build_payload(training_output, advanced_name, window=window)

    with BASELINE_ARTIFACT_PATH.open("wb") as f:
        pickle.dump(baseline_payload, f)
    with ADVANCED_ARTIFACT_PATH.open("wb") as f:
        pickle.dump(advanced_payload, f)

    baseline_acc = float(results[baseline_name]["accuracy"])
    advanced_acc = float(results[advanced_name]["accuracy"])
    deployed_name = advanced_name if advanced_acc >= baseline_acc else baseline_name
    deployed_payload = advanced_payload if deployed_name == advanced_name else baseline_payload
    deployed_payload = dict(deployed_payload)
    deployed_payload["deployment_selected_by"] = "holdout_accuracy"
    deployed_payload["deployment_candidates"] = {
        baseline_name: baseline_acc,
        advanced_name: advanced_acc,
    }

    with ARTIFACT_PATH.open("wb") as f:
        pickle.dump(deployed_payload, f)

    return {
        "baseline_path": BASELINE_ARTIFACT_PATH,
        "advanced_path": ADVANCED_ARTIFACT_PATH,
        "deploy_path": ARTIFACT_PATH,
        "baseline_model": baseline_name,
        "advanced_model": advanced_name,
        "deployed_model": deployed_name,
        "baseline_accuracy": baseline_acc,
        "advanced_accuracy": advanced_acc,
    }


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
    print(f"\nHoldout season: {results.get('holdout_season')}")
    print(
        f"Best Model: {best_model[0]} "
        f"(Accuracy: {best_model[1]['accuracy']:.4f})"
    )

    artifact_paths = save_artifacts(results, window=5)
    print(f"Saved baseline artifact: {artifact_paths['baseline_path'].name}")
    print(f"Saved advanced artifact: {artifact_paths['advanced_path'].name}")
    print(
        f"Deployment selected: {artifact_paths['deployed_model']} "
        f"(baseline={artifact_paths['baseline_accuracy']:.4f}, "
        f"advanced={artifact_paths['advanced_accuracy']:.4f})"
    )
    print(f"Saved deployable artifact: {artifact_paths['deploy_path'].name}")
    print("\nAdvanced training complete!")


if __name__ == "__main__":
    main()
