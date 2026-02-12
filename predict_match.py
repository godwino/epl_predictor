import argparse
import math
import pickle
from pathlib import Path

import pandas as pd

from epl_features import build_advanced_features, load_matches

DATA_DIR = Path(__file__).resolve().parent
ARTIFACT_PATH = DATA_DIR / "advanced_model.pkl"


def _parse_last4(label: str, value: str | None) -> float | None:
    """Parse a 4-match form string like WDLW into points-per-match in [0, 3]."""
    if value is None:
        return None
    text = value.strip().upper().replace(" ", "")
    if len(text) != 4 or any(ch not in {"W", "D", "L"} for ch in text):
        raise ValueError(f"{label} must be exactly 4 chars using only W/D/L (example: WDLW).")
    points = sum(3 if ch == "W" else 1 if ch == "D" else 0 for ch in text)
    return points / 4.0


def _apply_current_form_adjustment(probs: dict, home_last4: str | None, away_last4: str | None) -> dict:
    """Adjust probabilities using manual current-form input from last 4 results."""
    home_ppm = _parse_last4("home_last4", home_last4)
    away_ppm = _parse_last4("away_last4", away_last4)
    if home_ppm is None and away_ppm is None:
        return probs
    if home_ppm is None or away_ppm is None:
        raise ValueError("Provide both home_last4 and away_last4 together.")

    # Normalize form gap to [-1, 1] then apply light multiplicative bias.
    form_gap = (home_ppm - away_ppm) / 3.0
    h_mult = math.exp(0.45 * form_gap)
    a_mult = math.exp(-0.45 * form_gap)
    d_mult = max(0.75, 1.0 - 0.30 * abs(form_gap))

    adjusted = {
        "H": probs.get("H", 0.0) * h_mult,
        "D": probs.get("D", 0.0) * d_mult,
        "A": probs.get("A", 0.0) * a_mult,
    }
    total = adjusted["H"] + adjusted["D"] + adjusted["A"]
    if total <= 0:
        return probs
    return {k: v / total for k, v in adjusted.items()}


def _build_single_match_features(home: str, away: str, window: int = 5) -> pd.DataFrame:
    df = load_matches().copy()
    teams = set(df["HomeTeam"]).union(set(df["AwayTeam"]))
    if home not in teams:
        raise ValueError(f"Unknown home team: {home}")
    if away not in teams:
        raise ValueError(f"Unknown away team: {away}")
    if home == away:
        raise ValueError("Home and away teams must be different.")

    next_date = df["Date"].max() + pd.Timedelta(days=7)
    season_key = str(df["SeasonKey"].max())

    # Add one synthetic fixture so feature builder computes pre-match stats.
    upcoming = pd.DataFrame([{
        "Date": next_date,
        "SeasonKey": season_key,
        "HomeTeam": home,
        "AwayTeam": away,
        "FTHG": 0,
        "FTAG": 0,
        "FTR": "D",
        "HS": 0,
        "HST": 0,
        "HC": 0,
        "HF": 0,
        "HY": 0,
        "HR": 0,
        "AS": 0,
        "AST": 0,
        "AC": 0,
        "AF": 0,
        "AY": 0,
        "AR": 0,
    }])

    simulated = pd.concat([df, upcoming], ignore_index=True).sort_values("Date").reset_index(drop=True)
    feats = build_advanced_features(simulated, window=window)
    X_pred = feats.iloc[[-1]].drop(columns=["FTR", "Date"])
    return X_pred


def _predict_fixture(
    model,
    preprocessor,
    inv_label_map: dict,
    feature_columns: list,
    home: str,
    away: str,
    window: int,
    decision_thresholds: dict | None = None,
    home_last4: str | None = None,
    away_last4: str | None = None,
) -> dict:
    X_pred = _build_single_match_features(home, away, window=window)
    X_pred = X_pred.reindex(columns=feature_columns, fill_value=0.0)
    X_pred_processed = preprocessor.transform(X_pred)

    classes = list(getattr(model, "classes_", [0, 1, 2]))
    proba = model.predict_proba(X_pred_processed)[0]
    probs_by_label = {inv_label_map[int(cls)]: float(p) for cls, p in zip(classes, proba)}
    probs_by_label = _apply_current_form_adjustment(
        probs=probs_by_label,
        home_last4=home_last4,
        away_last4=away_last4,
    )
    thresholds = decision_thresholds or {"H": 0.5, "D": 0.5, "A": 0.5}
    scores = {
        "H": probs_by_label.get("H", 0.0) / max(float(thresholds.get("H", 0.5)), 1e-6),
        "D": probs_by_label.get("D", 0.0) / max(float(thresholds.get("D", 0.5)), 1e-6),
        "A": probs_by_label.get("A", 0.0) / max(float(thresholds.get("A", 0.5)), 1e-6),
    }
    predicted = max(scores, key=scores.get)

    return {
        "HomeTeam": home,
        "AwayTeam": away,
        "H": probs_by_label.get("H", 0.0),
        "D": probs_by_label.get("D", 0.0),
        "A": probs_by_label.get("A", 0.0),
        "Prediction": predicted,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict EPL match result probabilities (H/D/A).")
    parser.add_argument("--home", help="Home team name (must match dataset value).")
    parser.add_argument("--away", help="Away team name (must match dataset value).")
    parser.add_argument(
        "--fixtures-csv",
        help="CSV with columns HomeTeam,AwayTeam for batch prediction.",
    )
    parser.add_argument("--output-csv", help="Optional path to write batch predictions.")
    parser.add_argument("--home-last4", help="Optional home team recent form, 4 chars W/D/L (example: WDLW).")
    parser.add_argument("--away-last4", help="Optional away team recent form, 4 chars W/D/L (example: LDWW).")
    args = parser.parse_args()

    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"{ARTIFACT_PATH.name} not found. Run `python -B advanced_train.py` first."
        )

    with ARTIFACT_PATH.open("rb") as f:
        artifact = pickle.load(f)

    model = artifact["model"]
    preprocessor = artifact["preprocessor"]
    inv_label_map = artifact["inv_label_map"]
    window = artifact.get("window", 5)
    feature_columns = artifact["feature_columns"]
    decision_thresholds = artifact.get("decision_thresholds", {"H": 0.5, "D": 0.5, "A": 0.5})

    if args.fixtures_csv:
        fixtures = pd.read_csv(args.fixtures_csv)
        required = {"HomeTeam", "AwayTeam"}
        if not required.issubset(set(fixtures.columns)):
            raise ValueError("Fixtures CSV must include columns: HomeTeam, AwayTeam")

        rows = []
        for _, r in fixtures.iterrows():
            rows.append(
                _predict_fixture(
                    model=model,
                    preprocessor=preprocessor,
                    inv_label_map=inv_label_map,
                    feature_columns=feature_columns,
                    home=str(r["HomeTeam"]),
                    away=str(r["AwayTeam"]),
                    window=window,
                    decision_thresholds=decision_thresholds,
                    home_last4=str(r["HomeLast4"]) if "HomeLast4" in fixtures.columns and pd.notna(r["HomeLast4"]) else None,
                    away_last4=str(r["AwayLast4"]) if "AwayLast4" in fixtures.columns and pd.notna(r["AwayLast4"]) else None,
                )
            )

        out = pd.DataFrame(rows)
        print(f"Model: {artifact['model_name']}")
        print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        if args.output_csv:
            out.to_csv(args.output_csv, index=False)
            print(f"\nSaved: {args.output_csv}")
        return

    if not args.home or not args.away:
        raise ValueError("Provide --home and --away, or use --fixtures-csv for batch mode.")

    pred = _predict_fixture(
        model=model,
        preprocessor=preprocessor,
        inv_label_map=inv_label_map,
        feature_columns=feature_columns,
        home=args.home,
        away=args.away,
        window=window,
        decision_thresholds=decision_thresholds,
        home_last4=args.home_last4,
        away_last4=args.away_last4,
    )
    print(f"Model: {artifact['model_name']}")
    print(f"Fixture: {args.home} vs {args.away}")
    print("Probabilities:")
    print(f"  H (Home Win): {pred['H']:.4f}")
    print(f"  D (Draw):     {pred['D']:.4f}")
    print(f"  A (Away Win): {pred['A']:.4f}")
    print(f"Predicted Outcome: {pred['Prediction']}")


if __name__ == "__main__":
    main()
