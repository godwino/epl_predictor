import argparse
import pickle
from pathlib import Path

import pandas as pd

from advanced_train import build_advanced_features, load_matches

DATA_DIR = Path(__file__).resolve().parent
ARTIFACT_PATH = DATA_DIR / "advanced_model.pkl"


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
    model, preprocessor, inv_label_map: dict, feature_columns: list, home: str, away: str, window: int
) -> dict:
    X_pred = _build_single_match_features(home, away, window=window)
    X_pred = X_pred.reindex(columns=feature_columns, fill_value=0.0)
    X_pred_processed = preprocessor.transform(X_pred)

    classes = list(getattr(model, "classes_", [0, 1, 2]))
    proba = model.predict_proba(X_pred_processed)[0]
    probs_by_label = {inv_label_map[int(cls)]: float(p) for cls, p in zip(classes, proba)}

    return {
        "HomeTeam": home,
        "AwayTeam": away,
        "H": probs_by_label.get("H", 0.0),
        "D": probs_by_label.get("D", 0.0),
        "A": probs_by_label.get("A", 0.0),
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
    )
    print(f"Model: {artifact['model_name']}")
    print(f"Fixture: {args.home} vs {args.away}")
    print("Probabilities:")
    print(f"  H (Home Win): {pred['H']:.4f}")
    print(f"  D (Draw):     {pred['D']:.4f}")
    print(f"  A (Away Win): {pred['A']:.4f}")


if __name__ == "__main__":
    main()
