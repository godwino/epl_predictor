import re
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent


def _season_key(path: Path) -> str:
    m = re.search(r"season-(\d{4})\.csv", path.name)
    return m.group(1) if m else ""


def load_matches() -> pd.DataFrame:
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
    history = defaultdict(lambda: deque(maxlen=window))
    h2h_history = defaultdict(deque)
    rows = []

    def to_int(val):
        try:
            return int(val)
        except Exception:
            return 0

    def get_form_stats(team: str, decay=True):
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
        consistency = float(1.0 / (1.0 + np.std(pts) if len(pts) > 1 else 1.0))

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
            "h2h_h_pts": h2h["h2h_h_pts"],
            "h2h_draws": h2h["h2h_draws"],
            "h2h_advantage": h2h["h2h_advantage"],
            "FTR": r["FTR"],
        })

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

        h2h_key = tuple(sorted([home, away]))
        h2h_history[h2h_key].append((r["FTR"], home))

    return pd.DataFrame(rows)
