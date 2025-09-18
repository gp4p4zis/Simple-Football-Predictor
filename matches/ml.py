import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from .models import Match
from sklearn.preprocessing import StandardScaler

def normalize_probs(home, draw, away):
    """Convert odds to normalized implied probabilities."""
    if not home or not draw or not away:
        return None
    try:
        p_home, p_draw, p_away = 1/home, 1/draw, 1/away
        total = p_home + p_draw + p_away
        return p_home/total, p_draw/total, p_away/total
    except ZeroDivisionError:
        return None


def build_training_dataframe():
    matches = Match.objects.filter(status=Match.Status.FINISHED).exclude(home_score=None, away_score=None)
    rows = []
    for m in matches:
        rows.append({
            "season": m.kickoff_at.year,
            "home_team": m.home_team.name,
            "away_team": m.away_team.name,
            "home_score": m.home_score,
            "away_score": m.away_score,
            "HS": m.HS, "AS": m.AS,
            "HST": m.HST, "AST": m.AST,
            "HC": m.HC, "AC": m.AC,
            "B365H": m.B365H, "B365D": m.B365D, "B365A": m.B365A
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()

    df["result"] = df.apply(
        lambda r: "H" if r.home_score > r.away_score else ("A" if r.home_score < r.away_score else "D"),
        axis=1
    )
    return df


def compute_team_profiles(df, shrink_k: int = 30):
    """
    Compute average stats, average odds, and games played for teams at home/away.
    Applies shrinkage to account for teams with few games.
    """
    if df.empty:
        return pd.DataFrame()

    all_teams = pd.unique(df[["home_team", "away_team"]].values.ravel("K"))
    all_seasons = sorted(df["season"].unique())
    rows = []

    for team in all_teams:
        for season in all_seasons:
            # Home games
            home_games = df[(df["home_team"] == team) & (df["season"] == season)]
            rows.append({
                "team": team,
                "side": "home",
                "season": season,
                "games_played": len(home_games),
                "goals_scored_avg": home_games["home_score"].mean() if not home_games.empty else 0,
                "goals_conceded_avg": home_games["away_score"].mean() if not home_games.empty else 0,
                "shots_avg": home_games["HS"].mean() if not home_games.empty else 0,
                "shot_acc_avg": (home_games["HST"].sum()/home_games["HS"].sum() if home_games["HS"].sum()>0 else 0) if not home_games.empty else 0,
                "corners_avg": home_games["HC"].mean() if not home_games.empty else 0,
                "odds_win_avg": home_games["B365H"].mean(skipna=True) if not home_games.empty else 0,
                "odds_draw_avg": home_games["B365D"].mean(skipna=True) if not home_games.empty else 0,
            })

            # Away games
            away_games = df[(df["away_team"] == team) & (df["season"] == season)]
            rows.append({
                "team": team,
                "side": "away",
                "season": season,
                "games_played": len(away_games),
                "goals_scored_avg": away_games["away_score"].mean() if not away_games.empty else 0,
                "goals_conceded_avg": away_games["home_score"].mean() if not away_games.empty else 0,
                "shots_avg": away_games["AS"].mean() if not away_games.empty else 0,
                "shot_acc_avg": (away_games["AST"].sum()/away_games["AS"].sum() if away_games["AS"].sum()>0 else 0) if not away_games.empty else 0,
                "corners_avg": away_games["AC"].mean() if not away_games.empty else 0,
                "odds_win_avg": away_games["B365A"].mean(skipna=True) if not away_games.empty else 0,
                "odds_draw_avg": away_games["B365D"].mean(skipna=True) if not away_games.empty else 0,
            })

    profiles = pd.DataFrame(rows)

    # League averages for shrinkage
    league_means = profiles[[
        "goals_scored_avg","goals_conceded_avg","shots_avg",
        "shot_acc_avg","corners_avg","odds_win_avg","odds_draw_avg"
    ]].mean()

    # Aggregate across seasons
    profiles = profiles.groupby(["team","side"]).agg({
        "games_played":"sum",
        "goals_scored_avg":"mean",
        "goals_conceded_avg":"mean",
        "shots_avg":"mean",
        "shot_acc_avg":"mean",
        "corners_avg":"mean",
        "odds_win_avg":"mean",
        "odds_draw_avg":"mean"
    }).reset_index()

    # Apply shrinkage
    for col in ["goals_scored_avg","goals_conceded_avg","shots_avg","shot_acc_avg","corners_avg","odds_win_avg","odds_draw_avg"]:
        profiles[col] = (
            (profiles["games_played"] / (profiles["games_played"] + shrink_k)) * profiles[col] +
            (shrink_k / (profiles["games_played"] + shrink_k)) * league_means[col]
        )

    return profiles.set_index(["team","side"])

def train_model():
    df = build_training_dataframe()
    if df.empty:
        return None, None, None, None, None  # add scaler

    profiles = compute_team_profiles(df)

    feature_rows = []
    for _, row in df.iterrows():
        try:
            home_prof = profiles.loc[(row["home_team"], "home")]
            away_prof = profiles.loc[(row["away_team"], "away")]
        except KeyError:
            continue

        feature_rows.append({
            # home stats
            "home_goals_scored_avg": home_prof["goals_scored_avg"],
            "home_goals_conceded_avg": home_prof["goals_conceded_avg"],
            "home_shots_avg": home_prof["shots_avg"],
            "home_shot_acc_avg": home_prof["shot_acc_avg"],
            "home_corners_avg": home_prof["corners_avg"],
            "home_odds_win_avg": home_prof["odds_win_avg"],
            "home_odds_draw_avg": home_prof["odds_draw_avg"],

            # away stats
            "away_goals_scored_avg": away_prof["goals_scored_avg"],
            "away_goals_conceded_avg": away_prof["goals_conceded_avg"],
            "away_shots_avg": away_prof["shots_avg"],
            "away_shot_acc_avg": away_prof["shot_acc_avg"],
            "away_corners_avg": away_prof["corners_avg"],
            "away_odds_win_avg": away_prof["odds_win_avg"],
            "away_odds_draw_avg": away_prof["odds_draw_avg"],

            "result": row["result"],
        })

    feature_df = pd.DataFrame(feature_rows)
    if feature_df.empty:
        return None, None, None, None, None

    X = feature_df.drop(columns=["result"])
    y = feature_df["result"]

    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    if len(feature_df) < 5:
        model = LogisticRegression(max_iter=5000, multi_class="multinomial", class_weight="balanced")
        model.fit(X_scaled, y)
        acc = None
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        model = LogisticRegression(max_iter=5000, multi_class="multinomial", class_weight="balanced")
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)

    return model, acc, profiles, imputer, scaler



def probs_to_odds(probs: dict, margin: float = 0.06, draw_baseline: float = 0.34, adjustment_strength: float = 0.6) -> dict:
    """Convert model probabilities to realistic bookmaker odds."""
    prob_dict = probs.copy()
    for k in ["H","D","A"]:
        if k not in prob_dict:
            prob_dict[k] = 1/3 if k != "D" else draw_baseline

    favorite = max(prob_dict, key=prob_dict.get)
    fav_prob = prob_dict[favorite]

    for k in prob_dict:
        if k != favorite:
            prob_dict[k] = draw_baseline + (prob_dict[k] - draw_baseline) * adjustment_strength

    total_nonfav = sum(prob_dict[k] for k in ["H","D","A"] if k != favorite)
    prob_dict[favorite] = max(fav_prob, 1 - total_nonfav)

    return {k: round((1+margin)/prob_dict[k],2) if prob_dict[k]>0 else None for k in prob_dict}


def predict_match(model, home_team, away_team, profiles, imputer, scaler):
    """Predict probabilities for a match and convert them to bookmaker odds with 5% margin."""
    try:
        home_prof = profiles.loc[(home_team, "home")]
        away_prof = profiles.loc[(away_team, "away")]
    except KeyError:
        default_probs = {"H": 0.33, "D": 0.34, "A": 0.33}
        return {"probs": default_probs, "odds": probs_to_odds(default_probs, margin=0.05)}

    row = {
        "home_goals_scored_avg": home_prof["goals_scored_avg"],
        "home_goals_conceded_avg": home_prof["goals_conceded_avg"],
        "home_shots_avg": home_prof["shots_avg"],
        "home_shot_acc_avg": home_prof["shot_acc_avg"],
        "home_corners_avg": home_prof["corners_avg"],
        "home_odds_win_avg": home_prof["odds_win_avg"],
        "home_odds_draw_avg": home_prof["odds_draw_avg"],
        "away_goals_scored_avg": away_prof["goals_scored_avg"],
        "away_goals_conceded_avg": away_prof["goals_conceded_avg"],
        "away_shots_avg": away_prof["shots_avg"],
        "away_shot_acc_avg": away_prof["shot_acc_avg"],
        "away_corners_avg": away_prof["corners_avg"],
        "away_odds_win_avg": away_prof["odds_win_avg"],
        "away_odds_draw_avg": away_prof["odds_draw_avg"],
    }

    X_input = pd.DataFrame([row])
    X_imputed = imputer.transform(X_input)
    X_scaled = scaler.transform(X_imputed)

    probs_arr = model.predict_proba(X_scaled)[0]
    labels = model.classes_
    prob_dict = dict(zip(labels, probs_arr))

    odds_dict = probs_to_odds(prob_dict, margin=0.06)
    return {"probs": prob_dict, "odds": odds_dict}

