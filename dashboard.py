"""
KickCast Dashboard — Interactive World Cup 2026 Prediction Tool.

Run with: streamlit run dashboard.py

Features:
1. Head-to-Head: Select any two teams, see win/draw/loss probabilities
2. Tournament Overview: Win probabilities, group standings, advancement
3. Group Explorer: Detailed group-by-group predictions
4. Model Performance: Metrics comparison across all trained models
"""

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import joblib

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="KickCast — World Cup 2026 Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load data ─────────────────────────────────────────────────────────
WC_DIR = BASE / "data" / "raw" / "world_cup_2026"
SIM_DIR = BASE / "outputs" / "simulation_results"
ARTIFACTS = BASE / "outputs" / "model_artifacts"
SPLITS = BASE / "data" / "processed" / "splits"


@st.cache_resource
def load_model():
    """Load the best model for predictions."""
    # Try calibrated ensemble first, fall back to XGBoost
    for name in ["CalibratedEnsemble", "XGBoost_tuned_balanced"]:
        path = ARTIFACTS / f"{name}.joblib"
        if path.exists():
            return joblib.load(path), name
    return None, None


@st.cache_data
def load_data():
    groups = pd.read_csv(WC_DIR / "groups.csv")
    fixtures = pd.read_csv(WC_DIR / "fixtures.csv")
    bracket = pd.read_csv(WC_DIR / "knockout_bracket.csv")
    win_probs = pd.read_csv(SIM_DIR / "win_probabilities.csv")
    adv_probs = pd.read_csv(SIM_DIR / "advancement_probabilities.csv")
    gs_dist = pd.read_csv(SIM_DIR / "group_standings_distribution.csv")
    match_preds = pd.read_csv(SIM_DIR / "group_match_predictions.csv")
    feature_cols = pd.read_csv(SPLITS / "X_train.csv", nrows=0).columns.tolist()
    return groups, fixtures, bracket, win_probs, adv_probs, gs_dist, match_preds, feature_cols


@st.cache_resource
def load_team_state():
    """Load precomputed team state for predictions."""
    # Import feature builder
    sys.path.insert(0, str(BASE / "data" / "scripts"))
    from importlib import import_module
    fb_spec = import_module("02_build_features")
    from src.simulation import historical_name
    from collections import defaultdict, deque

    results = fb_spec.load_results()
    elo_hist = fb_spec.compute_elo_ratings(results)
    fifa_dates, fifa_by_date = fb_spec.load_fifa_rankings()
    players, vals = fb_spec.load_players_and_valuations()
    squad_agg, player_snap, top5_snap, quarters = fb_spec.precompute_squad_snapshots(
        players, vals, pd.Timestamp("2004-01-01")
    )
    wc_data = fb_spec.load_wc_history()

    ref_date = pd.Timestamp("2026-06-10")
    IMPORTANCE = fb_spec.IMPORTANCE

    # Build form tracker
    form_tracker = defaultdict(lambda: deque(maxlen=10))
    last_match_date = {}
    h2h_db = defaultdict(list)

    for _, row in results.iterrows():
        home, away = row["home_team"], row["away_team"]
        hs, aws = int(row["home_score"]), int(row["away_score"])
        date = row["date"]
        imp = IMPORTANCE.get(row["tournament"], 0.3)
        h_result = "win" if hs > aws else ("draw" if hs == aws else "loss")
        a_result = "win" if aws > hs else ("draw" if hs == aws else "loss")
        form_tracker[home].append({"r": h_result, "gf": hs, "ga": aws, "imp": imp})
        form_tracker[away].append({"r": a_result, "gf": aws, "ga": hs, "imp": imp})
        last_match_date[home] = date
        last_match_date[away] = date
        pair = tuple(sorted([home, away]))
        winner = home if hs > aws else (away if aws > hs else None)
        h2h_db[pair].append({"date": date, "winner": winner, "home": home})

    # Team confederation lookup
    team_conf = {}
    for dd in fifa_by_date.values():
        for t, d in dd.items():
            if d.get("conf"):
                team_conf[t] = d["conf"]

    return {
        "fb_spec": fb_spec,
        "elo_hist": elo_hist,
        "fifa_dates": fifa_dates,
        "fifa_by_date": fifa_by_date,
        "squad_agg": squad_agg,
        "quarters": quarters,
        "wc_data": wc_data,
        "form_tracker": form_tracker,
        "last_match_date": last_match_date,
        "h2h_db": h2h_db,
        "team_conf": team_conf,
        "ref_date": ref_date,
    }


def build_features(home, away, state, groups):
    """Build feature vector for a match between two teams."""
    from src.simulation import historical_name
    fb = state["fb_spec"]
    ref_date = state["ref_date"]

    f = {}
    f["match_importance"] = 1.0
    f["is_neutral"] = 1

    h_conf = state["team_conf"].get(home, "")
    a_conf = state["team_conf"].get(away, "")
    # Also check groups.csv confederations
    grp_conf = dict(zip(groups["team"], groups["confederation"]))
    if not h_conf:
        h_conf = grp_conf.get(home, "")
    if not a_conf:
        a_conf = grp_conf.get(away, "")
    f["same_confederation"] = int(h_conf != "" and h_conf == a_conf)
    f["home_continent_advantage"] = 0

    # Elo
    h_hist = historical_name(home)
    a_hist = historical_name(away)
    h_elo_data = state["elo_hist"].get(h_hist) or state["elo_hist"].get(home)
    a_elo_data = state["elo_hist"].get(a_hist) or state["elo_hist"].get(away)
    h_elo = h_elo_data[-1][1] if h_elo_data else 1500
    a_elo = a_elo_data[-1][1] if a_elo_data else 1500
    f["elo_diff"] = h_elo - a_elo

    h_mom = fb.get_elo_momentum(h_hist, ref_date, state["elo_hist"], n=5)
    a_mom = fb.get_elo_momentum(a_hist, ref_date, state["elo_hist"], n=5)
    f["elo_momentum_diff"] = (h_mom or 0) - (a_mom or 0)

    # FIFA Rankings
    h_rank, h_pts, _ = fb.get_fifa(h_hist, ref_date, state["fifa_dates"], state["fifa_by_date"])
    if h_rank is None:
        h_rank, h_pts, _ = fb.get_fifa(home, ref_date, state["fifa_dates"], state["fifa_by_date"])
    a_rank, a_pts, _ = fb.get_fifa(a_hist, ref_date, state["fifa_dates"], state["fifa_by_date"])
    if a_rank is None:
        a_rank, a_pts, _ = fb.get_fifa(away, ref_date, state["fifa_dates"], state["fifa_by_date"])
    f["rank_diff"] = (h_rank or 100) - (a_rank or 100)
    f["points_diff"] = (h_pts or 1000) - (a_pts or 1000)

    # Squad values
    h_sq = fb.get_squad(h_hist, ref_date, state["squad_agg"], state["quarters"])
    if h_sq is None:
        h_sq = fb.get_squad(home, ref_date, state["squad_agg"], state["quarters"])
    a_sq = fb.get_squad(a_hist, ref_date, state["squad_agg"], state["quarters"])
    if a_sq is None:
        a_sq = fb.get_squad(away, ref_date, state["squad_agg"], state["quarters"])

    log_t = lambda x: np.sign(x) * np.log1p(abs(x)) if x is not None and not (isinstance(x, float) and np.isnan(x)) else np.nan
    if h_sq and a_sq:
        for key in ["total", "top11", "attack", "mid", "def"]:
            f[f"squad_value_{key}_delta"] = log_t(h_sq[key] - a_sq[key])
        f["star_player_value_delta"] = log_t(h_sq["star"] - a_sq["star"])
        f["squad_depth_delta"] = log_t(h_sq["depth"] - a_sq["depth"])
    else:
        for col in ["squad_value_total_delta", "squad_value_top11_delta",
                     "squad_value_attack_delta", "squad_value_mid_delta",
                     "squad_value_def_delta", "star_player_value_delta",
                     "squad_depth_delta"]:
            f[col] = np.nan

    # Form
    def get_form(team):
        ht = historical_name(team)
        tf = state["form_tracker"].get(ht) or state["form_tracker"].get(team)
        if not tf or len(tf) == 0:
            return 0.5, 0.25, 0.0, 30.0
        wins = sum(1 for m in tf if m["r"] == "win")
        wr = wins / len(tf)
        wts = []
        for i, m in enumerate(reversed(list(tf))):
            decay = 0.9 ** i
            score = 1.0 if m["r"] == "win" else (0.5 if m["r"] == "draw" else 0.0)
            wts.append(decay * m["imp"] * score)
        wf = np.mean(wts) if wts else 0.25
        gds = [m["gf"] - m["ga"] for m in tf]
        gd = np.mean(gds)
        lm = state["last_match_date"].get(ht) or state["last_match_date"].get(team)
        days = (ref_date - lm).days if lm else 30
        return wr, wf, gd, days

    h_wr, h_wf, h_gd, h_rest = get_form(home)
    a_wr, a_wf, a_gd, a_rest = get_form(away)
    f["home_days_rest"] = h_rest
    f["away_days_rest"] = a_rest
    f["form_win_rate_diff"] = h_wr - a_wr
    f["form_weighted_diff"] = h_wf - a_wf
    f["goal_diff_delta"] = h_gd - a_gd

    # H2H
    pair = tuple(sorted([h_hist, a_hist]))
    hist = state["h2h_db"].get(pair, [])
    if not hist:
        pair = tuple(sorted([home, away]))
        hist = state["h2h_db"].get(pair, [])
    if hist:
        h_wins = sum(1 for h in hist if h["winner"] == h_hist or h["winner"] == home)
        draws = sum(1 for h in hist if h["winner"] is None)
        f["h2h_home_win_rate"] = h_wins / len(hist)
        f["h2h_draw_rate"] = draws / len(hist)
        f["h2h_matches_played"] = len(hist)
    else:
        f["h2h_home_win_rate"] = np.nan
        f["h2h_draw_rate"] = np.nan
        f["h2h_matches_played"] = 0

    f["tournament_wr_delta"] = 0.0

    # WC history
    h_wc = state["wc_data"].get(home, state["wc_data"].get(h_hist, {}))
    a_wc = state["wc_data"].get(away, state["wc_data"].get(a_hist, {}))
    for key, default in [("appearances", 0), ("knockout_rate", 0),
                          ("best_finish", 1), ("goals_per_game", 0)]:
        f[f"wc_{key}_diff"] = h_wc.get(key, default) - a_wc.get(key, default)

    f["injury_count_delta"] = 0
    f["injury_burden_delta"] = 0.0
    f["star_injury_flag"] = 0

    return f


# ── Confederation colors ──────────────────────────────────────────────
CONFED_COLORS = {
    "UEFA": "#003f88", "CONMEBOL": "#2a9d8f", "CONCACAF": "#e63946",
    "CAF": "#e9c46a", "AFC": "#f4a261", "OFC": "#8ecae6",
}


# ══════════════════════════════════════════════════════════════════════
# LOAD ALL DATA
# ══════════════════════════════════════════════════════════════════════
model, model_name = load_model()
groups, fixtures, bracket, win_probs, adv_probs, gs_dist, match_preds, feature_cols = load_data()
all_teams = sorted(groups["team"].tolist())
team_confed = dict(zip(groups["team"], groups["confederation"]))
team_group = dict(zip(groups["team"], groups["group"]))

# ── Sidebar ───────────────────────────────────────────────────────────
st.sidebar.title("KickCast")
st.sidebar.caption("2026 FIFA World Cup Predictor")
st.sidebar.markdown(f"**Model:** {model_name}")

page = st.sidebar.radio(
    "Navigate",
    ["Head-to-Head", "Tournament Overview", "Group Explorer", "Model Info"],
    index=0,
)

# ══════════════════════════════════════════════════════════════════════
# PAGE 1: HEAD-TO-HEAD
# ══════════════════════════════════════════════════════════════════════
if page == "Head-to-Head":
    st.title("Head-to-Head Match Predictor")
    st.markdown("Select any two teams to see predicted match probabilities.")

    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("Team A", all_teams, index=all_teams.index("France"))
    with col2:
        team_b = st.selectbox("Team B", all_teams, index=all_teams.index("Argentina"))

    if team_a == team_b:
        st.warning("Please select two different teams.")
    elif model is not None:
        with st.spinner("Loading team data..."):
            state = load_team_state()

        # Build features for both orientations and average (symmetric)
        f_ab = build_features(team_a, team_b, state, groups)
        f_ba = build_features(team_b, team_a, state, groups)
        X_ab = pd.DataFrame([f_ab]).reindex(columns=feature_cols)
        X_ba = pd.DataFrame([f_ba]).reindex(columns=feature_cols)

        proba_ab = model.predict_proba(X_ab)[0]
        proba_ba = model.predict_proba(X_ba)[0]

        # Symmetric average: AB's home win = A wins, BA's away win = A wins
        p_a_win = (proba_ab[0] + proba_ba[2]) / 2
        p_draw = (proba_ab[1] + proba_ba[1]) / 2
        p_b_win = (proba_ab[2] + proba_ba[0]) / 2
        total = p_a_win + p_draw + p_b_win
        p_a_win, p_draw, p_b_win = p_a_win / total, p_draw / total, p_b_win / total

        # Display probabilities
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(f"{team_a} Win", f"{p_a_win*100:.1f}%")
        with c2:
            st.metric("Draw", f"{p_draw*100:.1f}%")
        with c3:
            st.metric(f"{team_b} Win", f"{p_b_win*100:.1f}%")

        # Probability bar
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=["Match"], x=[p_a_win * 100], orientation="h",
            name=f"{team_a} Win", marker_color="#2a9d8f",
            text=f"{p_a_win*100:.1f}%", textposition="inside",
        ))
        fig.add_trace(go.Bar(
            y=["Match"], x=[p_draw * 100], orientation="h",
            name="Draw", marker_color="#e9c46a",
            text=f"{p_draw*100:.1f}%", textposition="inside",
        ))
        fig.add_trace(go.Bar(
            y=["Match"], x=[p_b_win * 100], orientation="h",
            name=f"{team_b} Win", marker_color="#e76f51",
            text=f"{p_b_win*100:.1f}%", textposition="inside",
        ))
        fig.update_layout(
            barmode="stack", height=120, margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(range=[0, 100], showticklabels=False),
            yaxis=dict(showticklabels=False),
            showlegend=True, legend=dict(orientation="h", y=-0.3),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Team comparison details
        st.markdown("### Team Comparison")
        from src.simulation import historical_name
        h_hist = historical_name(team_a)
        a_hist = historical_name(team_b)

        h_elo_data = state["elo_hist"].get(h_hist) or state["elo_hist"].get(team_a)
        a_elo_data = state["elo_hist"].get(a_hist) or state["elo_hist"].get(team_b)
        h_elo = h_elo_data[-1][1] if h_elo_data else 1500
        a_elo = a_elo_data[-1][1] if a_elo_data else 1500

        h_rank, h_pts, _ = state["fb_spec"].get_fifa(
            h_hist, state["ref_date"], state["fifa_dates"], state["fifa_by_date"])
        a_rank, a_pts, _ = state["fb_spec"].get_fifa(
            a_hist, state["ref_date"], state["fifa_dates"], state["fifa_by_date"])

        comp_df = pd.DataFrame({
            "Metric": ["Elo Rating", "FIFA Rank", "FIFA Points", "Confederation", "WC Group"],
            team_a: [f"{h_elo:.0f}", h_rank or "N/A", f"{h_pts:.0f}" if h_pts else "N/A",
                     team_confed.get(team_a, "?"), team_group.get(team_a, "?")],
            team_b: [f"{a_elo:.0f}", a_rank or "N/A", f"{a_pts:.0f}" if a_pts else "N/A",
                     team_confed.get(team_b, "?"), team_group.get(team_b, "?")],
        })
        st.dataframe(comp_df, hide_index=True, use_container_width=True)

        # H2H history
        pair = tuple(sorted([h_hist, a_hist]))
        hist = state["h2h_db"].get(pair, [])
        if not hist:
            pair = tuple(sorted([team_a, team_b]))
            hist = state["h2h_db"].get(pair, [])
        if hist:
            total_h2h = len(hist)
            a_wins = sum(1 for h in hist if h["winner"] == h_hist or h["winner"] == team_a)
            b_wins = sum(1 for h in hist if h["winner"] == a_hist or h["winner"] == team_b)
            draws = total_h2h - a_wins - b_wins
            st.markdown(f"### Head-to-Head Record ({total_h2h} matches)")
            hc1, hc2, hc3 = st.columns(3)
            hc1.metric(f"{team_a} Wins", a_wins)
            hc2.metric("Draws", draws)
            hc3.metric(f"{team_b} Wins", b_wins)
        else:
            st.info("No head-to-head history found between these teams.")

    else:
        st.error("No trained model found. Run the training pipeline first.")


# ══════════════════════════════════════════════════════════════════════
# PAGE 2: TOURNAMENT OVERVIEW
# ══════════════════════════════════════════════════════════════════════
elif page == "Tournament Overview":
    st.title("2026 FIFA World Cup — Tournament Predictions")

    # Win probabilities bar chart
    st.markdown("### Tournament Win Probabilities")
    wp = win_probs.sort_values("win_probability", ascending=True)
    wp["confederation"] = wp["team"].map(team_confed)
    wp["color"] = wp["confederation"].map(CONFED_COLORS).fillna("#adb5bd")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=wp["team"], x=wp["win_probability"], orientation="h",
        marker_color=wp["color"].tolist(),
        text=[f"{v:.1f}%" for v in wp["win_probability"]],
        textposition="outside", textfont_size=9,
    ))
    for confed, color in CONFED_COLORS.items():
        fig.add_trace(go.Bar(y=[None], x=[None], orientation="h",
                             marker_color=color, name=confed, showlegend=True))
    fig.update_layout(
        height=1100, width=800,
        xaxis_title="Win Probability (%)",
        showlegend=True, legend=dict(x=0.7, y=0.05),
        margin=dict(l=150),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Advancement probabilities table
    st.markdown("### Round-by-Round Advancement")
    adv_display = adv_probs.copy()
    for col in ["Group", "R32", "R16", "QF", "SF", "Final", "Winner"]:
        adv_display[col] = adv_display[col].map(lambda x: f"{x:.1f}%")
    st.dataframe(adv_display, hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE 3: GROUP EXPLORER
# ══════════════════════════════════════════════════════════════════════
elif page == "Group Explorer":
    st.title("Group Stage Explorer")

    selected_group = st.selectbox("Select Group", sorted(groups["group"].unique()))

    grp_teams = groups[groups["group"] == selected_group]["team"].tolist()
    grp_matches = match_preds[match_preds["group"] == selected_group]
    grp_standings = gs_dist[gs_dist["group"] == selected_group].sort_values("pos_1", ascending=False)

    # Standings distribution
    st.markdown(f"### Group {selected_group} — Position Probabilities")
    fig = go.Figure()
    pos_colors = ["#2a9d8f", "#264653", "#e9c46a", "#e63946"]
    for pos in [1, 2, 3, 4]:
        fig.add_trace(go.Bar(
            y=grp_standings["team"], x=grp_standings[f"pos_{pos}"],
            name=f"{pos}{'st' if pos==1 else 'nd' if pos==2 else 'rd' if pos==3 else 'th'}",
            orientation="h", marker_color=pos_colors[pos-1],
        ))
    fig.update_layout(barmode="stack", height=250, xaxis_title="Probability (%)",
                      legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(fig, use_container_width=True)

    # Match predictions
    st.markdown(f"### Group {selected_group} — Match Predictions")
    for _, match in grp_matches.iterrows():
        col1, col2, col3 = st.columns([2, 3, 1])
        with col1:
            st.write(f"**M{int(match['match_number'])}:** {match['home_team']} vs {match['away_team']}")
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Bar(y=[""], x=[match["p_home_win"]*100], orientation="h",
                                 name="Home", marker_color="#2a9d8f"))
            fig.add_trace(go.Bar(y=[""], x=[match["p_draw"]*100], orientation="h",
                                 name="Draw", marker_color="#e9c46a"))
            fig.add_trace(go.Bar(y=[""], x=[match["p_away_win"]*100], orientation="h",
                                 name="Away", marker_color="#e76f51"))
            fig.update_layout(barmode="stack", height=50, margin=dict(l=0,r=0,t=0,b=0),
                              showlegend=False, xaxis=dict(range=[0,100], showticklabels=False),
                              yaxis=dict(showticklabels=False))
            st.plotly_chart(fig, use_container_width=True, key=f"m{int(match['match_number'])}")
        with col3:
            st.write(f"**{match['predicted']}**")


# ══════════════════════════════════════════════════════════════════════
# PAGE 4: MODEL INFO
# ══════════════════════════════════════════════════════════════════════
elif page == "Model Info":
    st.title("Model Performance")

    # Load results summary
    for csv_name in ["enhanced_results_summary.csv", "results_summary.csv"]:
        path = ARTIFACTS / csv_name
        if path.exists():
            results = pd.read_csv(path)
            break
    else:
        st.error("No results summary found.")
        st.stop()

    results = results.sort_values("macro_f1", ascending=False)

    st.markdown("### Validation Set Metrics (sorted by Macro F1)")
    display_cols = ["model", "accuracy", "macro_f1", "log_loss", "auc_roc_ovr",
                    "acc_class_0", "acc_class_1", "acc_class_2"]
    display_cols = [c for c in display_cols if c in results.columns]
    st.dataframe(results[display_cols].round(4), hide_index=True, use_container_width=True)

    # Bar chart comparison
    st.markdown("### Model Comparison")
    metric = st.selectbox("Metric", ["macro_f1", "accuracy", "log_loss", "auc_roc_ovr"])
    ascending = metric == "log_loss"  # lower is better for log loss
    sorted_results = results.sort_values(metric, ascending=ascending)

    fig = px.bar(sorted_results, x=metric, y="model", orientation="h",
                 color=metric, color_continuous_scale="Viridis")
    fig.update_layout(height=500, yaxis=dict(categoryorder="total ascending" if not ascending else "total descending"))
    st.plotly_chart(fig, use_container_width=True)

    # Test set results if available
    test_path = ARTIFACTS / "enhanced_test_results.csv"
    if not test_path.exists():
        test_path = ARTIFACTS / "test_results.csv"
    if test_path.exists():
        st.markdown("### Test Set Results (2022 WC holdout)")
        test_results = pd.read_csv(test_path).sort_values("log_loss")
        st.dataframe(test_results[display_cols].round(4), hide_index=True, use_container_width=True)
