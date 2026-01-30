import asyncio
import aiohttp
import nest_asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from understat import Understat

nest_asyncio.apply()

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

st.set_page_config(page_title="xG Flow", layout="wide")

st.title("xG Flow")
st.caption("Fetch match data from Understat and plot xG momentum for a chosen match.")

LEAGUES = ["epl", "la_liga", "bundesliga", "serie_a", "ligue1", "rfpl"]

@st.cache_data(show_spinner=False)
def _get_team_results(team_name: str, season: int):
    async def _fetch():
        async with aiohttp.ClientSession(headers=HEADERS) as session:
            understat = Understat(session)
            return await understat.get_team_results(team_name, season)

    return asyncio.run(_fetch())


@st.cache_data(show_spinner=False)
def _get_teams(league_name: str, season: int):
    async def _fetch():
        async with aiohttp.ClientSession(headers=HEADERS) as session:
            understat = Understat(session)
            return await understat.get_teams(league_name, season)

    return asyncio.run(_fetch())


@st.cache_data(show_spinner=False)
def _get_match_shots(match_id: str):
    async def _fetch():
        async with aiohttp.ClientSession(headers=HEADERS) as session:
            understat = Understat(session)
            return await understat.get_match_shots(match_id)

    return asyncio.run(_fetch())


def _clean_shots_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["X", "Y", "xG", "minute"]
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _build_match_list_df(match_data: list) -> pd.DataFrame:
    df_matches = pd.DataFrame(match_data)
    if df_matches.empty:
        return df_matches

    df_matches = df_matches[df_matches["goals"].apply(lambda x: x is not None)]

    df_matches["opponent"] = np.where(
        df_matches["side"] == "h",
        df_matches["a"].apply(lambda x: x["title"]),
        df_matches["h"].apply(lambda x: x["title"]),
    )

    df_matches["result"] = df_matches.apply(
        lambda row: str(row["goals"]["h"]) + "-" + str(row["goals"]["a"]),
        axis=1,
    )

    return df_matches


def _get_flow_data(df: pd.DataFrame):
    minutes = [0] + df["minute"].tolist()
    xg_flow = [0] + df["xG_cumsum"].tolist()

    last_min = max(minutes[-1], 95)
    minutes.append(last_min)
    xg_flow.append(xg_flow[-1])

    return minutes, xg_flow


with st.sidebar:
    st.header("Inputs")
    league = st.selectbox("League", options=LEAGUES, index=0)
    season = st.number_input("Season year", min_value=2010, max_value=2030, value=2025, step=1)
    use_team_list = st.checkbox("Use team list", value=True)

    team_name = ""
    if use_team_list:
        try:
            teams = _get_teams(league, int(season))
            team_names = sorted({team.get("title") for team in teams if team.get("title")})
            if team_names:
                team_name = st.selectbox("Team", options=team_names)
            else:
                st.warning("No teams found for this league/season. Use manual entry.")
                use_team_list = False
        except Exception as exc:
            st.warning(f"Unable to load teams list: {exc}. Use manual entry.")
            use_team_list = False

    if not use_team_list:
        team_name = st.text_input("Team name", value="Liverpool")
    fetch_matches = st.button("Fetch matches")

if fetch_matches:
    if not team_name:
        st.error("Please enter a team name.")
    else:
        with st.spinner("Fetching matches..."):
            try:
                match_data = _get_team_results(team_name.strip(), int(season))
            except Exception as exc:
                st.error(f"Failed to fetch matches: {exc}")
                match_data = []

        df_matches = _build_match_list_df(match_data)

        if df_matches.empty:
            st.warning("No matches found. Check the team name and season.")
        else:
            st.subheader(f"Match list for {team_name.upper()}")
            display_cols = ["id", "datetime", "opponent", "result", "side"]
            st.dataframe(df_matches[display_cols].tail(10), use_container_width=True)

            match_id = st.selectbox(
                "Select a match ID",
                options=df_matches["id"].tolist(),
            )

            if match_id:
                with st.spinner("Fetching shot data..."):
                    try:
                        shot_data = _get_match_shots(str(match_id))
                    except Exception as exc:
                        st.error(f"Failed to fetch match shots: {exc}")
                        shot_data = None

                if shot_data:
                    df_home = pd.DataFrame(shot_data["h"])
                    df_away = pd.DataFrame(shot_data["a"])

                    if df_home.empty or df_away.empty:
                        st.warning("No shot data available for this match.")
                    else:
                        df_home = _clean_shots_df(df_home)
                        df_away = _clean_shots_df(df_away)

                        home_team_name = df_home["h_team"].iloc[0]
                        away_team_name = df_home["a_team"].iloc[0]

                        if home_team_name.lower() == team_name.strip().lower():
                            my_team_df = df_home
                            opp_df = df_away
                            opp_name = away_team_name
                            my_side = "Home"
                        else:
                            my_team_df = df_away
                            opp_df = df_home
                            opp_name = home_team_name
                            my_side = "Away"

                        st.info(f"Data Loaded: {team_name} ({my_side} vs {opp_name})")
                        st.write(f"Home Shots: {len(df_home)} | Away Shots: {len(df_away)}")

                        my_team_df = my_team_df.sort_values(by="minute")
                        my_team_df["xG_cumsum"] = my_team_df["xG"].cumsum()

                        opp_df = opp_df.sort_values(by="minute")
                        opp_df["xG_cumsum"] = opp_df["xG"].cumsum()

                        my_mins, my_flow = _get_flow_data(my_team_df)
                        opp_mins, opp_flow = _get_flow_data(opp_df)

                        fig, ax = plt.subplots(figsize=(12, 6))
                        fig.set_facecolor("#1e1e1e")
                        ax.set_facecolor("#1e1e1e")

                        ax.step(my_mins, my_flow, where="post", color="#c8102e", linewidth=3, label=team_name)
                        ax.step(opp_mins, opp_flow, where="post", color="#1f77b4", linewidth=3, label=opp_name)

                        ax.fill_between(my_mins, my_flow, step="post", color="#c8102e", alpha=0.2)
                        ax.fill_between(opp_mins, opp_flow, step="post", color="#1f77b4", alpha=0.2)

                        my_goals = my_team_df[my_team_df["result"] == "Goal"]
                        opp_goal = opp_df[opp_df["result"] == "Goal"]

                        for _, row in my_goals.iterrows():
                            ax.scatter(
                                row["minute"],
                                row["xG_cumsum"],
                                s=150,
                                color="gold",
                                edgecolors="red",
                                zorder=10,
                                label="Goal",
                            )

                        for _, row in opp_goal.iterrows():
                            ax.scatter(
                                row["minute"],
                                row["xG_cumsum"],
                                s=150,
                                color="white",
                                edgecolors="red",
                                zorder=10,
                                label="Goal",
                            )

                        max_xg = max(my_flow[-1], opp_flow[-1])
                        ax.set_ylim(0, max_xg + 0.5)
                        ax.set_xlim(0, 98)

                        ax.set_xlabel("Minute", fontsize=14, color="white")
                        ax.set_ylabel("Cumulative xG", fontsize=14, color="white")
                        ax.set_title(
                            f"xG Momentum: {team_name} vs {opp_name}",
                            fontsize=18,
                            color="white",
                            pad=15,
                        )

                        ax.tick_params(axis="x", colors="white")
                        ax.tick_params(axis="y", colors="white")
                        ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.3)

                        handles, labels = ax.get_legend_handles_labels()
                        by_label = dict(zip(labels, handles))
                        ax.legend(
                            by_label.values(),
                            by_label.keys(),
                            loc="upper left",
                            facecolor="#1e1e1e",
                            labelcolor="white",
                        )

                        st.pyplot(fig, use_container_width=True)
