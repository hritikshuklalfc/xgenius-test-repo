import streamlit as st
import asyncio
import aiohttp
import nest_asyncio
import pandas as pd
from understat import Understat
import numpy as np
import matplotlib.pyplot as plt

nest_asyncio.apply()

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def run_async(coro):
    return asyncio.get_event_loop().run_until_complete(coro)

async def xG_flow(team_name, season):
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        try:
            result = await understat.get_team_results(team_name, season)
            return result
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return []

def clean_df(df):
    cols = ["X", "Y", "xG", "minute"]
    for col in cols:
        df[col] = pd.to_numeric(df[col])
    return df

def get_flow_data(df):
    minutes = [0] + df["minute"].tolist()
    xg_flow = [0] + df["xG_cumsum"].tolist()
    last_min = max(minutes[-1], 95)
    minutes.append(last_min)
    xg_flow.append(xg_flow[-1])
    return minutes, xg_flow

def plot_xg_flow(my_team_df, opp_df, target_team, opp_name, my_side):
    my_team_df = my_team_df.sort_values(by="minute")
    my_team_df["xG_cumsum"] = my_team_df["xG"].cumsum()
    opp_df = opp_df.sort_values(by="minute")
    opp_df["xG_cumsum"] = opp_df["xG"].cumsum()
    my_mins, my_flow = get_flow_data(my_team_df)
    opp_mins, opp_flow = get_flow_data(opp_df)
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.set_facecolor("#1e1e1e")
    ax.set_facecolor("#1e1e1e")
    ax.step(my_mins, my_flow, where="post", color="#c8102e", linewidth=3, label=target_team)
    ax.step(opp_mins, opp_flow, where="post", color="#1f77b4", linewidth=3, label=opp_name)
    ax.fill_between(my_mins, my_flow, step="post", color="#c8102e", alpha=0.2)
    ax.fill_between(opp_mins, opp_flow, step="post", color="#1f77b4", alpha=0.2)
    my_goals = my_team_df[my_team_df["result"] == "Goal"]
    opp_goal = opp_df[opp_df["result"] == "Goal"]
    for _, row in my_goals.iterrows():
        ax.scatter(row["minute"], row["xG_cumsum"], s=150, color="gold", edgecolors="red", zorder=10, label="Goal")
    for _, row in opp_goal.iterrows():
        ax.scatter(row["minute"], row["xG_cumsum"], s=150, color="white", edgecolors="red", zorder=10, label="Goal")
    max_xg = max(my_flow[-1], opp_flow[-1])
    ax.set_ylim(0, max_xg + 0.5)
    ax.set_xlim(0, 98)
    ax.set_xlabel("Minute", fontsize=14, color="white")
    ax.set_ylabel("Cumulative xG", fontsize=14, color="white")
    ax.set_title(f"xG Momentum: {target_team} vs {opp_name}", fontsize=18, color="white", pad=15)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', facecolor='#1e1e1e', labelcolor='white')
    st.pyplot(fig)

def main():
    st.title("xG Flow Visualizer")
    st.write("Analyze and visualize xG momentum for football matches using Understat data.")
    team_name = st.text_input("Enter the name of the team (e.g., Liverpool, Arsenal):")
    season = st.text_input("Enter Season Year (e.g., 2024, 2025):")
    if st.button("Get Match List"):
        if not team_name or not season:
            st.warning("Please enter both team name and season.")
            return
        match_data = run_async(xG_flow(team_name, season))
        if not match_data:
            st.error("No match data found.")
            return
        df_matches = pd.DataFrame(match_data)
        df_matches = df_matches[df_matches["goals"].apply(lambda x: x is not None)]
        df_matches["opponent"] = np.where(
            df_matches["side"] == "h",
            df_matches["a"].apply(lambda x: x["title"]),
            df_matches["h"].apply(lambda x: x["title"])
        )
        df_matches["result"] = df_matches.apply(
            lambda row: str(row["goals"]["h"]) + "-" + str(row["goals"]["a"]), axis=1
        )
        display_cols = ["id", "datetime", "opponent", "result", "side"]
        st.dataframe(df_matches[display_cols].tail(10))
        match_id = st.text_input("Paste the Match Id here:")
        if st.button("Get xG Flow for Match"):
            if not match_id:
                st.warning("Please enter a match id.")
                return
            async def get_shot_info(match_id):
                async with aiohttp.ClientSession() as session:
                    understat = Understat(session)
                    shots = await understat.get_match_shots(match_id)
                    return shots
            shot_data = run_async(get_shot_info(match_id))
            df_home = pd.DataFrame(shot_data["h"])
            df_away = pd.DataFrame(shot_data["a"])
            df_home = clean_df(df_home)
            df_away = clean_df(df_away)
            home_team_name = df_home["h_team"].iloc[0]
            away_team_name = df_home["a_team"].iloc[0]
            if home_team_name == team_name:
                my_team_df = df_home
                opp_df = df_away
                opp_name = away_team_name
                my_side = "Home"
            else:
                my_team_df = df_away
                opp_df = df_home
                opp_name = home_team_name
                my_side = "Away"
            st.success(f"Data Loaded: {team_name} ({my_side} vs {opp_name})")
            st.write(f"Home Shots: {len(df_home)} | Away Shots: {len(df_away)}")
            st.dataframe(my_team_df.head(3))
            plot_xg_flow(my_team_df, opp_df, team_name, opp_name, my_side)

if __name__ == "__main__":
    main()
