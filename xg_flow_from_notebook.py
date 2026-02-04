# """Generate and visualize an xG flow chart for a chosen team and match.

# This script fetches match results and shot data from Understat, lets the user
# pick a match, and plots cumulative xG for both teams.
# """

# import asyncio
# import aiohttp
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import streamlit as st
# from understat import Understat
# import nest_asyncio

# nest_asyncio.apply()

# HEADERS = {
#     "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
# }


# # Data Loading
# async def xG_flow(team_name, season):
#     async with aiohttp.ClientSession() as session:
#         understat = Understat(session)
#         print(f"Searching for {team_name} matches in {season}")
#         try:
#             result = await understat.get_team_results(team_name, season)
#             return result
#         except Exception as e:
#             print(f"An error occured: {e}")
#             return []


# async def get_shot_info(match_id):
#     async with aiohttp.ClientSession() as session:
#         understat = Understat(session)
#         shots = await understat.get_match_shots(match_id)
#         return shots


# def clean_df(df):
#     cols = ["X", "Y", "xG", "minute"]
#     for col in cols:
#         df[col] = pd.to_numeric(df[col])
#     return df


# # Feature Engineering
# def get_flow_data(df):
#     minutes = [0] + df["minute"].tolist()
#     xg_flow = [0] + df["xG_cumsum"].tolist()

#     last_min = max(minutes[-1], 95)
#     minutes.append(last_min)
#     xg_flow.append(xg_flow[-1])

#     return minutes, xg_flow


# # Visualization
# def plot_xg_flow(my_team_df, opp_df, target_team, opp_name):
#     my_team_df = my_team_df.sort_values(by="minute")
#     my_team_df["xG_cumsum"] = my_team_df["xG"].cumsum()

#     opp_df = opp_df.sort_values(by="minute")
#     opp_df["xG_cumsum"] = opp_df["xG"].cumsum()

#     my_mins, my_flow = get_flow_data(my_team_df)
#     opp_mins, opp_flow = get_flow_data(opp_df)

#     fig, ax = plt.subplots(figsize=(12, 6))
#     fig.set_facecolor("#1e1e1e")
#     ax.set_facecolor("#1e1e1e")

#     ax.step(my_mins, my_flow, where="post", color="#c8102e", linewidth=3, label=target_team)
#     ax.step(opp_mins, opp_flow, where="post", color="#1f77b4", linewidth=3, label=opp_name)

#     ax.fill_between(my_mins, my_flow, step="post", color="#c8102e", alpha=0.2)
#     ax.fill_between(opp_mins, opp_flow, step="post", color="#1f77b4", alpha=0.2)

#     my_goals = my_team_df[my_team_df["result"] == "Goal"]
#     opp_goal = opp_df[opp_df["result"] == "Goal"]

#     for _, row in my_goals.iterrows():
#         ax.scatter(
#             row["minute"],
#             row["xG_cumsum"],
#             s=150,
#             color="gold",
#             edgecolors="red",
#             zorder=10,
#             label="Goal",
#         )

#     for _, row in opp_goal.iterrows():
#         ax.scatter(
#             row["minute"],
#             row["xG_cumsum"],
#             s=150,
#             color="white",
#             edgecolors="red",
#             zorder=10,
#             label="Goal",
#         )

#     max_xg = max(my_flow[-1], opp_flow[-1])
#     ax.set_ylim(0, max_xg + 0.5)
#     ax.set_xlim(0, 98)

#     ax.set_xlabel("Minute", fontsize=14, color="white")
#     ax.set_ylabel("Cumulative xG", fontsize=14, color="white")
#     ax.set_title(f"xG Momentum: {target_team} vs {opp_name}", fontsize=18, color="white", pad=15)

#     ax.tick_params(axis="x", colors="white")
#     ax.tick_params(axis="y", colors="white")
#     ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.3)

#     handles, labels = ax.get_legend_handles_labels()
#     by_label = dict(zip(labels, handles))
#     ax.legend(by_label.values(), by_label.keys(), loc="upper left", facecolor="#1e1e1e", labelcolor="white")

#     return fig


# @st.cache_data(ttl=3600, show_spinner=False)
# def fetch_matches(team_name, season):
#     return asyncio.run(xG_flow(team_name, season))


# @st.cache_data(ttl=3600, show_spinner=False)
# def fetch_shots(match_id):
#     return asyncio.run(get_shot_info(match_id))


# def build_matches_df(match_data):
#     df_matches = pd.DataFrame(match_data)
#     if df_matches.empty:
#         return df_matches

#     df_matches = df_matches[df_matches["goals"].apply(lambda x: x is not None)]

#     df_matches["opponent"] = np.where(
#         df_matches["side"] == "h",
#         df_matches["a"].apply(lambda x: x["title"]),
#         df_matches["h"].apply(lambda x: x["title"]),
#     )

#     df_matches["result"] = df_matches.apply(
#         lambda row: f"{row['goals']['h']}-{row['goals']['a']}", axis=1
#     )

#     return df_matches


# def main():
#     st.set_page_config(page_title="xG Flow", layout="wide")
#     st.title("xG Flow Chart")

#     with st.sidebar:
#         st.header("Inputs")
#         target_team = st.text_input("Team name", value="Liverpool")
#         target_season = st.number_input("Season year", min_value=2014, max_value=2026, value=2022, step=1)
#         load_matches = st.button("Load matches")

#     if load_matches and target_team:
#         with st.spinner("Fetching matches..."):
#             match_data = fetch_matches(target_team, int(target_season))
#         st.session_state["match_data"] = match_data

#     match_data = st.session_state.get("match_data")
#     if not match_data:
#         st.info("Enter a team and season, then click Load matches.")
#         return

#     df_matches = build_matches_df(match_data)
#     if df_matches.empty:
#         st.warning("No matches found for that team and season.")
#         return

#     display_cols = ["id", "datetime", "opponent", "result", "side"]
#     st.subheader("Match list")
#     st.dataframe(df_matches[display_cols].sort_values("datetime", ascending=False), width="stretch")

#     options = df_matches[display_cols].sort_values("datetime", ascending=False)
#     option_labels = options.apply(
#         lambda row: f"{row['datetime']} | {row['opponent']} | {row['result']} | {row['side']} | ID {row['id']}",
#         axis=1,
#     ).tolist()

#     selected_label = st.selectbox("Choose a match", options=option_labels)
#     selected_id = options.iloc[option_labels.index(selected_label)]["id"]

#     if st.button("Generate xG flow"):
#         with st.spinner("Fetching shots..."):
#             shot_data = fetch_shots(str(selected_id))

#         df_home = pd.DataFrame(shot_data["h"])
#         df_away = pd.DataFrame(shot_data["a"])

#         df_home = clean_df(df_home)
#         df_away = clean_df(df_away)

#         home_team_name = df_home["h_team"].iloc[0]
#         away_team_name = df_home["a_team"].iloc[0]

#         if home_team_name == target_team:
#             my_team_df = df_home
#             opp_df = df_away
#             opp_name = away_team_name
#             my_side = "Home"
#         else:
#             my_team_df = df_away
#             opp_df = df_home
#             opp_name = home_team_name
#             my_side = "Away"

#         st.caption(f"Data Loaded: {target_team} ({my_side} vs {opp_name})")
#         st.caption(f"Home Shots: {len(df_home)} | Away Shots: {len(df_away)}")

#         fig = plot_xg_flow(my_team_df, opp_df, target_team, opp_name)
#         st.pyplot(fig, width="stretch")


# if __name__ == "__main__":
#     main()



import asyncio
import aiohttp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import nest_asyncio
from understat import Understat

# --- FIX 1: Apply the patch for Streamlit Cloud's loop ---
nest_asyncio.apply()

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Data Loading
async def xG_flow(team_name, season):
    # Added trust_env=True to handle cloud network configs better
    async with aiohttp.ClientSession(headers=HEADERS, trust_env=True) as session:
        understat = Understat(session)
        try:
            result = await understat.get_team_results(team_name, season)
            return result
        except Exception as e:
            # This will show the exact technical error on your app screen
            st.error(f"Technical Error Details: {str(e)}")
            return []

async def get_shot_info(match_id):
    async with aiohttp.ClientSession(headers=HEADERS, trust_env=True) as session:
        understat = Understat(session)
        try:
            shots = await understat.get_match_shots(match_id)
            return shots
        except Exception as e:
            print(f"Error fetching shots: {e}")
            return {}

def clean_df(df):
    if df.empty:
        return df
    cols = ["X", "Y", "xG", "minute"]
    # Ensure columns exist before converting
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col])
    return df

# Feature Engineering
def get_flow_data(df):
    if df.empty:
        return [], []
    
    minutes = [0] + df["minute"].tolist()
    xg_flow = [0] + df["xG_cumsum"].tolist()

    last_min = max(minutes[-1], 95)
    minutes.append(last_min)
    xg_flow.append(xg_flow[-1])

    return minutes, xg_flow

# Visualization
def plot_xg_flow(my_team_df, opp_df, target_team, opp_name):
    # Handle empty data gracefully
    if my_team_df.empty or opp_df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No Data Available", ha='center')
        return fig

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

    # Check if 'result' column exists before filtering
    if "result" in my_team_df.columns:
        my_goals = my_team_df[my_team_df["result"] == "Goal"]
        for _, row in my_goals.iterrows():
            ax.scatter(row["minute"], row["xG_cumsum"], s=150, color="gold", edgecolors="red", zorder=10, label="Goal")
            
    if "result" in opp_df.columns:
        opp_goal = opp_df[opp_df["result"] == "Goal"]
        for _, row in opp_goal.iterrows():
            ax.scatter(row["minute"], row["xG_cumsum"], s=150, color="white", edgecolors="red", zorder=10, label="Goal")

    max_val_my = my_flow[-1] if my_flow else 0
    max_val_opp = opp_flow[-1] if opp_flow else 0
    max_xg = max(max_val_my, max_val_opp)
    
    ax.set_ylim(0, max_xg + 0.5)
    ax.set_xlim(0, 98)

    ax.set_xlabel("Minute", fontsize=14, color="white")
    ax.set_ylabel("Cumulative xG", fontsize=14, color="white")
    ax.set_title(f"xG Momentum: {target_team} vs {opp_name}", fontsize=18, color="white", pad=15)

    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")
    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.3)

    # Fix duplicate labels in legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left", facecolor="#1e1e1e", labelcolor="white")

    return fig

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_matches(team_name, season):
    # Using the loop patch
    return asyncio.run(xG_flow(team_name, season))

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_shots(match_id):
    return asyncio.run(get_shot_info(match_id))

def build_matches_df(match_data):
    if not match_data:
        return pd.DataFrame()
        
    df_matches = pd.DataFrame(match_data)
    if df_matches.empty:
        return df_matches

    # Safe check for 'goals' column
    if "goals" not in df_matches.columns:
        return pd.DataFrame()

    df_matches = df_matches[df_matches["goals"].apply(lambda x: x is not None)]

    df_matches["opponent"] = np.where(
        df_matches["side"] == "h",
        df_matches["a"].apply(lambda x: x.get("title") if isinstance(x, dict) else str(x)),
        df_matches["h"].apply(lambda x: x.get("title") if isinstance(x, dict) else str(x)),
    )

    df_matches["result"] = df_matches.apply(
        lambda row: f"{row['goals'].get('h')}-{row['goals'].get('a')}", axis=1
    )

    return df_matches

def main():
    st.set_page_config(page_title="xG Flow", layout="wide")
    st.title("xG Flow Chart")

    with st.sidebar:
        st.header("Inputs")
        target_team = st.text_input("Team name", value="Liverpool")
        target_season = st.number_input("Season year", min_value=2014, max_value=2026, value=2022, step=1)
        load_matches = st.button("Load matches")

    if load_matches and target_team:
        with st.spinner("Fetching matches..."):
            match_data = fetch_matches(target_team, int(target_season))
            if not match_data:
                st.error("Failed to fetch matches. The data source might be blocking the request.")
            else:
                st.session_state["match_data"] = match_data

    match_data = st.session_state.get("match_data")
    if not match_data:
        st.info("Enter a team and season, then click Load matches.")
        return

    df_matches = build_matches_df(match_data)
    if df_matches.empty:
        st.warning("No valid matches found or data format error.")
        return

    display_cols = ["id", "datetime", "opponent", "result", "side"]
    st.subheader("Match list")
    st.dataframe(df_matches[display_cols].sort_values("datetime", ascending=False), width="stretch")

    options = df_matches[display_cols].sort_values("datetime", ascending=False)
    option_labels = options.apply(
        lambda row: f"{row['datetime']} | {row['opponent']} | {row['result']} | {row['side']} | ID {row['id']}",
        axis=1,
    ).tolist()

    selected_label = st.selectbox("Choose a match", options=option_labels)
    
    if selected_label:
        selected_index = option_labels.index(selected_label)
        selected_id = options.iloc[selected_index]["id"]

        if st.button("Generate xG flow"):
            with st.spinner("Fetching shots..."):
                shot_data = fetch_shots(str(selected_id))

            if not shot_data or ("h" not in shot_data and "a" not in shot_data):
                st.error("Could not load shot data for this match.")
                return

            df_home = pd.DataFrame(shot_data["h"])
            df_away = pd.DataFrame(shot_data["a"])

            df_home = clean_df(df_home)
            df_away = clean_df(df_away)

            if df_home.empty or df_away.empty:
                 st.error("Dataframes are empty.")
                 return

            home_team_name = df_home["h_team"].iloc[0] if "h_team" in df_home else "Home"
            away_team_name = df_home["a_team"].iloc[0] if "a_team" in df_home else "Away"

            if home_team_name == target_team:
                my_team_df = df_home
                opp_df = df_away
                opp_name = away_team_name
                my_side = "Home"
            else:
                my_team_df = df_away
                opp_df = df_home
                opp_name = home_team_name
                my_side = "Away"

            st.caption(f"Data Loaded: {target_team} ({my_side} vs {opp_name})")
            st.caption(f"Home Shots: {len(df_home)} | Away Shots: {len(df_away)}")

            fig = plot_xg_flow(my_team_df, opp_df, target_team, opp_name)
            st.pyplot(fig, width="stretch")

if __name__ == "__main__":
    main()