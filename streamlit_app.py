import streamlit as st
import asyncio
import aiohttp
import pandas as pd
from understat import Understat
import numpy as np
import matplotlib.pyplot as plt
import nest_asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Page configuration
st.set_page_config(
    page_title="xG Flow Analyzer",
    page_icon="⚽",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #c8102e;
        color: white;
    }
    .stButton>button:hover {
        background-color: #a00d25;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("⚽ xG Flow Analyzer")
st.markdown("Analyze Expected Goals (xG) momentum for football matches")

# Headers for user agent
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Async function to get team results
async def xG_flow(team_name, season):
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        try:
            result = await understat.get_team_results(team_name, season)
            return result
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return []

# Async function to get shot information
async def get_shot_info(match_id):
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        shots = await understat.get_match_shots(match_id)
        return shots

# Function to clean dataframe
def clean_df(df):
    cols = ["X", "Y", "xG", "minute"]
    for col in cols:
        df[col] = pd.to_numeric(df[col])
    return df

# Function to get flow data
def get_flow_data(df):
    minutes = [0] + df["minute"].tolist()
    xg_flow = [0] + df["xG_cumsum"].tolist()
    
    last_min = max(minutes[-1], 95)
    minutes.append(last_min)
    xg_flow.append(xg_flow[-1])
    
    return minutes, xg_flow

# Initialize session state
if 'matches_loaded' not in st.session_state:
    st.session_state.matches_loaded = False
if 'match_data' not in st.session_state:
    st.session_state.match_data = None
if 'df_matches' not in st.session_state:
    st.session_state.df_matches = None
if 'target_team' not in st.session_state:
    st.session_state.target_team = ""

# Step 1: Team and Season Selection
st.header("Step 1: Select Team and Season")

col1, col2 = st.columns(2)
with col1:
    target_team = st.text_input(
        "Team Name",
        placeholder="e.g., Liverpool, Arsenal, Manchester City",
        help="Enter the team name as it appears on Understat"
    )
with col2:
    target_season = st.text_input(
        "Season Year",
        placeholder="e.g., 2024, 2023",
        help="Enter the year of the season"
    )

if st.button("Load Matches", type="primary"):
    if target_team and target_season:
        with st.spinner(f"Searching for {target_team} matches in {target_season}..."):
            try:
                match_data = asyncio.run(xG_flow(target_team, target_season))
                
                if match_data:
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
                    
                    st.session_state.matches_loaded = True
                    st.session_state.match_data = match_data
                    st.session_state.df_matches = df_matches
                    st.session_state.target_team = target_team
                    st.success(f"✅ Found {len(df_matches)} matches for {target_team}")
                else:
                    st.warning("No matches found. Please check the team name and season.")
            except Exception as e:
                st.error(f"Error loading matches: {e}")
    else:
        st.warning("Please enter both team name and season year.")

# Step 2: Match Selection and Visualization
if st.session_state.matches_loaded:
    st.header("Step 2: Select Match to Analyze")
    
    df_matches = st.session_state.df_matches
    
    # Display matches table
    st.subheader(f"Match List for {st.session_state.target_team.upper()}")
    display_cols = ["id", "datetime", "opponent", "result", "side"]
    st.dataframe(
        df_matches[display_cols].sort_values("datetime", ascending=False),
        use_container_width=True,
        hide_index=True
    )
    
    # Match selection
    match_id = st.text_input(
        "Match ID",
        placeholder="Enter the Match ID from the table above",
        help="Copy and paste the match ID from the table"
    )
    
    if st.button("Analyze Match", type="primary"):
        if match_id:
            with st.spinner("Loading match data..."):
                try:
                    # Get shot data
                    shot_data = asyncio.run(get_shot_info(match_id))
                    
                    df_home = pd.DataFrame(shot_data["h"])
                    df_away = pd.DataFrame(shot_data["a"])
                    
                    df_home = clean_df(df_home)
                    df_away = clean_df(df_away)
                    
                    home_team_name = df_home["h_team"].iloc[0]
                    away_team_name = df_home["a_team"].iloc[0]
                    
                    # Determine which team is the target team
                    if home_team_name == st.session_state.target_team:
                        my_team_df = df_home
                        opp_df = df_away
                        opp_name = away_team_name
                        my_side = "Home"
                    else:
                        my_team_df = df_away
                        opp_df = df_home
                        opp_name = home_team_name
                        my_side = "Away"
                    
                    st.success(f"Data Loaded: {st.session_state.target_team} ({my_side} vs {opp_name})")
                    st.info(f"Home Shots: {len(df_home)} | Away Shots: {len(df_away)}")
                    
                    # Calculate cumulative xG
                    my_team_df = my_team_df.sort_values(by="minute")
                    my_team_df["xG_cumsum"] = my_team_df["xG"].cumsum()
                    
                    opp_df = opp_df.sort_values(by="minute")
                    opp_df["xG_cumsum"] = opp_df["xG"].cumsum()
                    
                    # Get flow data
                    my_mins, my_flow = get_flow_data(my_team_df)
                    opp_mins, opp_flow = get_flow_data(opp_df)
                    
                    # Create visualization
                    fig, ax = plt.subplots(figsize=(14, 7))
                    fig.set_facecolor("#1e1e1e")
                    ax.set_facecolor("#1e1e1e")
                    
                    # Plot xG flow
                    ax.step(my_mins, my_flow, where="post", color="#c8102e", linewidth=3, label=st.session_state.target_team)
                    ax.step(opp_mins, opp_flow, where="post", color="#1f77b4", linewidth=3, label=opp_name)
                    
                    ax.fill_between(my_mins, my_flow, step="post", color="#c8102e", alpha=0.2)
                    ax.fill_between(opp_mins, opp_flow, step="post", color="#1f77b4", alpha=0.2)
                    
                    # Mark goals
                    my_goals = my_team_df[my_team_df["result"] == "Goal"]
                    opp_goals = opp_df[opp_df["result"] == "Goal"]
                    
                    for _, row in my_goals.iterrows():
                        ax.scatter(row["minute"], row["xG_cumsum"], s=150, color="gold", 
                                 edgecolors="red", zorder=10, label="Goal")
                    
                    for _, row in opp_goals.iterrows():
                        ax.scatter(row["minute"], row["xG_cumsum"], s=150, color="white", 
                                 edgecolors="red", zorder=10, label="Goal")
                    
                    # Formatting
                    max_xg = max(my_flow[-1], opp_flow[-1])
                    ax.set_ylim(0, max_xg + 0.5)
                    ax.set_xlim(0, 98)
                    
                    ax.set_xlabel("Minute", fontsize=14, color="white")
                    ax.set_ylabel("Cumulative xG", fontsize=14, color="white")
                    ax.set_title(f"xG Momentum: {st.session_state.target_team} vs {opp_name}", 
                               fontsize=18, color="white", pad=15)
                    
                    ax.tick_params(axis='x', colors='white')
                    ax.tick_params(axis='y', colors='white')
                    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
                    
                    # Legend without duplicates
                    handles, labels = ax.get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    ax.legend(by_label.values(), by_label.keys(), loc='upper left', 
                            facecolor='#1e1e1e', labelcolor='white')
                    
                    st.pyplot(fig)
                    
                    # Display match statistics
                    st.header("Match Statistics")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(f"{st.session_state.target_team} xG", f"{my_flow[-1]:.2f}")
                        st.metric(f"{st.session_state.target_team} Goals", len(my_goals))
                    
                    with col2:
                        st.metric(f"{opp_name} xG", f"{opp_flow[-1]:.2f}")
                        st.metric(f"{opp_name} Goals", len(opp_goals))
                    
                    with col3:
                        st.metric("xG Difference", f"{my_flow[-1] - opp_flow[-1]:.2f}")
                        st.metric("Total Shots", len(my_team_df) + len(opp_df))
                    
                except Exception as e:
                    st.error(f"Error analyzing match: {e}")
        else:
            st.warning("Please enter a Match ID.")

# Footer
st.markdown("---")
st.markdown("Data provided by [Understat](https://understat.com/) | Built with Streamlit")