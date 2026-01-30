import streamlit as st
import asyncio
import aiohttp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from understat import Understat

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
    }
    </style>
""", unsafe_allow_html=True)

# Headers configuration
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

# Async function to get match shots
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

# Main app
def main():
    st.title("⚽ xG Flow Analyzer")
    st.markdown("---")
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Match Selection")
        
        # Initialize session state
        if 'match_data' not in st.session_state:
            st.session_state.match_data = None
        if 'target_team' not in st.session_state:
            st.session_state.target_team = ""
        if 'target_season' not in st.session_state:
            st.session_state.target_season = ""
        
        target_team = st.text_input(
            "Team Name",
            value="Liverpool",
            help="Enter team name (e.g., Liverpool, Arsenal, Manchester City)"
        )
        
        target_season = st.text_input(
            "Season Year",
            value="2019",
            help="Enter season year (e.g., 2019, 2020, 2024)"
        )
        
        if st.button("🔍 Search Matches", type="primary"):
            with st.spinner(f"Searching for {target_team} matches in {target_season}..."):
                match_data = asyncio.run(xG_flow(target_team, target_season))
                
                if match_data:
                    st.session_state.match_data = match_data
                    st.session_state.target_team = target_team
                    st.session_state.target_season = target_season
                    st.success(f"Found {len(match_data)} matches!")
                else:
                    st.error("No matches found. Please check team name and season.")
    
    # Main content area
    if st.session_state.match_data:
        df_matches = pd.DataFrame(st.session_state.match_data)
        
        # Filter out matches without goals data
        df_matches = df_matches[df_matches["goals"].apply(lambda x: x is not None)]
        
        # Process match data
        df_matches["opponent"] = np.where(
            df_matches["side"] == "h",
            df_matches["a"].apply(lambda x: x["title"]),
            df_matches["h"].apply(lambda x: x["title"])
        )
        
        df_matches["result"] = df_matches.apply(
            lambda row: str(row["goals"]["h"]) + "-" + str(row["goals"]["a"]), axis=1
        )
        
        # Display match list
        st.subheader(f"📋 Match List for {st.session_state.target_team.upper()}")
        
        display_cols = ["id", "datetime", "opponent", "result", "side"]
        st.dataframe(
            df_matches[display_cols].tail(10),
            use_container_width=True,
            hide_index=True
        )
        
        # Match selection
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            match_id = st.selectbox(
                "Select a match to analyze:",
                options=df_matches["id"].tolist(),
                format_func=lambda x: f"{df_matches[df_matches['id']==x]['datetime'].values[0]} - {df_matches[df_matches['id']==x]['opponent'].values[0]} ({df_matches[df_matches['id']==x]['result'].values[0]})"
            )
        
        with col2:
            analyze_button = st.button("📊 Analyze Match", type="primary")
        
        if analyze_button and match_id:
            with st.spinner("Loading match data..."):
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
                
                st.info(f"**{st.session_state.target_team}** ({my_side} vs {opp_name})")
                st.write(f"Home Shots: {len(df_home)} | Away Shots: {len(df_away)}")
                
                # Prepare data for visualization
                my_team_df = my_team_df.sort_values(by="minute")
                my_team_df["xG_cumsum"] = my_team_df["xG"].cumsum()
                
                opp_df = opp_df.sort_values(by="minute")
                opp_df["xG_cumsum"] = opp_df["xG"].cumsum()
                
                my_mins, my_flow = get_flow_data(my_team_df)
                opp_mins, opp_flow = get_flow_data(opp_df)
                
                # Create visualization
                fig, ax = plt.subplots(figsize=(12, 6))
                fig.set_facecolor("#1e1e1e")
                ax.set_facecolor("#1e1e1e")
                
                ax.step(my_mins, my_flow, where="post", color="#c8102e", linewidth=3, label=st.session_state.target_team)
                ax.step(opp_mins, opp_flow, where="post", color="#1f77b4", linewidth=3, label=opp_name)
                
                ax.fill_between(my_mins, my_flow, step="post", color="#c8102e", alpha=0.2)
                ax.fill_between(opp_mins, opp_flow, step="post", color="#1f77b4", alpha=0.2)
                
                # Mark goals
                my_goals = my_team_df[my_team_df["result"] == "Goal"]
                opp_goals = opp_df[opp_df["result"] == "Goal"]
                
                for _, row in my_goals.iterrows():
                    ax.scatter(row["minute"], row["xG_cumsum"], s=150, color="gold", edgecolors="red", zorder=10)
                
                for _, row in opp_goals.iterrows():
                    ax.scatter(row["minute"], row["xG_cumsum"], s=150, color="white", edgecolors="red", zorder=10)
                
                # Set plot limits and labels
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
                
                # Add legend (remove duplicates)
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), 
                         loc='upper left', facecolor='#1e1e1e', labelcolor='white')
                
                # Display the plot
                st.pyplot(fig)
                
                # Display match statistics
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label=f"{st.session_state.target_team} Total xG",
                        value=f"{my_team_df['xG'].sum():.2f}"
                    )
                
                with col2:
                    st.metric(
                        label=f"{opp_name} Total xG",
                        value=f"{opp_df['xG'].sum():.2f}"
                    )
                
                with col3:
                    st.metric(
                        label="xG Difference",
                        value=f"{abs(my_team_df['xG'].sum() - opp_df['xG'].sum()):.2f}",
                        delta=f"{my_team_df['xG'].sum() - opp_df['xG'].sum():.2f}"
                    )
    
    else:
        # Welcome message
        st.info("👈 Enter a team name and season in the sidebar to get started!")
        
        st.markdown("""
        ### How to use this app:
        
        1. **Enter Team Name**: Type the name of a football team (e.g., Liverpool, Arsenal)
        2. **Enter Season**: Type the season year (e.g., 2019, 2024)
        3. **Click Search**: The app will fetch all matches for that team in the season
        4. **Select a Match**: Choose a match from the dropdown to analyze
        5. **View xG Flow**: See the cumulative expected goals (xG) flow throughout the match
        
        ### What is xG?
        
        Expected Goals (xG) is a metric that measures the quality of chances created in a football match. 
        The xG flow chart shows how the momentum of the match evolved minute by minute.
        
        - 🔴 **Red line**: Your selected team's xG accumulation
        - 🔵 **Blue line**: Opponent's xG accumulation
        - ⭐ **Gold/White markers**: Actual goals scored
        """)

if __name__ == "__main__":
    main()
