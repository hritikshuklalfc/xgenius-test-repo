import streamlit as st
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from understat import Understat
from mplsoccer import Pitch
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="LFC Analytics Hub")

# Dark Mode Style for Plots
plt.style.use('dark_background')

# Avoid "Event Loop is Closed" errors in Streamlit
import nest_asyncio
nest_asyncio.apply()

# --- 1. THE CACHED DATA LOADER (Prevents IP Bans) ---
@st.cache_data
def get_team_matches(team_name, season):
    """Fetches the schedule. Cached so it only runs once per team/season."""
    async def _fetch():
        async with aiohttp.ClientSession() as session:
            understat = Understat(session)
            return await understat.get_team_results(team_name, int(season))
    
    # We run the async loop safely inside this function
    return asyncio.run(_fetch())

@st.cache_data
def get_match_shots(match_id):
    """Fetches specific match shots. Cached for performance."""
    async def _fetch():
        async with aiohttp.ClientSession() as session:
            understat = Understat(session)
            return await understat.get_match_shots(match_id)
            
    return asyncio.run(_fetch())

# --- 2. THE SIDEBAR (User Inputs) ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/0/0c/Liverpool_FC.svg", width=100)
st.sidebar.title("Data Scout Pro")

target_team = st.sidebar.text_input("Team Name", value="Liverpool")
season = st.sidebar.text_input("Season (Year)", value="2024")
btn_load = st.sidebar.button("Find Matches")

# --- 3. THE MATCH SELECTOR ---
if btn_load or 'matches' in st.session_state:
    # Load data if button clicked or already loaded
    if btn_load:
        with st.spinner(f"Scouting {target_team}..."):
            st.session_state['matches'] = get_team_matches(target_team, season)

    matches_data = st.session_state.get('matches')
    
    if matches_data:
        df_matches = pd.DataFrame(matches_data)
        # Filter for played matches only
        df_matches = df_matches[df_matches['goals'].apply(lambda x: x is not None)]
        
        # Create a readable label for the dropdown
        df_matches['match_label'] = df_matches.apply(
            lambda x: f"{x['datetime'][:10]} | vs {x['a']['title'] if x['side']=='h' else x['h']['title']} ({x['goals']['h']}-{x['goals']['a']})", 
            axis=1
        )
        
        # The Dropdown Menu
        selected_match = st.selectbox("Select a Match:", df_matches['match_label'].tolist())
        
        # Get the ID of the selected match
        match_id = df_matches[df_matches['match_label'] == selected_match]['id'].values[0]

        # --- 4. THE VISUALIZATION ENGINE ---
        if st.button("Analyze Game"):
            shots_data = get_match_shots(match_id)
            
            # Process Data (Same logic as before)
            df_home = pd.DataFrame(shots_data['h'])
            df_away = pd.DataFrame(shots_data['a'])

            for df in [df_home, df_away]:
                for col in ['X', 'Y', 'xG', 'minute']:
                    df[col] = pd.to_numeric(df[col])

            # Who is who?
            home_team_name = df_home['h_team'].iloc[0]
            away_team_name = df_away['a_team'].iloc[0]

            if home_team_name == target_team:
                my_df, opp_df = df_home, df_away
                opp_name = away_team_name
            else:
                my_df, opp_df = df_away, df_home
                opp_name = home_team_name

            # --- TAB 1: SHOT MAP ---
            tab1, tab2 = st.tabs(["🎯 Shot Map", "📈 xG Flow"])
            
            with tab1:
                col1, col2 = st.columns([3, 1])
                with col1:
                    # Draw Pitch
                    pitch = Pitch(pitch_type='opta', pitch_color='#1e1e1e', line_color='#c7d5cc')
                    fig, ax = pitch.draw(figsize=(12, 8))
                    
                    # Plot My Team (Right)
                    pitch.scatter(my_df["X"]*100, my_df["Y"]*100, s=my_df["xG"]*500, 
                                c="#C8102E", edgecolors="white", ax=ax, label=target_team)
                    
                    # Plot Opponent (Left - Flipped)
                    pitch.scatter(100-(opp_df["X"]*100), 100-(opp_df["Y"]*100), s=opp_df["xG"]*500, 
                                c="#1f77b4", edgecolors="white", ax=ax, label=opp_name)
                    
                    # Goals (Stars)
                    my_goals = my_df[my_df["result"]=="Goal"]
                    pitch.scatter(my_goals["X"]*100, my_goals["Y"]*100, s=my_goals["xG"]*500, 
                                marker="*", c="gold", edgecolors="red", zorder=10, ax=ax)
                    
                    opp_goals = opp_df[opp_df["result"]=="Goal"]
                    pitch.scatter(100-(opp_goals["X"]*100), 100-(opp_goals["Y"]*100), s=opp_goals["xG"]*500, 
                                marker="*", c="white", edgecolors="blue", zorder=10, ax=ax)

                    ax.set_title(f"{target_team} vs {opp_name}", fontsize=20, color="white")
                    st.pyplot(fig)
                
                with col2:
                    st.metric(label=f"{target_team} xG", value=f"{my_df['xG'].sum():.2f}")
                    st.metric(label=f"{opp_name} xG", value=f"{opp_df['xG'].sum():.2f}")

            # --- TAB 2: xG FLOW ---
            with tab2:
                # Prepare Flow Data
                def get_flow(df):
                    df = df.sort_values(by='minute')
                    df['xG_cumsum'] = df['xG'].cumsum()
                    mins = [0] + df['minute'].tolist() + [95]
                    flow = [0] + df['xG_cumsum'].tolist() + [df['xG_cumsum'].iloc[-1]]
                    return mins, flow

                my_mins, my_flow = get_flow(my_df)
                opp_mins, opp_flow = get_flow(opp_df)

                fig2, ax2 = plt.subplots(figsize=(12, 6))
                fig2.set_facecolor('#1e1e1e')
                ax2.set_facecolor('#1e1e1e')
                
                ax2.step(my_mins, my_flow, where='post', color='#C8102E', linewidth=3, label=target_team)
                ax2.step(opp_mins, opp_flow, where='post', color='#1f77b4', linewidth=3, label=opp_name)
                ax2.fill_between(my_mins, my_flow, step="post", color='#C8102E', alpha=0.2)
                ax2.fill_between(opp_mins, opp_flow, step="post", color='#1f77b4', alpha=0.2)
                
                # Styling
                ax2.set_xlabel("Minute", color="white")
                ax2.set_ylabel("Cumulative xG", color="white")
                ax2.grid(alpha=0.3)
                ax2.tick_params(colors="white")
                ax2.legend(facecolor='#1e1e1e', labelcolor='white')
                
                st.pyplot(fig2)