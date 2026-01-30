import streamlit as st
import asyncio
import aiohttp
import json
import codecs
import re
import pandas as pd
from mplsoccer import Pitch
import matplotlib.pyplot as plt
import nest_asyncio

# Apply Async Patch
nest_asyncio.apply()

st.set_page_config(layout="wide", page_title="LFC Analytics Hub")
plt.style.use('dark_background')

# --- 1. THE MANUAL SCRAPER (No 'understat' library needed) ---
# This mimics exactly what the library does but without the installation headaches

async def fetch_understat_data(url):
    """Fetches the raw HTML and extracts the hidden JSON data."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            content = await response.text()
            return content

def parse_data(html, data_name):
    """Finds the JSON data inside the <script> tags."""
    # The data is hidden in strings like: var datesData = JSON.parse('...');
    pattern = re.compile(f"var {data_name} = JSON.parse\('(.*?)'\)")
    match = pattern.search(html)
    if match:
        # Understat obfuscates data with unicode escape sequences (e.g. \x22)
        # We decode it back to normal JSON
        byte_data = codecs.decode(match.group(1), 'unicode_escape')
        return json.loads(byte_data)
    return []

# --- 2. CACHED DATA LOADERS ---

@st.cache_data
def get_team_matches(team_name, season):
    async def _get():
        url = f"https://understat.com/team/{team_name}/{season}"
        html = await fetch_understat_data(url)
        return parse_data(html, "datesData") # 'datesData' holds the schedule
    return asyncio.run(_get())

@st.cache_data
def get_match_shots(match_id):
    async def _get():
        url = f"https://understat.com/match/{match_id}"
        html = await fetch_understat_data(url)
        return parse_data(html, "shotsData") # 'shotsData' holds the X,Y coords
    return asyncio.run(_get())

# --- 3. APP INTERFACE (Standard Streamlit) ---

st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/0/0c/Liverpool_FC.svg", width=100)
st.sidebar.title("Data Scout Pro (v2)")

# Inputs
target_team = st.sidebar.text_input("Team Name", value="Liverpool")
season = st.sidebar.text_input("Season (Year)", value="2024")
btn_load = st.sidebar.button("Find Matches")

if btn_load or 'matches' in st.session_state:
    if btn_load:
        with st.spinner(f"Scouting {target_team}..."):
            try:
                st.session_state['matches'] = get_team_matches(target_team.replace(" ", "_"), season)
            except Exception as e:
                st.error(f"Could not find team. Try 'Liverpool' or 'Manchester_City'. Error: {e}")

    matches_data = st.session_state.get('matches')
    
    if matches_data:
        df_matches = pd.DataFrame(matches_data)
        # Filter for played matches (where goals is not null)
        # Understat raw data uses 'goals' dictionary keys differently sometimes, handling safely:
        played_matches = df_matches[df_matches['goals'].apply(lambda x: x.get('h') is not None)]
        
        # Create Label
        played_matches['match_label'] = played_matches.apply(
            lambda x: f"{x['datetime'][:10]} | vs {x['a']['title'] if x['side']=='h' else x['h']['title']} ({x['goals']['h']}-{x['goals']['a']})", 
            axis=1
        )
        
        selected_match = st.selectbox("Select a Match:", played_matches['match_label'].tolist())
        match_id = played_matches[played_matches['match_label'] == selected_match]['id'].values[0]

        if st.button("Analyze Game"):
            shots_data = get_match_shots(match_id)
            
            # --- SAME LOGIC AS BEFORE ---
            df_home = pd.DataFrame(shots_data['h'])
            df_away = pd.DataFrame(shots_data['a'])

            for df in [df_home, df_away]:
                for col in ['X', 'Y', 'xG', 'minute']:
                    df[col] = pd.to_numeric(df[col])

            home_team_name = df_home['h_team'].iloc[0]
            away_team_name = df_away['a_team'].iloc[0]

            # Normalize team name for comparison (handle underscores)
            target_clean = target_team.replace("_", " ")
            
            if home_team_name == target_clean:
                my_df, opp_df = df_home, df_away
                opp_name = away_team_name
            else:
                my_df, opp_df = df_away, df_home
                opp_name = home_team_name

            # --- VIZ ---
            tab1, tab2 = st.tabs(["🎯 Shot Map", "📈 xG Flow"])
            
            with tab1:
                pitch = Pitch(pitch_type='opta', pitch_color='#1e1e1e', line_color='#c7d5cc')
                fig, ax = pitch.draw(figsize=(12, 8))
                
                # Plot My Team (Right)
                pitch.scatter(my_df["X"]*100, my_df["Y"]*100, s=my_df["xG"]*500, c="#C8102E", edgecolors="white", ax=ax)
                # Plot Opponent (Left)
                pitch.scatter(100-(opp_df["X"]*100), 100-(opp_df["Y"]*100), s=opp_df["xG"]*500, c="#1f77b4", edgecolors="white", ax=ax)
                
                # Goals
                my_goals = my_df[my_df["result"]=="Goal"]
                pitch.scatter(my_goals["X"]*100, my_goals["Y"]*100, s=my_goals["xG"]*500, marker="*", c="gold", edgecolors="red", zorder=10, ax=ax)
                
                opp_goals = opp_df[opp_df["result"]=="Goal"]
                pitch.scatter(100-(opp_goals["X"]*100), 100-(opp_goals["Y"]*100), s=opp_goals["xG"]*500, marker="*", c="white", edgecolors="blue", zorder=10, ax=ax)

                ax.set_title(f"{target_clean} ({my_df['xG'].sum():.2f}) vs {opp_name} ({opp_df['xG'].sum():.2f})", color="white", fontsize=20)
                st.pyplot(fig)

            with tab2:
                 # Calculate Flow
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
                ax2.step(my_mins, my_flow, where='post', color='#C8102E', linewidth=3, label=target_clean)
                ax2.step(opp_mins, opp_flow, where='post', color='#1f77b4', linewidth=3, label=opp_name)
                ax2.fill_between(my_mins, my_flow, step="post", color='#C8102E', alpha=0.2)
                ax2.fill_between(opp_mins, opp_flow, step="post", color='#1f77b4', alpha=0.2)
                
                ax2.set_xlabel("Minute", color="white")
                ax2.set_ylabel("Cumulative xG", color="white")
                ax2.tick_params(colors="white")
                ax2.grid(alpha=0.3)
                ax2.legend(facecolor='#1e1e1e', labelcolor='white')
                st.pyplot(fig2)