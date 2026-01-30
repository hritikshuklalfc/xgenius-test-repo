import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import requests
import plotly.graph_objects as go

# Page configuration - MUST BE FIRST
st.set_page_config(
    page_title="xG Flow Analyzer",
    page_icon="⚽",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {padding: 2rem;}
    .stButton>button {width: 100%;}
    </style>
""", unsafe_allow_html=True)

# Constants
BASE_URL = "https://understat.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

@st.cache_data(ttl=3600)
def extract_json_from_html(html, variable_name):
    """Extract JSON data from HTML script tags"""
    try:
        pattern = rf"var {variable_name}\s*=\s*JSON\.parse\('(.+?)'\)"
        match = re.search(pattern, html)
        if match:
            json_str = match.group(1)
            json_str = json_str.encode('utf-8').decode('unicode_escape')
            return json.loads(json_str)
    except Exception as e:
        st.error(f"Error parsing JSON: {str(e)}")
    return None

@st.cache_data(ttl=3600)
def get_team_results(team_name, season):
    """Get team results for a season"""
    try:
        url = f"{BASE_URL}/team/{team_name}/{season}"
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        data = extract_json_from_html(response.text, 'datesData')
        return data if data else []
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return []

@st.cache_data(ttl=3600)
def get_match_shots(match_id):
    """Get shot data for a match"""
    try:
        url = f"{BASE_URL}/match/{match_id}"
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        shots_data = extract_json_from_html(response.text, 'shotsData')
        return shots_data if shots_data else None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def clean_df(df):
    """Clean dataframe"""
    cols = ["X", "Y", "xG", "minute"]
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def get_flow_data(df):
    """Get flow data"""
    if len(df) == 0:
        return [0, 95], [0, 0]
    minutes = [0] + df["minute"].tolist()
    xg_flow = [0] + df["xG_cumsum"].tolist()
    last_min = max(minutes[-1], 95) if len(minutes) > 1 else 95
    minutes.append(last_min)
    xg_flow.append(xg_flow[-1])
    return minutes, xg_flow

def create_xg_plot(my_team_df, opp_df, target_team, opp_name):
    """Create xG flow plot using Plotly"""
    try:
        # Prepare data
        my_team_df = my_team_df.sort_values(by="minute").copy()
        my_team_df["xG_cumsum"] = my_team_df["xG"].cumsum()
        
        opp_df = opp_df.sort_values(by="minute").copy()
        opp_df["xG_cumsum"] = opp_df["xG"].cumsum()
        
        my_mins, my_flow = get_flow_data(my_team_df)
        opp_mins, opp_flow = get_flow_data(opp_df)
        
        # Create figure
        fig = go.Figure()
        
        # Add team lines
        fig.add_trace(go.Scatter(
            x=my_mins, y=my_flow,
            mode='lines',
            name=target_team,
            line=dict(color='#c8102e', width=3, shape='hv'),
            fill='tozeroy',
            fillcolor='rgba(200, 16, 46, 0.2)'
        ))
        
        fig.add_trace(go.Scatter(
            x=opp_mins, y=opp_flow,
            mode='lines',
            name=opp_name,
            line=dict(color='#1f77b4', width=3, shape='hv'),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.2)'
        ))
        
        # Add goal markers
        my_goals = my_team_df[my_team_df["result"] == "Goal"]
        if len(my_goals) > 0:
            fig.add_trace(go.Scatter(
                x=my_goals["minute"],
                y=my_goals["xG_cumsum"],
                mode='markers',
                name=f'{target_team} Goals',
                marker=dict(size=15, color='gold', line=dict(color='red', width=2)),
                showlegend=False
            ))
        
        opp_goals = opp_df[opp_df["result"] == "Goal"]
        if len(opp_goals) > 0:
            fig.add_trace(go.Scatter(
                x=opp_goals["minute"],
                y=opp_goals["xG_cumsum"],
                mode='markers',
                name=f'{opp_name} Goals',
                marker=dict(size=15, color='white', line=dict(color='red', width=2)),
                showlegend=False
            ))
        
        # Layout
        max_xg = max(my_flow[-1], opp_flow[-1]) if (my_flow[-1] > 0 or opp_flow[-1] > 0) else 1
        
        fig.update_layout(
            title=dict(
                text=f"xG Momentum: {target_team} vs {opp_name}",
                font=dict(size=20, color='white'),
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title="Minute",
                range=[0, 98],
                gridcolor='rgba(128, 128, 128, 0.2)',
                color='white'
            ),
            yaxis=dict(
                title="Cumulative xG",
                range=[0, max_xg + 0.5],
                gridcolor='rgba(128, 128, 128, 0.2)',
                color='white'
            ),
            plot_bgcolor='#1e1e1e',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(30, 30, 30, 0.8)',
                bordercolor='white',
                borderwidth=1
            ),
            hovermode='x unified',
            height=500
        )
        
        return fig, my_team_df, opp_df
        
    except Exception as e:
        st.error(f"Error creating plot: {str(e)}")
        return None, my_team_df, opp_df

def main():
    """Main application"""
    st.title("⚽ xG Flow Analyzer")
    st.markdown("Analyze Expected Goals (xG) momentum throughout football matches")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("🔍 Match Selection")
        
        if 'match_data' not in st.session_state:
            st.session_state.match_data = None
        if 'target_team' not in st.session_state:
            st.session_state.target_team = ""
        if 'target_season' not in st.session_state:
            st.session_state.target_season = ""
        
        st.markdown("### Team Information")
        target_team = st.text_input(
            "Team Name",
            value="Liverpool",
            help="Use underscores for spaces: Manchester_City"
        )
        
        target_season = st.text_input(
            "Season Year",
            value="2019",
            help="e.g., 2019, 2024"
        )
        
        st.markdown("---")
        
        if st.button("🔍 Search Matches", type="primary", use_container_width=True):
            if not target_team or not target_season:
                st.error("Enter both team and season!")
            else:
                with st.spinner(f"Searching {target_team} in {target_season}..."):
                    match_data = get_team_results(target_team, target_season)
                    if match_data and len(match_data) > 0:
                        st.session_state.match_data = match_data
                        st.session_state.target_team = target_team
                        st.session_state.target_season = target_season
                        st.success(f"✅ Found {len(match_data)} matches!")
                    else:
                        st.error("❌ No matches found")
                        st.info("Try: Liverpool, Arsenal, Manchester_City")
        
        st.markdown("---")
        st.markdown("### 💡 Common Teams")
        st.markdown("""
        - Liverpool
        - Arsenal
        - Manchester_City
        - Chelsea
        - Tottenham
        """)
    
    # Main content
    if st.session_state.match_data:
        try:
            df_matches = pd.DataFrame(st.session_state.match_data)
            
            if 'goals' in df_matches.columns:
                df_matches = df_matches[df_matches["goals"].apply(
                    lambda x: x is not None and isinstance(x, dict)
                )]
            
            if len(df_matches) == 0:
                st.warning("No valid match data")
                return
            
            df_matches["opponent"] = np.where(
                df_matches["side"] == "h",
                df_matches["a"].apply(lambda x: x.get("title", "") if isinstance(x, dict) else ""),
                df_matches["h"].apply(lambda x: x.get("title", "") if isinstance(x, dict) else "")
            )
            
            df_matches["result"] = df_matches.apply(
                lambda row: f"{row['goals']['h']}-{row['goals']['a']}" 
                if isinstance(row.get('goals'), dict) else "N/A", axis=1
            )
            
            st.subheader(f"📋 Matches - {st.session_state.target_team.upper()}")
            st.caption(f"Season {st.session_state.target_season} • {len(df_matches)} matches")
            
            display_cols = ["id", "datetime", "opponent", "result", "side"]
            st.dataframe(df_matches[display_cols], use_container_width=True, hide_index=True, height=300)
            
            st.markdown("---")
            st.subheader("📊 Analyze Match")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                match_id = st.selectbox(
                    "Select match:",
                    options=df_matches["id"].tolist(),
                    format_func=lambda x: (
                        f"{df_matches[df_matches['id']==x]['datetime'].values[0]} - "
                        f"{df_matches[df_matches['id']==x]['opponent'].values[0]} "
                        f"({df_matches[df_matches['id']==x]['result'].values[0]})"
                    )
                )
            
            with col2:
                analyze_button = st.button("📈 Analyze", type="primary", use_container_width=True)
            
            if analyze_button and match_id:
                with st.spinner("Loading..."):
                    shot_data = get_match_shots(match_id)
                    
                    if not shot_data or 'h' not in shot_data or 'a' not in shot_data:
                        st.error("Could not load data")
                        return
                    
                    df_home = pd.DataFrame(shot_data["h"])
                    df_away = pd.DataFrame(shot_data["a"])
                    
                    if len(df_home) == 0 and len(df_away) == 0:
                        st.warning("No shot data")
                        return
                    
                    df_home = clean_df(df_home)
                    df_away = clean_df(df_away)
                    
                    home_team = df_home["h_team"].iloc[0] if len(df_home) > 0 else "Home"
                    away_team = df_away["a_team"].iloc[0] if len(df_away) > 0 else "Away"
                    
                    target_clean = st.session_state.target_team.replace("_", " ")
                    
                    if home_team.lower() == target_clean.lower():
                        my_team_df = df_home
                        opp_df = df_away
                        opp_name = away_team
                        my_side = "Home"
                    else:
                        my_team_df = df_away
                        opp_df = df_home
                        opp_name = home_team
                        my_side = "Away"
                    
                    st.success(f"**{st.session_state.target_team}** ({my_side} vs {opp_name})")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Home Shots", len(df_home))
                    with col_b:
                        st.metric("Away Shots", len(df_away))
                    
                    st.markdown("---")
                    
                    # Create plot
                    fig, my_final, opp_final = create_xg_plot(
                        my_team_df, opp_df, st.session_state.target_team, opp_name
                    )
                    
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Stats
                    st.markdown("---")
                    st.subheader("📈 Statistics")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    my_xg = my_final['xG'].sum()
                    opp_xg = opp_final['xG'].sum()
                    
                    with col1:
                        st.metric(f"{st.session_state.target_team} xG", f"{my_xg:.2f}")
                    with col2:
                        st.metric(f"{opp_name} xG", f"{opp_xg:.2f}")
                    with col3:
                        st.metric("Difference", f"{abs(my_xg - opp_xg):.2f}", 
                                delta=f"{my_xg - opp_xg:.2f}")
                    
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    else:
        st.info("👈 Enter team name and season in sidebar")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 How to Use
            1. Enter team name (use underscores)
            2. Enter season (2014-2024)
            3. Click Search
            4. Select match
            5. Click Analyze
            """)
        
        with col2:
            st.markdown("""
            ### 📊 What is xG?
            **Expected Goals** = chance quality
            - 🔴 Red: Your team
            - 🔵 Blue: Opponent
            - ⭐ Stars: Goals
            """)
        
        st.markdown("---")
        st.caption("Data: Understat.com • Built with Streamlit")

if __name__ == "__main__":
    main()