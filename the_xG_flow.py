import streamlit as st
import pandas as pd
import numpy as np
import json
import re

# Configure matplotlib BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Streamlit
import matplotlib.pyplot as plt

# Must import requests AFTER matplotlib configuration
import requests

# Page configuration - MUST be first Streamlit command
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

# Base URL for Understat
BASE_URL = "https://understat.com"

# Headers
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
        if data:
            return data
        else:
            return []
    except requests.exceptions.Timeout:
        st.error("Request timed out. Please try again.")
        return []
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {str(e)}")
        return []
    except Exception as e:
        st.error(f"Error fetching team data: {str(e)}")
        return []

@st.cache_data(ttl=3600)
def get_match_shots(match_id):
    """Get shot data for a match"""
    try:
        url = f"{BASE_URL}/match/{match_id}"
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        
        shots_data = extract_json_from_html(response.text, 'shotsData')
        
        if shots_data:
            return shots_data
        else:
            return None
    except requests.exceptions.Timeout:
        st.error("Request timed out. Please try again.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error fetching match data: {str(e)}")
        return None

def clean_df(df):
    """Clean and convert dataframe columns to numeric"""
    cols = ["X", "Y", "xG", "minute"]
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def get_flow_data(df):
    """Calculate cumulative xG flow data"""
    if len(df) == 0:
        return [0, 95], [0, 0]
    
    minutes = [0] + df["minute"].tolist()
    xg_flow = [0] + df["xG_cumsum"].tolist()
    
    last_min = max(minutes[-1], 95) if len(minutes) > 1 else 95
    minutes.append(last_min)
    xg_flow.append(xg_flow[-1])
    
    return minutes, xg_flow

def create_xg_plot(my_team_df, opp_df, target_team, opp_name):
    """Create the xG flow visualization"""
    # Prepare data for visualization
    my_team_df = my_team_df.sort_values(by="minute").copy()
    my_team_df["xG_cumsum"] = my_team_df["xG"].cumsum()
    
    opp_df = opp_df.sort_values(by="minute").copy()
    opp_df["xG_cumsum"] = opp_df["xG"].cumsum()
    
    my_mins, my_flow = get_flow_data(my_team_df)
    opp_mins, opp_flow = get_flow_data(opp_df)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.set_facecolor("#1e1e1e")
    ax.set_facecolor("#1e1e1e")
    
    # Plot lines
    ax.step(my_mins, my_flow, where="post", color="#c8102e", linewidth=3, label=target_team)
    ax.step(opp_mins, opp_flow, where="post", color="#1f77b4", linewidth=3, label=opp_name)
    
    # Fill areas
    ax.fill_between(my_mins, my_flow, step="post", color="#c8102e", alpha=0.2)
    ax.fill_between(opp_mins, opp_flow, step="post", color="#1f77b4", alpha=0.2)
    
    # Mark goals
    my_goals = my_team_df[my_team_df["result"] == "Goal"]
    opp_goals = opp_df[opp_df["result"] == "Goal"]
    
    for _, row in my_goals.iterrows():
        ax.scatter(row["minute"], row["xG_cumsum"], s=150, color="gold", 
                  edgecolors="red", zorder=10, linewidths=2)
    
    for _, row in opp_goals.iterrows():
        ax.scatter(row["minute"], row["xG_cumsum"], s=150, color="white", 
                  edgecolors="red", zorder=10, linewidths=2)
    
    # Set plot limits and labels
    max_xg = max(my_flow[-1], opp_flow[-1]) if (my_flow[-1] > 0 or opp_flow[-1] > 0) else 1
    ax.set_ylim(0, max_xg + 0.5)
    ax.set_xlim(0, 98)
    
    ax.set_xlabel("Minute", fontsize=14, color="white", fontweight='bold')
    ax.set_ylabel("Cumulative xG", fontsize=14, color="white", fontweight='bold')
    ax.set_title(f"xG Momentum: {target_team} vs {opp_name}", 
               fontsize=18, color="white", pad=15, fontweight='bold')
    
    ax.tick_params(axis='x', colors='white', labelsize=11)
    ax.tick_params(axis='y', colors='white', labelsize=11)
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    
    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), 
             loc='upper left', facecolor='#1e1e1e', labelcolor='white', 
             fontsize=12, framealpha=0.9)
    
    plt.tight_layout()
    return fig, my_team_df, opp_df

def main():
    """Main application function"""
    st.title("⚽ xG Flow Analyzer")
    st.markdown("Analyze Expected Goals (xG) momentum throughout football matches")
    st.markdown("---")
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("🔍 Match Selection")
        
        # Initialize session state
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
            help="Use underscores for spaces: Manchester_City",
            placeholder="e.g., Liverpool, Arsenal, Chelsea"
        )
        
        target_season = st.text_input(
            "Season Year",
            value="2019",
            help="Enter season year",
            placeholder="e.g., 2019, 2020, 2024"
        )
        
        st.markdown("---")
        
        if st.button("🔍 Search Matches", type="primary", use_container_width=True):
            if not target_team or not target_season:
                st.error("Please enter both team name and season!")
            else:
                with st.spinner(f"Searching for {target_team} matches in {target_season}..."):
                    match_data = get_team_results(target_team, target_season)
                    
                    if match_data and len(match_data) > 0:
                        st.session_state.match_data = match_data
                        st.session_state.target_team = target_team
                        st.session_state.target_season = target_season
                        st.success(f"✅ Found {len(match_data)} matches!")
                    else:
                        st.error("❌ No matches found. Please check:")
                        st.info("• Team name spelling (use underscores for spaces)\n• Season year\n• Try: Liverpool, Arsenal, Manchester_City")
        
        # Help section
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.markdown("""
        **Common Team Names:**
        - Liverpool
        - Arsenal  
        - Manchester_City
        - Manchester_United
        - Chelsea
        - Tottenham
        
        **Seasons:** 2014-2024
        """)
    
    # Main content area
    if st.session_state.match_data:
        try:
            df_matches = pd.DataFrame(st.session_state.match_data)
            
            # Filter valid matches
            if 'goals' in df_matches.columns:
                df_matches = df_matches[df_matches["goals"].apply(
                    lambda x: x is not None and isinstance(x, dict)
                )]
            
            if len(df_matches) == 0:
                st.warning("⚠️ No valid match data found.")
                return
            
            # Process match data
            df_matches["opponent"] = np.where(
                df_matches["side"] == "h",
                df_matches["a"].apply(lambda x: x.get("title", "") if isinstance(x, dict) else ""),
                df_matches["h"].apply(lambda x: x.get("title", "") if isinstance(x, dict) else "")
            )
            
            df_matches["result"] = df_matches.apply(
                lambda row: f"{row['goals']['h']}-{row['goals']['a']}" 
                if isinstance(row.get('goals'), dict) else "N/A", 
                axis=1
            )
            
            # Display match list
            st.subheader(f"📋 Match List for {st.session_state.target_team.upper()}")
            st.caption(f"Season {st.session_state.target_season} • {len(df_matches)} matches")
            
            display_cols = ["id", "datetime", "opponent", "result", "side"]
            st.dataframe(
                df_matches[display_cols],
                use_container_width=True,
                hide_index=True,
                height=300
            )
            
            # Match selection
            st.markdown("---")
            st.subheader("📊 Analyze Match")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                match_id = st.selectbox(
                    "Select a match:",
                    options=df_matches["id"].tolist(),
                    format_func=lambda x: (
                        f"{df_matches[df_matches['id']==x]['datetime'].values[0]} - "
                        f"{df_matches[df_matches['id']==x]['opponent'].values[0]} "
                        f"({df_matches[df_matches['id']==x]['result'].values[0]})"
                    ),
                    key="match_selector"
                )
            
            with col2:
                analyze_button = st.button("📈 Analyze", type="primary", use_container_width=True)
            
            if analyze_button and match_id:
                with st.spinner("⏳ Loading match data..."):
                    shot_data = get_match_shots(match_id)
                    
                    if not shot_data or 'h' not in shot_data or 'a' not in shot_data:
                        st.error("❌ Could not load match data. Please try another match.")
                        return
                    
                    df_home = pd.DataFrame(shot_data["h"])
                    df_away = pd.DataFrame(shot_data["a"])
                    
                    if len(df_home) == 0 and len(df_away) == 0:
                        st.warning("⚠️ No shot data available for this match.")
                        return
                    
                    df_home = clean_df(df_home)
                    df_away = clean_df(df_away)
                    
                    home_team_name = df_home["h_team"].iloc[0] if len(df_home) > 0 else ""
                    away_team_name = df_away["a_team"].iloc[0] if len(df_away) > 0 else ""
                    
                    if not home_team_name:
                        home_team_name = df_away["h_team"].iloc[0] if len(df_away) > 0 else "Home"
                    if not away_team_name:
                        away_team_name = df_home["a_team"].iloc[0] if len(df_home) > 0 else "Away"
                    
                    # Determine target team
                    target_team_clean = st.session_state.target_team.replace("_", " ")
                    
                    if home_team_name.lower() == target_team_clean.lower():
                        my_team_df = df_home
                        opp_df = df_away
                        opp_name = away_team_name
                        my_side = "Home"
                    else:
                        my_team_df = df_away
                        opp_df = df_home
                        opp_name = home_team_name
                        my_side = "Away"
                    
                    # Display match info
                    st.success(f"✅ **{st.session_state.target_team}** ({my_side} vs {opp_name})")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Home Shots", len(df_home))
                    with col_b:
                        st.metric("Away Shots", len(df_away))
                    
                    st.markdown("---")
                    
                    # Create and display plot
                    fig, my_team_final, opp_final = create_xg_plot(
                        my_team_df, opp_df, 
                        st.session_state.target_team, 
                        opp_name
                    )
                    st.pyplot(fig)
                    plt.close(fig)  # Clean up
                    
                    # Display statistics
                    st.markdown("---")
                    st.subheader("📈 Match Statistics")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    my_xg = my_team_final['xG'].sum()
                    opp_xg = opp_final['xG'].sum()
                    xg_diff = my_xg - opp_xg
                    
                    with col1:
                        st.metric(
                            label=f"{st.session_state.target_team} Total xG",
                            value=f"{my_xg:.2f}",
                            help="Total Expected Goals"
                        )
                    
                    with col2:
                        st.metric(
                            label=f"{opp_name} Total xG",
                            value=f"{opp_xg:.2f}",
                            help="Total Expected Goals"
                        )
                    
                    with col3:
                        st.metric(
                            label="xG Difference",
                            value=f"{abs(xg_diff):.2f}",
                            delta=f"{xg_diff:.2f}",
                            help="Positive = your team had better chances"
                        )
                    
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.exception(e)
    
    else:
        # Welcome screen
        st.info("👈 **Get started:** Enter a team name and season in the sidebar")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 How to Use
            
            1. **Enter Team Name** in the sidebar
               - Use underscores for spaces
               - Example: `Manchester_City`
            
            2. **Enter Season Year**
               - Available: 2014-2024
               - Example: `2019`
            
            3. **Click Search** to load matches
            
            4. **Select a Match** from the list
            
            5. **Click Analyze** to view xG flow
            """)
        
        with col2:
            st.markdown("""
            ### 📊 What is xG?
            
            **Expected Goals (xG)** measures the quality of scoring chances in a match.
            
            The xG flow chart shows:
            - 🔴 **Red**: Your team's xG
            - 🔵 **Blue**: Opponent's xG  
            - ⭐ **Markers**: Actual goals
            
            Higher xG = Better chances created
            
            ### 🏆 Popular Teams
            - Liverpool
            - Arsenal
            - Manchester_City
            - Chelsea
            - Tottenham
            """)
        
        st.markdown("---")
        st.caption("Data source: Understat.com • Built with Streamlit")

if __name__ == "__main__":
    main()
