import asyncio
import aiohttp
import nest_asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import re
import json
import codecs
from bs4 import BeautifulSoup

# Import understat components
from understat import Understat
import understat.utils as understat_utils

nest_asyncio.apply()

# ============================================================================
# COMPREHENSIVE MONKEY-PATCHING SECTION
# ============================================================================

def robust_find_match(scripts, pattern):
    """Find pattern in scripts with comprehensive error handling"""
    if not scripts:
        return None
    
    for script in scripts:
        try:
            # Check if script exists and has string attribute
            if script is None:
                continue
            
            if not hasattr(script, 'string'):
                continue
                
            if script.string is None:
                continue
            
            # Try to match pattern
            match = re.search(pattern, script.string)
            if match:
                return match
                
        except (TypeError, AttributeError, re.error) as e:
            # Skip problematic scripts
            continue
    
    return None


def robust_decode_data(match):
    """Decode data from regex match with comprehensive error handling"""
    if match is None:
        return None
    
    try:
        # Try to get the matched group
        matched_text = match.group(1)
        if matched_text is None:
            return None
        
        # Try to decode
        byte_data = codecs.escape_decode(matched_text)
        if byte_data is None or len(byte_data) == 0:
            return None
        
        # Try to parse JSON
        decoded_string = byte_data[0].decode("utf-8")
        data = json.loads(decoded_string)
        
        return data
        
    except (AttributeError, TypeError, IndexError, UnicodeDecodeError, json.JSONDecodeError) as e:
        return None


async def robust_get_data(session, url, pattern_type):
    """Get data from URL with comprehensive error handling and fallbacks"""
    try:
        # Fetch HTML
        html = await understat_utils.fetch(session, url)
        
        if not html:
            raise ValueError(f"Empty response from {url}")
        
        # Parse HTML
        soup = BeautifulSoup(html, "html.parser")
        scripts = soup.find_all("script")
        
        if not scripts:
            raise ValueError(f"No scripts found in page")
        
        # Get pattern
        pattern = understat_utils.LEAGUE_URL_MAPPING.get(
            pattern_type, 
            understat_utils.LEAGUE_URL_MAPPING.get("playersData", "")
        )
        
        if not pattern:
            raise ValueError(f"No pattern found for {pattern_type}")
        
        # Find match
        match = robust_find_match(scripts, pattern)
        
        if match is None:
            raise ValueError(
                f"Could not find data pattern '{pattern_type}' in page. "
                f"This usually means the team/season doesn't exist on Understat or the website structure changed."
            )
        
        # Decode data
        data = robust_decode_data(match)
        
        if data is None:
            raise ValueError(f"Could not decode data for pattern '{pattern_type}'")
        
        # Ensure data is iterable (list or dict)
        if not isinstance(data, (list, dict)):
            raise ValueError(f"Unexpected data type: {type(data)}")
        
        return data
        
    except Exception as e:
        # Re-raise with more context
        raise ValueError(f"Error fetching {pattern_type} from {url}: {str(e)}")


# Apply patches to understat.utils
understat_utils.find_match = robust_find_match
understat_utils.decode_data = robust_decode_data
understat_utils.get_data = robust_get_data

# ============================================================================
# STREAMLIT APP CONFIGURATION
# ============================================================================

st.set_page_config(page_title="xG Flow Analyzer", page_icon="⚽", layout="wide")

st.markdown(
    """
    <style>
    .main { padding: 2rem; }
    .stButton>button {
        width: 100%;
        background-color: #c8102e;
        color: white;
    }
    .stButton>button:hover { background-color: #a00d25; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚽ xG Flow Analyzer")
st.markdown("Analyze Expected Goals (xG) momentum for football matches")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


async def robust_fetch(session, url):
    """Fetch URL with proper headers"""
    headers = {
        "X-Requested-With": "XMLHttpRequest",
        "User-Agent": HEADERS["User-Agent"],
    }
    try:
        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                raise ValueError(f"HTTP {response.status} from {url}")
            text = await response.text()
            return text
    except Exception as e:
        raise ValueError(f"Network error: {str(e)}")


understat_utils.fetch = robust_fetch

LEAGUES = ["epl", "la_liga", "bundesliga", "serie_a", "ligue1", "rfpl"]

# ============================================================================
# DATA FETCHING FUNCTIONS
# ============================================================================

@st.cache_data(show_spinner=False, ttl=3600)
def get_team_results(team_name, season):
    """Fetch team results with comprehensive error handling"""
    async def _fetch():
        async with aiohttp.ClientSession(headers=HEADERS) as session:
            understat = Understat(session)
            try:
                results = await understat.get_team_results(team_name, season)
                
                # Validate results
                if results is None:
                    raise ValueError("API returned None - team/season may not exist")
                
                if not isinstance(results, list):
                    raise ValueError(f"API returned unexpected type: {type(results)}")
                
                return results
                
            except Exception as e:
                raise

    try:
        return asyncio.run(_fetch())
    except Exception as e:
        # Provide user-friendly error message
        error_msg = str(e).lower()
        
        if "could not find data pattern" in error_msg:
            raise ValueError(
                f"Team '{team_name}' not found for season {season}. "
                "Please check the team name (case-sensitive) and try a different season."
            )
        elif "network error" in error_msg or "http" in error_msg:
            raise ValueError(
                "Network error - Understat.com may be down or blocking requests. "
                "Try again in a few minutes."
            )
        elif "none" in error_msg and "iterable" in error_msg:
            raise ValueError(
                f"No data available for '{team_name}' in season {season}. "
                "Try a different season year (2023, 2022, 2021)."
            )
        else:
            raise


@st.cache_data(show_spinner=False, ttl=3600)
def get_teams(league_name, season):
    """Fetch teams list with error handling"""
    async def _fetch():
        async with aiohttp.ClientSession(headers=HEADERS) as session:
            understat = Understat(session)
            teams = await understat.get_teams(league_name, season)
            
            if teams is None:
                return []
            
            return teams

    try:
        return asyncio.run(_fetch())
    except Exception:
        return []


@st.cache_data(show_spinner=False, ttl=3600)
def get_match_shots(match_id):
    """Fetch match shots with error handling"""
    async def _fetch():
        async with aiohttp.ClientSession(headers=HEADERS) as session:
            understat = Understat(session)
            shots = await understat.get_match_shots(match_id)
            
            if shots is None:
                raise ValueError("No shot data available for this match")
            
            if not isinstance(shots, dict) or 'h' not in shots or 'a' not in shots:
                raise ValueError("Invalid shot data structure")
            
            return shots

    return asyncio.run(_fetch())


# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

def clean_df(df):
    """Clean dataframe with error handling"""
    if df is None or len(df) == 0:
        return df
    
    cols = ["X", "Y", "xG", "minute"]
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows with NaN in critical columns
    df = df.dropna(subset=['xG', 'minute'])
    
    return df


def get_flow_data(df):
    """Generate flow data with error handling"""
    if df is None or len(df) == 0:
        return [0, 95], [0, 0]
    
    try:
        minutes = [0] + df["minute"].tolist()
        xg_flow = [0] + df["xG_cumsum"].tolist()

        last_min = max(minutes[-1] if minutes else 0, 95)
        minutes.append(last_min)
        xg_flow.append(xg_flow[-1])

        return minutes, xg_flow
    except Exception:
        return [0, 95], [0, 0]


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if "matches_loaded" not in st.session_state:
    st.session_state.matches_loaded = False
if "match_data" not in st.session_state:
    st.session_state.match_data = None
if "df_matches" not in st.session_state:
    st.session_state.df_matches = None
if "target_team" not in st.session_state:
    st.session_state.target_team = ""

# ============================================================================
# UI - STEP 1: TEAM SELECTION
# ============================================================================

st.header("Step 1: Select League, Team and Season")

col1, col2 = st.columns(2)
with col1:
    league = st.selectbox("League", options=LEAGUES, index=0)
with col2:
    target_season = st.text_input(
        "Season Year",
        placeholder="e.g., 2024, 2023",
        help="Enter the year of the season",
        value="2024"
    )

use_team_list = st.checkbox("Use team list", value=True)

target_team = ""
if use_team_list:
    if target_season:
        try:
            season_int = int(target_season)
            teams = get_teams(league, season_int)
            
            if teams and len(teams) > 0:
                team_names = sorted({
                    team.get("title") 
                    for team in teams 
                    if team and team.get("title")
                })
                
                if team_names:
                    target_team = st.selectbox("Team", options=team_names)
                else:
                    st.warning("No teams found. Use manual entry.")
                    use_team_list = False
            else:
                st.warning("Could not load teams list. Use manual entry.")
                use_team_list = False
                
        except ValueError:
            st.error("Please enter a valid year (e.g., 2024)")
        except Exception as exc:
            st.warning(f"Error loading teams: {str(exc)[:100]}. Use manual entry.")
            use_team_list = False
    else:
        st.info("Enter a season year to load the team list.")

if not use_team_list:
    target_team = st.text_input(
        "Team Name",
        placeholder="e.g., Liverpool, Arsenal, Manchester City",
        help="Enter the team name EXACTLY as it appears on Understat (case-sensitive)",
    )

# ============================================================================
# LOAD MATCHES BUTTON
# ============================================================================

if st.button("Load Matches", type="primary"):
    if target_team and target_season:
        with st.spinner(f"Searching for {target_team} matches in {target_season}..."):
            try:
                season_int = int(target_season)
                match_data = get_team_results(target_team, season_int)

                if match_data and len(match_data) > 0:
                    processed_matches = []
                    
                    for match in match_data:
                        try:
                            # Validate essential fields
                            if not match or not isinstance(match, dict):
                                continue
                            
                            match_id = match.get('id')
                            match_datetime = match.get('datetime')
                            
                            if not match_id or not match_datetime:
                                continue
                            
                            # Validate goals
                            goals = match.get('goals')
                            if not goals or not isinstance(goals, dict):
                                continue
                            
                            # Validate team data
                            h_data = match.get('h')
                            a_data = match.get('a')
                            
                            if not h_data or not isinstance(h_data, dict):
                                continue
                            if not a_data or not isinstance(a_data, dict):
                                continue
                            
                            # Get side
                            side = match.get('side', '')
                            if side not in ['h', 'a']:
                                continue
                            
                            # Extract opponent
                            if side == 'h':
                                opponent = a_data.get('title', 'Unknown')
                            else:
                                opponent = h_data.get('title', 'Unknown')
                            
                            if opponent == 'Unknown':
                                continue
                            
                            # Extract scores
                            h_goals = goals.get('h')
                            a_goals = goals.get('a')
                            
                            if h_goals is None or a_goals is None:
                                continue
                            
                            result = f"{h_goals}-{a_goals}"
                            
                            # Store processed match
                            processed_matches.append({
                                'id': match_id,
                                'datetime': match_datetime,
                                'opponent': opponent,
                                'result': result,
                                'side': side,
                                'h_goals': h_goals,
                                'a_goals': a_goals,
                            })
                            
                        except Exception:
                            continue
                    
                    if len(processed_matches) > 0:
                        df_matches = pd.DataFrame(processed_matches)
                        
                        st.session_state.matches_loaded = True
                        st.session_state.match_data = match_data
                        st.session_state.df_matches = df_matches
                        st.session_state.target_team = target_team
                        
                        st.success(f"✅ Found {len(df_matches)} matches for {target_team}")
                    else:
                        st.error("❌ No valid matches found with complete data.")
                        st.info("The API returned matches but they were missing required fields.")
                else:
                    st.warning(f"No matches found for {target_team} in {target_season}.")
                    st.info("Try: Different season year or verify team name on understat.com")
                    
            except ValueError as e:
                st.error(f"❌ {str(e)}")
                st.markdown("""
                **Common issues:**
                - ✅ Team name must match EXACTLY (case-sensitive): "Liverpool" not "liverpool"
                - ✅ Try seasons: 2024, 2023, 2022, 2021
                - ✅ Use the team dropdown instead of typing manually
                - ✅ Some teams may not have historical data
                """)
                
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)[:200]}")
                
                with st.expander("Show detailed error"):
                    import traceback
                    st.code(traceback.format_exc())
    else:
        st.warning("⚠️ Please enter both team name and season year.")

# ============================================================================
# UI - STEP 2: MATCH ANALYSIS
# ============================================================================

if st.session_state.matches_loaded:
    st.header("Step 2: Select Match to Analyze")

    df_matches = st.session_state.df_matches

    st.subheader(f"Match List for {st.session_state.target_team.upper()}")
    
    display_cols = ["id", "datetime", "opponent", "result", "side"]
    st.dataframe(
        df_matches[display_cols].sort_values("datetime", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    match_id = st.text_input(
        "Match ID",
        placeholder="Enter the Match ID from the table above",
        help="Copy the match ID from the table",
    )

    if st.button("Analyze Match", type="primary"):
        if match_id:
            with st.spinner("Loading match data..."):
                try:
                    shot_data = get_match_shots(match_id)

                    if not shot_data or 'h' not in shot_data or 'a' not in shot_data:
                        st.error("Invalid shot data received")
                        st.stop()

                    df_home = pd.DataFrame(shot_data["h"])
                    df_away = pd.DataFrame(shot_data["a"])

                    if len(df_home) == 0 and len(df_away) == 0:
                        st.error("No shot data available for this match")
                        st.stop()

                    df_home = clean_df(df_home)
                    df_away = clean_df(df_away)

                    # Validate required columns
                    required_cols = ["h_team", "a_team", "result", "minute", "xG"]
                    for col in required_cols:
                        if col not in df_home.columns and len(df_home) > 0:
                            st.error(f"Missing required column: {col}")
                            st.stop()

                    if len(df_home) > 0:
                        home_team_name = df_home["h_team"].iloc[0]
                        away_team_name = df_home["a_team"].iloc[0]
                    elif len(df_away) > 0:
                        home_team_name = df_away["h_team"].iloc[0]
                        away_team_name = df_away["a_team"].iloc[0]
                    else:
                        st.error("No valid shot data")
                        st.stop()

                    # Determine which team is "mine"
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

                    st.success(f"✅ Data Loaded: {st.session_state.target_team} ({my_side}) vs {opp_name}")
                    st.info(f"📊 Home Shots: {len(df_home)} | Away Shots: {len(df_away)}")

                    # Calculate cumulative xG
                    if len(my_team_df) > 0:
                        my_team_df = my_team_df.sort_values(by="minute")
                        my_team_df["xG_cumsum"] = my_team_df["xG"].cumsum()

                    if len(opp_df) > 0:
                        opp_df = opp_df.sort_values(by="minute")
                        opp_df["xG_cumsum"] = opp_df["xG"].cumsum()

                    # Get flow data
                    my_mins, my_flow = get_flow_data(my_team_df)
                    opp_mins, opp_flow = get_flow_data(opp_df)

                    # Create plot
                    fig, ax = plt.subplots(figsize=(14, 7))
                    fig.set_facecolor("#1e1e1e")
                    ax.set_facecolor("#1e1e1e")

                    ax.step(
                        my_mins,
                        my_flow,
                        where="post",
                        color="#c8102e",
                        linewidth=3,
                        label=st.session_state.target_team,
                    )
                    ax.step(
                        opp_mins,
                        opp_flow,
                        where="post",
                        color="#1f77b4",
                        linewidth=3,
                        label=opp_name,
                    )

                    ax.fill_between(my_mins, my_flow, step="post", color="#c8102e", alpha=0.2)
                    ax.fill_between(opp_mins, opp_flow, step="post", color="#1f77b4", alpha=0.2)

                    # Mark goals
                    if len(my_team_df) > 0 and "result" in my_team_df.columns:
                        my_goals = my_team_df[my_team_df["result"] == "Goal"]
                        for _, row in my_goals.iterrows():
                            ax.scatter(
                                row["minute"],
                                row["xG_cumsum"],
                                s=150,
                                color="gold",
                                edgecolors="red",
                                zorder=10,
                            )

                    if len(opp_df) > 0 and "result" in opp_df.columns:
                        opp_goals = opp_df[opp_df["result"] == "Goal"]
                        for _, row in opp_goals.iterrows():
                            ax.scatter(
                                row["minute"],
                                row["xG_cumsum"],
                                s=150,
                                color="white",
                                edgecolors="red",
                                zorder=10,
                            )

                    # Set limits
                    max_xg = max(my_flow[-1], opp_flow[-1]) if my_flow[-1] > 0 or opp_flow[-1] > 0 else 1
                    ax.set_ylim(0, max_xg + 0.5)
                    ax.set_xlim(0, 98)

                    # Labels and styling
                    ax.set_xlabel("Minute", fontsize=14, color="white")
                    ax.set_ylabel("Cumulative xG", fontsize=14, color="white")
                    ax.set_title(
                        f"xG Momentum: {st.session_state.target_team} vs {opp_name}",
                        fontsize=18,
                        color="white",
                        pad=15,
                    )

                    ax.tick_params(axis="x", colors="white")
                    ax.tick_params(axis="y", colors="white")
                    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.3)

                    # Legend
                    handles, labels = ax.get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    ax.legend(
                        by_label.values(),
                        by_label.keys(),
                        loc="upper left",
                        facecolor="#1e1e1e",
                        labelcolor="white",
                    )

                    st.pyplot(fig)

                    # Statistics
                    st.header("Match Statistics")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(f"{st.session_state.target_team} xG", f"{my_flow[-1]:.2f}")
                        my_goal_count = len(my_goals) if len(my_team_df) > 0 and "result" in my_team_df.columns else 0
                        st.metric(f"{st.session_state.target_team} Goals", my_goal_count)

                    with col2:
                        st.metric(f"{opp_name} xG", f"{opp_flow[-1]:.2f}")
                        opp_goal_count = len(opp_goals) if len(opp_df) > 0 and "result" in opp_df.columns else 0
                        st.metric(f"{opp_name} Goals", opp_goal_count)

                    with col3:
                        st.metric("xG Difference", f"{my_flow[-1] - opp_flow[-1]:+.2f}")
                        st.metric("Total Shots", len(my_team_df) + len(opp_df))

                except Exception as e:
                    st.error(f"❌ Error analyzing match: {str(e)[:200]}")
                    
                    with st.expander("Show detailed error"):
                        import traceback
                        st.code(traceback.format_exc())
        else:
            st.warning("⚠️ Please enter a Match ID from the table above.")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("Data provided by [Understat](https://understat.com/) | Built with Streamlit")