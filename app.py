import sys
import os
import json
import time
import requests
import streamlit as st
import pandas as pd
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import plotly.io as pio
import plotly.graph_objects as go
import numpy as np
from dotenv import load_dotenv
from databricks.sdk import WorkspaceClient
from agent_endpoint_client import AgentEndpointClient

# Load environment variables
load_dotenv()

# App Session Manager Class
class AppSessionManager:
    """Manages app-level session lifecycle"""
    
    def __init__(self, user_email: str):
        self.user_email = user_email
        self.session_id = f"session_{user_email}"
        print(f"[SESSION] Session ID: {self.session_id}")

def fix_chart_formatting(fig):
    """
    Fix common chart formatting issues:
    1. Remove scientific notation from axes
    2. Add padding to y-axis range (min-5%, max+5%)
    3. Sort bars by value (descending)
    4. Ensure numeric data types
    """
    try:
        # Get figure data
        if not fig.data:
            return fig
        
        # Check if x-axis contains dates
        has_date_x = False
        if fig.data and hasattr(fig.data[0], 'x') and fig.data[0].x is not None:
            # Check if x values look like dates
            x_sample = fig.data[0].x[0] if len(fig.data[0].x) > 0 else None
            if x_sample:
                # If it's a string that contains date-like patterns, or already a datetime
                has_date_x = (
                    isinstance(x_sample, str) and any(sep in str(x_sample) for sep in ['-', '/', ':']) 
                    or 'datetime' in str(type(x_sample)).lower()
                    or 'timestamp' in str(type(x_sample)).lower()
                )
        
        # Process each trace
        for trace in fig.data:
            # Convert y values to numeric if they're strings
            if hasattr(trace, 'y') and trace.y is not None:
                y_values = [float(v) if isinstance(v, str) else v for v in trace.y]
                trace.y = y_values
                
                # Check if this is a single data point chart
                if len(y_values) == 1:
                    # Don't show chart for single data point
                    return None
                
                # Calculate range with 5% padding
                if y_values:
                    y_min = min(y_values)
                    y_max = max(y_values)
                    y_range = y_max - y_min
                    
                    # Add 5% padding on each side
                    if y_range > 0:
                        padding = y_range * 0.05
                        range_min = y_min - padding
                        range_max = y_max + padding
                    else:
                        # If all values are the same, add absolute padding
                        range_min = y_min * 0.95 if y_min > 0 else y_min - abs(y_min) * 0.05
                        range_max = y_max * 1.05 if y_max > 0 else y_max + abs(y_max) * 0.05
                    
                    # Update y-axis range
                    fig.update_yaxes(range=[range_min, range_max])
            
            # For pie charts, check if there's only one value
            if trace.type == 'pie' and hasattr(trace, 'values') and trace.values is not None:
                if len(trace.values) == 1:
                    # Don't show pie chart for single value
                    return None
            
            # For bar charts, sort by value (descending) - but only if x-axis is not dates
            if trace.type == 'bar' and not has_date_x and hasattr(trace, 'y') and hasattr(trace, 'x'):
                try:
                    # Create pairs of (x, y) and sort by y descending
                    pairs = list(zip(trace.x, trace.y))
                    pairs_sorted = sorted(pairs, key=lambda p: float(p[1]) if p[1] is not None else 0, reverse=True)
                    
                    if pairs_sorted:
                        trace.x, trace.y = zip(*pairs_sorted)
                except Exception:
                    pass  # Keep original order if sorting fails
        
        # Disable scientific notation on axes, but only format as numbers if not dates
        if has_date_x:
            # Don't apply numeric formatting to x-axis if it contains dates
            fig.update_xaxes(exponentformat='none')
        else:
            # Apply numeric formatting to x-axis
            fig.update_xaxes(
                tickformat=',.0f',  # Format with comma separators, no decimals
                exponentformat='none'  # Disable scientific notation
            )
        
        # Always apply numeric formatting to y-axis (values)
        fig.update_yaxes(
            tickformat=',.0f',  # Format with comma separators, no decimals
            exponentformat='none'  # Disable scientific notation
        )
        
        # Add vertical spike line on hover only
        fig.update_xaxes(
            showspikes=True, 
            spikemode='across', 
            spikesnap='cursor', 
            spikedash='dot', 
            spikecolor='#FFFFFF', 
            spikethickness=1,
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)'
        )
        fig.update_yaxes(
            showspikes=False,
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)'
        )
        
        # Set hover mode to show closest data point and position label above point
        fig.update_layout(
            hovermode='x unified',
            hoverlabel=dict(
                bgcolor='rgba(26, 26, 26, 0.9)',
                font_size=12,
                font_color='white'
            )
        )
        
        # Update hover template to show actual values with proper formatting
        for trace in fig.data:
            if hasattr(trace, 'y') and trace.y is not None:
                # Get the y-axis title from the layout
                y_label = 'value'
                if hasattr(fig.layout, 'yaxis') and hasattr(fig.layout.yaxis, 'title') and fig.layout.yaxis.title:
                    if hasattr(fig.layout.yaxis.title, 'text') and fig.layout.yaxis.title.text:
                        y_label = fig.layout.yaxis.title.text
                
                # Check if values have decimals
                has_decimals = any(isinstance(v, float) and v != int(v) for v in trace.y if v is not None)
                
                if has_decimals:
                    # Show 2 decimal places for values with decimals
                    trace.update(hovertemplate=y_label + ': %{y:.2f}<extra></extra>')
                else:
                    # Show integers with comma separators
                    trace.update(hovertemplate=y_label + ': %{y:,.0f}<extra></extra>')
        
        return fig
        
    except Exception as e:
        print(f"Error fixing chart formatting: {e}")
        return fig  # Return original if fixing fails

# Get user information from Databricks App headers
def get_user_email():
    """Get user email from Databricks App context"""
    try:
        # Get email from Databricks App headers
        user_email = st.context.headers.get('X-Forwarded-Email')
        if user_email:
            return user_email
        
        # Fallback: try to get from other headers
        user_name = st.context.headers.get('X-Forwarded-User')
        if user_name:
            return user_name
        
        # If running locally/not in Databricks App, fallback to anonymous
        return "anonymous@local.dev"
    except Exception as e:
        print(f"Error getting user email: {e}")
        return "anonymous@local.dev"

def get_user_token():
    """
    Retrieves the X-Forwarded-Access-Token from the Streamlit context headers.
    
    Returns:
        str: The user's access token or None if not found.
    """
    try:
        # Use .get() to avoid KeyError if the header is missing
        headers = st.context.headers
        
        # Log all available headers for debugging
        print(f"Available headers: {list(headers.keys())}")
        
        # Try multiple possible token header names
        user_token = (
            headers.get("X-Forwarded-Access-Token") or
            headers.get("x-forwarded-access-token") or
            headers.get("Authorization") or
            headers.get("authorization")
        )
        print(f"token {user_token}")
        return user_token
    except Exception as e:
        print(f"Error getting user token: {e}")
        return None

# Get user email for session tracking
user_email = get_user_email()
user_token = get_user_token()
print(f"App user: {user_email}")
print(f"User token: {user_token[:20] + '...' if user_token else 'None'}")

# Get agent endpoint configuration from environment
AGENT_ENDPOINT_NAME = os.getenv("AGENT_ENDPOINT_NAME", "")
AGENT_ENDPOINT_URL = os.getenv("AGENT_ENDPOINT_URL", "https://5110247008182190.0.gcp.databricks.com/serving-endpoints/multi_agent_genie_20feb_v2/invocations")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "<YOUR_DATABRICKS_SERVICE_PRINCIPAL_TOKEN>")

ENV = os.getenv("ENV", "prod")
ENV =  "dev"

# Initialize agent endpoint client
try:
    # Check if we should use direct HTTP access or MLflow Deployments
    some_token = get_user_token()
    print(f"Retrieved user token for agent client initialization: {some_token[:20] + '...' if some_token else 'Not found'}")
    if AGENT_ENDPOINT_URL and DATABRICKS_TOKEN:
        # Use direct HTTP access
        agent = AgentEndpointClient(
            endpoint_url=AGENT_ENDPOINT_URL,
            access_token=some_token
        )
        client_initialized = True
        print(f"✅ Connected to agent endpoint via direct HTTP: {AGENT_ENDPOINT_URL}")
    else:
        # Fall back to MLflow Deployments
        workspace_client = WorkspaceClient()
        agent = AgentEndpointClient(
            agent_endpoint_name=AGENT_ENDPOINT_NAME, 
            workspace_client=workspace_client
        )
        client_initialized = True
        print(f"✅ Connected to agent endpoint via MLflow: {AGENT_ENDPOINT_NAME}")

except Exception as e:
    print(f"❌ Error initializing agent client: {e}")
    import traceback

    traceback.print_exc()
    client_initialized = False
    agent = None

# Initialize session manager
session_manager = AppSessionManager(user_email)
print(f"✅ Session initialized: {session_manager.session_id}")

# -------------------------
# Genie Space Configuration
# -------------------------
DATABRICKS_INSTANCE = "https://5110247008182190.0.gcp.databricks.com"

# Local dev fallback PAT (used when user token not available locally)
LOCAL_DEV_PAT = os.getenv("LOCAL_DEV_PAT", "<YOUR_DATABRICKS_PAT>")

GENIE_SPACES = {
    "VOD KPI": os.getenv("GENIE_SPACE_1", "01f05676badb125ba00d539d5b41abbe"),
    "Viewership": os.getenv("GENIE_SPACE_2", "01f0cf60adc9184e9742f67cbbbc08e1"),
}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_past_chats(token: str, spaces: dict) -> list:
    """
    Fetch past user queries from all Genie spaces in parallel:
    - All spaces fetched concurrently
    - All conversations within each space fetched concurrently
    """
    effective_token = token or LOCAL_DEV_PAT
    headers = {
        "Authorization": f"Bearer {effective_token}",
        "Content-Type": "application/json",
    }
    base_url = DATABRICKS_INSTANCE.rstrip("/")
    results = []

    def fetch_messages_for_conv(space_name, space_id, conv_id):
        """Fetch messages for a single conversation."""
        try:
            msg_url = f"{base_url}/api/2.0/genie/spaces/{space_id}/conversations/{conv_id}/messages"
            msg_resp = requests.get(msg_url, headers=headers, params={"page_size": 50}, timeout=8)
            msg_resp.raise_for_status()
            items = []
            for msg in msg_resp.json().get("messages", []):
                content = msg.get("content", "")
                if content:
                    items.append({
                        "space_name": space_name,
                        "query": content,
                        "timestamp_ms": msg.get("created_timestamp", 0) or 0,
                    })
            return items
        except Exception as e:
            print(f"[PAST CHATS] Error fetching messages for conv {conv_id}: {e}")
            return []

    def fetch_space(space_name, space_id):
        """Fetch all conversations + their messages for a single space."""
        space_results = []
        try:
            conv_url = f"{base_url}/api/2.0/genie/spaces/{space_id}/conversations"
            conv_resp = requests.get(conv_url, headers=headers, params={"page_size": 20}, timeout=8)
            conv_resp.raise_for_status()
            conversations = conv_resp.json().get("conversations", [])
            print(f"[PAST CHATS] Space '{space_name}': {len(conversations)} conversations")
            # Fetch all conversations in parallel
            with ThreadPoolExecutor(max_workers=5) as msg_ex:
                futures = [
                    msg_ex.submit(
                        fetch_messages_for_conv,
                        space_name,
                        space_id,
                        conv.get("conversation_id") or conv.get("id"),
                    )
                    for conv in conversations
                    if conv.get("conversation_id") or conv.get("id")
                ]
                for f in as_completed(futures):
                    space_results.extend(f.result())
        except Exception as e:
            print(f"[PAST CHATS] Error listing conversations for space '{space_name}': {e}")
        return space_results

    # Fetch all spaces in parallel
    with ThreadPoolExecutor(max_workers=len(spaces) or 1) as space_ex:
        futures = [
            space_ex.submit(fetch_space, space_name, space_id)
            for space_name, space_id in spaces.items()
            if space_id
        ]
        for f in as_completed(futures):
            results.extend(f.result())

    results.sort(key=lambda x: x["timestamp_ms"], reverse=True)
    return results


def _format_chat_timestamp(ts_ms: int) -> str:
    """Convert millisecond timestamp to relative label (Today / Yesterday / Older)."""
    if not ts_ms:
        return "Older conversations"
    import datetime
    now = datetime.datetime.now()
    dt = datetime.datetime.fromtimestamp(ts_ms / 1000)
    delta = (now.date() - dt.date()).days
    if delta == 0:
        return "Today"
    elif delta == 1:
        return "Yesterday"
    elif delta <= 7:
        return "Previous 7 days"
    else:
        return "Older conversations"


# Streamlit Page Config
st.set_page_config(
    page_title="CDI Genie Agent", layout="wide", initial_sidebar_state="expanded"
)

# Custom CSS - Crunchyroll theme
st.markdown(
    """
<style>
    /* Import Crunchyroll-like font */
    @import url('https://fonts.googleapis.com/css2?family=Lato:wght@400;700;900&display=swap');
    
    /* Dark background similar to Crunchyroll */
    .stApp {
        background-color: #0B0B0B !important;
        font-family: 'Lato', sans-serif !important;
    }
    
    /* Remove top padding from main block */
    .block-container {
        padding-top: 4rem !important;
    }
    
    /* Header styling */
    .main-header-wrapper {
        padding: 0.5rem 1rem;
        border: 2px solid #F47521;
        background: linear-gradient(135deg, #1a1a1a 0%, #0B0B0B 100%);
        box-shadow: 0 2px 8px rgba(244, 117, 33, 0.2);
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
        margin-bottom: 0rem;
        margin-top: 0rem;
    }
    .main-header-wrapper h1 {
        background: linear-gradient(90deg, #F47521, #FF8C42);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.25rem;
        font-weight: 700;
        margin: 0;
        font-family: 'Lato', sans-serif;
        letter-spacing: 0.5px;
    }
    
    /* Chat message styling - elegant cards */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 16px;
        max-width: 95%;
        width: 100%;
        border: none;
        font-family: 'Lato', sans-serif;
    }
    
    /* Mobile responsive - full width on small screens */
    @media (max-width: 768px) {
        [data-testid="stChatMessage"] {
            max-width: 100%;
            padding: 16px 20px;
        }
    }
    
    /* User messages - Crunchyroll orange accent */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg, #F47521 0%, #FF8C42 100%) !important;
        margin-left: auto;
        margin-right: 0;
        box-shadow: 0 4px 12px rgba(244, 117, 33, 0.4);
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) p,
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) div,
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) span,
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) * {
        color: #ffffff !important;
    }
    
    /* Assistant messages - dark with orange accent */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background-color: #1a1a1a !important;
        margin-right: auto;
        margin-left: 0;
        border-left: 3px solid #F47521;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) p,
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) div,
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) span {
        color: #ffffff !important;
    }
    
    /* Chart expander styling - Crunchyroll theme */
    [data-testid="stExpander"] {
        background-color: #1a1a1a !important;
        border: 2px solid #F47521 !important;
        border-radius: 12px !important;
        margin: 10px 0 !important;
    }
    
    [data-testid="stExpander"] summary {
        background-color: #2a2a2a !important;
        border-radius: 10px !important;
        padding: 14px 18px !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        color: #F47521 !important;
    }
    
    [data-testid="stExpander"] summary:hover {
        background-color: #3a3a3a !important;
    }
    
    [data-testid="stExpander"] div[role="button"] {
        background-color: transparent !important;
    }
    
    [data-testid="stExpander"] div[role="button"] p {
        color: #F47521 !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stExpander"] svg {
        fill: #F47521 !important;
    }
    
    .streamlit-expanderContent {
        background-color: #1a1a1a !important;
        padding: 20px !important;
    }
    
    /* Dataframe styling */
    [data-testid="stDataFrame"] {
        background-color: #1a1a1a !important;
    }
    
    /* Button styling - gradient */
    .stButton>button {
        background: linear-gradient(135deg, #F47521 0%, #FF8C42 100%);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 14px 28px;
        font-weight: 700;
        font-family: 'Lato', sans-serif;
        box-shadow: 0 4px 12px rgba(244, 117, 33, 0.3);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #FF8C42 0%, #F47521 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(244, 117, 33, 0.4);
    }
    
    /* Error message styling */
    .stAlert {
        background-color: #3a1810;
        color: #ffd4cc;
        border: 2px solid #F47521;
        border-radius: 12px;
        padding: 16px;
        font-family: 'Lato', sans-serif;
    }
    
    /* Text colors for Crunchyroll theme */
    p, div, span, h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-family: 'Lato', sans-serif;
    }
    
    /* Code blocks */
    code {
        background-color: #2a2a2a !important;
        color: #F47521 !important;
        padding: 0.25rem 0.5rem !important;
        border-radius: 6px !important;
        font-size: 0.9em !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1a1a1a !important;
        border-right: 2px solid #F47521 !important;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #F47521 !important;
        font-family: 'Lato', sans-serif;
        font-weight: 900;
        letter-spacing: 0.5px;
    }
    
    /* Chat input styling */
    [data-testid="stChatInput"] {
        border-color: #F47521 !important;
    }
    
    [data-testid="stChatInput"]:focus-within {
        border-color: #FF8C42 !important;
        box-shadow: 0 0 0 1px #F47521 !important;
    }
    
    [data-testid="stChatInput"] textarea {
        border-color: #F47521 !important;
    }
    
    [data-testid="stChatInput"] textarea:focus {
        border-color: #FF8C42 !important;
        box-shadow: 0 0 0 1px #F47521 !important;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-top-color: #F47521 !important;
    }
    
    .stSpinner > div > div {
        border-top-color: #F47521 !important;
    }
    
    /* Trace event styling */
    .trace-container {
        margin: 8px 0;
        padding: 12px 16px;
        border-radius: 8px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        animation: slideIn 0.3s ease-out;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    
    .trace-step {
        display: flex;
        align-items: center;
        gap: 12px;
        color: white;
        font-size: 14px;
        font-weight: 600;
    }
    
    .trace-icon {
        font-size: 18px;
        animation: pulse 1.5s infinite;
    }
    
    .trace-completed {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        box-shadow: 0 2px 8px rgba(56, 239, 125, 0.3);
    }
    
    .trace-error {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        box-shadow: 0 2px 8px rgba(235, 51, 73, 0.3);
    }
    
    .trace-details {
        font-size: 12px;
        opacity: 0.95;
        margin-top: 4px;
        font-weight: 400;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(-10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% { 
            transform: scale(1);
            opacity: 1;
        }
        50% { 
            transform: scale(1.1);
            opacity: 0.8;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)

# Helper function to render trace events
def render_trace(trace_data: dict):
    """Render a trace event in the UI with animated styling"""
    step = trace_data.get("step", "Processing")
    status = trace_data.get("status", "in_progress")
    details = trace_data.get("details", "")
    
    # Choose icon and CSS class based on status
    if status == "in_progress":
        icon = "⏳"
        css_class = ""
    elif status == "completed":
        icon = "✅"
        css_class = "trace-completed"
    else:
        icon = "❌"
        css_class = "trace-error"
    
    # Render trace HTML
    trace_html = f"""
    <div class="trace-container {css_class}">
        <div class="trace-step">
            <span class="trace-icon">{icon}</span>
            <strong>{step}</strong>
        </div>
        {f'<div class="trace-details">{details}</div>' if details else ''}
    </div>
    """
    
    return trace_html

# Sidebar — Past Chats
with st.sidebar:
    st.markdown(
        """
        <style>
        /* Sidebar background */
        [data-testid="stSidebar"] {
            background-color: #111111 !important;
            border-right: 1px solid #2a2a2a !important;
        }
        /* Past chats header */
        .past-chats-header {
            font-size: 1.1rem;
            font-weight: 700;
            color: #ffffff;
            padding: 0.5rem 0 0.75rem 0;
            letter-spacing: 0.3px;
        }
        /* Section label (Today / Yesterday / etc.) */
        .chat-section-label {
            font-size: 0.72rem;
            color: #888888;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            padding: 0.6rem 0 0.3rem 0;
        }
        /* Individual chat item */
        .chat-item {
            padding: 0.45rem 0.6rem;
            border-radius: 6px;
            cursor: pointer;
            color: #cccccc;
            font-size: 0.85rem;
            line-height: 1.35;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            transition: background 0.15s;
        }
        .chat-item:hover {
            background-color: #2a2a2a;
            color: #ffffff;
        }
        /* Space badge */
        .space-badge {
            display: inline-block;
            font-size: 0.65rem;
            font-weight: 700;
            color: #F47521;
            background: rgba(244, 117, 33, 0.12);
            border: 1px solid rgba(244, 117, 33, 0.3);
            border-radius: 4px;
            padding: 1px 5px;
            margin-right: 5px;
            vertical-align: middle;
        }
        /* Clear chat button */
        div[data-testid="stSidebar"] .stButton > button {
            background: transparent !important;
            color: #888888 !important;
            border: 1px solid #2a2a2a !important;
            border-radius: 6px !important;
            font-size: 0.8rem !important;
            padding: 4px 10px !important;
            box-shadow: none !important;
            font-weight: 400 !important;
            text-transform: none !important;
            letter-spacing: 0 !important;
        }
        div[data-testid="stSidebar"] .stButton > button:hover {
            background: #2a2a2a !important;
            color: #ffffff !important;
            transform: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header
    st.markdown('<div class="past-chats-header">Past chats</div>', unsafe_allow_html=True)

    # Use session_state to cache chats — avoids re-fetching on every st.rerun()
    chats_placeholder = st.empty()

    if "past_chats_cache" not in st.session_state:
        # Show skeleton while loading
        chats_placeholder.markdown(
            """
            <div style="padding:0.5rem 0;">
                <div style="background:#2a2a2a;border-radius:4px;height:11px;width:85%;margin-bottom:10px;"></div>
                <div style="background:#2a2a2a;border-radius:4px;height:11px;width:65%;margin-bottom:10px;"></div>
                <div style="background:#2a2a2a;border-radius:4px;height:11px;width:75%;margin-bottom:10px;"></div>
                <div style="background:#2a2a2a;border-radius:4px;height:11px;width:55%;margin-bottom:10px;"></div>
                <div style="background:#2a2a2a;border-radius:4px;height:11px;width:70%;margin-bottom:10px;"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # Parallel fetch (cached 5 min by @st.cache_data)
        past_chats = fetch_past_chats(user_token, GENIE_SPACES)
        st.session_state.past_chats_cache = past_chats
        chats_placeholder.empty()
    else:
        past_chats = st.session_state.past_chats_cache
        chats_placeholder.empty()

    def _render_chats(chats):
        if not chats:
            st.markdown(
                '<div style="color:#555;font-size:0.82rem;padding:1rem 0;">No past conversations found.</div>',
                unsafe_allow_html=True,
            )
            return
        time_order = ["Today", "Yesterday", "Previous 7 days", "Older conversations"]
        grouped: dict = defaultdict(lambda: defaultdict(list))
        for item in chats:
            label = _format_chat_timestamp(item["timestamp_ms"])
            grouped[label][item["space_name"]].append(item["query"])
        for time_label in time_order:
            if time_label not in grouped:
                continue
            st.markdown(f'<div class="chat-section-label">{time_label}</div>', unsafe_allow_html=True)
            for space_name, queries in grouped[time_label].items():
                for query in queries:
                    truncated = (query[:52] + "…") if len(query) > 55 else query
                    st.markdown(
                        f'<div class="chat-item">'
                        f'<span class="space-badge">{space_name}</span>'
                        f'{truncated}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    _render_chats(past_chats)

    st.markdown('<div style="margin-top:1.5rem;"></div>', unsafe_allow_html=True)
    is_processing = st.session_state.get("is_processing", False)
    if st.button("🗑️ Clear Chat", use_container_width=True, disabled=is_processing):
        st.session_state.messages = []
        st.session_state.pop("past_chats_cache", None)
        st.rerun()

# Header with Crunchyroll logo
logo_path = os.path.join(os.path.dirname(__file__), "assets", "CR-LOGO-RGB-HORIZONTAL-REGISTERED_Orange.png")
if os.path.exists(logo_path):
    # Center the logo with custom HTML/CSS
    with open(logo_path, "rb") as f:
        import base64
        logo_base64 = base64.b64encode(f.read()).decode()
    
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center; align-items: center; padding: 5px 0 15px 0;">
            <img src="data:image/png;base64,{logo_base64}" style="width: 300px; height: auto;">
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
<div class="main-header-wrapper">
    <h1>CDI Genie Agent</h1>
</div>
""",
        unsafe_allow_html=True,
    )

if not client_initialized:
    st.error(
        "Agent endpoint client initialization failed. Check your environment variables and ensure the agent endpoint is deployed."
    )
else:
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        print("[UI] Initialized empty messages list")
    
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False

    if not st.session_state.messages:
        # Derive display name from email: "abc.fds@some.com" → "Abc Fds"
        try:
            local_part = user_email.split("@")[0]
            display_name = " ".join(part.capitalize() for part in local_part.replace(".", " ").replace("_", " ").split())
        except Exception:
            display_name = "there"
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"Hi {display_name}! Ask me anything about your data to get started.",
                "table_data": None,
                "charts": None,
                "sql": None,
            }
        )
        print("[UI] Added welcome message")

    print(f"[UI] Starting to display {len(st.session_state.messages)} messages")
    
    # Display message history
    for idx, message in enumerate(st.session_state.messages):
        print(f"[UI] Displaying message {idx}: role={message['role']}, content_length={len(message['content'])}")
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Display SQL if present and ENV is dev
            if ENV == "dev" and message.get("sql"):
                with st.expander("🔍 SQL Query", expanded=False):
                    st.code(message["sql"], language="sql")

            # Display table data if present
            if message.get("table_data"):
                table_data = message["table_data"]
                with st.expander("Table Data", expanded=True):
                    df = pd.DataFrame(table_data["data"], columns=table_data["columns"])
                    st.dataframe(df)

            # Display charts if present
            if message.get("charts"):
                for idx, chart_info in enumerate(message["charts"]):
                    chart_type = chart_info.get("chart_type", "unknown")
                    plotly_json = chart_info.get("plotly_json")

                    if plotly_json:
                        # Generate unique key for this chart
                        msg_idx = st.session_state.messages.index(message)
                        chart_key = f"chart_{msg_idx}_{idx}"

                        try:
                            # Use plotly.io.from_json with skip_invalid to handle version incompatibilities
                            fig_json_str = json.dumps(plotly_json)
                            fig = pio.from_json(fig_json_str, skip_invalid=True)
                            
                            # Fix chart formatting issues
                            fig = fix_chart_formatting(fig)
                            
                            # Skip if chart was filtered out (single data point)
                            if fig is None:
                                continue

                            with st.expander(
                                f"{chart_type.capitalize()} Chart", expanded=True
                            ):
                                # Update layout for Crunchyroll theme
                                fig.update_layout(
                                    plot_bgcolor="#1a1a1a",
                                    paper_bgcolor="#1a1a1a",
                                    font=dict(
                                        color="#ffffff", size=14, family="Lato, sans-serif"
                                    ),
                                    title_font=dict(
                                        color="#F47521",
                                        size=18,
                                        family="Lato, sans-serif",
                                        weight="bold",
                                    ),
                                    xaxis=dict(
                                        gridcolor="#333333",
                                        zerolinecolor="#F47521",
                                        color="#ffffff",
                                        title_font=dict(
                                            size=14, color="#F47521", weight="bold"
                                        ),
                                        tickfont=dict(size=12, color="#ffffff"),
                                    ),
                                    yaxis=dict(
                                        gridcolor="#333333",
                                        zerolinecolor="#F47521",
                                        color="#ffffff",
                                        title_font=dict(
                                            size=14, color="#F47521", weight="bold"
                                        ),
                                        tickfont=dict(size=12, color="#ffffff"),
                                    ),
                                )

                                st.plotly_chart(
                                    fig, use_container_width=True, key=chart_key
                                )
                        except Exception as e:
                            print(f"Error displaying chart: {e}")

    # Chat input
    if prompt := st.chat_input("What is your question?"):
        print(f"[UI] User entered prompt: {prompt}")
        print(f"[UI] Current messages count: {len(st.session_state.messages)}")
        
        # Add user message to state
        st.session_state.messages.append(
            {"role": "user", "content": prompt, "table_data": None, "charts": None}
        )
        print(f"[UI] Added user message to state. New count: {len(st.session_state.messages)}")

        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)

        # Show assistant message with trace events
        with st.chat_message("assistant"):
            # Set processing state to True
            st.session_state.is_processing = True
            
            # Create containers for traces and response
            trace_container = st.empty()
            response_container = st.empty()
            
            try:
                print("[UI] Starting streaming query")
                
                # Track active traces and their order
                active_traces = {}
                trace_order = []
                final_response = None
                
                # Stream response with trace events - pass session_id
                for chunk in agent.query_stream(
                    prompt, 
                    user_email=user_email,
                    session_id=session_manager.session_id if session_manager else f"session_{user_email}"
                ):
                    try:
                        # Parse chunk as JSON
                        chunk_obj = json.loads(chunk)
                        
                        # Check if it's a trace event
                        if chunk_obj.get("type") == "trace":
                            step = chunk_obj.get("step")
                            status = chunk_obj.get("status")
                            
                            print(f"[UI] Trace event: {step} - {status}")
                            
                            # Track trace order
                            if step not in active_traces:
                                trace_order.append(step)
                            
                            # Update trace data
                            active_traces[step] = chunk_obj
                            
                            # Render all active traces in order
                            all_traces_html = ""
                            for trace_step in trace_order:
                                if trace_step in active_traces:
                                    all_traces_html += render_trace(active_traces[trace_step])
                            
                            # Update the trace container
                            trace_container.markdown(all_traces_html, unsafe_allow_html=True)
                            
                            # Add delay to show step-by-step progress
                            if status == "in_progress":
                                time.sleep(0.3)  # Pause to show in-progress state
                            elif status == "completed":
                                time.sleep(0.5)  # Longer pause to show completion
                        
                        # Check if it's the final response
                        elif "response" in chunk_obj:
                            print(f"[UI] Received final response")
                            final_response = chunk_obj
                            break
                    
                    except json.JSONDecodeError:
                        # Not JSON, might be plain text
                        print(f"[UI] Non-JSON chunk received: {chunk[:100]}")
                        continue
                
                # Keep traces visible for a moment before clearing
                if active_traces:
                    time.sleep(0.8)
                
                # Clear traces
                trace_container.empty()
                
                # Display final response
                if final_response:
                    response_text = final_response.get("response", "")
                    charts = final_response.get("charts", [])
                    table_data = final_response.get("table_data")
                    sql_query = final_response.get("sql")
                    
                    # Display response text
                    if response_text:
                        response_container.markdown(response_text)
                    
                    # Display SQL if present and ENV is dev
                    if ENV == "dev" and sql_query:
                        with st.expander("🔍 SQL Query", expanded=False):
                            st.code(sql_query, language="sql")
                    
                    # Display charts
                    if charts:
                        for i, chart in enumerate(charts):
                            try:
                                plotly_json = chart.get("plotly_json")
                                if plotly_json:
                                    fig_json_str = json.dumps(plotly_json)
                                    fig = pio.from_json(fig_json_str, skip_invalid=True)
                                    
                                    # Fix chart formatting issues
                                    fig = fix_chart_formatting(fig)
                                    
                                    # Skip if chart was filtered out (single data point)
                                    if fig is None:
                                        continue
                                    
                                    # Apply Crunchyroll theme
                                    fig.update_layout(
                                        plot_bgcolor="#1a1a1a",
                                        paper_bgcolor="#1a1a1a",
                                        font=dict(color="#ffffff", size=14, family="Lato, sans-serif"),
                                        title_font=dict(color="#F47521", size=18, family="Lato, sans-serif", weight="bold"),
                                        xaxis=dict(
                                            gridcolor="#333333",
                                            zerolinecolor="#F47521",
                                            color="#ffffff",
                                            title_font=dict(size=14, color="#F47521", weight="bold"),
                                            tickfont=dict(size=12, color="#ffffff"),
                                        ),
                                        yaxis=dict(
                                            gridcolor="#333333",
                                            zerolinecolor="#F47521",
                                            color="#ffffff",
                                            title_font=dict(size=14, color="#F47521", weight="bold"),
                                            tickfont=dict(size=12, color="#ffffff"),
                                        ),
                                    )
                                    
                                    st.plotly_chart(
                                        fig,
                                        use_container_width=True,
                                        key=f"chart_{len(st.session_state.messages)}_{i}"
                                    )
                            except Exception as e:
                                st.error(f"Error displaying chart: {e}")
                    
                    # Display table
                    if table_data and table_data.get("data"):
                        with st.expander("📊 View Data Table", expanded=False):
                            try:
                                df = pd.DataFrame(
                                    table_data.get("data", []),
                                    columns=table_data.get("columns", [])
                                )
                                st.dataframe(df, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error displaying table: {e}")
                    
                    # Store in session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text,
                        "charts": charts,
                        "table_data": table_data,
                        "sql": sql_query
                    })
                else:
                    # No response received
                    error_msg = "⚠️ No response received from agent"
                    response_container.warning(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "charts": None,
                        "table_data": None,
                        "sql": None
                    })
                
                # Reset processing state
                st.session_state.is_processing = False

            except Exception as e:
                print(f"[UI ERROR] Exception occurred: {str(e)}")
                import traceback
                traceback.print_exc()
                
                # Clear traces on error
                trace_container.empty()
                
                error_msg = f"❌ Error: {str(e)}"
                response_container.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "table_data": None,
                    "charts": None
                })
                
                # Reset processing state on error
                st.session_state.is_processing = False

        print("[UI] About to call st.rerun()")
        st.rerun()
