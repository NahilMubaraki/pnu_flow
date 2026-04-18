import sys
import os
from pathlib import Path
import streamlit as st
import pickle

# UI CONFIGURATION - must be first st command
st.set_page_config(page_title="PNU-Flow Navigation", layout="wide")

# PATH CONFIGURATION
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# IMPORT PIPELINE
query_route = None
PATHS = None
try:
    from pipelines.inference_pipeline import query_route
    from config import PATHS
except ImportError as e:
    st.error(f"Module loading error: {e}")
    st.stop()

# HELPER FUNCTIONS
def format_name(name):
    if not name:
        return name
    return name.replace('_', ' ').title()

st.title("PNU-Flow: Intelligent Indoor Navigation")

# SIDEBAR SETTINGS
st.sidebar.header("Navigation Settings")

# Load zone mapping for dropdowns
try:
    mapping_path = PATHS.artifacts_dir / "zone_mapping.pkl"
    with open(mapping_path, "rb") as f:
        zone_map_data = pickle.load(f)
    zones = sorted(list(zone_map_data.keys()))
except Exception:
    zones = ["main_entrance", "cafeteria", "corridor_A_G", "library", "lab_1"]

source = st.sidebar.selectbox("Select Start Point:", zones, key="src_key", format_func=format_name)
destination = st.sidebar.selectbox("Select Destination:", zones, key="dest_key", format_func=format_name)

# MAIN NAVIGATION LOGIC
if st.sidebar.button("Find Best Route"):
    with st.spinner('Calculating the optimal path using LSTM predictions...'):
        try:
            result = query_route(source, destination)

            col1, col2, col3 = st.columns(3)
            col1.metric("Estimated Time (ETA)", f"{result['eta_seconds']} sec")
            col2.metric("Distance Score", f"{result['distance_weighted_meters']} m")
            col3.metric("Model Confidence", f"{result['avg_model_confidence']*100:.1f}%")

            st.subheader("Recommended Path")
            clean_path = [format_name(step) for step in result['path']]
            st.success(" → ".join(clean_path))
            # --- NEW FEATURE: Route Congestion Details (Heatmap) ---
            with st.expander("🔍 View Route Congestion Details"):
                st.caption("Real-time occupancy predictions along your path:")
                for step in result['path']:
                    # Get the occupancy prediction from the LSTM model
                    occ_val = result['occupancy_predictions'].get(step, 0.0)
                    occ_pct = occ_val * 100
                    
                    # Determine the status icon based on congestion levels
                    if occ_pct < 5.0:
                        status_icon = "🟢" # Very quiet
                    elif occ_pct < 15.0:
                        status_icon = "🟡" # Moderate traffic
                    else:
                        status_icon = "🔴" # Heavy traffic
                        
                    # Display the zone name and its occupancy percentage
                    st.write(f"{status_icon} **{format_name(step)}** — {occ_pct:.1f}%")
                    # Render a progress bar for a quick visual cue
                    st.progress(float(min(occ_val, 1.0)))
            # -------------------------------------------------------

            if result.get('used_shortest_path_fallback'):
                st.warning("Note: Using standard shortest path due to low model confidence.")
            else:
                st.info("Route optimized based on current occupancy (Quiet Route).")

            st.subheader("Suggested Study Spot")
            study = result.get('study_spot')
            if isinstance(study, dict) and 'zone' in study:
                occ = study.get('occupancy_pct', 0) * 100
                st.write(f"The best place to study right now is **{format_name(study['zone'])}** with an estimated occupancy of **{occ:.1f}%**.")
            elif isinstance(study, str):
                st.write(format_name(study))
            else:
                st.info("No specific study spot recommendation available at this moment.")

        except Exception as e:
            st.error(f"Error during inference: {e}")

# SYSTEM MONITORING
st.sidebar.markdown("")
st.sidebar.subheader("System Health")
st.sidebar.write("LSTM Model: Loaded")
st.sidebar.write("Graph Engine: Active")