import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import time

# Professional Page Layout Configuration
st.set_page_config(
    page_title="Industrial Elastocaloric SCADA Control Facility", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MASTER CORE STATE ENGINE INITIALIZATION ---
if "cycle_count" not in st.session_state:
    st.session_state.cycle_count = 0
if "chamber_temp" not in st.session_state:
    st.session_state.chamber_temp = 25.0
if "temp_history" not in st.session_state:
    st.session_state.temp_history = [25.0]
if "current_stage_idx" not in st.session_state:
    st.session_state.current_stage_idx = 0  

# Expanded Multi-Node Thermal State Variables
if "wire_max_temp" not in st.session_state:
    st.session_state.wire_max_temp = 25.0
if "wire_min_temp" not in st.session_state:
    st.session_state.wire_min_temp = 25.0
if "air_max_temp" not in st.session_state:
    st.session_state.air_max_temp = 25.0
if "air_min_temp" not in st.session_state:
    st.session_state.air_min_temp = 25.0

# Mechanical Tracking Array
if "stress" not in st.session_state:
    st.session_state.stress = 0.0
if "strain" not in st.session_state:
    st.session_state.strain = 0.0
if "cop" not in st.session_state:
    st.session_state.cop = 0.0
if "auto_running" not in st.session_state:
    st.session_state.auto_running = False

# ================= NAVIGATION DASHBOARD TABS =================
tab1, tab2 = st.tabs(["🖥️ Front-End Live Presentation & SCADA Graphics", "⚙️ System Back-End Parameters"])

# ================= TAB 2: BACK-END PARAMETERS =================
with tab2:
    st.markdown("### 🎛️ User-Defined Engineering Configuration Panel")
    st.markdown("---")
    
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.markdown("#### **Material & Core Matrix Assembly**")
        material_selection = st.selectbox("Active Core Shape Memory Element Type:", ["Nitinol (Nickel-Titanium Baseline)", "Cu-Al-Ni Crystal Alloy", "Fe-Mn-Si Smart Element", "Elastocaloric Cooling Polymer"])
        num_wires = st.number_input("Bundle Core Element Quantity (Wires):", min_value=1, max_value=20, value=5, step=1)
        wire_len = st.number_input("Custom Core Length Parameter (mm):", min_value=10.0, max_value=500.0, value=150.0, step=10.0)
        wire_dia = st.slider("Wire Diameter Profile Bounds (mm):", 0.5, 5.0, 2.0, step=0.1)
        
    with col_b2:
        st.markdown("#### **Thermodynamic Boundary Variables**")
        force_mode = st.selectbox("Actuator Operation Stroke Profile Setting:", ["Uniaxial Tension (Tensile Stretching)", "Uniaxial Compression (Structural Pressing)"])
        ambient_temp = st.number_input("Atmospheric Ambient Starting Temperature (°C):", min_value=15.0, max_value=45.0, value=25.0, step=1.0)
        target_limit = st.number_input("Sub-Zero Vault Target Freeze Goal (°C):", min_value=-40.0, max_value=10.0, value=-18.0, step=1.0)

    st.success("✅ Back-End Parameters Locked Securely into Volatile Microprocessor Memory State.")

# Synchronize Ambient Thermal Baselines on Reset Block
if st.session_state.cycle_count == 0 and len(st.session_state.temp_history) == 1:
    st.session_state.chamber_temp = ambient_temp
    st.session_state.temp_history = [ambient_temp]
    st.session_state.wire_max_temp = ambient_temp
    st.session_state.wire_min_temp = ambient_temp
    st.session_state.air_max_temp = ambient_temp
    st.session_state.air_min_temp = ambient_temp

# SIDEBAR QUICK OVERRIDES
st.sidebar.header("🕹️ CONTROL RUN TIME")
strain_limit = st.sidebar.slider("Peak Stroke Strain (ε):", 2.0, 8.0, 5.0, step=0.5, format="%.1f %%")
cycle_speed = st.sidebar.slider("Sequence Delay Spacing (seconds):", 0.1, 3.0, 0.8, step=0.1, format="%.1f sec/phase")
fan_flow = st.sidebar.slider("Convective Fan Flow Volume (CFM):", 10, 100, 50, step=5)
st.sidebar.markdown("---")
mode = st.sidebar.radio("Select Automation Framework Architecture:", ["Manual Diagnostics", "Automated Cycling Loop"])

# ================= CYCLIC PHYSICS STEP SEQUENCER =================
def execute_step_physics(stage_idx):
    st.session_state.current_stage_idx = stage_idx
    mass_factor = (num_wires / 5.0) * (wire_len / 150.0)
    
    mat_scalar = 1.2 if material_selection == "Cu-Al-Ni Crystal Alloy" else (1.5 if material_selection == "Elastocaloric Cooling Polymer" else 1.0)
    latent_delta = 12.5 * (strain_limit / 5.0) * mat_scalar
    direction_sign = -1.0 if force_mode == "Uniaxial Compression (Structural Pressing)" else 1.0
    
    if stage_idx == 1: 
        st.session_state.strain = strain_limit
        st.session_state.stress = direction_sign * (450.0 + (strain_limit * 12))
        st.session_state.wire_max_temp = ambient_temp + latent_delta
        st.session_state.air_max_temp = ambient_temp + (latent_delta * 0.4)
        
    elif stage_idx == 2: 
        st.session_state.strain = strain_limit
        st.session_state.stress = direction_sign * 410.0
        cooling_efficiency = (fan_flow / 100.0)
        st.session_state.wire_max_temp -= (st.session_state.wire_max_temp - ambient_temp) * cooling_efficiency
        st.session_state.air_max_temp -= (st.session_state.air_max_temp - ambient_temp) * cooling_efficiency
        
    elif stage_idx == 3: 
        st.session_state.strain = 0.0
        st.session_state.stress = direction_sign * 120.0
        st.session_state.wire_min_temp = ambient_temp - latent_delta
        st.session_state.air_min_temp = ambient_temp - (latent_delta * 0.5)
        
    elif stage_idx == 4: 
        st.session_state.strain = 0.0
        st.session_state.stress = 0.0
        
        if st.session_state.chamber_temp > target_limit:
            efficiency_scalar = 0.05 * (2.0 / wire_dia) * (fan_flow / 50.0) * mass_factor
            st.session_state.chamber_temp -= (st.session_state.chamber_temp - target_limit) * efficiency_scalar
        
        st.session_state.air_min_temp = min(st.session_state.air_min_temp, st.session_state.chamber_temp)
        st.session_state.wire_min_temp = min(st.session_state.wire_min_temp, st.session_state.chamber_temp - 2.0)
        
        st.session_state.temp_history.append(st.session_state.chamber_temp)
        if len(st.session_state.temp_history) > 40:
            st.session_state.temp_history.pop(0)
            
        st.session_state.cop = abs((ambient_temp - st.session_state.chamber_temp) / (strain_limit * 0.4 + 0.1))

# ================= TAB 1: FRONT-END PRESENTATION =================
with tab1:
    st.markdown("### 🎚️ Master Control Panel Overrides")
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
    
    with ctrl_col1:
        if mode == "Automated Cycling Loop":
            if not st.session_state.auto_running:
                if st.button("🚀 Start Continuous Sequence Loop", type="primary", use_container_width=True):
                    st.session_state.auto_running = True
                    st.rerun()
            else:
                if st.button("🛑 Halt Sequence Loop", type="secondary", use_container_width=True):
                    st.session_state.auto_running = False
                    st.rerun()
        else:
            st.markdown("**Manual Single-Phase Operations Active**")
            
    with ctrl_col2:
        if st.button("🔄 Emergency Master System Reset", type="secondary", use_container_width=True):
            st.session_state.cycle_count = 0
            st.session_state.chamber_temp = ambient_temp
            st.session_state.temp_history = [ambient_temp]
            st.session_state.wire_max_temp = ambient_temp
            st.session_state.wire_min_temp = ambient_temp
            st.session_state.air_max_temp = ambient_temp
            st.session_state.air_min_temp = ambient_temp
            st.session_state.stress = 0.0
            st.session_state.strain = 0.0
            st.session_state.cop = 0.0
            st.session_state.auto_running = False
            st.session_state.current_stage_idx = 0
            st.rerun()
            
    with ctrl_col3:
        st.info(f"**Controller Core Protocol:** {mode}")

    st.markdown("---")
    
    # ================= ⚡ ANIMATED SCADA PIPING AND STATUS TILES =================
    st.markdown("### 🚦 Live SCADA Flow Manifold Tracking Pane")
    s1, s2, s3, s4 = st.columns(4)
    
    with s1:
        if st.session_state.current_stage_idx == 1:
            with st.status("🔥 PHASE 1: CHARGING ACTUATOR", state="running", expanded=True):
                st.markdown("**Core Status:** Stretching Crystal Lattice")
                st.markdown("**Exhaust Valve:** `CLOSED 🔴`")
                st.markdown("**Circulation Valve:** `CLOSED 🔴`")
                st.markdown("⚡ *Latent Heat Packet Released*")
        else:
            with st.status("Phase 1: Actuator Charge", state="complete", expanded=False):
                st.write("Node Standby")

    with s2:
        if st.session_state.current_stage_idx == 2:
            with st.status("💨 PHASE 2: EXHAUST OPEN", state="running", expanded=True):
                st.markdown("**Core Status:** Holding Peak Deflection")
                st.markdown("**Exhaust Valve:** `OPEN 🟢 [Venting]`")
                st.markdown("**Circulation Valve:** `CLOSED 🔴`")
                st.markdown(f"🍃 *Flushing Core Heat at {fan_flow} CFM*")
        else:
            with st.status("Phase 2: Thermal Exhaust", state="complete", expanded=False):
                st.write("Node Standby")

    with s3:
        if st.session_state.current_stage_idx == 3:
            with st.status("❄️ PHASE 3: DISCHARGE CORE", state="running", expanded=True):
                st.markdown("**Core Status:** Structural Relaxation Snapping")
                st.markdown("**Exhaust Valve:** `CLOSED 🔴`")
                st.markdown("**Circulation Valve:** `CLOSED 🔴`")
                st.markdown("🧊 *Adiabatic Latent Temperature Plunge*")
        else:
            with st.status("Phase 3: Actuator Discharge", state="complete", expanded=False):
