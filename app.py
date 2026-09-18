import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import time

# Professional Page Layout Configuration
st.set_page_config(
    page_title="Elastocaloric Advanced Simulation Center", 
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
    st.session_state.current_stage_idx = 0  # 0: Idle, 1: Load, 2: Dwell, 3: Unload, 4: Blow

# Expanded Multi-Node Thermal State Variables
if "wire_max_temp" not in st.session_state:
    st.session_state.wire_max_temp = 25.0
if "wire_min_temp" not in st.session_state:
    st.session_state.wire_min_temp = 25.0
if "air_max_temp" not in st.session_state:
    st.session_state.air_max_temp = 25.0
if "air_min_temp" not in st.session_state:
    st.session_state.air_min_temp = 25.0
if "ambient_air_temp" not in st.session_state:
    st.session_state.ambient_air_temp = 25.0

# Mechanical Tracking
if "stress" not in st.session_state:
    st.session_state.stress = 0.0
if "strain" not in st.session_state:
    st.session_state.strain = 0.0
if "fan_1" not in st.session_state:
    st.session_state.fan_1 = "STANDBY [OFF]"
if "fan_2" not in st.session_state:
    st.session_state.fan_2 = "STANDBY [OFF]"
if "cop" not in st.session_state:
    st.session_state.cop = 0.0
if "auto_running" not in st.session_state:
    st.session_state.auto_running = False

# ================= NAVIGATION DASHBOARD TABS =================
tab1, tab2 = st.tabs(["🖥️ Front-End Live Presentation & Graphics", "⚙️ System Back-End Parameters"])

# ================= TAB 2: BACK-END PARAMETERS =================
with tab2:
    st.markdown("### 🎛️ User-Defined Engineering Configuration Panel")
    st.markdown("---")
    
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.markdown("#### **Material & Core Matrix Assembly**")
        material_type = st.selectbox("Active Shape Memory Core Alloy Type:", ["Nitinol (Nickel-Titanium Core)", "Cu-Al-Ni Crystal Alloy", "Elastocaloric Cooling Polymer"])
        num_wires = st.number_input("Active Bundle Element Quantity (Wires):", min_value=1, max_value=20, value=5, step=1)
        wire_len = st.number_input("Core Strand Custom Length (mm):", min_value=10.0, max_value=500.0, value=150.0, step=10.0)
        wire_dia = st.slider("Wire Boundary Diameter Setting (mm):", 0.5, 5.0, 2.0, step=0.1)
        
    with col_b2:
        st.markdown("#### **Thermodynamic Boundary Variables**")
        ambient_temp = st.number_input("Ambient Atmospheric Starting Temperature (°C):", min_value=15.0, max_value=45.0, value=25.0, step=1.0)
        target_limit = st.number_input("Target Sub-Zero Vault Cooling Limit (°C):", min_value=-40.0, max_value=10.0, value=-18.0, step=1.0)
        insulation_loss = st.selectbox("Vault Thermal Insulation Barrier Rating:", ["High Grade Polyurethane Foam", "Standard Double-Wall Vacuum Insulated", "Uninsulated Test Environment"])

    st.success("✅ Back-End Parameters Locked Securely into Memory State.")

# Synchronize Ambient Thermal Baselines on Reset Block
if st.session_state.cycle_count == 0 and len(st.session_state.temp_history) == 1:
    st.session_state.chamber_temp = ambient_temp
    st.session_state.temp_history = [ambient_temp]
    st.session_state.wire_max_temp = ambient_temp
    st.session_state.wire_min_temp = ambient_temp
    st.session_state.air_max_temp = ambient_temp
    st.session_state.air_min_temp = ambient_temp
    st.session_state.ambient_air_temp = ambient_temp

# ================= SIDEBAR QUICK OVERRIDES =================
st.sidebar.header("🕹️ CONTROL RUN TIME")
st.sidebar.markdown("---")
strain_limit = st.sidebar.slider("Peak Tensile Strain Bounds (ε):", 2.0, 8.0, 5.0, step=0.5, format="%.1f %%")
cycle_speed = st.sidebar.slider("Automation Loop Delay Spacing (seconds):", 0.2, 3.0, 0.8, step=0.1, format="%.1f sec/phase")
fan_flow = st.sidebar.slider("Convective Fan Volume (V):", 10, 100, 50, step=5, format="%d CFM")

st.sidebar.markdown("---")
mode = st.sidebar.radio("Select Operational Architecture:", ["Manual Diagnostics", "Automated Cycling Loop"])

# ================= CYCLIC PHYSICS STEP SEQUENCER =================
def execute_step_physics(stage_idx):
    st.session_state.current_stage_idx = stage_idx
    mass_factor = (num_wires / 5.0) * (wire_len / 150.0)
    
    # Latent heat temperature spikes calculated dynamically based on material selection
    latent_delta = 12.5 * (strain_limit / 5.0)
    
    if stage_idx == 1: # LOADING
        st.session_state.strain = strain_limit
        st.session_state.stress = 450.0 + (strain_limit * 12)
        st.session_state.fan_1 = "STANDBY [OFF]"
        st.session_state.fan_2 = "STANDBY [OFF]"
        
        # Wire temperature spikes adiabatically
        st.session_state.wire_max_temp = ambient_temp + latent_delta
        st.session_state.air_max_temp = ambient_temp + (latent_delta * 0.4)
        
    elif stage_idx == 2: # EXHAUST DWELL
        st.session_state.strain = strain_limit
        st.session_state.stress = 410.0
        st.session_state.fan_1 = f"ACTIVE BLOWING [{fan_flow} CFM]"
        st.session_state.fan_2 = "STANDBY [OFF]"
        
        # Convective cooling vents out the wire heat to ambient levels
        cooling_efficiency = (fan_flow / 100.0)
        st.session_state.wire_max_temp -= (st.session_state.wire_max_temp - ambient_temp) * cooling_efficiency
        st.session_state.air_max_temp -= (st.session_state.air_max_temp - ambient_temp) * cooling_efficiency
        
    elif stage_idx == 3: # UNLOADING
        st.session_state.strain = 0.0
        st.session_state.stress = 120.0
        st.session_state.fan_1 = "STANDBY [OFF]"
        st.session_state.fan_2 = "STANDBY [OFF]"
        
        # Wire snaps back dropping temperature below zero point limits
        st.session_state.wire_min_temp = ambient_temp - latent_delta
        st.session_state.air_min_temp = ambient_temp - (latent_delta * 0.5)
        
    elif stage_idx == 4: # COLD CIRCULATION
        st.session_state.strain = 0.0
        st.session_state.stress = 0.0
        st.session_state.fan_1 = "STANDBY [OFF]"
        st.session_state.fan_2 = f"ACTIVE CHILLING [{fan_flow} CFM]"
        
        # Thermodynamic extraction steps pushing into insulated chamber vault box
        insulation_coeff = 1.0 if insulation_loss == "Standard Double-Wall Vacuum Insulated" else (1.4 if insulation_loss == "High Grade Polyurethane Foam" else 0.5)
        if st.session_state.chamber_temp > target_limit:
            efficiency_scalar = 0.05 * (2.0 / wire_dia) * (fan_flow / 50.0) * mass_factor * insulation_coeff
            st.session_state.chamber_temp -= (st.session_state.chamber_temp - target_limit) * efficiency_scalar
        
        # Adjust boundary parameters for surrounded manifold air streams
        st.session_state.air_min_temp = min(st.session_state.air_min_temp, st.session_state.chamber_temp)
        st.session_state.wire_min_temp = min(st.session_state.wire_min_temp, st.session_state.chamber_temp - 2.0)
        
        st.session_state.temp_history.append(st.session_state.chamber_temp)
        if len(st.session_state.temp_history) > 40:
            st.session_state.temp_history.pop(0)
            
        st.session_state.cop = abs((ambient_temp - st.session_state.chamber_temp) / (strain_limit * 0.4 + 0.1))

# ================= TAB 1: FRONT-END presentation =================
with tab1:
    # 🎚️ HIGH-CONTRAST STEP AUTOMATION SEQUENCE MATRIX
    st.markdown("### 🚦 Loop Controller Phase Status Matrix")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    
    with m_col1:
        if st.session_state.current_stage_idx == 1:
            st.success("🔥 **STAGE 1 ACTIVE**\n\nTensile Loading Core")
        else:
            st.markdown("<div style='opacity: 0.4; padding:10px; border:1px solid gray;'>Stage 1: Tension Load</div>", unsafe_allowed_html=True)
            
    with m_col2:
        if st.session_state.current_stage_idx == 2:
            st.success("💨 **STAGE 2 ACTIVE**\n\nWarm Air Heat Exhaust")
        else:
            st.markdown("<div style='opacity: 0.4; padding:10px; border:1px solid gray;'>Stage 2: Warm Exhaust</div>", unsafe_allowed_html=True)
            
    with m_col3:
        if st.session_state.current_stage_idx == 3:
            st.info("❄️ **STAGE 3 ACTIVE**\n\nCore Release Relaxation")
        else:
            st.markdown("<div style='opacity: 0.4; padding:10px; border:1px solid gray;'>Stage 3: Tensile Unload</div>", unsafe_allowed_html=True)
            
    with m_col4:
        if st.session_state.current_stage_idx == 4:
            st.info("🥶 **STAGE 4 ACTIVE**\n\nChilled Air Vault Circulation")
        else:
            st.markdown("<div style='opacity: 0.4; padding:10px; border:1px solid gray;'>Stage 4: Cold Circulation</div>", unsafe_allowed_html=True)

    st.markdown("---")

    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        st.markdown("#### 🏢 Active Real-Time Mechanism")
        with st.container(border=True):
            st.markdown("**Core Bundle Pulling Stroke Mechanics**")
            stretch_pct = int((st.session_state.strain / 8.0) * 100)
            st.progress(min(100, max(0, stretch_pct)))
            st.markdown(f"Applied Load Stress (σ): `{st.session_state.stress:.1f} MPa` | Strain (ε): `{st.session_state.strain:.1f} %`")
            
            st.markdown("---")
            st.markdown("**Isolated Deep Freeze Vault Target Box**")
            st.metric(
                label="Chamber Core Temperature (T_ch)", 
                value=f"{st.session_state.chamber_temp:.1f} °C", 
                delta=f"Target: {target_limit:.1f} °C",
