import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import time

# Professional Page Layout Configuration
st.set_page_config(page_title="Elastocaloric System Simulator", layout="wide")

st.title("📊 Industrial Elastocaloric Air-Cooled Refrigeration Dashboard")
st.markdown("---")

# Initialize Session States for Continuous Cycles
if "cycle_count" not in st.session_state:
    st.session_state.cycle_count = 0
if "chamber_temp" not in st.session_state:
    st.session_state.chamber_temp = 25.0
if "temp_history" not in st.session_state:
    st.session_state.temp_history = [25.0]
if "current_stage_idx" not in st.session_state:
    st.session_state.current_stage_idx = 0
if "stress" not in st.session_state:
    st.session_state.stress = 0.0
if "strain" not in st.session_state:
    st.session_state.strain = 0.0
if "fan_1" not in st.session_state:
    st.session_state.fan_1 = "OFF"
if "fan_2" not in st.session_state:
    st.session_state.fan_2 = "OFF"
if "cop" not in st.session_state:
    st.session_state.cop = 0.0

# ================= USER-DEFINED REQUIREMENTS CONFIGURATION SECTION =================
st.markdown("### ⚙️ System Custom Specifications & Parameters")
config_col1, config_col2, config_col3 = st.columns(3)

with config_col1:
    num_wires = st.number_input("Core Matrix Element Quantity (Wires):", min_value=1, max_value=20, value=5, step=1)
with config_col2:
    wire_len = st.number_input("Custom Core Strand Length (mm):", min_value=10.0, max_value=500.0, value=150.0, step=10.0)
with config_col3:
    ambient_temp = st.number_input("Ambient Atmospheric Starting Temperature (°C):", min_value=15.0, max_value=45.0, value=25.0, step=1.0)

st.markdown("---")

# Reset thermal baseline parameters if simulation metrics zero out
if st.session_state.cycle_count == 0 and len(st.session_state.temp_history) == 1:
    st.session_state.chamber_temp = ambient_temp
    st.session_state.temp_history = [ambient_temp]

# SIDEBAR SYSTEM INPUTS
st.sidebar.header("🛠️ RUN PARAMETERS")
wire_dia = st.sidebar.slider("Wire Diameter (mm)", 0.5, 5.0, 2.0, step=0.1)
strain_limit = st.sidebar.slider("Peak Tensile Strain Limit (%)", 2.0, 8.0, 5.0, step=0.5)
cycle_speed = st.sidebar.slider("Cycle Frequency Speed (Hz)", 0.2, 3.0, 1.0, step=0.1)
fan_flow = st.sidebar.slider("Cooling Fan Flow Rate (CFM)", 10, 100, 50, step=5)

st.sidebar.markdown("---")
st.sidebar.header("🕹️ CYCLE OPERATION CONTROL")
mode = st.sidebar.radio("Control Architecture Mode", ["Manual Diagnostics", "Automatic Mode"])

# PHYSICS ENGINE FUNCTIONS
def execute_physics(stage_idx):
    st.session_state.current_stage_idx = stage_idx
    target_temp = -18.0
    mass_factor = (num_wires / 5.0) * (wire_len / 150.0)
    
    if stage_idx == 1:
        st.session_state.strain = strain_limit
        st.session_state.stress = 450.0 + (strain_limit * 15)
        st.session_state.fan_1 = "OFF"
        st.session_state.fan_2 = "OFF"
        
    elif stage_idx == 2:
        st.session_state.strain = strain_limit
        st.session_state.stress = 420.0
        st.session_state.fan_1 = f"RUNNING ({fan_flow} CFM)"
        st.session_state.fan_2 = "OFF"
        
    elif stage_idx == 3:
        st.session_state.strain = 0.0
        st.session_state.stress = 150.0
        st.session_state.fan_1 = "OFF"
        st.session_state.fan_2 = "OFF"
        
    elif stage_idx == 4:
        st.session_state.strain = 0.0
        st.session_state.stress = 0.0
        st.session_state.fan_1 = "OFF"
        st.session_state.fan_2 = f"RUNNING ({fan_flow} CFM)"
        
        # Calculate heat drawdown curve using custom specs inputs
        if st.session_state.chamber_temp > target_temp:
            efficiency_loss = 0.06 * (2.0 / wire_dia) * (fan_flow / 50.0) * mass_factor
            st.session_state.chamber_temp -= (st.session_state.chamber_temp - target_temp) * efficiency_loss
            
        st.session_state.temp_history.append(st.session_state.chamber_temp)
        if len(st.session_state.temp_history) > 40:
            st.session_state.temp_history.pop(0)
            
        st.session_state.cop = abs((ambient_temp - st.session_state.chamber_temp) / (strain_limit * 0.42 + 0.15))

# ================= DYNAMIC VISUAL STAGE CONTROLLER MATRIX =================
st.markdown("### 🚦 Controller Phase Sequence Status Matrix")
m_col1, m_col2, m_col3, m_col4 = st.columns(4)

with m_col1:
    if st.session_state.current_stage_idx == 1:
        st.success("🔥 **STAGE 1 ACTIVE**\n\nTensile Loading Applied")
    else:
        st.info("Stage 1: Tension Load")
        
with m_col2:
    if st.session_state.current_stage_idx == 2:
        st.success("💨 **STAGE 2 ACTIVE**\n\nWarm Air Exhaust Open")
    else:
        st.info("Stage 2: Warm Exhaust")
        
with m_col3:
    if st.session_state.current_stage_idx == 3:
        st.success("❄️ **STAGE 3 ACTIVE**\n\nTensile Release Executed")
    else:
        st.info("Stage 3: Tensile Unload")
        
with m_col4:
    if st.session_state.current_stage_idx == 4:
        st.success("🥶 **STAGE 4 ACTIVE**\n\nChilled Air Vault Circulation")
    else:
        st.info("Stage 4: Cold Circulation")

st.markdown("---")

# CORE INTERFACE GRID
col1, col2, col3 = st.columns(3)

# Column 1: Chamber Status Indicators
with col1:
    st.subheader("🏢 Structural Compartments")
    st.info(f"**Pulling Chamber Status**\n\nActive Elements: {num_wires} Wires ({wire_dia}mm × {wire_len}mm)\n\nApplied Mechanical Stress: **{st.session_state.stress:.1f} MPa**\n\nStroke Strain: **{st.session_state.strain:.1f} %**")
    
    with st.container(border=True):
        st.markdown("### ❄️ Enclosed Internal Chamber Vault")
        st.metric(label="Chamber Core Temperature", value=f"{st.session_state.chamber_temp:.1f}°C")
        st.markdown("**Target Freeze Goal:** -18.0°C")

# Column 2: System Operational Status 
with col2:
    st.subheader("📊 System Real-Time Metrics")
    st.metric(label="Completed Cycles Counter", value=st.session_state.cycle_count)
    st.metric(label="Calculated System COP", value=f"{st.session_state.cop:.2f}")
    st.text(f"Exhaust Fan (Warm Air): {st.session_state.fan_1}")
    st.text(f"Circulation Fan (Cold Air): {st.session_state.fan_2}")

# Column 3: Diagnostic Trigger Switches
with col3:
    st.subheader("🎮 Interactive Controls")
    
    if mode == "Manual Diagnostics":
        st.markdown("*Click the sequential operational phases manually to review thermodynamic actions:*")
        if st.button("Step 1: Apply Tensile Load", use_container_width=True): execute_physics(1)
        if st.button("Step 2: Trigger Warm Air Blow", use_container_width=True): execute_physics(2)
        if st.button("Step 3: Trigger Released Load", use_container_width=True): execute_physics(3)
        if st.button("Step 4: Trigger Chilled Cold Blow", use_container_width=True): 
            execute_physics(4)
            st.session_state.cycle_count += 1
            st.rerun()
    else:
        st.markdown("*Automation engine active. Click process below to execute sequential loop loops:*")
        if st.button("🚀 Process 5 Continuous Cycles", type="primary", use_container_width=True):
            for _ in range(5):
                execute_physics(1)
                execute_physics(2)
                execute_physics(3)
                execute_physics(4)
                st.session_state.cycle_count += 1
            st.rerun()
                
    if st.button("Reset Entire Simulation", type="secondary", use_container_width=True):
        st.session_state.cycle_count = 0
        st.session_state.chamber_temp = ambient_temp
        st.session_state.temp_history = [ambient_temp]
        st.session_state.current_stage_idx = 0
        st.session_state.stress = 0.0
        st.session_state.strain = 0.0
        st.session_state.cop = 0.0
        st.rerun()

# SCIENTIFIC CHART PANELS
st.markdown("---")
st.subheader("📈 Scientific Performance Graphs")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.patch.set_facecolor('#0E1117')

# Graph 1: Nitinol Hysteresis Loop
ax1.set_facecolor('#1E1E1E')
ax1.set_title("Nitinol Stress-Strain Structural Hysteresis", color='white', fontsize=12)
ax1.set_xlabel("Strain (%)", color='white')
ax1.set_ylabel("Stress (MPa)", color='white')
ax1.tick_params(colors='white')
ax1.set_xlim(0, 8)
ax1.set_ylim(0, 600)
ax1.grid(True, color='#333333')

if st.session_state.strain > 0 or st.session_state.cycle_count > 0:
    strain_pts = [0, max(2.0, st.session_state.strain * 0.3), max(4.0, st.session_state.strain), max(4.0, st.session_state.strain), 0]
    stress_pts = [0, 200, max(400.0, st.session_state.stress), 100, 0]
    ax1.plot(strain_pts, stress_pts, '#FFA500', lw=3, marker='o')

# Graph 2: Chamber Temperature Drop History Curve
ax2.set_facecolor('#1E1E1E')
ax2.set_title("Cooling Chamber Profile Curve", color='white', fontsize=12)
ax2.set_xlabel("Time History Data Steps", color='white')
ax2.set_ylabel("Chamber Temperature (°C)", color='white')
ax2.tick_params(colors='white')
ax2.set_xlim(0, 40)
ax2.set_ylim(-22, 28)
ax2.grid(True, color='#333333')
ax2.plot(st.session_state.temp_history, '#00BFFF', lw=3, marker='s')
ax2.axhline(-18.0, color='red', linestyle='--', alpha=0.7, label='Target (-18°C)')

st.pyplot(fig)
