import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import time

# Page Layout Configuration
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
if "current_stage" not in st.session_state:
    st.session_state.current_stage = "System Ready"
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

# SIDEBAR SYSTEM INPUTS
st.sidebar.header("🛠️ SYSTEM CONFIGURATION")
st.sidebar.markdown("**Fixed Geometry Baseline:**\n* 5 Parallel Wires\n* 150mm Length")

wire_dia = st.sidebar.slider("Wire Diameter (mm)", 0.5, 5.0, 2.0, step=0.1)
strain_limit = st.sidebar.slider("Peak Tensile Strain Limit (%)", 2.0, 8.0, 5.0, step=0.5)
cycle_speed = st.sidebar.slider("Cycle Frequency Speed (Hz)", 0.2, 3.0, 1.0, step=0.1)
fan_flow = st.sidebar.slider("Cooling Fan Flow Rate (CFM)", 10, 100, 50, step=5)

st.sidebar.markdown("---")
st.sidebar.header("🕹️ CYCLE OPERATION CONTROL")
mode = st.sidebar.radio("Control Architecture Mode", ["Manual Diagnostics", "Automatic Mode"])

# PHYSICS ENGINE FUNCTIONS
def execute_physics(stage_idx):
    target_temp = -18.0
    
    if stage_idx == 1:
        st.session_state.current_stage = "Stage 1: Tensile Loading (Adiabatic Exothermic)"
        st.session_state.strain = strain_limit
        st.session_state.stress = 450.0 + (strain_limit * 15)
        st.session_state.fan_1 = "OFF"
        st.session_state.fan_2 = "OFF"
        
    elif stage_idx == 2:
        st.session_state.current_stage = "Stage 2: Warm Air Blow (Heat Rejection Open)"
        st.session_state.strain = strain_limit
        st.session_state.stress = 420.0
        st.session_state.fan_1 = f"RUNNING ({fan_flow} CFM)"
        st.session_state.fan_2 = "OFF"
        
    elif stage_idx == 3:
        st.session_state.current_stage = "Stage 3: Released Load (Adiabatic Endothermic)"
        st.session_state.strain = 0.0
        st.session_state.stress = 150.0
        st.session_state.fan_1 = "OFF"
        st.session_state.fan_2 = "OFF"
        
    elif stage_idx == 4:
        st.session_state.current_stage = "Stage 4: Chilled Cold Blow (Refrigeration Transfer)"
        st.session_state.strain = 0.0
        st.session_state.stress = 0.0
        st.session_state.fan_1 = "OFF"
        st.session_state.fan_2 = f"RUNNING ({fan_flow} CFM)"
        
        # Calculate heat drawdown curve toward -18 target limits
        if st.session_state.chamber_temp > target_temp:
            efficiency_loss = 0.06 * (2.0 / wire_dia) * (fan_flow / 50.0)
            st.session_state.chamber_temp -= (st.session_state.chamber_temp - target_temp) * efficiency_loss
            
        st.session_state.temp_history.append(st.session_state.chamber_temp)
        if len(st.session_state.temp_history) > 40:
            st.session_state.temp_history.pop(0)
            
        st.session_state.cop = abs((25.0 - st.session_state.chamber_temp) / (strain_limit * 0.42 + 0.15))

# CORE INTERFACE GRID
col1, col2, col3 = st.columns(3)

# Column 1: Chamber Status Indicators
with col1:
    st.subheader("🏢 Structural Compartments")
    st.info(f"**Pulling Chamber Status**\n\nActive Core: 5 Wires (2mm × 150mm)\n\nApplied Mechanical Stress: **{st.session_state.stress:.1f} MPa**\n\nStroke Strain: **{st.session_state.strain:.1f} %**")
    
    with st.container(border=True):
        st.markdown("### ❄️ Isolated Sub-Zero Vault")
        st.metric(label="Chamber Temperature", value=f"{st.session_state.chamber_temp:.1f}°C")
        st.markdown("**Target Freeze Goal:** -18.0°C")

# Column 2: System Operational Status 
with col2:
    st.subheader("📊 System Real-Time Metrics")
    st.metric(label="Completed Cycles Counter", value=st.session_state.cycle_count)
    st.metric(label="Calculated System COP", value=f"{st.session_state.cop:.2f}")
    st.text(f"Exhaust Fan (Warm Air): {st.session_state.fan_1}")
    st.text(f"Circulation Fan (Cold Air): {st.session_state.fan_2}")
    st.warning(f"**Controller Status:**\n{st.session_state.current_stage}")

# Column 3: Diagnostic Trigger Switches
with col3:
    st.subheader("🎮 Interactive Controls")
    
    if mode == "Manual Diagnostics":
        st.markdown("*Click the sequential operational phases manually to review thermodynamic actions:*")
        if st.button("Step 1: Apply Tensile Load"): execute_physics(1)
        if st.button("Step 2: Trigger Warm Air Blow"): execute_physics(2)
        if st.button("Step 3: Trigger Released Load"): execute_physics(3)
        if st.button("Step 4: Trigger Chilled Cold Blow"): 
            execute_physics(4)
            st.session_state.cycle_count += 1
    else:
        st.markdown("*Automatic Mode Active. Click button below to process continuous cooling test strings:*")
        if st.button("🚀 Process 5 Continuous Cycles"):
            for _ in range(5):
                execute_physics(1)
                execute_physics(2)
                execute_physics(3)
                execute_physics(4)
                st.session_state.cycle_count += 1
                
    if st.button("Reset Entire Simulation", type="primary"):
        st.session_state.cycle_count = 0
        st.session_state.chamber_temp = 25.0
        st.session_state.temp_history = [25.0]
        st.session_state.current_stage = "System Reset Done"
        st.session_state.stress = 0.0
        st.session_state.strain = 0.0
        st.session_state.cop = 0.0

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
