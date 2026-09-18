import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Professional Page Layout Configuration
st.set_page_config(
    page_title="Elastocaloric System Simulator", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- EXPERT-GRADE STYLING INTERFACE ---
st.title("🎛️ Industrial Elastocaloric Refrigeration Control Center")
st.markdown("##### **SOLID-STATE THERMODYNAMIC SIMULATION ENGINE • DESIGN VERIFICATION PANEL**")
st.markdown("---")

# Initialize Session States for Continuous Cycles
if "cycle_count" not in st.session_state:
    st.session_state.cycle_count = 0
if "chamber_temp" not in st.session_state:
    st.session_state.chamber_temp = 25.0
if "temp_history" not in st.session_state:
    st.session_state.temp_history = [25.0]
if "current_stage" not in st.session_state:
    st.session_state.current_stage = "System Initialized & Ready"
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

# ================= SIDEBAR CRITICAL SYSTEM INPUTS =================
st.sidebar.header("🛠️ CORE SPECIFICATIONS")
st.sidebar.markdown("---")
st.sidebar.markdown("### **Fixed Geometry Baseline**")
st.sidebar.markdown("- **Active Bundle Elements:** 5 Wires")
st.sidebar.markdown("- **Core Strand Length:** 150.0 mm")
st.sidebar.markdown("---")

st.sidebar.markdown("### **Adjustable Boundary Conditions**")
wire_dia = st.sidebar.slider("Wire Diameter (d)", min_value=0.5, max_value=5.0, value=2.0, step=0.1, format="%.1f mm")
strain_limit = st.sidebar.slider("Peak Tensile Strain (ε)", min_value=2.0, max_value=8.0, value=5.0, step=0.5, format="%.1f %%")
cycle_speed = st.sidebar.slider("Actuation Frequency (f)", min_value=0.2, max_value=3.0, value=1.0, step=0.1, format="%.1f Hz")
fan_flow = st.sidebar.slider("Convective Fan Flow (V)", min_value=10, max_value=100, value=50, step=5, format="%d CFM")

st.sidebar.markdown("---")
st.sidebar.markdown("### **Control Logic Overrides**")
mode = st.sidebar.radio("Select Loop Operation Mode:", ["Manual Diagnostics", "Automated Cycling Loop"])

# ================= CYCLIC THERMODYNAMIC ENGINE =================
def execute_physics_stage(stage_idx):
    target_temp = -18.0
    
    if stage_idx == 1:
        st.session_state.current_stage = "STAGE 1: Uniaxial Tension Applied [Adiabatic Exothermic Phase]"
        st.session_state.strain = strain_limit
        st.session_state.stress = 450.0 + (strain_limit * 15)
        st.session_state.fan_1 = "STANDBY [OFF]"
        st.session_state.fan_2 = "STANDBY [OFF]"
        
    elif stage_idx == 2:
        st.session_state.current_stage = "STAGE 2: Constant Strain Thermal Dwell [Exhaust Air Active]"
        st.session_state.strain = strain_limit
        st.session_state.stress = 420.0
        st.session_state.fan_1 = f"ACTIVE [V = {fan_flow} CFM]"
        st.session_state.fan_2 = "STANDBY [OFF]"
        
    elif stage_idx == 3:
        st.session_state.current_stage = "STAGE 3: Uniaxial Relaxation Executed [Adiabatic Endothermic Phase]"
        st.session_state.strain = 0.0
        st.session_state.stress = 150.0
        st.session_state.fan_1 = "STANDBY [OFF]"
        st.session_state.fan_2 = "STANDBY [OFF]"
        
    elif stage_idx == 4:
        st.session_state.current_stage = "STAGE 4: Constant Strain Cooling Blow [Circulation Blower Active]"
        st.session_state.strain = 0.0
        st.session_state.stress = 0.0
        st.session_state.fan_1 = "STANDBY [OFF]"
        st.session_state.fan_2 = f"ACTIVE [V = {fan_flow} CFM]"
        
        # Newtonian Transient Heat Extraction Equations
        if st.session_state.chamber_temp > target_temp:
            heat_transfer_coeff = 0.06 * (2.0 / wire_dia) * (fan_flow / 50.0)
            st.session_state.chamber_temp -= (st.session_state.chamber_temp - target_temp) * heat_transfer_coeff
            
        st.session_state.temp_history.append(st.session_state.chamber_temp)
        if len(st.session_state.temp_history) > 40:
            st.session_state.temp_history.pop(0)
            
        st.session_state.cop = abs((25.0 - st.session_state.chamber_temp) / (strain_limit * 0.42 + 0.15))

# ================= HIGH-READABILITY COMPARTMENT GRID =================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🏢 Core Physical Status")
    with st.container(border=True):
        st.markdown("#### **Actuator Mechanics**")
        st.markdown(f"**Structural Stress (σ):** `{st.session_state.stress:.1f} MPa`")
        st.markdown(f"**Displacement Strain (ε):** `{st.session_state.strain:.1f} %`")
        st.markdown("---")
        st.markdown("#### **Sub-Zero Isolated Vault**")
        st.metric(
            label="Internal Chamber Temp (T_ch)", 
            value=f"{st.session_state.chamber_temp:.1f} °C", 
            delta=f"Target Limit: -18.0 °C",
            delta_color="inverse"
        )

with col2:
    st.markdown("### 📊 Thermodynamic Performance")
    with st.container(border=True):
        st.markdown("#### **Real-Time Efficiency Metrics**")
        st.metric(label="Total Completed Cycles (N)", value=st.session_state.cycle_count)
        st.metric(label="Coefficient of Performance (COP)", value=f"{st.session_state.cop:.2f}")
        st.markdown("---")
        st.markdown("#### **Convective Air Manifold**")
        st.markdown(f"**Exhaust Circuit Blower:** `{st.session_state.fan_1}`")
        st.markdown(f"**Circulation Circuit Blower:** `{st.session_state.fan_2}`")

with col3:
    st.markdown("### 🎮 Control Interface")
    with st.container(border=True):
        st.markdown("#### **Cycle Execution Framework**")
        st.info(f"**PLC Status Indicator:**\n\n{st.session_state.current_stage}")
        st.markdown("---")
        
        if mode == "Manual Diagnostics":
            st.markdown("**Diagnostic Step Sequences:**")
            if st.button("Trigger Stage 1: Load Core", use_container_width=True): execute_physics_stage(1)
            if st.button("Trigger Stage 2: Warm Exhaust", use_container_width=True): execute_physics_stage(2)
            if st.button("Trigger Stage 3: Unload Core", use_container_width=True): execute_physics_stage(3)
            if st.button("Trigger Stage 4: Cold Circulation", use_container_width=True): 
                execute_physics_stage(4)
                st.session_state.cycle_count += 1
        else:
            st.markdown("**Continuous System Diagnostics:**")
            if st.button("🚀 Execute 5 Autonomous Cycles", type="primary", use_container_width=True):
                for _ in range(5):
                    for stage_num in range(1, 5):
                        execute_physics_stage(stage_num)
                    st.session_state.cycle_count += 1
                    
        if st.button("Emergency System Reset", type="secondary", use_container_width=True):
            st.session_state.cycle_count = 0
            st.session_state.chamber_temp = 25.0
            st.session_state.temp_history = [25.0]
            st.session_state.current_stage = "System Initialized & Ready"
            st.session_state.stress = 0.0
            st.session_state.strain = 0.0
            st.session_state.cop = 0.0
            st.rerun()

# ================= SCIENTIFIC COMPREHENSIVE CHARTS =================
st.markdown("---")
st.markdown("### 📈 Real-Time Data Acquisition & Multi-Plot Transient Charts")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
fig.patch.set_facecolor('#0E1117')

# Graph 1: Nitinol Material Mechanical Response
ax1.set_facecolor('#1E1E1E')
ax1.set_title("Nitinol Phase Boundary Stress-Strain Hysteresis Loop", color='white', fontsize=12, pad=10)
ax1.set_xlabel("Strain ε (%)", color='white', fontsize=10)
ax1.set_ylabel("Stress σ (MPa)", color='white', fontsize=10)
ax1.tick_params(colors='white', labelsize=9)
ax1.set_xlim(0, 8)
ax1.set_ylim(0, 600)
ax1.grid(True, color='#333333', linestyle=':')

if st.session_state.strain > 0 or st.session_state.cycle_count > 0:
    strain_pts = [0, max(2.0, st.session_state.strain * 0.3), max(4.0, st.session_state.strain), max(4.0, st.session_state.strain), 0]
    stress_pts = [0, 200, max(400.0, st.session_state.stress), 100, 0]
    ax1.plot(strain_pts, stress_pts, color='#FFA500', lw=3, marker='o', label='Active Loop Path')
    ax1.fill(strain_pts, stress_pts, color='#FFA500', alpha=0.1)
    ax1.legend(loc="upper left")

# Graph 2: Transient Chamber Thermal Drawdown Curve
ax2.set_facecolor('#1E1E1E')
ax2.set_title("Chamber Thermal Drawdown Profile Curve [T_ch vs Time]", color='white', fontsize=12, pad=10)
ax2.set_xlabel("Data Sampling Intervals (t)", color='white', fontsize=10)
ax2.set_ylabel("Temperature T (°C)", color='white', fontsize=10)
ax2.tick_params(colors='white', labelsize=9)
ax2.set_xlim(0, 40)
ax2.set_ylim(-22, 28)
ax2.grid(True, color='#333333', linestyle=':')
ax2.plot(st.session_state.temp_history, color='#00BFFF', lw=3, marker='s', label='Chamber Temperature')
ax2.axhline(-18.0, color='#FF3333', linestyle='--', alpha=0.8, lw=2, label='Target Freeze Barrier (-18°C)')
ax2.legend(loc="upper right")

st.pyplot(fig)
