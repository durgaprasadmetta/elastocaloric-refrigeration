"""
app.py
================================================================================
"Advanced Elastocaloric Refrigeration Using NiTi Alloy" SCADA/HMI Application
================================================================================
An engineering-grade, real physics-based simulation, data-acquisition, and 
SCADA/HMI research application developed for Mechanical Engineering projects.

Run:      streamlit run app.py
Requires: streamlit numpy pandas matplotlib
"""

from __future__ import annotations

import io
import math
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Force non-interactive visualization backend context to prevent canvas locks
matplotlib.use("Agg")

# ==============================================================================
# 1. PHYSICAL CONSTANTS & CONFIGURATIONS
# ==============================================================================
CFM_TO_M3S = 4.719474e-4  
AIR_RHO = 1.184           
AIR_CP = 1005.0           
KELVIN = 273.15           
NOMINAL_CFM = 50.0        

@st.cache_data
def get_material_database():
    return {
        "NiTi (Nitinol, 50.8 at.% Ni)": {"rho": 6450.0, "cp": 470.0, "ds_tr": 40.0, "eps_tr": 0.055, "sigma_limit": 800.0, "cc_slope": 6.5, "sigma_ref": 420.0, "sigma_hyst": 160.0},
        "Cu-Al-Ni single crystal": {"rho": 7100.0, "cp": 400.0, "ds_tr": 22.0, "eps_tr": 0.045, "sigma_limit": 350.0, "cc_slope": 2.2, "sigma_ref": 180.0, "sigma_hyst": 45.0},
        "Fe-Mn-Si alloy": {"rho": 7200.0, "cp": 520.0, "ds_tr": 12.0, "eps_tr": 0.030, "sigma_limit": 600.0, "cc_slope": 1.6, "sigma_ref": 280.0, "sigma_hyst": 140.0},
        "Natural rubber": {"rho": 950.0, "cp": 1900.0, "ds_tr": 28.0, "eps_tr": 3.000, "sigma_limit": 18.0, "cc_slope": 0.02, "sigma_ref": 3.0, "sigma_hyst": 1.2}
    }

# ==============================================================================
# 2. RUNNABLE SOLVER INTERFACE
# ==============================================================================
def execute_scada_simulation(
    rho, cp, ds_tr, eps_tr, sigma_limit, cc_slope, sigma_ref, sigma_hyst,
    form, n_elements, length_mm, od_mm, id_mm, strain_pct, phase_time_s,
    ambient_c, target_c, flow_cfm, mode_compression
) -> pd.DataFrame:
    
    dt = 0.4
    duration_s = 120.0
    steps = int(duration_s / dt)
    cycle_period = 4.0 * phase_time_s
    
    # Calculate geometric parameters
    or_m = (od_mm * 1e-3) / 2.0
    ir_m = (id_mm * 1e-3) / 2.0
    a_single = math.pi * (or_m**2) if form == "wire" else math.pi * (or_m**2 - ir_m**2)
    a_total = a_single * n_elements
    vol = a_total * (length_mm * 1e-3)
    mass = vol * rho
    cap = mass * cp
    wetted_area = (math.pi * od_mm * 1e-3 if form == "wire" else math.pi * (od_mm + id_mm) * 1e-3) * length_mm * 1e-3 * n_elements
    
    # Precompute thermal conductances
    ha = 620.0 * ((max(flow_cfm, 1.0) / NOMINAL_CFM) ** 0.6) * wetted_area
    mdot_cp = flow_cfm * CFM_TO_M3S * AIR_RHO * AIR_CP
    ua_hx = (1.0 - math.exp(-ha / mdot_cp)) * mdot_cp if mdot_cp > 0 else 0.0
    ua_idle = 9.0 * wetted_area
    
    # State flags
    time_s, t_element, t_chamber = 0.0, ambient_c, ambient_c
    accumulated_cooling_j, accumulated_work_j = 0.0, 0.0
    inst_cooling_w, inst_cop = 0.0, 0.0
    
    records = []
    
    for step in range(steps + 1):
        time_within_cycle = time_s % cycle_period
        current_cycle = int(time_s // cycle_period)
        
        if time_within_cycle < phase_time_s:
            phase = 1
            strain = (time_within_cycle / phase_time_s) * (strain_pct / 100.0)
            stress = (sigma_ref + cc_slope * (t_element - 25.0)) + (sigma_hyst / 2.0)
        elif time_within_cycle < 2.0 * phase_time_s:
            phase = 2
            strain = strain_pct / 100.0
            stress = (sigma_ref + cc_slope * (t_element - 25.0)) + (sigma_hyst / 2.0)
        elif time_within_cycle < 3.0 * phase_time_s:
            phase = 3
            fraction = (time_within_cycle - 2.0 * phase_time_s) / phase_time_s
            strain = (1.0 - fraction) * (strain_pct / 100.0)
            stress = max(0.0, (sigma_ref + cc_slope * (t_element - 25.0)) - (sigma_hyst / 2.0))
        else:
            phase = 4
            strain = 0.0
            stress = 0.0
            
        if mode_compression:
            stress = -abs(stress)
            
        force_kn = abs(stress * 1e6 * a_total) / 1000.0
        disp_mm = strain * length_mm
        
        # Latent heat spikes
        q_latent = 0.0
        if phase == 1 and time_within_cycle < dt:
            q_latent = (t_chamber + KELVIN) * ds_tr * min(1.0, strain / eps_tr) * cap / dt
        elif phase == 3 and abs(time_within_cycle - 2.0 * phase_time_s) < dt:
            q_latent = -(t_chamber + KELVIN) * ds_tr * min(1.0, (strain_pct / 100.0) / eps_tr) * cap / dt
            
        # Convection path routing
        ua = ua_hx if phase in (2, 4) else ua_idle
        t_in = t_chamber if phase == 4 else ambient_c
        
        # Thermal derivative stepping
        t_element += ((-ua * (t_element - t_in) + q_latent) / max(1.0, cap)) * dt
        t_out = t_element + (t_in - t_element) * math.exp(-ua / max(1.0, mdot_cp)) if ua > 0 else t_element
        
        # Chamber energy balance
        q_leak = 0.35 * (ambient_c - t_chamber) * dt
        if phase == 4:
            q_cool = ua * (t_element - t_chamber) * dt
            if q_cool < 0:
                accumulated_cooling_j += abs(q_cool)
            t_chamber += ((q_cool + q_leak) / (2.0 * 4184.0))
        else:
            t_chamber += (q_leak / (2.0 * 4184.0))
            
        # Cycle reset configurations
        if abs(time_within_cycle - (cycle_period - dt)) < 1e-3:
            inst_cooling_w = accumulated_cooling_j / cycle_period
            work_input = max(10.0, (sigma_hyst * 1e6 * min(strain_pct/100.0, eps_tr) * vol) / 0.70 + (14.0 * cycle_period))
            inst_cop = accumulated_cooling_j / work_input
            accumulated_cooling_j = 0.0
            
        records.append({
            "time_s": time_s, "cycle": current_cycle, "phase": phase, "chamber_c": t_chamber,
            "element_c": t_element, "air_inlet_c": t_in, "air_outlet_c": t_out, "strain_pct": strain * 100.0,
            "stress_mpa": stress, "force_kn": force_kn, "displacement_mm": disp_mm, "cooling_w": inst_cooling_w,
            "cop": inst_cop, "safety_status": 1.0 if abs(stress) <= sigma_limit else 0.0
        })
        time_s += dt
        
    return pd.DataFrame(records)

# ==============================================================================
# 3. INTERFACE DEPLOYMENT BLOCK
# ==============================================================================
def main():
    st.set_page_config(page_title="NiTi SCADA HMI", layout="wide")
    
    st.title("🎛️ NiTi Elastocaloric Refrigeration SCADA Project")
    st.markdown("---")
    
    # Initialize reactive storage dictionaries safely
    db = get_material_database()
    
    st.sidebar.header("🛠️ Test Rig Controls")
    op_mode = st.sidebar.radio("MONITORING MODE:", ["SIMULATION MODE", "EXPERIMENT MODE"])
    
    mat_selection = st.sidebar.selectbox("Active Alloy Matrix:", list(db.keys()))
    mat_props = db[mat_selection]
    
    with st.sidebar.expander("📝 Modify Material Constants"):
        rho = st.number_input("Density (kg/m³):", value=mat_props["rho"])
        cp = st.number_input("Specific Heat (J/kg·K):", value=mat_props["cp"])
        ds_tr = st.number_input("Entropy ΔS (J/kg·K):", value=mat_props["ds_tr"])
        eps_tr = st.number_input("Max Trans Strain:", value=mat_props["eps_tr"], format="%.3f")
        sigma_limit = st.number_input("Stress Yield Limit (MPa):", value=mat_props["sigma_limit"])
        cc_slope = st.number_input("CC Slope (MPa/K):", value=mat_props["cc_slope"])
        sigma_ref = st.number_input("Plateau Base Stress (MPa):", value=mat_props["sigma_ref"])
        sigma_hyst = st.number_input("Hysteresis (MPa):", value=mat_props["sigma_hyst"])

    st.sidebar.subheader("📐 Element Geometry")
    form = st.sidebar.radio("Profile Geometry Factor:", ["tube", "wire"])
    n_elements = st.sidebar.number_input("Element Count:", min_value=1, value=5)
    length_mm = st.sidebar.number_input("Length (mm):", value=150.0)
    od_mm = st.sidebar.number_input("Outer Dia (mm):", value=12.0)
    id_mm = st.sidebar.number_input("Inner Dia (mm):", value=10.0)
    
    st.sidebar.subheader("🔄 Mechanical Controls")
    strain_pct = st.sidebar.slider("Applied Stroke (Strain %):", 0.5, 10.0, 5.5, 0.1)
    phase_time_s = st.sidebar.slider("Phase Steps Interval (s):", 0.1, 5.0, 0.8, 0.1)
    mode_compression = st.sidebar.checkbox("Compression Profile Vector")
    
    st.sidebar.subheader("🎯 Chamber Boundaries")
    ambient_c = st.sidebar.number_input("Ambient Environment Base (°C):", value=25.0)
    target_c = st.sidebar.number_input("Chamber Target Level (°C):", value=10.0)
    flow_cfm = st.sidebar.slider("Volumetric Air Flow (CFM):", 10.0, 200.0, 50.0, 5.0)

    # Process solver triggers safely behind a responsive spinner
    with st.spinner("Processing transient non-linear engineering matrix equations..."):
        df = execute_scada_simulation(
            rho, cp, ds_tr, eps_tr, sigma_limit, cc_slope, sigma_ref, sigma_hyst,
            form, n_elements, length_mm, od_mm, id_mm, strain_pct, phase_time_s,
            ambient_c, target_c, flow_cfm, mode_compression
        )

    # Telemetry Instrumentation Readouts Rendering Section
    st.markdown("### 📊 Real-Time SCADA Dashboard Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Chamber Core Temp", f"{df['chamber_c'].iloc[-1]:.2f} °C")
    m2.metric("NiTi Element Node", f"{df['element_c'].iloc[-1]:.2f} °C")
    m3.metric("Mechanical Work Net", f"{df['force_kn'].iloc[-1]:.2f} kN")
