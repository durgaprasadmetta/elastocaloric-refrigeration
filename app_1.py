"""
app.py
=================================================================
NiTi elastocaloric refrigeration demonstrator — physics + HMI
+ rule-based "material expert" recommendation agent.

Run:      streamlit run app.py
Requires: streamlit numpy pandas matplotlib

MODEL
-----
Two lumped capacities integrated in time:

    C_e dT_e/dt = -UA_hx (T_e - T_sink) + Q_tr        element
    C_c dT_c/dt = -UA_cold (T_c - T_e) + UA_par (T_a - T_c)

Adiabatic elastocaloric swing:   dT_ad = T ds_tr xi / cp
Transformation stress:           sigma(T) = sigma_ref + C_cc (T - T_ref)
Exchanger:                       NTU-effectiveness on the gas side
=================================================================
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.use("Agg")

# =================================================================
# PART 1 — PHYSICS CORE
# =================================================================

CFM_TO_M3S = 4.719474e-4
AIR_RHO = 1.184
AIR_CP = 1005.0
KELVIN = 273.15
NOMINAL_CFM = 50.0

PHASE_NAMES = {1: "Load", 2: "Reject heat", 3: "Unload", 4: "Absorb heat"}


@dataclass(frozen=True)
class Material:
    key: str
    name: str
    rho: float
    cp: float
    ds_tr: float
    eps_tr: float
    sigma_ref_mpa: float
    t_ref_c: float
    cc_slope: float
    sigma_hyst_mpa: float
    sigma_limit_mpa: float
    af_c: float
    fatigue_cycles: int

    def dt_adiabatic(self, t_c: float, eps: float) -> float:
        xi = min(1.0, max(0.0, eps / self.eps_tr))
        return (t_c + KELVIN) * self.ds_tr * xi / self.cp

    def plateau_stress_mpa(self, t_c: float) -> float:
        return self.sigma_ref_mpa + self.cc_slope * (t_c - self.t_ref_c)


MATERIALS: Dict[str, Material] = {m.key: m for m in [
    Material("niti", "NiTi (Nitinol, 50.8 at.% Ni)",
             6450, 470, 40.0, 0.055, 420, 25.0, 6.5, 160, 800, -5.0, 100_000),
    Material("cualni", "Cu-Al-Ni single crystal",
             7100, 400, 22.0, 0.045, 180, 25.0, 2.2, 45, 350, 5.0, 20_000),
    Material("femnsi", "Fe-Mn-Si shape memory alloy",
             7200, 520, 12.0, 0.030, 280, 25.0, 1.6, 140, 600, 20.0, 500_000),
    Material("rubber", "Natural rubber (elastocaloric polymer)",
             950, 1900, 28.0, 3.00, 3.0, 25.0, 0.02, 1.2, 18, -60.0, 1_000_000),
]}


@dataclass(frozen=True)
class Exchanger:
    key: str
    name: str
    h_ref: float
    liquid: bool
    pump_ref_w: float


EXCHANGERS: Dict[str, Exchanger] = {e.key: e for e in [
    Exchanger("bare_air", "Bare tube, forced air", 90.0, False, 8.0),
    Exchanger("finned_air", "Finned plate, forced air", 620.0, False, 14.0),
    Exchanger("liquid", "Water loop, counterflow", 3200.0, True, 22.0),
]}


@dataclass(frozen=True)
class Config:
    material_key: str = "niti"
    exchanger_key: str = "finned_air"
    form: str = "tube"
    n_elements: int = 5
    length_mm: float = 150.0
    od_mm: float = 12.0
    id_mm: float = 10.0
    strain_pct: float = 5.0
    mode_compression: bool = False
    phase_time_s: float = 0.8
    flow_cfm: float = 50.0
    ambient_c: float = 25.0
    target_c: float = 10.0
    chamber_capacity: float = 800.0
    insulation_ua: float = 0.35
    actuator_efficiency: float = 0.70
    regen_effectiveness: float = 0.0   # 0 = no regenerator (original model)
    fixed_end: str = "right"           # "right" or "left" — the end held mechanically fixed

    @property
    def displacement_end(self) -> str:
        return "left" if self.fixed_end == "right" else "right"

    @property
    def material(self) -> Material:
        return MATERIALS[self.material_key]

    @property
    def exchanger(self) -> Exchanger:
        return EXCHANGERS[self.exchanger_key]

    @property
    def strain(self) -> float:
        return self.strain_pct / 100.0

    @property
    def section_area(self) -> float:
        od = self.od_mm * 1e-3
        idm = min(self.id_mm, self.od_mm - 0.2) * 1e-3
        a = (math.pi / 4.0 * od ** 2 if self.form == "wire"
             else math.pi / 4.0 * (od ** 2 - idm ** 2))
        return a * self.n_elements

    @property
    def volume(self) -> float:
        return self.section_area * self.length_mm * 1e-3

    @property
    def mass(self) -> float:
        return self.volume * self.material.rho

    @property
    def capacity(self) -> float:
        return self.mass * self.material.cp

    @property
    def wetted_area(self) -> float:
        od = self.od_mm * 1e-3
        idm = min(self.id_mm, self.od_mm - 0.2) * 1e-3
        per = math.pi * od if self.form == "wire" else math.pi * (od + idm)
        return per * self.length_mm * 1e-3 * self.n_elements

    @property
    def ua_hx(self) -> float:
        ex = self.exchanger
        scale = (max(self.flow_cfm, 1.0) / NOMINAL_CFM) ** 0.6
        ha = ex.h_ref * scale * self.wetted_area
        if ex.liquid:
            return ha
        mdot_cp = self.flow_cfm * CFM_TO_M3S * AIR_RHO * AIR_CP
        if mdot_cp <= 0:
            return 0.0
        return (1.0 - math.exp(-ha / mdot_cp)) * mdot_cp

    @property
    def ua_idle(self) -> float:
        """Natural convection while the fluid path is closed."""
        return 9.0 * self.wetted_area

    @property
    def regen_gain(self) -> float:
        """Span-extension factor from regeneration, capped at 5x."""
        eps = min(max(self.regen_effectiveness, 0.0), 0.8)
        return 1.0 / (1.0 - eps)

    def effective_dt_adiabatic(self, t_c: float) -> float:
        return self.material.dt_adiabatic(t_c, self.strain) * self.regen_gain

    @property
    def floor_c(self) -> float:
        """Ideal single-stage floor."""
        return self.ambient_c - self.effective_dt_adiabatic(self.ambient_c)

    @property
    def pump_power(self) -> float:
        return self.exchanger.pump_ref_w * (max(self.flow_cfm, 1.0) / NOMINAL_CFM) ** 2

    @property
    def cycle_time(self) -> float:
        return 4.0 * self.phase_time_s

    @property
    def hysteresis_work(self) -> float:
        eps = min(self.strain, self.material.eps_tr)
        return self.material.sigma_hyst_mpa * 1e6 * eps * self.volume

    @property
    def work_per_cycle(self) -> float:
        return (self.hysteresis_work / self.actuator_efficiency + self.pump_power * self.cycle_time)


# =================================================================
# PART 2 — PHYSICS ENGINE & SOLVER
# =================================================================

def run_simulation(cfg: Config, duration_s: float = 300.0) -> pd.DataFrame:
    """
    Executes a finite-difference time integration of the elastocaloric rig.
    Tracks structural mechanics and dual-node thermal capacities across 4 phases.
    """
    dt = 0.4  # Step size matching your exact data spacing
    steps = int(duration_s / dt)
    
    # Initialize baseline operational nodes matching row zero
    st_t = 0.0
    st_cycle = 0
    st_phase = 1
    st_t_element = cfg.ambient_c
    st_t_chamber = cfg.ambient_c
    st_t_fluid = cfg.ambient_c
    st_strain_pct = 0.0
    st_stress_mpa = 0.0
    
    # Tracking registers for cyclic integration loops
    cycle_cooling_j = 0.0
    current_cop = 0.0
    current_cooling_w = 0.0
    
    history: List[Dict[str, float]] = []
    
    # Capture step 0 row profile values
    history.append({
        "time_s": st_t, "cycle": float(st_cycle), "chamber_c": st_t_chamber,
        "element_c": st_t_element, "coolant_c": st_t_fluid, "strain_pct": st_strain_pct,
        "stress_mpa": st_stress_mpa, "cop": current_cop, "cooling_w": current_cooling_w
    })
    
    for _ in range(1, steps + 1):
        st_t += dt
        
        cycle_time_elapsed = st_t % cfg.cycle_time
        st_cycle = int(st_t // cfg.cycle_time)
        
        # Operational phase switching map definitions
        if cycle_time_elapsed < cfg.phase_time_s:
            st_phase = 1
        elif cycle_time_elapsed < 2.0 * cfg.phase_time_s:
            st_phase = 2
        elif cycle_time_elapsed < 3.0 * cfg.phase_time_s:
            st_phase = 3
        else:
            st_phase = 4
            
        # --- Kinematic & Structural Mechanical Sub-Solver ---
        if st_phase in:
            st_strain_pct = cfg.strain_pct
            st_stress_mpa = cfg.material.plateau_stress_mpa(st_t_element) + (cfg.material.sigma_hyst_mpa / 2.0)
        elif st_phase == 3:
            st_strain_pct = 0.0
            st_stress_mpa = cfg.material.plateau_stress_mpa(st_t_element) - (cfg.material.sigma_hyst_mpa / 2.0)
            if st_stress_mpa < 0:
                st_stress_mpa = 0.0
        else:
            st_strain_pct = 0.0
            st_stress_mpa = 0.0
            
        if cfg.mode_compression:
            st_stress_mpa = -abs(st_stress_mpa)
            
        # --- Transient Lumped Thermal Capacitance Solver ---
        q_tr = 0.0
        if cycle_time_elapsed >= 0.0 and cycle_time_elapsed < dt:
            q_tr = cfg.effective_dt_adiabatic(st_t_chamber) * cfg.capacity / dt
        elif cycle_time_elapsed >= 2.0 * cfg.phase_time_s and cycle_time_elapsed < 2.0 * cfg.phase_time_s + dt:
            q_tr = -cfg.effective_dt_adiabatic(st_t_chamber) * cfg.capacity / dt
            
        # Establish thermal boundary conditions depending on flow state routing
        if st_phase == 2:
            ua_active = cfg.ua_hx
            t_sink = cfg.ambient_c
        elif st_phase == 4:
            ua_active = cfg.ua_hx
            t_sink = st_t_chamber
        else:
            ua_active = cfg.ua_idle
            t_sink = cfg.ambient_c
            
        # Thermal derivative updates execution step routines
        dt_element = (-ua_active * (st_t_element - t_sink) + q_tr) / cfg.capacity
