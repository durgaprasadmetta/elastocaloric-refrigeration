"""
app.py
================================================================================
"Advanced Elastocaloric Refrigeration Using NiTi Alloy" SCADA/HMI Application
================================================================================
A professional-grade real physics-based simulation, data-acquisition, and 
SCADA/HMI research application developed for Mechanical Engineering projects.
Monitors coupled thermomechanical, structural, air-side convective transport, 
and closed environmental chamber thermal energy balances.

Run:      streamlit run app.py
Requires: streamlit numpy pandas matplotlib
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

# Force non-interactive visualization backend context
matplotlib.use("Agg")

# ==============================================================================
# 1. PHYSICAL CONSTANTS & ENGINEERING CONVERSIONS
# ==============================================================================
CFM_TO_M3S = 4.719474e-4  # Conversion from cubic feet per minute to m³/s
AIR_RHO = 1.184           # Atmospheric density of dry air at 25°C (kg/m³)
AIR_CP = 1005.0           # Specific heat capacity of air at constant pressure (J/kg·K)
KELVIN = 273.15           # Absolute zero temperature offset definition
NOMINAL_CFM = 50.0        # Reference baseline volumetric flow rate (CFM)

PHASE_MAP = {
    1: "Phase 1: Mechanical Loading (Adiabatic Heating)",
    2: "Phase 2: Heat Rejection (Convective Air Venting)",
    3: "Phase 3: Mechanical Unloading (Adiabatic Cooling)",
    4: "Phase 4: Heat Absorption (Product Chamber Pull-Down)"
}

# ==============================================================================
# 2. DEFINITION DATA MODELS & DATABASES
# ==============================================================================
@dataclass(frozen=True)
class MaterialProperties:
    key: str
    name: str
    rho: float             # Density (kg/m³)
    cp: float              # Specific heat capacity (J/kg·K)
    ds_tr: float           # Transformation entropy change (J/kg·K)
    eps_tr: float          # Maximum transformation strain value (dimensionless)
    sigma_ref_mpa: float   # Reference transformation plateau stress (MPa)
    t_ref_c: float         # Reference engineering temperature for stress model (°C)
    cc_slope: float        # Clausius-Clapeyron thermomechanical slope (MPa/K)
    sigma_hyst_mpa: float  # Mechanical loading/unloading hysteresis envelope (MPa)
    sigma_limit_mpa: float # Structural working mechanical stress yield limit (MPa)
    af_c: float            # Austenite finish temperature threshold (°C)
    fatigue_cycles: int    # Indicative mechanical operational life metric (cycles)

    def calculate_adiabatic_swing(self, current_temp_c: float, applied_strain: float) -> float:
        """Computes structural lattice temperature step using entropy balance."""
        transformation_fraction = min(1.0, max(0.0, applied_strain / self.eps_tr))
        absolute_temp_k = current_temp_c + KELVIN
        return absolute_temp_k * self.ds_tr * transformation_fraction / self.cp

    def calculate_plateau_stress(self, current_temp_c: float) -> float:
        """Determines thermodynamic forward transformation stress plateau baseline."""
        return self.sigma_ref_mpa + self.cc_slope * (current_temp_c - self.t_ref_c)


MATERIAL_DATABASE: Dict[str, MaterialProperties] = {
    "niti": MaterialProperties(
        "niti", "NiTi (Nitinol, 50.8 at.% Ni Alloy)",
        6450.0, 470.0, 40.0, 0.055, 420.0, 25.0, 6.5, 160.0, 800.0, -5.0, 100000
    ),
    "cualni": MaterialProperties(
        "cualni", "Cu-Al-Ni Shape Memory Single Crystal",
        7100.0, 400.0, 22.0, 0.045, 180.0, 25.0, 2.2, 45.0, 350.0, 5.0, 20000
    ),
    "femnsi": MaterialProperties(
        "femnsi", "Fe-Mn-Si Ferrous Shape Memory Alloy",
        7200.0, 520.0, 12.0, 0.030, 280.0, 25.0, 1.6, 140.0, 600.0, 20.0, 500000
    ),
    "rubber": MaterialProperties(
        "rubber", "Natural Rubber (Elastocaloric Polymer Matrix)",
        950.0, 1900.0, 28.0, 3.000, 3.0, 25.0, 0.02, 1.2, 18.0, -60.0, 1000000
    ),
}

@dataclass(frozen=True)
class ExchangerConfiguration:
    key: str
    name: str
    convective_h_ref: float  # Base heat transfer coefficient (W/m²·K)
    parasitic_w: float       # Auxiliary mechanical/electrical fan power footprint (W)


EXCHANGER_DATABASE: Dict[str, ExchangerConfiguration] = {
    "bare_air": ExchangerConfiguration("bare_air", "Bare Tube matrix, Forced Convection Air", 90.0, 8.0),
    "finned_air": ExchangerConfiguration("finned_air", "Finned Surface Grid, Forced Convection Air", 620.0, 14.0),
}

# ==============================================================================
# 3. ADVANCED FLUID LOOP THERMOMECHANICAL SYSTEM LAYOUT CONFIGURATOR
# ==============================================================================
@dataclass(frozen=True)
class SystemConfiguration:
    material_key: str = "niti"
    exchanger_key: str = "finned_air"
    element_form: str = "tube"
    n_elements: int = 5
    length_mm: float = 150.0
    od_mm: float = 12.0
    id_mm: float = 10.0
    applied_strain_pct: float = 5.0
    mode_compression: bool = False
    phase_time_s: float = 0.8
    flow_rate_cfm: float = 50.0
    ambient_temp_c: float = 25.0
    target_temp_c: float = 10.0
    chamber_fluid_mass_kg: float = 2.0  # Thermal equivalent water-mass mass load
    insulation_ua_w_k: float = 0.35     # Enclosure parasitic leak coefficient
    actuator_efficiency: float = 0.70   # Mechanical drive powertrain efficiency
    regen_effectiveness: float = 0.0    # Solid-state regeneration recovery factor
    fixed_boundary_end: str = "right"   # Rig boundary kinematics labeling definition

    @property
    def operational_material(self) -> MaterialProperties:
        return MATERIAL_DATABASE[self.material_key]

    @property
    def structural_exchanger(self) -> ExchangerConfiguration:
        return EXCHANGER_DATABASE[self.exchanger_key]

    @property
    def applied_strain(self) -> float:
        return self.applied_strain_pct / 100.0

    @property
    def elements_cross_section_area_m2(self) -> float:
        outer_radius_m = (self.od_mm * 1e-3) / 2.0
        inner_radius_m = (self.id_mm * 1e-3) / 2.0
        if self.element_form == "wire":
            single_area = math.pi * (outer_radius_m ** 2)
        else:
            single_area = math.pi * (outer_radius_m ** 2 - inner_radius_m ** 2)
        return single_area * self.n_elements

    @property
    def structural_volume_m3(self) -> float:
        return self.elements_cross_section_area_m2 * (self.length_mm * 1e-3)

    @property
    def structural_mass_kg(self) -> float:
        return self.structural_volume_m3 * self.operational_material.rho

    @property
    def thermal_capacitance_j_k(self) -> float:
        return self.structural_mass_kg * self.operational_material.cp

    @property
    def wetted_surface_area_m2(self) -> float:
        outer_dia_m = self.od_mm * 1e-3
        inner_dia_m = self.id_mm * 1e-3
        length_m = self.length_mm * 1e-3
        if self.element_form == "wire":
            perimeter = math.pi * outer_dia_m
        else:
            perimeter = math.pi * (outer_dia_m + inner_dia_m)
        return perimeter * length_m * self.n_elements

    @property
    def convective_ua_hx_w_k(self) -> float:
        ex = self.structural_exchanger
        reynolds_scaling = (max(self.flow_rate_cfm, 1.0) / NOMINAL_CFM) ** 0.6
        local_h_coefficient = ex.convective_h_ref * reynolds_scaling
        total_ha = local_h_coefficient * self.wetted_surface_area_m2
        
        mass_flow_rate_air = self.flow_rate_cfm * CFM_TO_M3S * AIR_RHO
        heat_capacity_flow_air = mass_flow_rate_air * AIR_CP
        if heat_capacity_flow_air <= 0:
            return 0.0
        # NTU-Effectiveness fluid loop thermal boundary implementation
        effectiveness = 1.0 - math.exp(-total_ha / heat_capacity_flow_air)
        return effectiveness * heat_capacity_flow_air

    @property
    def natural_convection_ua_w_k(self) -> float:
        """Parasitic heat transfer baseline during closed/stopped valve steps."""
        return 9.0 * self.wetted_surface_area_m2

    @property
    def cycle_period_s(self) -> float:
        return 4.0 * self.phase_time_s

    @property
    def intrinsic_hysteresis_loss_j(self) -> float:
        clamped_strain = min(self.applied_strain, self.operational_material.eps_tr)
        stress_hysteresis_pa = self.operational_material.sigma_hyst_mpa * 1e6
        return stress_hysteresis_pa * clamped_strain * self.structural_volume_m3


# ==============================================================================
# 4. SOLVER COMPONENT: TRANSIENT PHYSICS LOOP
# ==============================================================================
def execute_scada_physics_simulation(cfg: SystemConfiguration, run_duration_s: float = 240.0) -> pd.DataFrame:
    """
    Executes a high-fidelity numerical transient finite-difference time integration
    of the thermodynamic, structural mechanical, convective air-transport, and 
    enclosure thermal load network models. Output parameters are grounded explicitly.
    """
    dt = 0.4  # Step size (seconds) matching system data logging framework
    steps = int(run_duration_s / dt)
    
    # Establish initialization baselines matching standard thermodynamic starting points
    time_s = 0.0
    current_cycle = 0
    current_phase = 1
    
    t_element = cfg.ambient_temp_c
    t_chamber = cfg.ambient_temp_c
    t_fluid_in = cfg.ambient_temp_c
    t_fluid_out = cfg.ambient_temp_c
    
    strain_state = 0.0
    stress_state_mpa = 0.0
    
    accumulated_cooling_joules = 0.0
    accumulated_work_joules = 0.0
    
