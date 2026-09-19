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
from dataclasses import dataclass, fields, replace
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
    custom_area_mm2: float = 50.0      # used only when form == "custom"
    custom_perimeter_mm: float = 30.0  # used only when form == "custom"

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
        """
        Cross-sectional area of the whole bundle, m2. Not hard-coded to
        one geometry: tube and wire are computed from diameters, and
        "custom" takes the person's own measured or specified area
        directly, so the physics below adapts to whatever element the
        person actually has rather than forcing a predefined shape.
        """
        if self.form == "custom":
            return self.custom_area_mm2 * 1e-6 * self.n_elements
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
        """Heat-transfer surface, m2 — same custom-geometry adaptation
        as section_area above."""
        if self.form == "custom":
            return (self.custom_perimeter_mm * 1e-3 * self.length_mm
                    * 1e-3 * self.n_elements)
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
        """
        Span-extension factor from regeneration, capped at 5x (80%
        effectiveness). This is a simplified proxy — real counter-flow
        AMR-style regenerator analysis is more involved — but it captures
        the qualitative effect correctly: recycling sensible heat between
        the hot and cold halves of the cycle lets the machine pump across
        a span larger than a single element's adiabatic swing, and the
        gain grows sharply as effectiveness approaches 1.
        """
        eps = min(max(self.regen_effectiveness, 0.0), 0.8)
        return 1.0 / (1.0 - eps)

    def effective_dt_adiabatic(self, t_c: float) -> float:
        return self.material.dt_adiabatic(t_c, self.strain) * self.regen_gain

    @property
    def floor_c(self) -> float:
        """Ideal single-stage floor: ambient minus the (regen-boosted)
        adiabatic swing."""
        return self.ambient_c - self.effective_dt_adiabatic(self.ambient_c)

    @property
    def pump_power(self) -> float:
        return self.exchanger.pump_ref_w * (
            max(self.flow_cfm, 1.0) / NOMINAL_CFM) ** 2

    @property
    def cycle_time(self) -> float:
        return 4.0 * self.phase_time_s

    @property
    def hysteresis_work(self) -> float:
        eps = min(self.strain, self.material.eps_tr)
        return self.material.sigma_hyst_mpa * 1e6 * eps * self.volume

    @property
    def work_per_cycle(self) -> float:
        return (self.hysteresis_work / self.actuator_efficiency
                + self.pump_power * self.cycle_time)


@dataclass
class State:
    t: float = 0.0
    phase: int = 0
    cycle: int = 0
    t_element: float = 25.0
    t_chamber: float = 25.0
    t_fluid: float = 25.0
    strain_pct: float = 0.0
    stress_mpa: float = 0.0
    q_cold_cycle: float = 0.0
    q_cold_rate: float = 0.0
    cop: float = 0.0
    target_reached: bool = False


def initial_state(cfg: Config) -> State:
    return State(t_element=cfg.ambient_c, t_chamber=cfg.ambient_c,
                 t_fluid=cfg.ambient_c)


def _relax(t_hot, t_sink, ua, cap, dt):
    """Exact exponential relaxation of one capacity toward a sink."""
    if ua <= 0 or cap <= 0:
        return t_hot
    return t_sink + (t_hot - t_sink) * math.exp(-ua * dt / cap)


def step_phase(cfg: Config, st_in: State, phase: int,
               substeps: int = 24) -> State:
    """Advance the machine through one phase. Returns a new State."""
    mat = cfg.material
    s = replace(st_in, phase=phase)
    dt = cfg.phase_time_s / substeps

    if phase == 1:
        s.strain_pct = cfg.strain_pct
        s.t_element += cfg.effective_dt_adiabatic(s.t_element)
        s.stress_mpa = mat.plateau_stress_mpa(s.t_element) + \
            mat.sigma_hyst_mpa / 2.0
    elif phase == 2:
        s.strain_pct = cfg.strain_pct
        s.stress_mpa = mat.plateau_stress_mpa(s.t_element) + \
            mat.sigma_hyst_mpa / 2.0
    elif phase == 3:
        s.strain_pct = 0.0
        s.t_element -= cfg.effective_dt_adiabatic(s.t_element)
        s.stress_mpa = max(0.0, mat.plateau_stress_mpa(s.t_element)
                           - mat.sigma_hyst_mpa / 2.0)
    else:
        s.strain_pct = 0.0
        s.stress_mpa = 0.0

    if cfg.mode_compression:
        s.stress_mpa = -abs(s.stress_mpa)

    ua_active, ua_off, q_cold = cfg.ua_hx, cfg.ua_idle, 0.0

    for _ in range(substeps):
        if phase == 2:
            s.t_element = _relax(s.t_element, cfg.ambient_c,
                                 ua_active, cfg.capacity, dt)
            s.t_fluid = cfg.ambient_c + 0.45 * (s.t_element - cfg.ambient_c)
        elif phase == 4:
            dq = ua_active * (s.t_chamber - s.t_element) * dt
            q_cold += max(0.0, dq)
            s.t_element += dq / cfg.capacity
            s.t_chamber -= dq / cfg.chamber_capacity
            s.t_fluid = s.t_chamber - 0.45 * (s.t_chamber - s.t_element)
        else:
            s.t_element = _relax(s.t_element, cfg.ambient_c,
                                 ua_off, cfg.capacity, dt)
            s.t_fluid += (cfg.ambient_c - s.t_fluid) * 0.25

        s.t_chamber = _relax(s.t_chamber, cfg.ambient_c,
                             cfg.insulation_ua, cfg.chamber_capacity, dt)

    s.q_cold_cycle += q_cold
    s.t += cfg.phase_time_s

    if phase == 4:
        s.q_cold_rate = s.q_cold_cycle / cfg.cycle_time
        w = cfg.work_per_cycle
        s.cop = s.q_cold_cycle / w if w > 0 else 0.0
        s.q_cold_cycle = 0.0
        s.cycle += 1

    if s.t_chamber <= cfg.target_c:
        s.t_chamber = cfg.target_c
        s.target_reached = True

    return s


def step_cycle(cfg: Config, s: State) -> State:
    for phase in (1, 2, 3, 4):
        s = step_phase(cfg, s, phase)
    return s


def predict_envelope(cfg: Config, max_cycles: int = 4000) -> Dict[str, float]:
    """Head-less solve with the target suppressed: what can this rig hold?"""
    probe = replace(cfg, target_c=-273.0)
    s = initial_state(probe)
    t_reached, prev, cop, n = math.nan, s.t_chamber, 0.0, 0

    for n in range(1, max_cycles + 1):
        s = step_cycle(probe, s)
        cop = s.cop
        if math.isnan(t_reached) and s.t_chamber <= cfg.target_c:
            t_reached = s.t
        if abs(prev - s.t_chamber) < 1e-4 and n > 20:
            break
        prev = s.t_chamber

    return {
        "t_min_c": s.t_chamber,
        "time_to_target_s": t_reached,
        "cycles_to_target": (t_reached / probe.cycle_time
                             if not math.isnan(t_reached) else math.nan),
        "steady_cop": cop,
        "steady_lift_w": s.q_cold_rate,
        "cycles_simulated": n,
        "reachable": s.t_chamber <= cfg.target_c + 1e-6,
        "dt_adiabatic": cfg.effective_dt_adiabatic(cfg.ambient_c),
    }


def check_interlocks(cfg: Config, s: State) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    mat = cfg.material
    if abs(s.stress_mpa) > mat.sigma_limit_mpa:
        out.append(("alarm", f"Stress {abs(s.stress_mpa):.0f} MPa is above "
                             f"the {mat.sigma_limit_mpa:.0f} MPa working limit."))
    if cfg.strain > mat.eps_tr * 1.15:
        out.append(("alarm", f"Strain {cfg.strain_pct:.1f}% exceeds the "
                             f"transformation plateau "
                             f"({mat.eps_tr * 100:.1f}%); plastic deformation "
                             f"is likely."))
    if s.t_element < mat.af_c:
        out.append(("alarm", f"Element at {s.t_element:.1f} °C is below Af "
                             f"({mat.af_c:.0f} °C). Strain recovery is not "
                             f"guaranteed."))
    if cfg.target_c < mat.af_c + 5:
        out.append(("warn", "Target is close to the austenite finish "
                            "temperature; superelastic behaviour degrades."))
    if cfg.form == "tube" and cfg.id_mm >= cfg.od_mm - 0.2:
        out.append(("warn", "Wall thickness is below 0.1 mm."))
    if s.cycle > mat.fatigue_cycles:
        out.append(("warn", f"Cycle count is past the indicative fatigue "
                            f"life ({mat.fatigue_cycles:,})."))
    return out


def search_recommendation(base_cfg, material_key, target_temp, ambient,
                          max_strain_pct=None, prefer_speed=True):
    """
    Rule-based expert search. Keeps the current rig geometry (elements,
    length, diameters, chamber, exchanger flow) and searches over strain,
    phase duration and exchanger type to find settings that can actually
    hold `target_temp`. Returns up to 3 (Config, envelope) pairs, best
    first, or [] if nothing in the search range works.
    """
    mat = MATERIALS[material_key]
    fracs = (0.5, 0.75, 1.0)
    strain_choices = sorted({round(mat.eps_tr * 100 * f, 1) for f in fracs})
    if max_strain_pct:
        filtered = [s for s in strain_choices if s <= max_strain_pct]
        strain_choices = filtered or strain_choices[:1]
    phase_choices = [0.5, 1.0, 2.0]

    candidates = []
    for strain in strain_choices:
        for phase in phase_choices:
            for hx in list(EXCHANGERS):
                c = replace(base_cfg, material_key=material_key,
                           exchanger_key=hx, strain_pct=strain,
                           phase_time_s=phase, ambient_c=ambient,
                           target_c=target_temp)
                e = predict_envelope(c)
                if e["reachable"]:
                    candidates.append((c, e))

    if not candidates:
        return []
    candidates.sort(key=lambda ce: ce[1]["time_to_target_s"] if prefer_speed
                    else -ce[1]["steady_cop"])
    return candidates[:3]


# =================================================================
# PART 2 — INTERFACE
# =================================================================

st.set_page_config(page_title="Elastocaloric rig control", page_icon="◆",
                   layout="wide", initial_sidebar_state="expanded")


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def get_theme(dark: bool) -> dict:
    if dark:
        return dict(INK="#E8EEF3", STEEL="#93A5B4", COLD="#35C3E8",
                    WARM="#F2A65A", GOOD="#4ADE9B", LINE="#2A3B47",
                    PAPER="#16222B", APP_BG="#0B141A")
    return dict(INK="#12232E", STEEL="#4A6572", COLD="#0E7C9B",
               WARM="#B45309", GOOD="#1F7A5A", LINE="#C7D2DB",
               PAPER="#FFFFFF", APP_BG="#E8EDF1")


# ---------------------------------------------------------------
# Factory defaults for every configuration widget. Used both to
# seed the app on first load and to power "Restore factory defaults".
# ---------------------------------------------------------------

DEFAULTS = {
    "w_theme": True,                 # False = light, True = dark — dark is
                                      # the default per spec; reset never
                                      # touches this key (see _THEME_KEY).
    "w_material": "niti",
    "w_form": "tube",
    "w_custom_area": 50.0,
    "w_custom_perimeter": 30.0,
    "w_n_elements": 5,
    "w_length": 150.0,
    "w_od": 12.0,
    "w_id": 10.0,
    "w_strain": round(MATERIALS["niti"].eps_tr * 100 * 0.9, 1),
    # Boundary-condition panel. "w_bc_disp_mm" defaults to whatever the
    # standard strain/length defaults above already imply, purely so the
    # two views agree with each other if a person switches straight to
    # User Defined without touching anything first.
    "w_bc_mode": "Fixed + Displacement",
    "w_bc_fixed_end": "Left",
    "w_bc_direction": "Tension (elongation)",
    "w_bc_disp_mm": round(
        150.0 * (MATERIALS["niti"].eps_tr * 100 * 0.9) / 100, 2),
    "w_compression": False,
    "w_phase_time": 10.0,
    "w_eta": 0.70,
    "w_hx": "finned_air",
    "w_flow": 50.0,
    "w_ambient": 25.0,
    "w_target": 10.0,
    "w_chamber_cap": 800.0,
    "w_insulation": 0.35,
    "w_regen": 0.0,
    "w_auto": True,
    "w_cycles_update": 4,
}

# If a factory-reset was requested on the previous run, apply it now,
# BEFORE any widget is instantiated, so every control snaps back.
# Display preference, not a rig setting — a reset should never change
# how the screen looks, only what it's simulating. Kept out of the
# factory-restore loop below on purpose.
_THEME_KEY = "w_theme"

if st.session_state.get("_do_restore"):
    for _k, _v in DEFAULTS.items():
        if _k == _THEME_KEY:
            continue
        st.session_state[_k] = _v
    # Also wipe anything the material expert or the combination-comparison
    # panel left behind (their result tables, picks, prior inputs), so a
    # factory reset genuinely clears the whole screen, not just the
    # sidebar sliders.
    for _k in list(st.session_state.keys()):
        if _k.startswith("ae_"):
            del st.session_state[_k]
    st.session_state["_do_restore"] = False
    st.session_state["_pending_full_reset"] = True

# Any code that wants to change a "w_*" sidebar setting from a button
# click (the expert agent, the comparison panel) must NOT write to
# st.session_state["w_..."] directly from inside that button's handler —
# by the time that handler runs, the sidebar has already instantiated a
# widget bound to that key for this script run, and Streamlit raises
# StreamlitWidgetAlreadyInstantiatedError if you try to overwrite it
# afterward. Instead those handlers set "_pending_apply" and call
# st.rerun(); this block picks it up and applies it here, before the
# sidebar widgets exist for the new run, which is the only point where
# it's safe to do.
if st.session_state.get("_pending_apply"):
    for _k, _v in st.session_state.pop("_pending_apply").items():
        st.session_state[_k] = _v

# Seed any missing keys (first load only — setdefault is a no-op after).
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)


def style_axes(ax, theme: dict):
    ax.set_facecolor(theme["PAPER"])
    ax.figure.patch.set_facecolor(theme["PAPER"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["LINE"])
    ax.tick_params(colors=theme["STEEL"], labelsize=8)
    ax.grid(True, color=theme["LINE"], linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(theme["STEEL"])
    ax.yaxis.label.set_color(theme["STEEL"])
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)


# --- sidebar: configuration ---------------------------------------
with st.sidebar:
    st.markdown("### Reset")
    if st.button("↺ Restore factory defaults", use_container_width=True,
                 type="primary",
                 help="Resets every rig setting — material, geometry, "
                      "loading, exchanger, thermal, control mode — and "
                      "clears the run. Your dark/light theme choice is a "
                      "display preference, not a rig setting, so it is "
                      "left untouched."):
        st.session_state["_do_restore"] = True
        st.rerun()
    st.caption(
        "Resets settings **and** the run. For clearing just the run data "
        "without touching your settings, use **Reset run data** in the "
        "control bar below the dashboard."
    )

    st.markdown("### Display")
    dark_mode = st.toggle("Dark theme", key="w_theme")

    st.markdown("### Rig configuration")
    mat_key = st.selectbox("Active material", list(MATERIALS),
                           format_func=lambda k: MATERIALS[k].name,
                           key="w_material")
    mat = MATERIALS[mat_key]
    form = st.radio("Element form", ["tube", "wire", "custom"],
                    horizontal=True, key="w_form",
                    help="Tube and wire compute cross-section and surface "
                         "area from diameters. Custom lets you enter a "
                         "measured or specified area directly, for "
                         "geometries that don't fit either preset.")
    n_elements = st.number_input("Elements in bundle", 1, 40, step=1,
                                 key="w_n_elements")
    length_mm = st.number_input("Active length (mm)", 20.0, 500.0, step=5.0,
                                key="w_length")
    if form == "custom":
        custom_area_mm2 = st.number_input(
            "Cross-section area, per element (mm²)", 0.5, 500.0, step=1.0,
            key="w_custom_area",
            help="The load-bearing cross-section of one element.")
        custom_perimeter_mm = st.number_input(
            "Wetted perimeter, per element (mm)", 0.5, 200.0, step=1.0,
            key="w_custom_perimeter",
            help="The perimeter exposed to the coolant, per element.")
        od_mm, id_mm = 12.0, 10.0   # unused in this branch; kept as a
                                    # stable default so Config always has
                                    # a valid value regardless of form.
    else:
        custom_area_mm2, custom_perimeter_mm = 50.0, 30.0
        od_mm = st.number_input("Outer diameter (mm)", 0.5, 30.0, step=0.5,
                                key="w_od")
        id_mm = st.number_input("Bore diameter (mm)", 0.1, 29.0, step=0.5,
                                disabled=(form == "wire"), key="w_id")

    st.markdown("### Boundary Conditions")
    st.caption(
        "One end of the NiTi element is always Fixed; the opposite end is "
        "always Displacement-controlled. This rule can't be broken from "
        "this control — picking a Fixed end automatically makes the "
        "other end the Displacement end."
    )
    bc_mode = st.selectbox(
        "Support configuration",
        ["Fixed + Displacement", "User Defined"], key="w_bc_mode",
        help="Fixed + Displacement: the standard setup — magnitude and "
             "direction follow the strain and loading-direction controls "
             "below. User Defined: specify the displacement end, "
             "direction and magnitude directly.")
    bc_fixed_end = st.radio(
        "Fixed end", ["Left", "Right"], horizontal=True, key="w_bc_fixed_end")
    bc_other_end = "Right" if bc_fixed_end == "Left" else "Left"
    st.caption(f"Fixed end: **{bc_fixed_end}**  ·  "
              f"Displacement end: **{bc_other_end}** (set automatically)")

    if bc_mode == "User Defined":
        bc_direction = st.radio(
            "Displacement direction",
            ["Tension (elongation)", "Compression (contraction)"],
            key="w_bc_direction")
        bc_disp_max_mm = round(length_mm * 0.10, 2)
        bc_disp_mm = st.number_input(
            "Displacement magnitude (mm)", 0.05, max(0.1, bc_disp_max_mm),
            step=0.05, key="w_bc_disp_mm",
            help=f"Bounded to 10% of the {length_mm:.0f} mm active length "
                 f"as a generic mechanical-stroke limit.")
        _bc_eff_strain_preview = round(bc_disp_mm / length_mm * 100, 2) \
            if length_mm > 0 else 0.0
        _mat_preview = MATERIALS[mat_key]
        st.caption(f"Equivalent strain: **{_bc_eff_strain_preview:.2f}%** "
                  f"of a {length_mm:.0f} mm element.")
        if _bc_eff_strain_preview > _mat_preview.eps_tr * 100 * 1.15:
            st.warning(
                f"This displacement implies {_bc_eff_strain_preview:.1f}% "
                f"strain, past {_mat_preview.name.split('(')[0].strip()}'s "
                f"{_mat_preview.eps_tr*100:.1f}% transformation plateau. "
                f"Reduce the displacement or expect a plastic-deformation "
                f"alarm below.")
    else:
        bc_direction = None
        bc_disp_mm = None
        st.caption(
            "Magnitude and direction are taken from **Applied strain** and "
            "**Compressive loading** in the Loading section below.")

    st.markdown("### Loading")
    strain_mn, strain_mx = 0.5, max(8.0, mat.eps_tr * 100 * 1.3)
    st.session_state["w_strain"] = min(max(
        st.session_state["w_strain"], strain_mn), strain_mx)
    _bc_overrides_loading = (bc_mode == "User Defined")
    strain_pct = st.slider(
        "Applied strain (%)", strain_mn, strain_mx, step=0.1, key="w_strain",
        disabled=_bc_overrides_loading,
        help=("Set by the Boundary Conditions panel above while User "
              "Defined is selected." if _bc_overrides_loading else
              f"Transformation plateau ends near {mat.eps_tr * 100:.1f}%."))
    compression = st.toggle("Compressive loading", key="w_compression",
                            disabled=_bc_overrides_loading)
    if _bc_overrides_loading:
        st.caption("↑ Controlled by Boundary Conditions above.")
    phase_time = st.slider("Phase duration (s)", 0.2, 60.0, step=0.1,
                           key="w_phase_time")
    eta_act = st.slider("Actuator efficiency", 0.3, 0.95, step=0.05,
                        key="w_eta")

    st.markdown("### Heat transfer")
    hx_key = st.selectbox("Exchanger", list(EXCHANGERS),
                          format_func=lambda k: EXCHANGERS[k].name,
                          key="w_hx")
    flow_cfm = st.slider("Coolant flow (CFM equivalent)", 10.0, 150.0,
                         step=5.0, key="w_flow")

    st.markdown("### Enclosure (cold box)")
    ambient_c = st.number_input("Ambient (°C)", 5.0, 45.0, step=1.0,
                                key="w_ambient")
    target_c = st.number_input("Target (°C)", -40.0, 40.0, step=1.0,
                               key="w_target")
    chamber_cap = st.number_input("Chamber heat capacity (J/K)",
                                  50.0, 20000.0, step=50.0,
                                  key="w_chamber_cap",
                                  help="Air, payload and inner wall combined.")
    insulation_ua = st.number_input("Cabinet loss (W/K)", 0.05, 5.0,
                                    step=0.05, key="w_insulation")

    st.markdown("### Advanced")
    regen_eff = st.slider(
        "Regenerator effectiveness", 0.0, 0.8, step=0.05, key="w_regen",
        help="0 = no regenerator (a single element, exactly as before). "
             "Recycling heat between the hot and cold halves of the cycle "
             "lets the machine pump across a span far larger than one "
             "element's own adiabatic swing — this is the single biggest "
             "lever for reaching colder targets or raising COP, more "
             "effective than strain, exchanger choice or phase timing "
             "alone.")
    if regen_eff > 0:
        st.caption(f"Effective temperature swing is being multiplied "
                  f"×{1/(1-min(regen_eff,0.8)):.2f} by the regenerator.")

    st.markdown("### Control")
    auto_mode = st.toggle("Automatic sequencing", key="w_auto")
    cycles_per_update = st.slider("Cycles per screen update", 1, 20,
                                  key="w_cycles_update")

theme = get_theme(dark_mode)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

/* Tell Streamlit's own components (labels, captions, sidebar text,
   buttons, widget option text) which colors to use. These CSS custom
   properties are what Streamlit's native widgets actually read from —
   setting color on html/body alone does not reach them, which is why
   text could go white-on-white after switching themes. */
:root, .stApp {{
    --text-color: {theme["INK"]};
    --background-color: {theme["APP_BG"]};
    --secondary-background-color: {theme["PAPER"]};
}}

html, body, [class*="css"] {{ font-family:'IBM Plex Sans',system-ui,sans-serif;
  color:{theme["INK"]}; }}
.stApp {{ background:{theme["APP_BG"]}; }}
.block-container {{ max-width:1480px; padding-top:1.2rem; }}

/* Explicit safety net: force every native Streamlit text element to
   the current theme's ink color, regardless of what the browser's own
   light/dark preference or a previous render left behind. */
section[data-testid="stSidebar"] {{
    background:{theme["APP_BG"]};
}}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] {{
    color:{theme["INK"]} !important;
}}
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p,
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p,
[data-testid="stRadio"] label p,
[data-testid="stSelectbox"] label p,
[data-testid="stNumberInput"] label p,
[data-testid="stSlider"] label p {{
    color:{theme["INK"]} !important;
}}
/* Radio / selectbox option text and dropdown menus */
div[role="radiogroup"] label span,
li[role="option"], div[data-baseweb="select"] * {{
    color:{theme["INK"]} !important;
}}
/* Secondary (non-primary) buttons should read in ink, not a stuck white */
.stButton button[kind="secondary"] p {{
    color:{theme["INK"]} !important;
}}
.titlebar {{ display:flex; align-items:baseline; gap:18px;
  border-bottom:2px solid {theme["INK"]}; padding-bottom:10px; margin-bottom:22px; }}
.titlebar h1 {{ font-size:24px; font-weight:600; color:{theme["INK"]}; margin:0; }}
.titlebar span {{ font-size:13px; color:{theme["STEEL"]}; }}
.verdict {{ background:{theme["PAPER"]}; border:1px solid {theme["LINE"]};
  border-left:5px solid {theme["COLD"]}; border-radius:4px; padding:20px 24px; }}
.verdict.blocked {{ border-left-color:{theme["WARM"]}; }}
.verdict h2 {{ font-size:20px; font-weight:600; color:{theme["INK"]}; margin:0 0 6px; }}
.verdict p {{ font-size:14px; color:{theme["STEEL"]}; margin:0; line-height:1.6;
  max-width:78ch; }}
.figure {{ font-family:'IBM Plex Mono',monospace; font-variant-numeric:tabular-nums;
  font-size:26px; font-weight:500; color:{theme["INK"]}; }}
.figure small {{ font-size:13px; color:{theme["STEEL"]}; font-weight:400; }}
.cap {{ font-size:12px; color:{theme["STEEL"]}; margin-bottom:2px; }}
.step {{ background:{theme["PAPER"]}; border:1px solid {theme["LINE"]}; border-radius:4px;
  padding:14px 16px; min-height:124px; }}
.step.active {{ border:1px solid {theme["WARM"]};
  background:{hex_to_rgba(theme["WARM"], 0.08)}; }}
.step.done {{ border-left:4px solid {theme["GOOD"]}; }}
.step .n {{ font-family:'IBM Plex Mono',monospace; font-size:12px; color:{theme["STEEL"]}; }}
.step .t {{ font-size:15px; font-weight:600; color:{theme["INK"]}; margin-top:6px; }}
.step .d {{ font-size:12px; color:{theme["STEEL"]}; margin-top:6px; line-height:1.5; }}
.chip {{ display:inline-block; padding:3px 10px; border-radius:3px;
  font-size:11px; font-weight:600; }}
.chip.run {{ background:{hex_to_rgba(theme["WARM"], 0.16)}; color:{theme["WARM"]}; }}
.chip.hold {{ background:{hex_to_rgba(theme["GOOD"], 0.16)}; color:{theme["GOOD"]}; }}
.chip.idle {{ background:{hex_to_rgba(theme["STEEL"], 0.16)}; color:{theme["STEEL"]}; }}
.extreme {{ background:{theme["PAPER"]}; border:1px solid {theme["LINE"]};
  border-radius:4px; padding:10px 14px; }}
footer, #MainMenu {{ visibility:hidden; }}

/* -----------------------------------------------------------------
   THEME ENFORCEMENT
   Streamlit renders some elements (dropdown menus, multiselect popovers,
   tags) in a portal attached to the page body rather than inside this
   app's own container, and Streamlit has its own separate light/dark
   setting that can silently disagree with this toggle. Both are forced
   below with !important so text can never end up matching its own
   background regardless of what the browser or a prior render left
   behind. Rules here are written to be no more specific than a single
   class + universal selector, so the app's own semantic colors above
   (.figure, .chip, .verdict, .step) — being more specific selectors —
   still win where they're meant to.
   ------------------------------------------------------------------ */
.stApp, .stApp * {{ color:{theme["INK"]} !important; }}
/* These are more specific than ".stApp *" above (two classes beats one
   class + universal), so they win automatically without needing to
   "undo" the blanket rule first. */
.figure {{ color:{theme["INK"]} !important; }}
.figure small {{ color:{theme["STEEL"]} !important; }}
.cap {{ color:{theme["STEEL"]} !important; }}
.verdict h2 {{ color:{theme["INK"]} !important; }}
.verdict p {{ color:{theme["STEEL"]} !important; }}
.step .n {{ color:{theme["STEEL"]} !important; }}
.step .t {{ color:{theme["INK"]} !important; }}
.step .d {{ color:{theme["STEEL"]} !important; }}
.chip.run {{ color:{theme["WARM"]} !important; }}
.chip.hold {{ color:{theme["GOOD"]} !important; }}
.chip.idle {{ color:{theme["STEEL"]} !important; }}

section[data-testid="stSidebar"] {{ background:{theme["APP_BG"]} !important; }}

/* Dropdown / multiselect popovers are portalled to <body>, outside
   .stApp, so they need their own explicit background + text pairing. */
div[data-baseweb="popover"],
div[data-baseweb="popover"] ul,
ul[role="listbox"] {{
    background:{theme["PAPER"]} !important;
}}
div[data-baseweb="popover"] *,
li[role="option"],
div[data-baseweb="tag"] {{
    color:{theme["INK"]} !important;
}}
li[role="option"]:hover,
li[aria-selected="true"] {{
    background:{hex_to_rgba(theme["COLD"], 0.14)} !important;
}}
div[data-baseweb="tag"] {{
    background:{hex_to_rgba(theme["COLD"], 0.18)} !important;
}}

/* Inputs, selects and their placeholder/value text */
input, textarea,
div[data-baseweb="select"] > div {{
    background:{theme["PAPER"]} !important;
    color:{theme["INK"]} !important;
}}
</style>
""", unsafe_allow_html=True)

# Resolve the two possible sources of strain/direction into the single
# pair the physics actually uses. This is a plain local-variable choice,
# made once per script run — it does not write back into either widget's
# session_state key, so it can't trigger the
# StreamlitWidgetAlreadyInstantiatedError class of bug and needs no
# st.rerun() to take effect. Changing Support configuration therefore
# updates the simulation on the very next run, live, without any reset.
if bc_mode == "User Defined":
    effective_strain_pct = round(bc_disp_mm / length_mm * 100, 3) \
        if length_mm > 0 else strain_pct
    effective_compression = (bc_direction == "Compression (contraction)")
else:
    effective_strain_pct = strain_pct
    effective_compression = compression

cfg = Config(material_key=mat_key, exchanger_key=hx_key, form=form,
             n_elements=int(n_elements), length_mm=length_mm, od_mm=od_mm,
             id_mm=id_mm, strain_pct=effective_strain_pct,
             mode_compression=effective_compression,
             phase_time_s=phase_time, flow_cfm=flow_cfm, ambient_c=ambient_c,
             target_c=target_c, chamber_capacity=chamber_cap,
             insulation_ua=insulation_ua, actuator_efficiency=eta_act,
             regen_effectiveness=regen_eff,
             custom_area_mm2=custom_area_mm2,
             custom_perimeter_mm=custom_perimeter_mm)

# --- run-time session state (separate from the widget/config state) ----
MAX_POINTS = 2400

# A person's saved snapshots are their own archive — not a rig setting
# (untouched by "Restore factory defaults") and not run telemetry
# (untouched by "Reset run data"). Only the explicit "Clear saved runs"
# button below empties it.
st.session_state.setdefault("saved_runs", [])
st.session_state.setdefault("has_downloaded", False)


def blank_log():
    return {k: [] for k in ("t", "chamber", "element", "fluid", "strain",
                            "stress", "cop", "cycle", "lift")}


def event(msg: str):
    st.session_state.events.insert(
        0, f"{datetime.now():%H:%M:%S}  {msg}")
    del st.session_state.events[40:]


def record():
    s: State = st.session_state.state
    log = st.session_state.log
    for key, val in (("t", s.t), ("chamber", s.t_chamber),
                     ("element", s.t_element), ("fluid", s.t_fluid),
                     ("strain", s.strain_pct), ("stress", s.stress_mpa),
                     ("cop", s.cop), ("cycle", s.cycle),
                     ("lift", s.q_cold_rate)):
        log[key].append(val)
    if len(log["t"]) > MAX_POINTS:
        for key in log:
            del log[key][0]

    ex = st.session_state.extremes
    ex["chamber_max"] = max(ex["chamber_max"], s.t_chamber)
    ex["chamber_min"] = min(ex["chamber_min"], s.t_chamber)
    ex["element_max"] = max(ex["element_max"], s.t_element)
    ex["element_min"] = min(ex["element_min"], s.t_element)


def reset(reason: str = "Controller reset."):
    """Resets the RUN only — cycle count, history, energy tally,
    max/min tracking. Does not touch the sidebar settings."""
    st.session_state.state = initial_state(cfg)
    st.session_state.running = False
    st.session_state.log = blank_log()
    st.session_state.events = []
    st.session_state.energy = [0.0, 0.0]        # [Q_cold J, W_in J]
    st.session_state.extremes = {
        "chamber_max": cfg.ambient_c, "chamber_min": cfg.ambient_c,
        "element_max": cfg.ambient_c, "element_min": cfg.ambient_c,
    }
    record()
    event(reason)


need_reset = "state" not in st.session_state
full_restore_pending = st.session_state.pop("_pending_full_reset", False)
if need_reset or full_restore_pending:
    reset("Factory defaults restored — all settings and run data cleared."
          if full_restore_pending else "Controller initialised.")

state: State = st.session_state.state


def advance_phase():
    """Run the next phase. Always reads the live object out of session
    state — a stale local reference would silently skip updates."""
    cur: State = st.session_state.state
    nxt = cur.phase % 4 + 1
    st.session_state.state = step_phase(cfg, cur, nxt)
    s = st.session_state.state
    if nxt == 4:
        st.session_state.energy[0] += s.q_cold_rate * cfg.cycle_time
        st.session_state.energy[1] += cfg.work_per_cycle
    record()
    return s


# --- capability analysis -----------------------------------------
@st.cache_data(show_spinner=False)
def envelope(cfg_key: tuple):
    return predict_envelope(Config(*cfg_key))


env = envelope(tuple(getattr(cfg, f.name) for f in fields(cfg)))

# Computed here (not later, where they were previously) so the AI
# Engineering Assistant glossary below — which explains phase duration
# using these exact numbers — has them available.
tau = cfg.capacity / cfg.ua_hx if cfg.ua_hx > 0 else float("inf")
effectiveness = 1.0 - math.exp(-cfg.phase_time_s / tau) if tau > 0 else 0.0

# Also computed here, early, for the same reason as tau/effectiveness
# above — the AI Engineering Assistant glossary and the active-config
# summary both reference this before the old "live readouts" section
# (which used to be the only place it was computed) has run.
q_tot, w_tot = st.session_state.energy
avg_cop = q_tot / w_tot if w_tot > 0 else 0.0

st.markdown(f"""
<div class="titlebar"><h1>Elastocaloric refrigeration rig</h1>
<span>{mat.name} &nbsp;·&nbsp; {EXCHANGERS[hx_key].name}
&nbsp;·&nbsp; {cfg.n_elements} × {cfg.length_mm:.0f} mm</span></div>
""", unsafe_allow_html=True)


def render_boundary_diagram(theme_d, fixed_end, is_compression,
                            magnitude_mm, magnitude_pct, mode_label):
    """
    Builds the Fixed-end / Displacement-end diagram. Which side carries
    which label is decided purely by `fixed_end` — there is no code path
    that can mark both ends, or neither, as Fixed: the function always
    derives exactly one Displacement end as "whichever end is not
    Fixed", so the invalid states the spec rules out (both Fixed, both
    Displacement, neither valid) are structurally unreachable here.
    """
    rod_color = theme_d["WARM"] if is_compression else theme_d["COLD"]
    verb = "shortens" if is_compression else "elongates"
    sign = "−" if is_compression else "+"
    disp_label = (f"DISPLACEMENT<br><span style='font-weight:400;"
                  f"font-size:11px'>{sign}{magnitude_mm:.2f} mm "
                  f"({sign}{magnitude_pct:.2f}% strain) · element "
                  f"{verb}</span>")

    wall = (f'<div style="display:flex;flex-direction:column;'
           f'align-items:center;gap:6px;min-width:92px">'
           f'<div style="font-size:11px;font-weight:700;'
           f'color:{theme_d["INK"]};letter-spacing:0.5px">FIXED</div>'
           f'<div style="width:14px;height:54px;'
           f'background:repeating-linear-gradient(135deg,'
           f'{theme_d["INK"]},{theme_d["INK"]} 3px,transparent 3px,'
           f'transparent 8px);border-radius:2px"></div></div>')

    arrow_char = "&#8592;" if (fixed_end == "Right") != is_compression \
        else "&#8594;"
    disp_end = (f'<div style="display:flex;flex-direction:column;'
               f'align-items:center;gap:6px;min-width:150px">'
               f'<div style="font-size:11px;font-weight:700;'
               f'color:{rod_color};letter-spacing:0.5px;text-align:center">'
               f'{disp_label}</div>'
               f'<div style="font-size:26px;color:{rod_color};'
               f'line-height:1">{arrow_char}</div></div>')

    rod = (f'<div style="flex:1;height:16px;border-radius:8px;'
          f'background:linear-gradient(90deg,{rod_color}66,{rod_color});'
          f'margin:0 4px"></div>')

    parts = [wall, rod, disp_end] if fixed_end == "Left" \
        else [disp_end, rod, wall]

    return (f'<div class="verdict" style="display:flex;align-items:center;'
           f'justify-content:center;gap:10px;padding:22px 24px">'
           f'{"".join(parts)}</div>'
           f'<div style="text-align:center;font-size:12px;'
           f'color:{theme_d["STEEL"]};margin-top:8px">'
           f'Support configuration: <b>{mode_label}</b></div>')


st.markdown(render_boundary_diagram(
    theme, bc_fixed_end, effective_compression,
    cfg.strain * cfg.length_mm, effective_strain_pct, bc_mode),
    unsafe_allow_html=True)

# ---------------------------------------------------------------------
# ACTIVE CONFIGURATION — the values actually in effect right now, as
# opposed to whatever a sidebar widget happens to show. In this app's
# architecture there's no separate "staged, not yet applied" state:
# every sidebar control takes effect on the very next script run, so
# these numbers ARE what's live. Shown here so the main screen states
# them plainly rather than requiring a trip to the sidebar to confirm.
# ---------------------------------------------------------------------
_direction_label = "Compression" if effective_compression else "Tension"
_form_label = {"tube": "Tube", "wire": "Wire", "custom": "Custom"}.get(
    cfg.form, cfg.form.title())
active_rows = [
    ("Element", f"{_form_label} · {cfg.n_elements} × {cfg.length_mm:.0f} mm"),
    ("Phase duration", f"{cfg.phase_time_s:.1f} s"),
    ("Displacement", f"{cfg.strain * cfg.length_mm:.2f} mm ({_direction_label})"),
    ("Target temperature", f"{cfg.target_c:.1f} °C"),
    ("Ambient temperature", f"{cfg.ambient_c:.1f} °C"),
]
st.markdown(
    f'<div style="background:{theme["PAPER"]};border:1px solid '
    f'{theme["LINE"]};border-radius:6px;padding:10px 18px;margin-top:8px;'
    f'display:flex;flex-wrap:wrap;gap:22px;font-size:12.5px;'
    f'color:{theme["STEEL"]}">' +
    "".join(
        f'<span><b style="color:{theme["INK"]}">{label}:</b> {value}</span>'
        for label, value in active_rows
    ) + '</div>', unsafe_allow_html=True)
st.caption("↑ Currently active — set in the sidebar, applied immediately.")
st.write("")

with st.expander("Quick guide", expanded=False):
    st.markdown(
        "- The rig runs a **four-phase cycle**: load → reject heat → "
        "unload → absorb heat. One full pass of all four is one cycle.\n"
        "- **Start sequencing** runs it automatically; **Step one phase** "
        "(shown when Automatic sequencing is off) lets you advance one "
        "phase at a time.\n"
        "- **Run to steady state** solves forward instantly without "
        "redrawing every step — use it when you want the answer, not the "
        "animation.\n"
        "- **Reset run data** clears the current run but keeps your "
        "settings. **Restore factory defaults** (top of the sidebar) "
        "clears everything, including material and geometry — use it if "
        "the rig ends up in a state you don't understand.\n"
        "- The **Enclosure & wire extremes** panel below tracks the "
        "highest and lowest temperature reached since the last reset."
    )

with st.expander("🤖 AI Engineering Assistant", expanded=False):
    st.caption(
        "Rule-based — reads this app's own physics engine and live state "
        "directly, so it can't invent a wrong number, but it also can't "
        "hold a free-form conversation or read an uploaded file/image "
        "(that needs a connected AI model, which this deployment doesn't "
        "have configured)."
    )

    st.markdown("#### Beginner mode — what does this do?")
    GLOSSARY = {
        "Phase duration": lambda: (
            f"How long each of the four steps (load, reject heat, unload, "
            f"absorb heat) runs before moving to the next. Right now it's "
            f"**{cfg.phase_time_s:.1f} s**. Your element's own thermal time "
            f"constant is **{tau:.1f} s**, so at the current setting only "
            f"about **{effectiveness*100:.0f}%** of the available "
            f"temperature swing is transferred each phase — shorter phases "
            f"run more cycles per minute but move less heat each time; "
            f"longer phases move more heat per cycle but pull down slower "
            f"in wall-clock time."),
        "Applied strain / displacement": lambda: (
            f"How far the element is mechanically stretched or compressed. "
            f"Right now that's **{effective_strain_pct:.2f}%** strain, "
            f"which is **{cfg.strain * cfg.length_mm:.2f} mm** of "
            f"displacement on a {cfg.length_mm:.0f} mm element. This "
            f"directly sets the adiabatic temperature swing "
            f"(**{env['dt_adiabatic']:.1f} K** at present) — more strain "
            f"means more cooling per cycle, up to the material's "
            f"transformation plateau at {mat.eps_tr*100:.1f}%; beyond "
            f"that you get no extra cooling, only extra stress."),
        "Fixed / Displacement ends": lambda: (
            f"One end of the element must stay still (**Fixed: "
            f"{bc_fixed_end}**) so the mechanical stroke has something to "
            f"push or pull against; the opposite end "
            f"(**Displacement: {bc_other_end}**) is the one actually "
            f"moved to strain the material. Without a fixed reference "
            f"end, 'displacement' wouldn't mean anything — both ends "
            f"would just move together."),
        "Target temperature": lambda: (
            f"The chamber temperature the controller is trying to reach — "
            f"currently **{cfg.target_c:.1f} °C**, against an ambient of "
            f"**{cfg.ambient_c:.1f} °C**. The verdict banner above tells "
            f"you directly whether this configuration can physically "
            f"reach it, and if not, by how much it falls short."),
        "Exchanger": lambda: (
            f"How heat actually moves between the element and its "
            f"surroundings — currently **{EXCHANGERS[hx_key].name}**, "
            f"giving a conductance of **{cfg.ua_hx:.1f} W/K**. Higher "
            f"conductance pulls down faster but also lets the element "
            f"return closer to ambient each cycle, which raises the "
            f"floor temperature it can ultimately sustain."),
        "Regenerator effectiveness": lambda: (
            f"Recycles heat between the hot and cold halves of the cycle "
            f"so the machine can pump across a span larger than one "
            f"element's own adiabatic swing. Currently set to "
            f"**{cfg.regen_effectiveness*100:.0f}%**"
            + (", i.e. off — a single element with no heat recycling."
               if cfg.regen_effectiveness == 0 else
               f", multiplying the effective swing by "
               f"×{cfg.regen_gain:.2f}.")),
        "COP (coefficient of performance)": lambda: (
            f"Cooling energy delivered divided by work put in. Right now "
            f"the run-average is **{avg_cop:.2f}** — below 1 is normal "
            f"for a single-stage, non-regenerative elastocaloric cycle "
            f"like this one; refrigerators you buy are usually 2-4 "
            f"because they're highly optimized multi-stage systems."),
        "Why a graph/number looks wrong": lambda: (
            "Most often this means the configuration genuinely can't do "
            "what's being asked of it — check the verdict banner near the "
            "top first. If chamber temperature is flat, either the target "
            "was already reached (see the green banner) or the machine "
            "hit its physical floor (see 'Sustainable minimum' in the key "
            "figures row)."),
    }
    glossary_pick = st.selectbox(
        "Pick a control or concept to have it explained using your "
        "current numbers:", list(GLOSSARY), key="ai_glossary_pick")
    st.info(GLOSSARY[glossary_pick]())

    st.divider()
    st.markdown("#### Recommend settings for a target")
    st.markdown(
        "Tell me the material and the temperature you're trying to reach. "
        "I'll search for settings that can actually get there, show you the "
        "values in plain terms, and ask before changing anything."
    )
    ae1, ae2, ae3 = st.columns(3)
    with ae1:
        ae_material = st.selectbox(
            "Material", list(MATERIALS),
            format_func=lambda k: MATERIALS[k].name, key="ae_material")
    with ae2:
        ae_target = st.number_input(
            "Temperature you want (°C)", -40.0, 40.0, 5.0, 1.0, key="ae_target")
    with ae3:
        ae_ambient = st.number_input(
            "Room temperature (°C)", 5.0, 45.0, 25.0, 1.0, key="ae_ambient")

    ae_priority = st.radio(
        "What matters more to you?",
        ["Reach it fast", "Best efficiency (COP)"],
        horizontal=True, key="ae_priority")

    if st.button("Get recommendation", key="ae_go"):
        st.session_state["ae_results"] = search_recommendation(
            cfg, ae_material, ae_target, ae_ambient,
            prefer_speed=(ae_priority == "Reach it fast"))
        st.session_state["ae_ran"] = True

    if st.session_state.get("ae_ran"):
        results = st.session_state.get("ae_results", [])
        if not results:
            st.error(
                f"With {MATERIALS[ae_material].name} and your current rig "
                f"size, I can't find settings that hold {ae_target:.1f} °C. "
                f"Try a warmer target, or a material with a bigger "
                f"transformation swing, such as NiTi."
            )
        else:
            best_c, best_e = results[0]
            mo = best_c.material
            st.success(f"Recommended setup — {mo.name}")
            st.markdown(
                f"- **Applied strain:** {best_c.strain_pct:.1f} %\n"
                f"- **Phase duration:** {best_c.phase_time_s:.1f} s\n"
                f"- **Exchanger:** {EXCHANGERS[best_c.exchanger_key].name}\n"
                f"- **Expected result:** reaches {best_c.target_c:.1f} °C in "
                f"about {best_e['cycles_to_target']:.0f} cycles "
                f"(~{best_e['time_to_target_s']/60:.1f} min), run-average COP "
                f"around {best_e['steady_cop']:.2f}.\n\n"
                f"Type these into the sidebar yourself, or let me set them "
                f"for you."
            )
            aa1, aa2 = st.columns(2)
            with aa1:
                if st.button("✅ Yes, apply automatically", key="ae_apply",
                             type="primary", use_container_width=True):
                    st.session_state["_pending_apply"] = {
                        "w_material": best_c.material_key,
                        "w_strain": best_c.strain_pct,
                        "w_phase_time": best_c.phase_time_s,
                        "w_hx": best_c.exchanger_key,
                        "w_ambient": best_c.ambient_c,
                        "w_target": best_c.target_c,
                    }
                    st.session_state["ae_ran"] = False
                    st.session_state["_pending_full_reset"] = True
                    event("Settings applied automatically by the material "
                          "expert.")
                    st.rerun()
            with aa2:
                if st.button("No, I'll set it myself", key="ae_skip",
                             use_container_width=True):
                    st.info("No problem — use the values listed above.")


if env["reachable"]:
    headline = f"This configuration can hold {cfg.target_c:.1f} °C."
    body = (f"Predicted pull-down takes {env['cycles_to_target']:.0f} cycles "
            f"({env['time_to_target_s']/60:.1f} min). The lowest temperature "
            f"the cold box can sustain is {env['t_min_c']:.1f} °C, so there is "
            f"{cfg.target_c - env['t_min_c']:.1f} K of margin against the "
            f"{cfg.insulation_ua:.2f} W/K cabinet load.")
else:
    headline = f"This configuration cannot reach {cfg.target_c:.1f} °C."
    body = (f"The cold box settles at {env['t_min_c']:.1f} °C, "
            f"{abs(cfg.target_c - env['t_min_c']):.1f} K short. The adiabatic "
            f"swing is only {env['dt_adiabatic']:.1f} K at "
            f"{cfg.strain_pct:.1f}% strain, and a single-stage cycle without "
            f"regeneration cannot pump much past it. Raise the strain toward "
            f"{mat.eps_tr*100:.1f}%, cut the cabinet load, or add a "
            f"regenerator or a second stage.")

st.markdown(f'<div class="verdict {"" if env["reachable"] else "blocked"}">'
            f'<h2>{headline}</h2><p>{body}</p></div>', unsafe_allow_html=True)

for col, cap, value, unit in zip(
        st.columns(5),
        ["Adiabatic swing ΔT<sub>ad</sub>", "Ideal single-stage floor",
         "Sustainable minimum", "Transfer effectiveness",
         "Steady cooling duty"],
        [f"{env['dt_adiabatic']:.1f}", f"{cfg.floor_c:.1f}",
         f"{env['t_min_c']:.1f}", f"{effectiveness*100:.0f}",
         f"{env['steady_lift_w']:.1f}"],
        ["K", "°C", "°C", f"% &nbsp;(τ = {tau:.1f} s)", "W"]):
    with col:
        st.markdown(f'<div class="cap">{cap}</div><div class="figure">'
                    f'{value} <small>{unit}</small></div>',
                    unsafe_allow_html=True)

if effectiveness < 0.35:
    st.info(f"Each phase lasts {cfg.phase_time_s:.1f} s against a thermal time "
            f"constant of {tau:.1f} s, so only {effectiveness*100:.0f}% of the "
            "available temperature swing is transferred before the phase ends. "
            "Longer phases or a higher-conductance exchanger raise the "
            "coefficient of performance, at the cost of pull-down speed.")

st.divider()

# --- controls -----------------------------------------------------
c1, c2, c3, c4, c5, c6 = st.columns([1.3, 1.3, 1.3, 1, 1, 1])
running = st.session_state.get("running", False)

with c1:
    if state.target_reached:
        st.button("Holding at target", disabled=True, use_container_width=True)
    elif running:
        if st.button("Stop", use_container_width=True):
            st.session_state.running = False
            event("Sequencing stopped by operator.")
            st.rerun()
    else:
        label = "Start sequencing" if auto_mode else "Step one phase"
        if st.button(label, type="primary", use_container_width=True):
            if auto_mode:
                st.session_state.running = True
                event("Automatic sequencing started.")
            else:
                s = advance_phase()
                event(f"Phase {s.phase} — {PHASE_NAMES[s.phase]}.")
            st.rerun()

with c2:
    if st.button("Run to steady state", use_container_width=True,
                 help="Solve forward until the chamber stops changing."):
        guard = 0
        while not st.session_state.state.target_reached and guard < 6000:
            before = st.session_state.state.t_chamber
            for _ in range(4):
                advance_phase()
            guard += 4
            if abs(before - st.session_state.state.t_chamber) < 5e-5:
                break
        st.session_state.running = False
        event(f"Batch solve finished after "
              f"{st.session_state.state.cycle} cycles.")
        st.rerun()

with c3:
    if st.button("Reset run data", use_container_width=True,
                 help="Clears the run history and cycle count but keeps "
                      "your material and rig settings."):
        reset("Run data reset by operator.")
        st.rerun()

with c4:
    st.markdown(f'<div class="cap">Cycles</div>'
                f'<div class="figure">{state.cycle}</div>',
                unsafe_allow_html=True)
with c5:
    st.markdown(f'<div class="cap">Gap to target</div><div class="figure">'
                f'{state.t_chamber - cfg.target_c:+.2f} <small>K</small></div>',
                unsafe_allow_html=True)
with c6:
    chip = ("hold", "Target hold") if state.target_reached else (
        ("run", "Sequencing") if running else ("idle", "Idle"))
    st.markdown(f'<div class="cap">Controller</div><div style="margin-top:8px">'
                f'<span class="chip {chip[0]}">{chip[1]}</span></div>',
                unsafe_allow_html=True)

# --- the four-step sequence ---------------------------------------
DESCRIPTIONS = {
    1: "Stress drives the austenite-to-martensite transformation. Latent heat "
       "appears as a temperature rise in the element.",
    2: "The hot element is swept by the coolant and returns toward ambient "
       "while the strain is held.",
    3: "The stress is released. The reverse transformation absorbs latent heat "
       "and the element goes below ambient.",
    4: "The cold element is coupled to the cold box and pulls heat out of it. "
       "This is the useful cooling.",
}

cols = st.columns(4)
for i in range(1, 5):
    cls = "active" if state.phase == i else (
        "done" if state.cycle > 0 or state.phase > i else "")
    with cols[i - 1]:
        st.markdown(f'<div class="step {cls}"><div class="n">{i}</div>'
                    f'<div class="t">{PHASE_NAMES[i]}</div>'
                    f'<div class="d">{DESCRIPTIONS[i]}</div></div>',
                    unsafe_allow_html=True)
st.write("")

# --- live readouts -------------------------------------------------
for col, (cap, value, unit) in zip(st.columns(6), [
        ("Cold box", f"{state.t_chamber:.2f}", "°C"),
        ("Element", f"{state.t_element:.2f}", "°C"),
        ("Coolant", f"{state.t_fluid:.2f}", "°C"),
        ("Stress", f"{state.stress_mpa:.0f}", "MPa"),
        ("Cooling duty", f"{state.q_cold_rate:.1f}", "W"),
        ("COP, run average", f"{avg_cop:.2f}", "")]):
    with col:
        st.markdown(f'<div class="cap">{cap}</div><div class="figure">'
                    f'{value} <small>{unit}</small></div>',
                    unsafe_allow_html=True)

for severity, message in check_interlocks(cfg, state):
    (st.error if severity == "alarm" else st.warning)(message)

st.write("")

# --- enclosure & wire (element) extremes ---------------------------
st.markdown("**Enclosure & wire extremes**  "
            "<span style='color:%s;font-size:12px'>(since last reset)</span>"
            % theme["STEEL"], unsafe_allow_html=True)

ex = st.session_state.extremes
for col, (cap, value) in zip(st.columns(4), [
        ("Enclosure — maximum", f"{ex['chamber_max']:.2f} °C"),
        ("Enclosure — minimum", f"{ex['chamber_min']:.2f} °C"),
        ("Wire / element — maximum", f"{ex['element_max']:.2f} °C"),
        ("Wire / element — minimum", f"{ex['element_min']:.2f} °C")]):
    with col:
        st.markdown(f'<div class="extreme"><div class="cap">{cap}</div>'
                    f'<div class="figure" style="font-size:20px">{value}</div>'
                    f'</div>', unsafe_allow_html=True)

st.write("")

# --- trends ---------------------------------------------------------
log = st.session_state.log
g1, g2 = st.columns([1.55, 1])

with g1:
    st.markdown("**Pull-down**")
    fig, ax = plt.subplots(figsize=(8.2, 3.5))
    style_axes(ax, theme)
    t = np.asarray(log["t"])
    ax.plot(t, log["element"], color=theme["WARM"], lw=1.0, alpha=0.75,
            label="Element")
    ax.plot(t, log["chamber"], color=theme["COLD"], lw=2.2, label="Cold box")
    ax.axhline(cfg.target_c, color=theme["GOOD"], ls="--", lw=1.2,
              label="Target")
    ax.axhline(cfg.ambient_c, color=theme["STEEL"], ls=":", lw=1.0,
              label="Ambient")
    ax.set_xlabel("Elapsed time (s)"); ax.set_ylabel("Temperature (°C)")
    leg = ax.legend(frameon=False, fontsize=8, ncols=4, loc="upper right")
    for text in leg.get_texts():
        text.set_color(theme["STEEL"])
    st.pyplot(fig, clear_figure=True); plt.close(fig)

with g2:
    st.markdown("**Stress–strain path**")
    fig2, ax2 = plt.subplots(figsize=(5.2, 3.5))
    style_axes(ax2, theme)
    ax2.plot(log["strain"][-160:], log["stress"][-160:],
             color=theme["INK"], lw=1.4, alpha=0.85)
    ax2.scatter([state.strain_pct], [state.stress_mpa],
                s=55, color=theme["WARM"], zorder=5)
    ax2.set_xlabel("Strain (%)"); ax2.set_ylabel("Stress (MPa)")
    st.pyplot(fig2, clear_figure=True); plt.close(fig2)

h1, h2 = st.columns([1.55, 1])
with h1:
    st.markdown("**Coefficient of performance per cycle**")
    fig3, ax3 = plt.subplots(figsize=(8.2, 2.4))
    style_axes(ax3, theme)
    cyc, cop = np.asarray(log["cycle"]), np.asarray(log["cop"])
    mask = cyc > 0
    ax3.plot(cyc[mask], cop[mask], color=theme["COLD"], lw=1.6)
    ax3.set_xlabel("Cycle"); ax3.set_ylabel("COP (–)")
    st.pyplot(fig3, clear_figure=True); plt.close(fig3)

with h2:
    st.markdown("**Controller log**")
    _log_lines = st.session_state.events or ["No events yet."]
    _log_html = "<br>".join(
        f'<span style="color:{theme["STEEL"]}">{ln.split("  ", 1)[0]}</span>'
        f'&nbsp;&nbsp;{ln.split("  ", 1)[1] if "  " in ln else ln}'
        if "  " in ln else ln
        for ln in _log_lines
    )
    st.markdown(
        f'<div style="background:{theme["PAPER"]};border:1px solid '
        f'{theme["LINE"]};border-radius:6px;padding:12px 14px;'
        f'max-height:220px;overflow-y:auto;font-family:\'IBM Plex Mono\','
        f'monospace;font-size:12px;line-height:1.7;color:{theme["INK"]}">'
        f'{_log_html}</div>',
        unsafe_allow_html=True)

# --- automatic sequencer: one screen update per N cycles ------------
if auto_mode and st.session_state.get("running") and not state.target_reached:
    for _ in range(cycles_per_update * 4):
        if advance_phase().target_reached:
            break
    s = st.session_state.state
    if s.target_reached:
        st.session_state.running = False
        event(f"Target {cfg.target_c:.1f} °C reached after {s.cycle} cycles. "
              f"Sequencing stopped.")
        st.toast(f"Target reached in {s.cycle} cycles.", icon="✅")
    st.rerun()

# --- record and export ----------------------------------------------
st.divider()
st.markdown("**Test record**")

frame = pd.DataFrame({
    "time_s": log["t"], "cycle": log["cycle"], "chamber_c": log["chamber"],
    "element_c": log["element"], "coolant_c": log["fluid"],
    "strain_pct": log["strain"], "stress_mpa": log["stress"],
    "cop": log["cop"], "cooling_w": log["lift"],
})


def build_report(rows_extra=None) -> str:
    rows = [
        ("Material", mat.name),
        ("Element form", f"{cfg.n_elements} × {cfg.form}, {cfg.length_mm:.0f} mm"),
        ("Cross-section", f"{cfg.section_area*1e6:.1f} mm²"),
        ("Active mass", f"{cfg.mass*1e3:.1f} g"),
        ("Heat-transfer area", f"{cfg.wetted_area*1e4:.0f} cm²"),
        ("Exchanger", EXCHANGERS[hx_key].name),
        ("Conductance UA", f"{cfg.ua_hx:.1f} W/K"),
        ("Thermal time constant", f"{tau:.2f} s"),
        ("Applied strain", f"{cfg.strain_pct:.2f} %"),
        ("Regenerator effectiveness", f"{cfg.regen_effectiveness*100:.0f} %"),
        ("Boundary condition", f"{bc_mode}"),
        ("Fixed end", f"{bc_fixed_end}"),
        ("Displacement end", f"{bc_other_end}"),
        ("Displacement direction",
         "Compression" if effective_compression else "Tension"),
        ("Displacement magnitude",
         f"{cfg.strain * cfg.length_mm:.2f} mm"),
        ("Phase duration", f"{cfg.phase_time_s:.2f} s"),
        ("Ambient", f"{cfg.ambient_c:.1f} °C"),
        ("Target", f"{cfg.target_c:.1f} °C"),
        ("Adiabatic ΔT", f"{env['dt_adiabatic']:.2f} K"),
        ("Sustainable minimum", f"{env['t_min_c']:.2f} °C"),
        ("Cycles completed", f"{state.cycle}"),
        ("Cold box now", f"{state.t_chamber:.2f} °C"),
        ("Enclosure max / min", f"{ex['chamber_max']:.2f} / {ex['chamber_min']:.2f} °C"),
        ("Wire max / min", f"{ex['element_max']:.2f} / {ex['element_min']:.2f} °C"),
        ("Run-average COP", f"{avg_cop:.3f}"),
        ("Cooling energy delivered", f"{q_tot/1000:.2f} kJ"),
        ("Work input", f"{w_tot/1000:.2f} kJ"),
    ]
    table = "\n".join(f"| {k} | {v} |" for k, v in rows)
    verdict = "reached" if state.target_reached else "not reached"
    return (f"# Elastocaloric rig test record\n\nGenerated "
            f"{datetime.now():%Y-%m-%d %H:%M}\n\n## Result\n\nTarget "
            f"{cfg.target_c:.1f} °C was {verdict} after {state.cycle} cycles. "
            f"{headline} {body}\n\n## Configuration and values\n\n"
            f"| Quantity | Value |\n|---|---|\n{table}\n\n## Basis\n\n"
            f"Two-capacity lumped model with Clausius-Clapeyron transformation "
            f"stress and an NTU-effectiveness exchanger. Values are simulated, "
            f"not measured.\n")


# ---------------------------------------------------------------------
# SAVE PANEL — a plain-language stage bar so it's obvious what to do
# and where you stand: simulate it, save it here, then download it to
# your own device (the only step that actually leaves a permanent copy
# on your computer — everything before that lives only in this browser
# tab and disappears if you close it).
# ---------------------------------------------------------------------
st.markdown("### 💾 Save your results")

stage1_done = state.cycle > 0 or len(log["t"]) > 1
stage2_done = len(st.session_state.saved_runs) > 0
stage3_done = st.session_state.has_downloaded

def _stage_class(done, active):
    if done:
        return "done"
    if active:
        return "active"
    return "pending"

STAGE_INFO = [
    ("1", "Run a simulation", "Click Start sequencing or Run to steady "
     "state above, on any target — reaching it isn't required."),
    ("2", "Save it here", "Give this run a name and click Save. It "
     "joins a list below, right on this screen."),
    ("3", "Download it", "Download the file for any saved run. This is "
     "the step that actually puts a copy on your device — the saved "
     "list disappears if you close this browser tab."),
]
stage_states = [
    stage1_done,
    stage1_done and stage2_done,
    stage1_done and stage2_done and stage3_done,
]
scols = st.columns(3)
for i, (num, title, desc) in enumerate(STAGE_INFO):
    active = (not stage_states[i]) and (i == 0 or stage_states[i - 1])
    cls = _stage_class(stage_states[i], active)
    with scols[i]:
        st.markdown(
            f'<div class="step {cls}"><div class="n">STAGE {num}</div>'
            f'<div class="t">{"✓ " if stage_states[i] else ""}{title}</div>'
            f'<div class="d">{desc}</div></div>',
            unsafe_allow_html=True)

st.write("")

if not stage1_done:
    st.info("Run at least one phase above, then come back here to save it.")
else:
    sv1, sv2 = st.columns([3, 1])
    with sv1:
        default_name = (
            f"{mat.name.split('(')[0].strip()} → {cfg.target_c:.0f}°C "
            f"({datetime.now():%H:%M})")
        save_name = st.text_input("Name this run", value=default_name,
                                  key="save_name_input")
    with sv2:
        st.write("")
        st.write("")
        do_save = st.button("💾 Save this run", type="primary",
                            use_container_width=True)

    if do_save:
        snapshot = {
            "name": save_name or default_name,
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "material": mat.name,
            "target_c": cfg.target_c,
            "reached": state.target_reached,
            "cycles": state.cycle,
            "chamber_c": state.t_chamber,
            "cop": avg_cop,
            "csv": frame.to_csv(index=False),
            "report": build_report(),
        }
        st.session_state.saved_runs.append(snapshot)
        event(f"Run saved as \"{snapshot['name']}\".")
        st.success(f"Saved as \"{snapshot['name']}\".")
        st.rerun()

    if st.session_state.saved_runs:
        st.markdown(f"**Your saved runs ({len(st.session_state.saved_runs)})**")
        for i, run in enumerate(reversed(st.session_state.saved_runs)):
            badge = ("✅ Reached target" if run["reached"]
                     else "⚠️ Did not reach target")
            with st.expander(f"{run['name']}  —  {badge}", expanded=False):
                rc1, rc2, rc3, rc4 = st.columns(4)
                rc1.markdown(f'<div class="cap">Material</div>'
                             f'<div class="figure" style="font-size:16px">'
                             f'{run["material"]}</div>', unsafe_allow_html=True)
                rc2.markdown(f'<div class="cap">Target</div>'
                             f'<div class="figure" style="font-size:16px">'
                             f'{run["target_c"]:.1f} °C</div>',
                             unsafe_allow_html=True)
                rc3.markdown(f'<div class="cap">Cycles</div>'
                             f'<div class="figure" style="font-size:16px">'
                             f'{run["cycles"]}</div>', unsafe_allow_html=True)
                rc4.markdown(f'<div class="cap">COP</div>'
                             f'<div class="figure" style="font-size:16px">'
                             f'{run["cop"]:.2f}</div>', unsafe_allow_html=True)
                st.caption(f"Saved {run['saved_at']}")

                dc1, dc2 = st.columns(2)
                orig_idx = len(st.session_state.saved_runs) - 1 - i
                with dc1:
                    if st.download_button(
                            "Download data (CSV)", run["csv"].encode(),
                            f"{run['name'].replace(' ', '_')}.csv",
                            "text/csv", key=f"dl_csv_{orig_idx}",
                            use_container_width=True):
                        st.session_state.has_downloaded = True
                with dc2:
                    if st.download_button(
                            "Download report (Markdown)",
                            run["report"].encode(),
                            f"{run['name'].replace(' ', '_')}.md",
                            "text/markdown", key=f"dl_md_{orig_idx}",
                            use_container_width=True):
                        st.session_state.has_downloaded = True

        if st.button("🗑 Clear all saved runs", key="clear_saved"):
            st.session_state.saved_runs = []
            event("Cleared all saved runs.")
            st.rerun()

st.write("")
st.caption("Quick download without saving a named copy first:")



e1, e2, e3 = st.columns(3)
with e1:
    if st.download_button("Download time series (CSV)",
                          frame.to_csv(index=False).encode(),
                          f"ecx_run_{datetime.now():%Y%m%d_%H%M}.csv",
                          "text/csv", use_container_width=True):
        st.session_state.has_downloaded = True
with e2:
    if st.download_button("Download test record (Markdown)",
                          build_report().encode(),
                          f"ecx_report_{datetime.now():%Y%m%d_%H%M}.md",
                          "text/markdown", use_container_width=True):
        st.session_state.has_downloaded = True
with e3:
    buf = io.StringIO(); frame.describe().to_csv(buf)
    if st.download_button("Download summary statistics (CSV)",
                          buf.getvalue().encode(), "ecx_summary.csv",
                          "text/csv", use_container_width=True):
        st.session_state.has_downloaded = True

with st.expander("Last 25 logged steps"):
    st.dataframe(frame.tail(25), use_container_width=True, hide_index=True)

st.caption("Simulated process values. Experimental validation requires measured "
           "transformation temperatures, the stress–strain hysteresis of the "
           "actual element, heat-transfer coefficients, thermal masses and "
           "actuator losses.")
