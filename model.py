"""
niti_elastocaloric.model
=========================
Lumped-parameter physics core for the NiTi elastocaloric refrigeration
demonstrator, plus the Config/Geometry data model and reporting helpers
that app.py builds its pages from.

PHYSICS (same idealized lumped-capacity model used throughout this
project): two thermal capacities — the element and the chamber — are
integrated in time through four phases per cycle (load, reject heat,
unload, absorb heat). The element's adiabatic temperature swing comes
from the elastocaloric effect (ΔT ≈ T·ΔS·ξ/Cp); heat exchange with the
air stream uses an NTU-style exponential relaxation; the chamber loses
heat to ambient through its insulation conductance.

This is a reference/teaching model — not a substitute for measured
material data or experimental validation (see the Configuration page's
"Assumptions and limitations" note in app.py).
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, fields, replace
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

__all__ = [
    "MATERIALS", "PHASES", "Phase", "Material", "Geometry", "Config",
    "State", "PhysicsEngine", "Simulation",
    "build_config", "config_frame", "geometry_table", "active_alarms",
    "run_simulation", "summarize_cycles", "run_parameter_sweep",
]

KELVIN = 273.15
CFM_TO_M3S = 4.719474e-4
AIR_RHO = 1.184
AIR_CP = 1005.0
NOMINAL_CFM = 50.0


class Phase(enum.Enum):
    LOAD = "LOAD"
    REJECT_HEAT = "REJECT HEAT"
    UNLOAD = "UNLOAD"
    ABSORB_HEAT = "ABSORB HEAT"


PHASES: List[Phase] = list(Phase)


# =================================================================
# Material & geometry
# =================================================================

@dataclass(frozen=True)
class Material:
    name: str
    density_kg_m3: float
    specific_heat_j_kgk: float
    transformation_entropy_j_kgk: float
    transformation_strain: float          # fraction, e.g. 0.055 = 5.5%
    stress_ref_mpa: float
    stress_ref_c: float
    clausius_clapeyron_mpa_k: float
    hysteresis_mpa: float
    stress_limit_mpa: float
    af_c: float
    fatigue_cycles: int

    def plateau_stress_mpa(self, element_c: float) -> float:
        return self.stress_ref_mpa + self.clausius_clapeyron_mpa_k * (
            element_c - self.stress_ref_c)

    def adiabatic_dt(self, element_c: float, strain: float) -> float:
        if self.transformation_strain <= 0:
            return 0.0
        xi = min(1.0, max(0.0, strain / self.transformation_strain))
        return (element_c + KELVIN) * self.transformation_entropy_j_kgk * xi \
            / self.specific_heat_j_kgk


MATERIALS: Dict[str, Material] = {
    "NiTi": Material("NiTi", 6450, 470, 40.0, 0.055, 420, 25.0, 6.5, 160,
                     800, -5.0, 100_000),
    "Cu-Al-Ni": Material("Cu-Al-Ni", 7100, 400, 22.0, 0.045, 180, 25.0, 2.2,
                         45, 350, 5.0, 20_000),
    "Fe-Mn-Si": Material("Fe-Mn-Si", 7200, 520, 12.0, 0.030, 280, 25.0, 1.6,
                         140, 600, 20.0, 500_000),
    "Natural Rubber": Material("Natural Rubber", 950, 1900, 28.0, 3.00, 3.0,
                               25.0, 0.02, 1.2, 18, -60.0, 1_000_000),
}


@dataclass(frozen=True)
class Geometry:
    form: str          # "Tube" or "Wire"
    n_elements: int
    od_mm: float
    id_mm: float
    length_mm: float

    @property
    def _bore_mm(self) -> float:
        return min(self.id_mm, max(self.od_mm - 0.05, 0.0)) \
            if self.form == "Tube" else 0.0

    @property
    def cross_section_area_m2(self) -> float:
        od = self.od_mm * 1e-3
        if self.form == "Tube":
            idm = self._bore_mm * 1e-3
            a = math.pi / 4.0 * (od ** 2 - idm ** 2)
        else:
            a = math.pi / 4.0 * od ** 2
        return max(a, 0.0) * self.n_elements

    @property
    def wetted_area_m2(self) -> float:
        od = self.od_mm * 1e-3
        if self.form == "Tube":
            idm = self._bore_mm * 1e-3
            per = math.pi * (od + idm)
        else:
            per = math.pi * od
        return per * self.length_mm * 1e-3 * self.n_elements

    @property
    def volume_m3(self) -> float:
        return self.cross_section_area_m2 * self.length_mm * 1e-3

    @property
    def wall_thickness_mm(self) -> float:
        if self.form != "Tube":
            return self.od_mm / 2.0
        return max(0.0, (self.od_mm - self.id_mm) / 2.0)


# =================================================================
# Config
# =================================================================

@dataclass(frozen=True)
class Config:
    material_key: str = "NiTi"
    form: str = "Tube"
    n_elements: int = 5
    od_mm: float = 12.0
    id_mm: float = 10.0
    length_mm: float = 150.0
    max_strain: float = 0.05              # fraction
    phase_time_s: float = 5.0
    ambient_c: float = 25.0
    target_c: float = 10.0
    chamber_capacity_j_k: float = 800.0
    insulation_ua_w_k: float = 0.35
    exchanger: str = "Finned forced air"
    h_w_m2k: float = 620.0
    air_flow_cfm: float = 50.0
    n_cycles: int = 40
    dt_s: float = 0.5

    @property
    def material(self) -> Material:
        return MATERIALS[self.material_key]

    @property
    def geometry(self) -> Geometry:
        return Geometry(self.form, self.n_elements, self.od_mm, self.id_mm,
                        self.length_mm)

    @property
    def mass_kg(self) -> float:
        return self.geometry.volume_m3 * self.material.density_kg_m3

    @property
    def thermal_capacity_j_k(self) -> float:
        return max(self.mass_kg * self.material.specific_heat_j_kgk, 1e-9)

    @property
    def fixed_target_is_valid(self) -> bool:
        return self.target_c < self.ambient_c

    @property
    def safe_min_c(self) -> float:
        return self.material.af_c + 5.0

    @property
    def safe_max_c(self) -> float:
        return self.ambient_c + 40.0

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def build_config(**kwargs) -> Config:
    """Validates raw sidebar inputs and returns a Config, or raises
    ValueError with a message meant to be shown directly to the user."""
    material_key = kwargs["material_key"]
    if material_key not in MATERIALS:
        raise ValueError(f"Unknown material '{material_key}'.")
    form = kwargs["form"]
    if form not in ("Tube", "Wire"):
        raise ValueError(f"Unknown element form '{form}'.")
    od_mm, id_mm = float(kwargs["od_mm"]), float(kwargs["id_mm"])
    if od_mm <= 0:
        raise ValueError("Outer / wire diameter must be positive.")
    if form == "Tube" and id_mm >= od_mm:
        raise ValueError("Inner diameter must be smaller than outer diameter for a tube.")
    length_mm = float(kwargs["length_mm"])
    if length_mm <= 0:
        raise ValueError("Element length must be positive.")
    n_elements = int(kwargs["n_elements"])
    if n_elements < 1:
        raise ValueError("Number of elements must be at least 1.")
    max_strain = float(kwargs["max_strain"])
    if max_strain <= 0:
        raise ValueError("Maximum strain must be positive.")
    phase_time_s = float(kwargs["phase_time_s"])
    if phase_time_s <= 0:
        raise ValueError("Phase time must be positive.")
    chamber_capacity_j_k = float(kwargs["chamber_capacity_j_k"])
    if chamber_capacity_j_k <= 0:
        raise ValueError("Chamber thermal capacity must be positive.")
    insulation_ua_w_k = float(kwargs["insulation_ua_w_k"])
    if insulation_ua_w_k < 0:
        raise ValueError("Insulation UA cannot be negative.")
    h_w_m2k = float(kwargs["h_w_m2k"])
    if h_w_m2k <= 0:
        raise ValueError("Heat-transfer coefficient must be positive.")
    air_flow_cfm = float(kwargs["air_flow_cfm"])
    if air_flow_cfm <= 0:
        raise ValueError("Air flow must be positive.")
    n_cycles = int(kwargs["n_cycles"])
    if n_cycles < 1:
        raise ValueError("Number of cycles must be at least 1.")
    dt_s = float(kwargs["dt_s"])
    if dt_s <= 0:
        raise ValueError("Recording interval must be positive.")

    return Config(
        material_key=material_key, form=form, n_elements=n_elements,
        od_mm=od_mm, id_mm=id_mm, length_mm=length_mm,
        max_strain=max_strain, phase_time_s=phase_time_s,
        ambient_c=float(kwargs["ambient_c"]), target_c=float(kwargs["target_c"]),
        chamber_capacity_j_k=chamber_capacity_j_k,
        insulation_ua_w_k=insulation_ua_w_k, exchanger=kwargs["exchanger"],
        h_w_m2k=h_w_m2k, air_flow_cfm=air_flow_cfm, n_cycles=n_cycles,
        dt_s=dt_s,
    )


def config_frame(config: Config) -> pd.DataFrame:
    """Full calculated-configuration table for the Configuration page."""
    m = config.material
    rows = [
        ("Material", m.name),
        ("Element form", config.form),
        ("Number of elements", config.n_elements),
        ("Outer / wire diameter (mm)", config.od_mm),
        ("Inner diameter (mm)", config.id_mm if config.form == "Tube" else "—"),
        ("Length (mm)", config.length_mm),
        ("Maximum strain (%)", round(config.max_strain * 100, 3)),
        ("Transformation strain (%)", round(m.transformation_strain * 100, 3)),
        ("Phase time (s)", config.phase_time_s),
        ("Ambient (°C)", config.ambient_c),
        ("Target (°C)", config.target_c),
        ("Chamber thermal capacity (J/K)", config.chamber_capacity_j_k),
        ("Insulation UA (W/K)", config.insulation_ua_w_k),
        ("Exchanger", config.exchanger),
        ("Heat-transfer coefficient (W/m²K)", config.h_w_m2k),
        ("Air flow (CFM)", config.air_flow_cfm),
        ("Number of cycles", config.n_cycles),
        ("Recording interval (s)", config.dt_s),
        ("Adiabatic ΔT at ambient (K)",
         round(m.adiabatic_dt(config.ambient_c, config.max_strain), 3)),
        ("Element mass (g)", round(config.mass_kg * 1e3, 2)),
        ("Element thermal capacity (J/K)", round(config.thermal_capacity_j_k, 2)),
    ]
    return pd.DataFrame(rows, columns=["Parameter", "Value"])


def geometry_table(config: Config) -> pd.DataFrame:
    """Engineering-properties table for the Dashboard page."""
    g = config.geometry
    rows = [
        ("Form", g.form, ""),
        ("Elements", g.n_elements, "count"),
        ("Cross-section area", round(g.cross_section_area_m2 * 1e6, 2), "mm²"),
        ("Wetted (heat-transfer) area", round(g.wetted_area_m2 * 1e4, 2), "cm²"),
        ("Volume", round(g.volume_m3 * 1e6, 3), "cm³"),
        ("Mass", round(config.mass_kg * 1e3, 2), "g"),
        ("Thermal capacity", round(config.thermal_capacity_j_k, 2), "J/K"),
        ("Wall thickness", round(g.wall_thickness_mm, 3), "mm"),
    ]
    return pd.DataFrame(rows, columns=["Property", "Value", "Unit"])


# =================================================================
# State & physics engine
# =================================================================

@dataclass
class State:
    time_s: float = 0.0
    cycle: int = 0
    phase_index: int = 0
    chamber_c: float = 25.0
    element_c: float = 25.0
    air_in_c: float = 25.0
    air_out_c: float = 25.0
    strain_pct: float = 0.0
    stress_mpa: float = 0.0
    force_n: float = 0.0
    displacement_mm: float = 0.0
    cooling_w: float = 0.0
    cop: float = 0.0
    _cooling_energy_cycle_j: float = 0.0
    _work_energy_cycle_j: float = 0.0
    target_reached: bool = False
    target_time_s: Optional[float] = None
    target_cycle: Optional[int] = None
    target_phase: Optional[str] = None

    @property
    def phase(self) -> Phase:
        return PHASES[self.phase_index % len(PHASES)]


def _relax(t_hot: float, t_sink: float, ua: float, cap: float, dt: float) -> float:
    if ua <= 0 or cap <= 0:
        return t_hot
    return t_sink + (t_hot - t_sink) * math.exp(-ua * dt / cap)


class PhysicsEngine:
    """Advances a State through one phase per `.step()` call. Keeping one
    phase per call (rather than one whole cycle) is what lets app.py's
    START / SINGLE CYCLE / STEP PHASE controls all share this one method."""

    SUBSTEPS = 20

    def __init__(self, config: Config, state: Optional[State] = None):
        self.config = config
        self.state = state if state is not None else State(
            chamber_c=config.ambient_c, element_c=config.ambient_c,
            air_in_c=config.ambient_c, air_out_c=config.ambient_c)

    def _ua_active(self) -> float:
        cfg = self.config
        g = cfg.geometry
        scale = (max(cfg.air_flow_cfm, 1.0) / NOMINAL_CFM) ** 0.6
        ha = cfg.h_w_m2k * scale * g.wetted_area_m2
        mdot_cp = cfg.air_flow_cfm * CFM_TO_M3S * AIR_RHO * AIR_CP
        if mdot_cp <= 0:
            return 0.0
        return (1.0 - math.exp(-ha / mdot_cp)) * mdot_cp

    def _ua_idle(self) -> float:
        return 6.0 * self.config.geometry.wetted_area_m2

    def step(self) -> dict:
        cfg, s, mat = self.config, self.state, self.config.material
        phase = PHASES[s.phase_index % len(PHASES)]
        substeps = self.SUBSTEPS
        dt = cfg.phase_time_s / substeps
        cap = cfg.thermal_capacity_j_k

        if phase is Phase.LOAD:
            s.strain_pct = cfg.max_strain * 100
            s.element_c += mat.adiabatic_dt(s.element_c, cfg.max_strain)
            s.stress_mpa = mat.plateau_stress_mpa(s.element_c) + mat.hysteresis_mpa / 2
        elif phase is Phase.REJECT_HEAT:
            s.strain_pct = cfg.max_strain * 100
            s.stress_mpa = mat.plateau_stress_mpa(s.element_c) + mat.hysteresis_mpa / 2
        elif phase is Phase.UNLOAD:
            s.strain_pct = 0.0
            s.element_c -= mat.adiabatic_dt(s.element_c, cfg.max_strain)
            s.stress_mpa = max(0.0, mat.plateau_stress_mpa(s.element_c) - mat.hysteresis_mpa / 2)
        else:  # ABSORB_HEAT
            s.strain_pct = 0.0
            s.stress_mpa = 0.0

        ua_active, ua_idle = self._ua_active(), self._ua_idle()
        q_cold = 0.0
        for _ in range(substeps):
            if phase is Phase.REJECT_HEAT:
                s.element_c = _relax(s.element_c, cfg.ambient_c, ua_active, cap, dt)
                s.air_in_c = cfg.ambient_c
                s.air_out_c = cfg.ambient_c + 0.5 * (s.element_c - cfg.ambient_c)
            elif phase is Phase.ABSORB_HEAT:
                dq = max(0.0, ua_active * (s.chamber_c - s.element_c) * dt)
                q_cold += dq
                s.element_c += dq / cap
                s.chamber_c -= dq / cfg.chamber_capacity_j_k
                s.air_in_c = s.chamber_c
                s.air_out_c = s.chamber_c - 0.5 * (s.chamber_c - s.element_c)
            else:
                s.element_c = _relax(s.element_c, cfg.ambient_c, ua_idle, cap, dt)
                s.air_in_c = s.air_out_c = cfg.ambient_c
            s.chamber_c = _relax(s.chamber_c, cfg.ambient_c, cfg.insulation_ua_w_k,
                                 cfg.chamber_capacity_j_k, dt)

        s.time_s += cfg.phase_time_s

        strain = cfg.max_strain if phase in (Phase.LOAD, Phase.REJECT_HEAT) else 0.0
        s.force_n = abs(s.stress_mpa) * 1e6 * cfg.geometry.cross_section_area_m2
        s.displacement_mm = strain * cfg.length_mm

        s._cooling_energy_cycle_j += q_cold
        if phase is Phase.LOAD:
            eps_work = min(cfg.max_strain, mat.transformation_strain)
            s._work_energy_cycle_j += (mat.hysteresis_mpa * 1e6 * eps_work
                                       * cfg.geometry.volume_m3)
        s.cooling_w = q_cold / cfg.phase_time_s if cfg.phase_time_s > 0 else 0.0

        cycle_complete = phase is Phase.ABSORB_HEAT
        if cycle_complete:
            s.cop = (s._cooling_energy_cycle_j / s._work_energy_cycle_j
                    if s._work_energy_cycle_j > 0 else 0.0)

        if not s.target_reached and s.chamber_c <= cfg.target_c:
            s.target_reached = True
            s.target_time_s = s.time_s
            s.target_cycle = s.cycle
            s.target_phase = phase.value
            s.chamber_c = cfg.target_c

        record = {
            "time_s": s.time_s, "cycle": s.cycle, "phase": phase.value,
            "chamber_c": s.chamber_c, "element_c": s.element_c,
            "air_in_c": s.air_in_c, "air_out_c": s.air_out_c,
            "strain_pct": s.strain_pct, "stress_mpa": s.stress_mpa,
            "force_n": s.force_n, "displacement_mm": s.displacement_mm,
            "cooling_w": s.cooling_w, "cop": s.cop,
            "target_reached": s.target_reached,
        }

        s.phase_index += 1
        if cycle_complete:
            s.cycle += 1
            s._cooling_energy_cycle_j = 0.0
            s._work_energy_cycle_j = 0.0

        return record


@dataclass
class Simulation:
    config: Config
    readings: pd.DataFrame
    state: State
    cycle_summary: pd.DataFrame


def run_simulation(config: Config) -> Simulation:
    engine = PhysicsEngine(config)
    records = [engine.step() for _ in range(config.n_cycles * len(PHASES))]
    readings = pd.DataFrame(records)
    return Simulation(config=config, readings=readings, state=engine.state,
                      cycle_summary=summarize_cycles(readings))


def summarize_cycles(readings: pd.DataFrame) -> pd.DataFrame:
    """Per-cycle COP / chamber temperature / cooling energy, for the
    Cycle Analysis page."""
    if readings.empty:
        return pd.DataFrame(columns=["cycle", "cop", "chamber_c", "cooling_energy_j"])
    phase_dt = (float(readings["time_s"].iloc[1] - readings["time_s"].iloc[0])
               if len(readings) > 1 else float(readings["time_s"].iloc[0] or 1.0))
    rows = []
    for cycle, group in readings.groupby("cycle"):
        rows.append({
            "cycle": int(cycle),
            "cop": float(group["cop"].iloc[-1]),
            "chamber_c": float(group["chamber_c"].iloc[-1]),
            "cooling_energy_j": float(group["cooling_w"].clip(lower=0).sum() * phase_dt),
        })
    return pd.DataFrame(rows)


# =================================================================
# Alarms
# =================================================================

def active_alarms(config: Config, state: Optional[State]) -> List[Tuple[str, str]]:
    alarms: List[Tuple[str, str]] = []
    mat = config.material
    if state is not None:
        if abs(state.stress_mpa) > mat.stress_limit_mpa:
            alarms.append(("ALARM", f"Stress {abs(state.stress_mpa):.0f} MPa "
                          f"exceeds the {mat.stress_limit_mpa:.0f} MPa working limit."))
        if state.element_c < mat.af_c:
            alarms.append(("ALARM", f"Element at {state.element_c:.1f} °C is below "
                          f"Af ({mat.af_c:.0f} °C); strain recovery is not guaranteed."))
        if state.cycle > mat.fatigue_cycles:
            alarms.append(("WARNING", f"Cycle count is past the indicative fatigue "
                          f"life ({mat.fatigue_cycles:,})."))
    if config.max_strain > mat.transformation_strain * 1.15:
        alarms.append(("ALARM", f"Maximum strain {config.max_strain*100:.1f}% exceeds "
                      f"the {mat.transformation_strain*100:.1f}% transformation plateau."))
    if not config.fixed_target_is_valid:
        alarms.append(("WARNING", "Target temperature is not below ambient; "
                      "cooling mode cannot reach it."))
    if config.target_c < mat.af_c + 5:
        alarms.append(("WARNING", "Target is close to the austenite finish "
                      "temperature; superelastic behaviour degrades."))
    if config.form == "Tube" and config.geometry.wall_thickness_mm < 0.1:
        alarms.append(("WARNING", "Wall thickness is below 0.1 mm."))
    return alarms


# =================================================================
# Parameter sweeps
# =================================================================

_PARAMETER_FIELD = {
    "Material": "material_key",
    "Number of elements": "n_elements",
    "Strain (%)": "max_strain",
    "Air flow (CFM)": "air_flow_cfm",
    "Heat transfer coefficient": "h_w_m2k",
    "Phase time (s)": "phase_time_s",
    "Target (°C)": "target_c",
}


def _coerce_value(parameter: str, raw) -> object:
    if parameter == "Material":
        raw_str = str(raw).strip()
        for key in MATERIALS:
            if key.lower() == raw_str.lower():
                return key
        raise ValueError(f"Unknown material '{raw}'.")
    if parameter == "Number of elements":
        return int(float(raw))
    if parameter == "Strain (%)":
        return float(raw) / 100.0
    return float(raw)


def _probe(config: Config, max_cycles: int = 150) -> dict:
    """Head-less run used by the sweep: how does this candidate behave?"""
    engine = PhysicsEngine(config)
    min_chamber = config.ambient_c
    time_to_target = None
    last_cop = 0.0
    last_cooling_w = 0.0
    prev_chamber = config.ambient_c
    for i in range(max_cycles * len(PHASES)):
        record = engine.step()
        min_chamber = min(min_chamber, record["chamber_c"])
        if record["phase"] == Phase.ABSORB_HEAT.value:
            last_cop = record["cop"]
            last_cooling_w = record["cooling_w"]
        if record["target_reached"] and time_to_target is None:
            time_to_target = record["time_s"]
        if (i > 20 * len(PHASES) and i % len(PHASES) == 0
                and abs(record["chamber_c"] - prev_chamber) < 1e-4):
            break
        if i % len(PHASES) == 0:
            prev_chamber = record["chamber_c"]
    return {
        "COP": last_cop,
        "Cooling power (W)": last_cooling_w,
        "Minimum chamber °C": min_chamber,
        "Time to target (s)": time_to_target if time_to_target is not None else float("nan"),
    }


def run_parameter_sweep(config: Config, parameter: str, values: Sequence) -> pd.DataFrame:
    """Varies one field across `values` (strings or numbers), keeping
    everything else fixed, and reports COP / cooling power / achievable
    minimum / time-to-target for each candidate."""
    field_name = _PARAMETER_FIELD.get(parameter)
    if field_name is None:
        raise ValueError(f"Unknown sweep parameter '{parameter}'.")
    rows = []
    for raw in values:
        try:
            value = _coerce_value(parameter, raw)
            candidate = replace(config, **{field_name: value})
            result = _probe(candidate)
        except Exception as exc:
            result = {"COP": float("nan"), "Cooling power (W)": float("nan"),
                      "Minimum chamber °C": float("nan"),
                      "Time to target (s)": float("nan"), "Error": str(exc)}
        rows.append({parameter: raw, **result})
    return pd.DataFrame(rows)
