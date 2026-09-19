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
ARCHITECTURE NOTES (read this before touching the app)
=================================================================
This build adds three things on top of the original single-user demo,
without changing the physics in PART 1:

  1. MULTI-USER DATA ISOLATION
     Streamlit already gives every browser tab/session its own private
     `st.session_state` — two people on the same URL never share Python
     objects at runtime. What the *original* app was missing was
     PERSISTENCE: session_state disappears on refresh/close, and there
     was nowhere per-user to keep it. This build adds a small SQLite
     database (PART 1.5) with every table keyed by `user_id`, and every
     read/write goes through that key. That's what makes "own saved
     configuration / own files / own AI context" survive a refresh or a
     reopened tab, and what makes it correct with N concurrent users.

     IDENTITY: this demo authenticates with a simple, self-chosen user
     ID (a name/handle) carried in the URL's `?u=` query parameter —
     there is no password. That is intentionally lightweight so the
     app keeps working as a single shared URL with zero extra
     infrastructure. It is NOT meant to be a production auth system:
     anyone who knows another person's `u` value could open their
     session. If you deploy this somewhere that matters, put a real
     identity provider in front of it (e.g. `st.experimental_user`
     behind Streamlit Community Cloud SSO, or an external auth proxy)
     and pass the verified identity into `get_or_create_user_id()`
     instead of trusting the query string.

  2. SETTINGS / DASHBOARD SEPARATION + DRAFT VS ACTIVE
     All the previously-editable sidebar controls now live on the
     "User Defined Settings" tab and write into `st.session_state`
     keys prefixed `d_...` (draft only). Nothing there is live. The
     dashboard, calculations, graphs and simulation all read from
     `st.session_state["active"]`, a plain dict, which only changes
     when "Apply changes" validates the draft and commits it — at
     which point it is also written to SQLite for that user.

  3. THEME
     A single `theme(...)` dict of semantic colors feeds one CSS
     block (`inject_css`) that is intentionally broad — it targets
     Streamlit's actual DOM (data-testid selectors), not just this
     app's own custom divs, so widgets, alerts, tables, tabs, popovers
     and dynamically-created components all follow the same theme.
=================================================================
"""

from __future__ import annotations

import io
import json
import math
import sqlite3
import threading
import uuid
from dataclasses import dataclass, fields, replace
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.use("Agg")

# =================================================================
# PART 1 — PHYSICS CORE  (unchanged from the original app)
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
# PART 1.5 — PERSISTENCE LAYER  (SQLite, every table keyed by user_id)
# =================================================================
# This is the piece the original single-user demo did not have. It is
# a plain file-backed SQLite database so the app keeps working as one
# shared URL with no external service required; swap `_conn()` for a
# real managed database (Postgres, etc.) if you need true multi-writer
# concurrency at scale — the schema and call sites below don't change.

DB_PATH = "ecx_scada.db"
_DB_LOCK = threading.Lock()

CONFIG_FIELDS: List[str] = [f.name for f in fields(Config)]


def config_to_dict(cfg: Config) -> dict:
    return {k: getattr(cfg, k) for k in CONFIG_FIELDS}


def dict_to_config(d: dict) -> Config:
    return Config(**{k: d[k] for k in CONFIG_FIELDS if k in d})


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=15)
    c.execute("PRAGMA journal_mode=WAL")
    return c


def init_db() -> None:
    with _DB_LOCK, _conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS users(
            user_id TEXT PRIMARY KEY, created_at TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS configs(
            user_id TEXT PRIMARY KEY, cfg_json TEXT, updated_at TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS prefs(
            user_id TEXT PRIMARY KEY, dark INTEGER)""")
        c.execute("""CREATE TABLE IF NOT EXISTS saved_runs(
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT,
            name TEXT, payload_json TEXT, saved_at TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS user_files(
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT,
            filename TEXT, mime TEXT, content BLOB, note TEXT,
            uploaded_at TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS ai_context(
            user_id TEXT PRIMARY KEY, ctx_json TEXT, updated_at TEXT)""")


def ensure_user(user_id: str) -> None:
    with _DB_LOCK, _conn() as c:
        c.execute("INSERT OR IGNORE INTO users(user_id, created_at) VALUES (?,?)",
                 (user_id, datetime.now().isoformat()))


def load_active_config(user_id: str) -> Optional[dict]:
    with _DB_LOCK, _conn() as c:
        row = c.execute("SELECT cfg_json FROM configs WHERE user_id=?",
                        (user_id,)).fetchone()
    return json.loads(row[0]) if row else None


def save_active_config(user_id: str, cfg_dict: dict) -> None:
    with _DB_LOCK, _conn() as c:
        c.execute("""INSERT INTO configs(user_id, cfg_json, updated_at)
                     VALUES(?,?,?)
                     ON CONFLICT(user_id) DO UPDATE SET
                     cfg_json=excluded.cfg_json, updated_at=excluded.updated_at""",
                 (user_id, json.dumps(cfg_dict), datetime.now().isoformat()))


def delete_active_config(user_id: str) -> None:
    with _DB_LOCK, _conn() as c:
        c.execute("DELETE FROM configs WHERE user_id=?", (user_id,))


def load_pref_dark(user_id: str, default: bool = True) -> bool:
    with _DB_LOCK, _conn() as c:
        row = c.execute("SELECT dark FROM prefs WHERE user_id=?",
                        (user_id,)).fetchone()
    return bool(row[0]) if row else default


def save_pref_dark(user_id: str, dark: bool) -> None:
    with _DB_LOCK, _conn() as c:
        c.execute("""INSERT INTO prefs(user_id, dark) VALUES(?,?)
                     ON CONFLICT(user_id) DO UPDATE SET dark=excluded.dark""",
                 (user_id, int(dark)))


def list_saved_runs(user_id: str) -> List[dict]:
    with _DB_LOCK, _conn() as c:
        rows = c.execute("""SELECT id, name, payload_json, saved_at
                            FROM saved_runs WHERE user_id=?
                            ORDER BY id DESC""", (user_id,)).fetchall()
    out = []
    for rid, name, payload, saved_at in rows:
        d = json.loads(payload)
        d.update({"id": rid, "name": name, "saved_at": saved_at})
        out.append(d)
    return out


def add_saved_run(user_id: str, name: str, payload: dict) -> None:
    with _DB_LOCK, _conn() as c:
        c.execute("""INSERT INTO saved_runs(user_id, name, payload_json, saved_at)
                     VALUES (?,?,?,?)""",
                 (user_id, name, json.dumps(payload), datetime.now().isoformat()))


def delete_saved_run(user_id: str, run_id: int) -> None:
    with _DB_LOCK, _conn() as c:
        c.execute("DELETE FROM saved_runs WHERE user_id=? AND id=?",
                 (user_id, run_id))


def clear_saved_runs(user_id: str) -> None:
    with _DB_LOCK, _conn() as c:
        c.execute("DELETE FROM saved_runs WHERE user_id=?", (user_id,))


def list_user_files(user_id: str) -> List[Tuple]:
    with _DB_LOCK, _conn() as c:
        return c.execute("""SELECT id, filename, mime, note, uploaded_at,
                            length(content) FROM user_files
                            WHERE user_id=? ORDER BY id DESC""",
                         (user_id,)).fetchall()


def add_user_file(user_id: str, filename: str, mime: str, content: bytes,
                  note: str = "") -> None:
    with _DB_LOCK, _conn() as c:
        c.execute("""INSERT INTO user_files(user_id, filename, mime, content,
                     note, uploaded_at) VALUES (?,?,?,?,?,?)""",
                 (user_id, filename, mime, content, note,
                  datetime.now().isoformat()))


def get_user_file(user_id: str, file_id: int) -> Optional[Tuple]:
    with _DB_LOCK, _conn() as c:
        return c.execute("""SELECT filename, mime, content FROM user_files
                            WHERE user_id=? AND id=?""",
                         (user_id, file_id)).fetchone()


def delete_user_file(user_id: str, file_id: int) -> None:
    with _DB_LOCK, _conn() as c:
        c.execute("DELETE FROM user_files WHERE user_id=? AND id=?",
                 (user_id, file_id))


def load_ai_context(user_id: str) -> dict:
    with _DB_LOCK, _conn() as c:
        row = c.execute("SELECT ctx_json FROM ai_context WHERE user_id=?",
                        (user_id,)).fetchone()
    return json.loads(row[0]) if row else {}


def save_ai_context(user_id: str, ctx: dict) -> None:
    with _DB_LOCK, _conn() as c:
        c.execute("""INSERT INTO ai_context(user_id, ctx_json, updated_at)
                     VALUES(?,?,?)
                     ON CONFLICT(user_id) DO UPDATE SET
                     ctx_json=excluded.ctx_json, updated_at=excluded.updated_at""",
                 (user_id, json.dumps(ctx), datetime.now().isoformat()))


init_db()

# ---------------------------------------------------------------
# Factory / default ACTIVE configuration. This is the full superset of
# what "User Defined Settings" edits: every Config() field, plus a
# handful of UI-only extras (boundary-condition mode, auto-sequencing,
# free-text research notes) that aren't physics parameters but still
# belong to "the current user's configuration".
# ---------------------------------------------------------------
_default_material = MATERIALS["niti"]
_default_strain_pct = round(_default_material.eps_tr * 100 * 0.9, 1)
_default_length_mm = 150.0
_default_cfg = Config(strain_pct=_default_strain_pct, length_mm=_default_length_mm)

ACTIVE_DEFAULTS: dict = {
    **config_to_dict(_default_cfg),
    "bc_mode": "Fixed + Displacement",
    "bc_fixed_end": "Left",
    "bc_direction": "Tension (elongation)",
    "bc_disp_mm": round(_default_length_mm * _default_strain_pct / 100, 2),
    "auto_mode": True,
    "cycles_per_update": 4,
    "experimental_notes": "",
    "material_notes": "",
}


def full_active_dict(user_id: str) -> dict:
    """Loads this user's persisted active config, filling in any keys
    that don't exist yet (new user, or app upgraded with new fields)."""
    stored = load_active_config(user_id) or {}
    merged = {**ACTIVE_DEFAULTS, **stored}
    return merged


def validate_settings(d: dict) -> List[str]:
    """Rejects physically-inconsistent settings before they can become
    active. Returns a list of human-readable errors; empty = valid."""
    errs = []
    mat = MATERIALS[d["material_key"]]
    if d["form"] == "tube" and d["id_mm"] >= d["od_mm"]:
        errs.append("Bore diameter must be smaller than outer diameter for a tube.")
    if d["form"] != "custom" and d["od_mm"] <= 0:
        errs.append("Outer diameter must be positive.")
    if d["form"] == "custom" and d.get("custom_area_mm2", 0) <= 0:
        errs.append("Custom cross-section area must be positive.")
    if d["strain_pct"] <= 0:
        errs.append("Applied strain must be positive.")
    if d["strain_pct"] > mat.eps_tr * 100 * 1.5:
        errs.append(f"Applied strain ({d['strain_pct']:.1f}%) is unreasonably "
                    f"far past {mat.name.split('(')[0].strip()}'s "
                    f"{mat.eps_tr*100:.1f}% transformation plateau.")
    if d["phase_time_s"] <= 0:
        errs.append("Phase duration must be positive.")
    if d["n_elements"] < 1:
        errs.append("Bundle must have at least one element.")
    if d["length_mm"] <= 0:
        errs.append("Active length must be positive.")
    if d["chamber_capacity"] <= 0:
        errs.append("Chamber heat capacity must be positive.")
    if not (-273.0 < d["target_c"] < d["ambient_c"] + 60):
        errs.append("Target temperature is out of a physically sane range.")
    if d["bc_mode"] == "User Defined":
        disp_max = max(0.1, d["length_mm"] * 0.10)
        if not (0 < d.get("bc_disp_mm", 0) <= disp_max):
            errs.append(f"Displacement magnitude must be between 0 and "
                        f"{disp_max:.2f} mm (10% of the active length).")
    return errs


def settings_to_config(d: dict) -> Config:
    """Resolves boundary-condition mode into the strain/compression the
    physics engine actually uses, then builds a Config."""
    if d["bc_mode"] == "User Defined":
        eff_strain = round(d["bc_disp_mm"] / d["length_mm"] * 100, 3) \
            if d["length_mm"] > 0 else d["strain_pct"]
        eff_compression = (d["bc_direction"] == "Compression (contraction)")
    else:
        eff_strain = d["strain_pct"]
        eff_compression = d["mode_compression"]
    return Config(**{
        **{k: d[k] for k in CONFIG_FIELDS if k not in
           ("strain_pct", "mode_compression")},
        "strain_pct": eff_strain,
        "mode_compression": eff_compression,
    })


# =================================================================
# PART 2 — USER IDENTITY
# =================================================================

def _slugify(raw: str) -> str:
    s = "".join(ch.lower() if ch.isalnum() else "-" for ch in raw.strip())
    s = "-".join(filter(None, s.split("-")))
    return s[:60]


def get_query_user() -> Optional[str]:
    try:
        val = st.query_params.get("u")
    except Exception:
        qp = st.experimental_get_query_params()
        val = qp.get("u", [None])[0]
    return val or None


def set_query_user(user_id: str) -> None:
    try:
        st.query_params["u"] = user_id
    except Exception:
        st.experimental_set_query_params(u=user_id)


def clear_query_user() -> None:
    try:
        if "u" in st.query_params:
            del st.query_params["u"]
    except Exception:
        st.experimental_set_query_params()


# =================================================================
# PART 3 — INTERFACE
# =================================================================

st.set_page_config(page_title="Elastocaloric rig control", page_icon="◆",
                   layout="wide", initial_sidebar_state="collapsed")

# --- sign-in gate: must happen before anything user-scoped runs ------
_uid = get_query_user()
if not _uid:
    st.markdown("## ◆ Elastocaloric rig control — sign in")
    st.caption(
        "Enter any handle to identify your session — no password. Two "
        "people using different handles on this same shared link get "
        "completely separate settings, files, runs and AI history. "
        "Using the same handle again (e.g. after closing the tab) "
        "restores that session's saved configuration.")
    with st.form("signin"):
        handle = st.text_input("Your name or handle", max_chars=60)
        go = st.form_submit_button("Continue", type="primary")
    if go:
        uid = _slugify(handle) or uuid.uuid4().hex[:8]
        ensure_user(uid)
        set_query_user(uid)
        st.rerun()
    st.stop()

USER_ID = _uid
ensure_user(USER_ID)


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def get_theme(dark: bool) -> dict:
    """Semantic theme tokens. Every color the UI uses is named here
    once; nothing downstream should hardcode a hex value."""
    if dark:
        base = dict(INK="#E8EEF3", STEEL="#93A5B4", COLD="#35C3E8",
                    WARM="#F2A65A", GOOD="#4ADE9B", LINE="#2A3B47",
                    PAPER="#16222B", APP_BG="#0B141A",
                    ERROR="#F87171", COLD_HOVER="#1BA9CE")
    else:
        base = dict(INK="#12232E", STEEL="#4A6572", COLD="#0E7C9B",
                   WARM="#B45309", GOOD="#1F7A5A", LINE="#C7D2DB",
                   PAPER="#FFFFFF", APP_BG="#E8EDF1",
                   ERROR="#C0392B", COLD_HOVER="#0B657F")
    base["SUCCESS"] = base["GOOD"]
    base["WARNING"] = base["WARM"]
    base["INFOC"] = base["COLD"]
    base["BTN_BG"] = base["PAPER"]
    base["BTN_TEXT"] = base["INK"]
    return base


# --- persisted per-user state, loaded once per session ---------------
if "active" not in st.session_state:
    st.session_state["active"] = full_active_dict(USER_ID)
if "theme_dark" not in st.session_state:
    st.session_state["theme_dark"] = load_pref_dark(USER_ID)
if "ai_ctx" not in st.session_state:
    st.session_state["ai_ctx"] = load_ai_context(USER_ID)

# Apply requests raised from the AI assistant (or a factory-restore on
# the Settings tab) are staged here and consumed at the very top of the
# script, BEFORE the Settings tab's widgets are instantiated for this
# run — writing into an already-instantiated widget's session_state key
# raises StreamlitWidgetAlreadyInstantiatedError, so this staging +
# rerun pattern is the safe way to change a "d_*" value programmatically.
if st.session_state.get("_pending_settings"):
    for k, v in st.session_state.pop("_pending_settings").items():
        st.session_state[f"d_{k}"] = v

if st.session_state.get("_pending_apply_active"):
    new_active = st.session_state.pop("_pending_apply_active")
    st.session_state["active"] = new_active
    save_active_config(USER_ID, new_active)
    st.session_state["_run_reset_reason"] = st.session_state.pop(
        "_pending_apply_reason", "Configuration applied.")

for k, v in ACTIVE_DEFAULTS.items():
    st.session_state.setdefault(f"d_{k}", st.session_state["active"].get(k, v))

theme = get_theme(st.session_state["theme_dark"])


def inject_css(t: dict) -> None:
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root, .stApp {{
    --text-color: {t["INK"]};
    --background-color: {t["APP_BG"]};
    --secondary-background-color: {t["PAPER"]};
}}
html, body, [class*="css"] {{ font-family:'IBM Plex Sans',system-ui,sans-serif; }}
.stApp {{ background:{t["APP_BG"]}; }}
.block-container {{ max-width:1480px; padding-top:1.0rem; }}
footer, #MainMenu {{ visibility:hidden; }}

/* -----------------------------------------------------------------
   BLANKET RULE + targeted overrides. Streamlit renders many widgets
   (selects, sliders, dataframes, alerts, popovers) in generic divs
   whose own default styling is theme-agnostic, so a blanket text
   color plus explicit background rules for every widget family is
   the reliable way to avoid dark-on-dark / light-on-light patches —
   a single CSS patch per bug report does not scale to this many
   component types.
   ------------------------------------------------------------------ */
.stApp, .stApp * {{ color:{t["INK"]} !important; }}

/* Headings / captions / labels */
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{
    color:{t["STEEL"]} !important;
}}
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label {{
    color:{t["INK"]} !important;
}}

/* Tabs */
[data-testid="stTabs"] [data-baseweb="tab-list"] {{
    background:{t["APP_BG"]} !important; gap:4px;
    border-bottom:1px solid {t["LINE"]} !important;
}}
[data-testid="stTabs"] button[role="tab"] {{
    background:transparent !important; color:{t["STEEL"]} !important;
    border-radius:4px 4px 0 0 !important;
}}
[data-testid="stTabs"] button[role="tab"] p {{ color:inherit !important; }}
[data-testid="stTabs"] button[aria-selected="true"] {{
    color:{t["COLD"]} !important; border-bottom:2px solid {t["COLD"]} !important;
}}

/* Buttons — primary / secondary / download / form-submit */
.stButton button[kind="primary"], .stButton button[kind="primary"] p,
[data-testid="baseButton-primary"], [data-testid="baseButton-primary"] p {{
    background:{t["COLD"]} !important; color:#FFFFFF !important;
    border:1px solid {t["COLD"]} !important;
}}
.stButton button[kind="primary"]:hover {{ background:{t["COLD_HOVER"]} !important; }}
.stButton button[kind="secondary"], .stButton button[kind="secondary"] p,
[data-testid="baseButton-secondary"], [data-testid="baseButton-secondary"] p,
div[data-testid="stDownloadButton"] button,
div[data-testid="stDownloadButton"] button p,
div[data-testid="stFormSubmitButton"] button,
div[data-testid="stFormSubmitButton"] button p {{
    background:{t["BTN_BG"]} !important; color:{t["BTN_TEXT"]} !important;
    border:1px solid {t["LINE"]} !important;
}}
.stButton button:hover, div[data-testid="stDownloadButton"] button:hover {{
    border-color:{t["COLD"]} !important;
    background:{hex_to_rgba(t["COLD"], 0.10)} !important;
}}
button:disabled, button:disabled p {{ opacity:0.45 !important; }}

/* Inputs: text, number, textarea, select, multiselect, file uploader */
input, textarea,
div[data-baseweb="select"] > div,
div[data-baseweb="base-input"] {{
    background:{t["PAPER"]} !important; color:{t["INK"]} !important;
    border-color:{t["LINE"]} !important;
}}
[data-testid="stFileUploaderDropzone"] {{
    background:{t["PAPER"]} !important; border:1px dashed {t["LINE"]} !important;
}}
[data-testid="stFileUploaderDropzone"] * {{ color:{t["STEEL"]} !important; }}
[data-testid="stFileUploaderDropzone"] button {{
    background:{t["BTN_BG"]} !important; color:{t["BTN_TEXT"]} !important;
    border:1px solid {t["LINE"]} !important;
}}

/* Portalled popovers / dropdown menus (rendered outside .stApp) */
div[data-baseweb="popover"], div[data-baseweb="popover"] ul,
ul[role="listbox"] {{ background:{t["PAPER"]} !important; }}
div[data-baseweb="popover"] *, li[role="option"], div[data-baseweb="tag"] {{
    color:{t["INK"]} !important;
}}
li[role="option"]:hover, li[aria-selected="true"] {{
    background:{hex_to_rgba(t["COLD"], 0.14)} !important;
}}
div[data-baseweb="tag"] {{ background:{hex_to_rgba(t["COLD"], 0.18)} !important; }}

/* Sliders */
div[data-baseweb="slider"] [role="slider"] {{ background:{t["COLD"]} !important; }}
div[data-baseweb="slider"] > div > div {{ background:{t["LINE"]} !important; }}
div[data-baseweb="slider"] > div > div > div {{ background:{t["COLD"]} !important; }}

/* Checkbox / radio */
[data-testid="stCheckbox"] label, [data-testid="stRadio"] label {{
    color:{t["INK"]} !important;
}}
[data-baseweb="radio"] div:first-child, [data-baseweb="checkbox"] div:first-child {{
    border-color:{t["STEEL"]} !important;
}}

/* Alerts: success / info / warning / error */
[data-testid="stAlert"] {{
    background:{t["PAPER"]} !important; border:1px solid {t["LINE"]} !important;
}}
div[data-testid="stAlertContentSuccess"] {{ border-left:4px solid {t["SUCCESS"]} !important; }}
div[data-testid="stAlertContentInfo"] {{ border-left:4px solid {t["INFOC"]} !important; }}
div[data-testid="stAlertContentWarning"] {{ border-left:4px solid {t["WARNING"]} !important; }}
div[data-testid="stAlertContentError"] {{ border-left:4px solid {t["ERROR"]} !important; }}

/* Dataframes / tables */
[data-testid="stDataFrame"], [data-testid="stTable"] {{
    background:{t["PAPER"]} !important;
}}
[data-testid="stDataFrame"] * {{ color:{t["INK"]} !important; }}

/* Metrics */
[data-testid="stMetric"] {{
    background:{t["PAPER"]} !important; border:1px solid {t["LINE"]} !important;
    border-radius:6px; padding:8px 12px;
}}
[data-testid="stMetricLabel"] * {{ color:{t["STEEL"]} !important; }}
[data-testid="stMetricValue"] * {{ color:{t["INK"]} !important; }}

/* Expander */
[data-testid="stExpander"] {{
    background:{t["PAPER"]} !important; border:1px solid {t["LINE"]} !important;
    border-radius:6px;
}}
[data-testid="stExpander"] summary p {{ color:{t["INK"]} !important; }}

/* Toast */
[data-testid="stToast"] {{
    background:{t["PAPER"]} !important; color:{t["INK"]} !important;
    border:1px solid {t["LINE"]} !important;
}}

/* App-specific semantic classes */
.titlebar {{ display:flex; align-items:baseline; gap:18px;
  border-bottom:2px solid {t["INK"]}; padding-bottom:10px; margin-bottom:18px; }}
.titlebar h1 {{ font-size:24px; font-weight:600; margin:0; }}
.titlebar span {{ font-size:13px; color:{t["STEEL"]} !important; }}
.verdict {{ background:{t["PAPER"]}; border:1px solid {t["LINE"]};
  border-left:5px solid {t["COLD"]}; border-radius:4px; padding:20px 24px; }}
.verdict.blocked {{ border-left-color:{t["WARM"]}; }}
.verdict h2 {{ font-size:20px; font-weight:600; margin:0 0 6px; }}
.verdict p {{ font-size:14px; color:{t["STEEL"]} !important; margin:0;
  line-height:1.6; max-width:78ch; }}
.figure {{ font-family:'IBM Plex Mono',monospace; font-variant-numeric:tabular-nums;
  font-size:26px; font-weight:500; }}
.figure small {{ font-size:13px; color:{t["STEEL"]} !important; font-weight:400; }}
.cap {{ font-size:12px; color:{t["STEEL"]} !important; margin-bottom:2px; }}
.step {{ background:{t["PAPER"]}; border:1px solid {t["LINE"]}; border-radius:4px;
  padding:14px 16px; min-height:118px; }}
.step.active {{ border:1px solid {t["WARM"]};
  background:{hex_to_rgba(t["WARM"], 0.08)}; }}
.step.done {{ border-left:4px solid {t["GOOD"]}; }}
.step .n {{ font-family:'IBM Plex Mono',monospace; font-size:12px;
  color:{t["STEEL"]} !important; }}
.step .t {{ font-size:15px; font-weight:600; margin-top:6px; }}
.step .d {{ font-size:12px; color:{t["STEEL"]} !important; margin-top:6px;
  line-height:1.5; }}
.chip {{ display:inline-block; padding:3px 10px; border-radius:3px;
  font-size:11px; font-weight:600; }}
.chip.run {{ background:{hex_to_rgba(t["WARM"], 0.16)}; color:{t["WARM"]} !important; }}
.chip.hold {{ background:{hex_to_rgba(t["GOOD"], 0.16)}; color:{t["GOOD"]} !important; }}
.chip.idle {{ background:{hex_to_rgba(t["STEEL"], 0.16)}; color:{t["STEEL"]} !important; }}
.extreme {{ background:{t["PAPER"]}; border:1px solid {t["LINE"]};
  border-radius:4px; padding:10px 14px; }}
.ctrl-log {{ background:{t["PAPER"]}; border:1px solid {t["LINE"]};
  border-radius:6px; padding:12px 14px; max-height:220px; overflow-y:auto;
  font-family:'IBM Plex Mono',monospace; font-size:12px; line-height:1.7; }}
.ctrl-log .ts {{ color:{t["STEEL"]} !important; }}
.activebar {{ background:{t["PAPER"]}; border:1px solid {t["LINE"]};
  border-radius:6px; padding:10px 18px; margin-top:8px; display:flex;
  flex-wrap:wrap; gap:22px; font-size:12.5px; color:{t["STEEL"]} !important; }}
.activebar b {{ color:{t["INK"]} !important; }}
.unsaved-banner {{ background:{hex_to_rgba(t["WARM"], 0.12)};
  border:1px solid {t["WARM"]}; border-radius:6px; padding:10px 16px;
  margin-bottom:14px; font-size:13.5px; }}
</style>
""", unsafe_allow_html=True)


inject_css(theme)

# ---------------------------------------------------------------------
# Header: title, signed-in-as badge, theme toggle, sign out
# ---------------------------------------------------------------------
hc1, hc2, hc3 = st.columns([5, 2, 1.4])
with hc1:
    st.markdown('<div class="titlebar"><h1>Elastocaloric refrigeration rig'
               '</h1><span>NiTi SCADA / HMI</span></div>',
               unsafe_allow_html=True)
with hc2:
    st.caption(f"Signed in as **{USER_ID}** — this handle's settings, files "
              f"and runs are private to it.")
with hc3:
    new_dark = st.toggle("Dark theme", value=st.session_state["theme_dark"],
                         key="theme_toggle")
    if new_dark != st.session_state["theme_dark"]:
        st.session_state["theme_dark"] = new_dark
        save_pref_dark(USER_ID, new_dark)
        st.rerun()
    if st.button("Sign out", use_container_width=True):
        clear_query_user()
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ---------------------------------------------------------------------
# Build the Config currently in effect for THIS user from st.session_
# state["active"] — the single source of truth every other part of the
# app (dashboard, simulation, graphs, AI assistant) reads from.
# ---------------------------------------------------------------------
active = st.session_state["active"]
cfg = settings_to_config(active)
mat = cfg.material


# --- run-time (per-user, per-session) state ---------------------------
def blank_log():
    return {k: [] for k in ("t", "chamber", "element", "fluid", "strain",
                            "stress", "cop", "cycle", "lift")}


def event(msg: str):
    st.session_state.events.insert(0, f"{datetime.now():%H:%M:%S}  {msg}")
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
    if len(log["t"]) > 2400:
        for key in log:
            del log[key][0]
    ex = st.session_state.extremes
    ex["chamber_max"] = max(ex["chamber_max"], s.t_chamber)
    ex["chamber_min"] = min(ex["chamber_min"], s.t_chamber)
    ex["element_max"] = max(ex["element_max"], s.t_element)
    ex["element_min"] = min(ex["element_min"], s.t_element)


def reset(reason: str = "Controller reset."):
    """Resets the RUN only (cycle count, history, energy tally, extremes)
    for this user's session. Never touches the persisted active config."""
    st.session_state.state = initial_state(cfg)
    st.session_state.running = False
    st.session_state.log = blank_log()
    st.session_state.events = []
    st.session_state.energy = [0.0, 0.0]
    st.session_state.extremes = {
        "chamber_max": cfg.ambient_c, "chamber_min": cfg.ambient_c,
        "element_max": cfg.ambient_c, "element_min": cfg.ambient_c,
    }
    record()
    event(reason)


need_reset = "state" not in st.session_state
run_reset_reason = st.session_state.pop("_run_reset_reason", None)
if need_reset or run_reset_reason:
    reset(run_reset_reason or "Controller initialised.")

state: State = st.session_state.state


def advance_phase():
    cur: State = st.session_state.state
    nxt = cur.phase % 4 + 1
    st.session_state.state = step_phase(cfg, cur, nxt)
    s = st.session_state.state
    if nxt == 4:
        st.session_state.energy[0] += s.q_cold_rate * cfg.cycle_time
        st.session_state.energy[1] += cfg.work_per_cycle
    record()
    return s


st.session_state.setdefault("saved_runs_cache_key", None)

env = predict_envelope(cfg)
tau = cfg.capacity / cfg.ua_hx if cfg.ua_hx > 0 else float("inf")
effectiveness = 1.0 - math.exp(-cfg.phase_time_s / tau) if tau > 0 else 0.0
q_tot, w_tot = st.session_state.energy
avg_cop = q_tot / w_tot if w_tot > 0 else 0.0

_direction_label = "Compression" if cfg.mode_compression else "Tension"
_form_label = {"tube": "Tube", "wire": "Wire", "custom": "Custom"}.get(
    cfg.form, cfg.form.title())


def render_boundary_diagram(theme_d, fixed_end, is_compression,
                            magnitude_mm, magnitude_pct, mode_label):
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
           f'letter-spacing:0.5px">FIXED</div>'
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
    parts = [wall, rod, disp_end] if fixed_end == "Left" else [disp_end, rod, wall]
    return (f'<div class="verdict" style="display:flex;align-items:center;'
           f'justify-content:center;gap:10px;padding:22px 24px">'
           f'{"".join(parts)}</div>'
           f'<div style="text-align:center;font-size:12px;'
           f'color:{theme_d["STEEL"]};margin-top:8px">'
           f'Support configuration: <b>{mode_label}</b></div>')


# =================================================================
# TAB NAVIGATION
# =================================================================
tab_dash, tab_settings, tab_ai = st.tabs(
    ["📊 SCADA Dashboard", "⚙ User Defined Settings", "🤖 AI Engineering Assistant"])

# -----------------------------------------------------------------
# TAB 1 — SCADA / OPERATING DASHBOARD
# Monitoring, active values, live readings, graphs, controller status,
# results. No detailed editable configuration lives here.
# -----------------------------------------------------------------
with tab_dash:
    active_rows = [
        ("Element", f"{_form_label} · {cfg.n_elements} × {cfg.length_mm:.0f} mm"),
        ("Phase duration", f"{cfg.phase_time_s:.1f} s"),
        ("Displacement", f"{cfg.strain * cfg.length_mm:.2f} mm ({_direction_label})"),
        ("Target temperature", f"{cfg.target_c:.1f} °C"),
        ("Ambient temperature", f"{cfg.ambient_c:.1f} °C"),
        ("Material", mat.name), ("Exchanger", cfg.exchanger.name),
    ]
    st.markdown(
        '<div class="activebar">' +
        "".join(f"<span><b>{label}:</b> {value}</span>"
               for label, value in active_rows) + "</div>",
        unsafe_allow_html=True)
    st.caption("↑ Currently active configuration — edit it in "
              "**User Defined Settings**, then Apply.")
    st.write("")

    st.markdown(render_boundary_diagram(
        theme, active["bc_fixed_end"], cfg.mode_compression,
        cfg.strain * cfg.length_mm,
        cfg.strain_pct, active["bc_mode"]), unsafe_allow_html=True)

    if env["reachable"]:
        headline = f"This configuration can hold {cfg.target_c:.1f} °C."
        body = (f"Predicted pull-down takes {env['cycles_to_target']:.0f} cycles "
                f"({env['time_to_target_s']/60:.1f} min). The lowest temperature "
                f"the cold box can sustain is {env['t_min_c']:.1f} °C, so there "
                f"is {cfg.target_c - env['t_min_c']:.1f} K of margin against "
                f"the {cfg.insulation_ua:.2f} W/K cabinet load.")
    else:
        headline = f"This configuration cannot reach {cfg.target_c:.1f} °C."
        body = (f"The cold box settles at {env['t_min_c']:.1f} °C, "
                f"{abs(cfg.target_c - env['t_min_c']):.1f} K short. The adiabatic "
                f"swing is only {env['dt_adiabatic']:.1f} K at "
                f"{cfg.strain_pct:.1f}% strain. Raise the strain toward "
                f"{mat.eps_tr*100:.1f}% in User Defined Settings, cut the "
                f"cabinet load, or add a regenerator.")
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
        st.info(f"Each phase lasts {cfg.phase_time_s:.1f} s against a thermal "
               f"time constant of {tau:.1f} s, so only "
               f"{effectiveness*100:.0f}% of the available swing transfers "
               f"before the phase ends.")

    st.divider()

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
            label = "Start sequencing" if active["auto_mode"] else "Step one phase"
            if st.button(label, type="primary", use_container_width=True):
                if active["auto_mode"]:
                    st.session_state.running = True
                    event("Automatic sequencing started.")
                else:
                    s = advance_phase()
                    event(f"Phase {s.phase} — {PHASE_NAMES[s.phase]}.")
                st.rerun()
    with c2:
        if st.button("Run to steady state", use_container_width=True):
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
                     help="Clears run history and cycle count only — your "
                          "settings and saved runs are untouched."):
            reset("Run data reset by operator.")
            st.rerun()
    with c4:
        st.markdown(f'<div class="cap">Cycles</div><div class="figure">'
                   f'{state.cycle}</div>', unsafe_allow_html=True)
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

    DESCRIPTIONS = {
        1: "Stress drives the austenite-to-martensite transformation. Latent "
           "heat appears as a temperature rise in the element.",
        2: "The hot element is swept by the coolant and returns toward "
           "ambient while the strain is held.",
        3: "The stress is released. The reverse transformation absorbs "
           "latent heat and the element goes below ambient.",
        4: "The cold element is coupled to the cold box and pulls heat out "
           "of it. This is the useful cooling.",
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

    st.markdown(f"**Enclosure & wire extremes**  "
               f"<span style='color:{theme['STEEL']};font-size:12px'>"
               f"(since last reset)</span>", unsafe_allow_html=True)
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

    log = st.session_state.log
    g1, g2 = st.columns([1.55, 1])
    with g1:
        st.markdown("**Pull-down**")
        fig, ax = plt.subplots(figsize=(8.2, 3.5))
        ax.set_facecolor(theme["PAPER"]); ax.figure.patch.set_facecolor(theme["PAPER"])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(theme["LINE"])
        ax.tick_params(colors=theme["STEEL"], labelsize=8)
        ax.grid(True, color=theme["LINE"], linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        t_arr = np.asarray(log["t"])
        ax.plot(t_arr, log["element"], color=theme["WARM"], lw=1.0, alpha=0.75,
               label="Element")
        ax.plot(t_arr, log["chamber"], color=theme["COLD"], lw=2.2, label="Cold box")
        ax.axhline(cfg.target_c, color=theme["GOOD"], ls="--", lw=1.2, label="Target")
        ax.axhline(cfg.ambient_c, color=theme["STEEL"], ls=":", lw=1.0, label="Ambient")
        ax.set_xlabel("Elapsed time (s)", color=theme["STEEL"])
        ax.set_ylabel("Temperature (°C)", color=theme["STEEL"])
        leg = ax.legend(frameon=False, fontsize=8, ncols=4, loc="upper right")
        for text in leg.get_texts():
            text.set_color(theme["STEEL"])
        st.pyplot(fig, clear_figure=True); plt.close(fig)
    with g2:
        st.markdown("**Stress–strain path**")
        fig2, ax2 = plt.subplots(figsize=(5.2, 3.5))
        ax2.set_facecolor(theme["PAPER"]); ax2.figure.patch.set_facecolor(theme["PAPER"])
        for side in ("top", "right"):
            ax2.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax2.spines[side].set_color(theme["LINE"])
        ax2.tick_params(colors=theme["STEEL"], labelsize=8)
        ax2.grid(True, color=theme["LINE"], linewidth=0.6, alpha=0.7)
        ax2.plot(log["strain"][-160:], log["stress"][-160:],
                color=theme["INK"], lw=1.4, alpha=0.85)
        ax2.scatter([state.strain_pct], [state.stress_mpa],
                   s=55, color=theme["WARM"], zorder=5)
        ax2.set_xlabel("Strain (%)", color=theme["STEEL"])
        ax2.set_ylabel("Stress (MPa)", color=theme["STEEL"])
        st.pyplot(fig2, clear_figure=True); plt.close(fig2)

    h1, h2 = st.columns([1.55, 1])
    with h1:
        st.markdown("**Coefficient of performance per cycle**")
        fig3, ax3 = plt.subplots(figsize=(8.2, 2.4))
        ax3.set_facecolor(theme["PAPER"]); ax3.figure.patch.set_facecolor(theme["PAPER"])
        for side in ("top", "right"):
            ax3.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax3.spines[side].set_color(theme["LINE"])
        ax3.tick_params(colors=theme["STEEL"], labelsize=8)
        ax3.grid(True, color=theme["LINE"], linewidth=0.6, alpha=0.7)
        cyc, cop_arr = np.asarray(log["cycle"]), np.asarray(log["cop"])
        mask = cyc > 0
        ax3.plot(cyc[mask], cop_arr[mask], color=theme["COLD"], lw=1.6)
        ax3.set_xlabel("Cycle", color=theme["STEEL"])
        ax3.set_ylabel("COP (–)", color=theme["STEEL"])
        st.pyplot(fig3, clear_figure=True); plt.close(fig3)
    with h2:
        st.markdown("**Controller log**")
        lines = st.session_state.events or ["No events yet."]
        html_lines = []
        for ln in lines:
            if "  " in ln:
                ts, rest = ln.split("  ", 1)
                html_lines.append(f'<span class="ts">{ts}</span>&nbsp;&nbsp;{rest}')
            else:
                html_lines.append(ln)
        st.markdown(f'<div class="ctrl-log">{"<br>".join(html_lines)}</div>',
                   unsafe_allow_html=True)

    if active["auto_mode"] and st.session_state.get("running") and not state.target_reached:
        for _ in range(active["cycles_per_update"] * 4):
            if advance_phase().target_reached:
                break
        s = st.session_state.state
        if s.target_reached:
            st.session_state.running = False
            event(f"Target {cfg.target_c:.1f} °C reached after {s.cycle} "
                 f"cycles. Sequencing stopped.")
            st.toast(f"Target reached in {s.cycle} cycles.", icon="✅")
        st.rerun()

    # --- results / export --------------------------------------------
    st.divider()
    st.markdown("**Test record**")
    frame = pd.DataFrame({
        "time_s": log["t"], "cycle": log["cycle"], "chamber_c": log["chamber"],
        "element_c": log["element"], "coolant_c": log["fluid"],
        "strain_pct": log["strain"], "stress_mpa": log["stress"],
        "cop": log["cop"], "cooling_w": log["lift"],
    })

    def build_report() -> str:
        rows = [
            ("User", USER_ID), ("Material", mat.name),
            ("Element form", f"{cfg.n_elements} × {cfg.form}, {cfg.length_mm:.0f} mm"),
            ("Cross-section", f"{cfg.section_area*1e6:.1f} mm²"),
            ("Active mass", f"{cfg.mass*1e3:.1f} g"),
            ("Heat-transfer area", f"{cfg.wetted_area*1e4:.0f} cm²"),
            ("Exchanger", cfg.exchanger.name),
            ("Conductance UA", f"{cfg.ua_hx:.1f} W/K"),
            ("Thermal time constant", f"{tau:.2f} s"),
            ("Applied strain", f"{cfg.strain_pct:.2f} %"),
            ("Regenerator effectiveness", f"{cfg.regen_effectiveness*100:.0f} %"),
            ("Boundary condition", active["bc_mode"]),
            ("Fixed end", active["bc_fixed_end"]),
            ("Displacement direction", _direction_label),
            ("Displacement magnitude", f"{cfg.strain * cfg.length_mm:.2f} mm"),
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
               f"{datetime.now():%Y-%m-%d %H:%M} for user `{USER_ID}`\n\n"
               f"## Result\n\nTarget {cfg.target_c:.1f} °C was {verdict} "
               f"after {state.cycle} cycles. {headline} {body}\n\n"
               f"## Configuration and values\n\n"
               f"| Quantity | Value |\n|---|---|\n{table}\n\n## Basis\n\n"
               f"Two-capacity lumped model with Clausius-Clapeyron "
               f"transformation stress and an NTU-effectiveness exchanger. "
               f"Values are simulated, not measured.\n")

    stage1_done = state.cycle > 0 or len(log["t"]) > 1
    saved = list_saved_runs(USER_ID)
    stage2_done = len(saved) > 0
    stage3_done = st.session_state.get("has_downloaded", False)

    def _stage_class(done, active_):
        return "done" if done else ("active" if active_ else "pending")

    STAGE_INFO = [
        ("1", "Run a simulation", "Click Start sequencing or Run to steady "
         "state above."),
        ("2", "Save it here", "Give this run a name and click Save. It's "
         "stored to your account and survives a refresh."),
        ("3", "Download it", "Download the file for any saved run."),
    ]
    stage_states = [stage1_done, stage1_done and stage2_done,
                    stage1_done and stage2_done and stage3_done]
    scols = st.columns(3)
    for i, (num, title, desc) in enumerate(STAGE_INFO):
        act = (not stage_states[i]) and (i == 0 or stage_states[i - 1])
        cls = _stage_class(stage_states[i], act)
        with scols[i]:
            st.markdown(f'<div class="step {cls}"><div class="n">STAGE {num}</div>'
                       f'<div class="t">{"✓ " if stage_states[i] else ""}{title}</div>'
                       f'<div class="d">{desc}</div></div>', unsafe_allow_html=True)
    st.write("")

    if not stage1_done:
        st.info("Run at least one phase above, then come back here to save it.")
    else:
        sv1, sv2 = st.columns([3, 1])
        with sv1:
            default_name = (f"{mat.name.split('(')[0].strip()} → "
                           f"{cfg.target_c:.0f}°C ({datetime.now():%H:%M})")
            save_name = st.text_input("Name this run", value=default_name,
                                      key="save_name_input")
        with sv2:
            st.write(""); st.write("")
            do_save = st.button("💾 Save this run", type="primary",
                                use_container_width=True)
        if do_save:
            add_saved_run(USER_ID, save_name or default_name, {
                "material": mat.name, "target_c": cfg.target_c,
                "reached": state.target_reached, "cycles": state.cycle,
                "chamber_c": state.t_chamber, "cop": avg_cop,
                "csv": frame.to_csv(index=False), "report": build_report(),
            })
            event(f'Run saved as "{save_name or default_name}".')
            st.success(f'Saved as "{save_name or default_name}".')
            st.rerun()

        if saved:
            st.markdown(f"**Your saved runs ({len(saved)})**")
            for run in saved:
                badge = ("✅ Reached target" if run["reached"]
                         else "⚠️ Did not reach target")
                with st.expander(f"{run['name']}  —  {badge}", expanded=False):
                    rc1, rc2, rc3, rc4 = st.columns(4)
                    rc1.markdown(f'<div class="cap">Material</div><div class="figure" '
                               f'style="font-size:16px">{run["material"]}</div>',
                               unsafe_allow_html=True)
                    rc2.markdown(f'<div class="cap">Target</div><div class="figure" '
                               f'style="font-size:16px">{run["target_c"]:.1f} °C</div>',
                               unsafe_allow_html=True)
                    rc3.markdown(f'<div class="cap">Cycles</div><div class="figure" '
                               f'style="font-size:16px">{run["cycles"]}</div>',
                               unsafe_allow_html=True)
                    rc4.markdown(f'<div class="cap">COP</div><div class="figure" '
                               f'style="font-size:16px">{run["cop"]:.2f}</div>',
                               unsafe_allow_html=True)
                    st.caption(f"Saved {run['saved_at']}")
                    dc1, dc2, dc3 = st.columns(3)
                    with dc1:
                        if st.download_button("Download CSV",
                                             run["csv"].encode(),
                                             f"{run['name'].replace(' ', '_')}.csv",
                                             "text/csv", key=f"dl_csv_{run['id']}",
                                             use_container_width=True):
                            st.session_state.has_downloaded = True
                    with dc2:
                        if st.download_button("Download report",
                                             run["report"].encode(),
                                             f"{run['name'].replace(' ', '_')}.md",
                                             "text/markdown", key=f"dl_md_{run['id']}",
                                             use_container_width=True):
                            st.session_state.has_downloaded = True
                    with dc3:
                        if st.button("🗑 Delete", key=f"del_run_{run['id']}",
                                    use_container_width=True):
                            delete_saved_run(USER_ID, run["id"])
                            event(f'Deleted saved run "{run["name"]}".')
                            st.rerun()
            if st.button("🗑 Clear all my saved runs", key="clear_saved"):
                clear_saved_runs(USER_ID)
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

    st.caption("Simulated process values. Experimental validation requires "
              "measured transformation temperatures, the stress–strain "
              "hysteresis of the actual element, heat-transfer coefficients, "
              "thermal masses and actuator losses.")

# -----------------------------------------------------------------
# TAB 2 — USER DEFINED SETTINGS
# Detailed editable configuration, data input, file/image input.
# Everything here writes to "d_*" draft keys only; the dashboard keeps
# showing the ACTIVE config until "Apply changes" is clicked.
# -----------------------------------------------------------------
with tab_settings:
    d = {k: st.session_state[f"d_{k}"] for k in ACTIVE_DEFAULTS}
    unsaved = any(d[k] != active.get(k) for k in ACTIVE_DEFAULTS)

    def _top_apply_bar():
        if unsaved:
            st.markdown('<div class="unsaved-banner">You have unsaved changes '
                       "here — the SCADA dashboard is still running your "
                       "previously applied configuration. Click "
                       "<b>Apply changes</b> below to activate the draft.</div>",
                       unsafe_allow_html=True)
        bcol1, bcol2, bcol3 = st.columns([1.4, 1.4, 3])
        apply_clicked = bcol1.button("✅ Apply changes", type="primary",
                                     use_container_width=True,
                                     disabled=not unsaved, key="apply_top")
        discard_clicked = bcol2.button("↩ Discard draft", use_container_width=True,
                                       disabled=not unsaved, key="discard_top")
        return apply_clicked, discard_clicked

    apply1, discard1 = _top_apply_bar()

    st.markdown("### Operating parameters")
    oc1, oc2, oc3 = st.columns(3)
    with oc1:
        st.selectbox("Active material", list(MATERIALS),
                    format_func=lambda k: MATERIALS[k].name, key="d_material_key")
        d_mat = MATERIALS[st.session_state["d_material_key"]]
        st.number_input("Target temperature (°C)", -40.0, 40.0, step=1.0,
                        key="d_target_c")
    with oc2:
        st.slider("Phase duration (s)", 0.2, 60.0, step=0.1, key="d_phase_time_s")
        st.number_input("Ambient temperature (°C)", 5.0, 45.0, step=1.0,
                        key="d_ambient_c")
    with oc3:
        st.toggle("Automatic sequencing", key="d_auto_mode")
        st.slider("Cycles per screen update", 1, 20, key="d_cycles_per_update")

    st.markdown("### Boundary conditions & displacement")
    st.caption("One end is always Fixed; the opposite end is always "
              "Displacement-controlled.")
    bc1, bc2 = st.columns(2)
    with bc1:
        st.selectbox("Support configuration",
                    ["Fixed + Displacement", "User Defined"], key="d_bc_mode")
        st.radio("Fixed end", ["Left", "Right"], horizontal=True,
                key="d_bc_fixed_end")
    with bc2:
        strain_mn = 0.5
        strain_mx = max(8.0, d_mat.eps_tr * 100 * 1.3)
        st.session_state["d_strain_pct"] = min(max(
            st.session_state["d_strain_pct"], strain_mn), strain_mx)
        bc_overrides = st.session_state["d_bc_mode"] == "User Defined"
        st.slider("Applied strain (%)", strain_mn, strain_mx, step=0.1,
                 key="d_strain_pct", disabled=bc_overrides,
                 help=(f"Transformation plateau ends near "
                       f"{d_mat.eps_tr*100:.1f}%."))
        st.toggle("Compressive loading", key="d_mode_compression",
                 disabled=bc_overrides)
    if bc_overrides:
        st.radio("Displacement direction",
                ["Tension (elongation)", "Compression (contraction)"],
                key="d_bc_direction")
        disp_max = max(0.1, round(st.session_state["d_length_mm"] * 0.10, 2))
        st.number_input("Displacement magnitude (mm)", 0.05, disp_max, step=0.05,
                       key="d_bc_disp_mm",
                       help=f"Bounded to 10% of the active length.")

    st.markdown("### Mechanical configuration")
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.radio("Element form", ["tube", "wire", "custom"], horizontal=True,
                key="d_form")
        st.number_input("Elements in bundle", 1, 40, step=1, key="d_n_elements")
    with mc2:
        st.number_input("Active length (mm)", 20.0, 500.0, step=5.0,
                       key="d_length_mm")
        if st.session_state["d_form"] == "custom":
            st.number_input("Cross-section area, per element (mm²)", 0.5, 500.0,
                           step=1.0, key="d_custom_area_mm2")
        else:
            st.number_input("Outer diameter (mm)", 0.5, 30.0, step=0.5, key="d_od_mm")
    with mc3:
        if st.session_state["d_form"] == "custom":
            st.number_input("Wetted perimeter, per element (mm)", 0.5, 200.0,
                           step=1.0, key="d_custom_perimeter_mm")
        else:
            st.number_input("Bore diameter (mm)", 0.1, 29.0, step=0.5,
                           disabled=(st.session_state["d_form"] == "wire"),
                           key="d_id_mm")
        st.slider("Actuator efficiency", 0.3, 0.95, step=0.05, key="d_actuator_efficiency")

    st.markdown("### Heat transfer & enclosure")
    hc1_, hc2_, hc3_ = st.columns(3)
    with hc1_:
        st.selectbox("Exchanger", list(EXCHANGERS),
                    format_func=lambda k: EXCHANGERS[k].name, key="d_exchanger_key")
        st.slider("Coolant flow (CFM equivalent)", 10.0, 150.0, step=5.0,
                 key="d_flow_cfm")
    with hc2_:
        st.number_input("Chamber heat capacity (J/K)", 50.0, 20000.0, step=50.0,
                       key="d_chamber_capacity")
        st.number_input("Cabinet loss (W/K)", 0.05, 5.0, step=0.05,
                       key="d_insulation_ua")
    with hc3_:
        st.slider("Regenerator effectiveness", 0.0, 0.8, step=0.05,
                 key="d_regen_effectiveness",
                 help="0 = no regenerator. Recycling heat between the hot "
                      "and cold halves lets the machine pump across a span "
                      "larger than one element's own adiabatic swing.")

    st.markdown("### Material / research data")
    st.text_area("Material notes / user-defined properties / experimental "
                "parameters", key="d_material_notes", height=90,
                placeholder="e.g. measured Af/As/Ms/Mf, batch number, "
                            "vendor datasheet deviations…")

    st.markdown("### Data input — numerical / sensor / experimental measurements")
    st.text_area("Free-form numerical or experimental data", key="d_experimental_notes",
                height=110,
                placeholder="Paste sensor readings, measured stress-strain "
                            "points, lab notes, etc. Saved with your account "
                            "on Apply.")

    apply2, discard2 = False, False
    st.markdown("### ")
    apply2, discard2 = _top_apply_bar()

    if discard1 or discard2:
        st.session_state["_pending_settings"] = dict(active)
        st.rerun()

    if apply1 or apply2:
        new_active = dict(active)
        for k in ACTIVE_DEFAULTS:
            src_key = {"material_key": "d_material_key",
                      "exchanger_key": "d_exchanger_key"}.get(k, f"d_{k}")
            if src_key in st.session_state:
                new_active[k] = st.session_state[src_key]
        errs = validate_settings(new_active)
        if errs:
            for e_ in errs:
                st.error(e_)
        else:
            st.session_state["_pending_apply_active"] = new_active
            st.session_state["_pending_apply_reason"] = (
                "Settings applied — run data reset to start cleanly under "
                "the new configuration.")
            event_msg_holder = new_active
            st.success("Configuration validated and applied.")
            st.rerun()

    st.divider()
    st.markdown("### File / image input")
    st.caption("CSV, Excel, PDF, TXT and image files. Stored to your account "
              "only — other users cannot see or list them.")
    uploads = st.file_uploader(
        "Upload research or experimental files", accept_multiple_files=True,
        type=["csv", "xlsx", "xls", "pdf", "txt", "png", "jpg", "jpeg"],
        key="settings_uploader")
    if uploads:
        note = st.text_input("Optional note for these files", key="upload_note")
        if st.button("💾 Save uploaded files to my account"):
            for uf in uploads:
                add_user_file(USER_ID, uf.name, uf.type or "application/octet-stream",
                             uf.getvalue(), note)
            event(f"{len(uploads)} file(s) uploaded.")
            st.success(f"Saved {len(uploads)} file(s).")
            st.rerun()

    files = list_user_files(USER_ID)
    if files:
        st.markdown(f"**Your files ({len(files)})**")
        for fid, fname, mime, note, uploaded_at, size in files:
            fc1, fc2, fc3 = st.columns([3, 1, 1])
            fc1.write(f"📄 **{fname}**  ·  {size/1024:.1f} KB  ·  {uploaded_at[:16]}"
                     + (f"  ·  _{note}_" if note else ""))
            with fc2:
                row = get_user_file(USER_ID, fid)
                if row:
                    _, mime_, content_ = row
                    st.download_button("Download", content_, fname, mime_,
                                      key=f"dlf_{fid}", use_container_width=True)
            with fc3:
                if st.button("🗑 Delete", key=f"delf_{fid}", use_container_width=True):
                    delete_user_file(USER_ID, fid)
                    event(f'Deleted file "{fname}".')
                    st.rerun()
    else:
        st.caption("No files uploaded yet.")

    st.divider()
    st.markdown("### Reset")
    rc1, rc2 = st.columns(2)
    with rc1:
        st.caption("Restores every draft field here to factory defaults. "
                  "Nothing becomes active until you click Apply.")
        if st.button("↺ Restore factory defaults (draft only)"):
            st.session_state["_pending_settings"] = dict(ACTIVE_DEFAULTS)
            st.rerun()
    with rc2:
        st.caption("⚠️ Permanently deletes your saved cloud configuration, "
                  "saved runs and uploaded files. Cannot be undone.")
        confirm = st.checkbox("I understand this is permanent", key="confirm_wipe")
        if st.button("🗑 Clear my saved cloud settings", disabled=not confirm,
                     type="primary"):
            delete_active_config(USER_ID)
            clear_saved_runs(USER_ID)
            for fid, *_ in list_user_files(USER_ID):
                delete_user_file(USER_ID, fid)
            fresh = dict(ACTIVE_DEFAULTS)
            st.session_state["_pending_apply_active"] = fresh
            st.session_state["_pending_apply_reason"] = (
                "Cloud settings, saved runs and files cleared.")
            st.session_state["_pending_settings"] = fresh
            st.session_state["confirm_wipe"] = False
            st.rerun()

# -----------------------------------------------------------------
# TAB 3 — AI ENGINEERING ASSISTANT
# -----------------------------------------------------------------
with tab_ai:
    st.caption(
        "Rule-based — reads this app's own physics engine and your live "
        "state directly, so it can't invent a wrong number, but it also "
        "can't hold a free-form conversation or read an uploaded file/image "
        "(that needs a connected AI model, which this deployment doesn't "
        "have configured). Your recommendation history below is saved to "
        "your account.")

    st.markdown("#### Beginner mode — what does this do?")
    GLOSSARY = {
        "Phase duration": lambda: (
            f"How long each of the four steps runs before moving to the "
            f"next. Right now it's **{cfg.phase_time_s:.1f} s** against a "
            f"thermal time constant of **{tau:.1f} s**, so about "
            f"**{effectiveness*100:.0f}%** of the available swing transfers "
            f"each phase."),
        "Applied strain / displacement": lambda: (
            f"Right now that's **{cfg.strain_pct:.2f}%** strain "
            f"(**{cfg.strain * cfg.length_mm:.2f} mm** on a "
            f"{cfg.length_mm:.0f} mm element), giving an adiabatic swing of "
            f"**{env['dt_adiabatic']:.1f} K**. The plateau is at "
            f"{mat.eps_tr*100:.1f}%; more strain beyond that adds stress, "
            f"not cooling."),
        "Fixed / Displacement ends": lambda: (
            f"One end (**Fixed: {active['bc_fixed_end']}**) stays still so "
            f"the mechanical stroke has something to push or pull against; "
            f"the opposite end is the Displacement end."),
        "Target temperature": lambda: (
            f"Currently **{cfg.target_c:.1f} °C** against ambient "
            f"**{cfg.ambient_c:.1f} °C**. The verdict banner on the "
            f"Dashboard tab says whether this configuration can reach it."),
        "Exchanger": lambda: (
            f"Currently **{cfg.exchanger.name}**, giving a conductance of "
            f"**{cfg.ua_hx:.1f} W/K**. Higher conductance pulls down faster "
            f"but also raises the sustainable floor temperature."),
        "Regenerator effectiveness": lambda: (
            f"Currently **{cfg.regen_effectiveness*100:.0f}%**"
            + (", i.e. off." if cfg.regen_effectiveness == 0 else
               f", multiplying the effective swing by "
               f"×{cfg.regen_gain:.2f}.")),
        "COP (coefficient of performance)": lambda: (
            f"Run-average is **{avg_cop:.2f}**. Below 1 is normal for a "
            f"single-stage, non-regenerative elastocaloric cycle."),
    }
    glossary_pick = st.selectbox("Pick a concept to have it explained with "
                                 "your current numbers:", list(GLOSSARY),
                                 key="ai_glossary_pick")
    st.info(GLOSSARY[glossary_pick]())

    st.divider()
    st.markdown("#### Recommend settings for a target")
    ae1, ae2, ae3 = st.columns(3)
    with ae1:
        ae_material = st.selectbox("Material", list(MATERIALS),
                                   format_func=lambda k: MATERIALS[k].name,
                                   key="ae_material")
    with ae2:
        ae_target = st.number_input("Temperature you want (°C)", -40.0, 40.0,
                                    5.0, 1.0, key="ae_target")
    with ae3:
        ae_ambient = st.number_input("Room temperature (°C)", 5.0, 45.0,
                                     25.0, 1.0, key="ae_ambient")
    ae_priority = st.radio("What matters more to you?",
                          ["Reach it fast", "Best efficiency (COP)"],
                          horizontal=True, key="ae_priority")

    if st.button("Get recommendation", key="ae_go"):
        results = search_recommendation(
            cfg, ae_material, ae_target, ae_ambient,
            prefer_speed=(ae_priority == "Reach it fast"))
        st.session_state["ae_results"] = [
            (config_to_dict(c), e) for c, e in results]
        st.session_state["ae_ran"] = True
        ctx = st.session_state["ai_ctx"]
        ctx.setdefault("history", []).insert(0, {
            "at": datetime.now().isoformat(), "material": ae_material,
            "target_c": ae_target, "ambient_c": ae_ambient,
            "priority": ae_priority, "found": len(results) > 0})
        ctx["history"] = ctx["history"][:20]
        st.session_state["ai_ctx"] = ctx
        save_ai_context(USER_ID, ctx)

    if st.session_state.get("ae_ran"):
        results = st.session_state.get("ae_results", [])
        if not results:
            st.error(f"With {MATERIALS[ae_material].name} and your current "
                    f"rig size, I can't find settings that hold "
                    f"{ae_target:.1f} °C. Try a warmer target or a material "
                    f"with a bigger transformation swing, such as NiTi.")
        else:
            best_dict, best_e = results[0]
            best_c = dict_to_config(best_dict)
            mo = best_c.material
            st.success(f"Recommended setup — {mo.name}")
            st.markdown(
                f"- **Applied strain:** {best_c.strain_pct:.1f} %\n"
                f"- **Phase duration:** {best_c.phase_time_s:.1f} s\n"
                f"- **Exchanger:** {best_c.exchanger.name}\n"
                f"- **Expected result:** reaches {best_c.target_c:.1f} °C in "
                f"about {best_e['cycles_to_target']:.0f} cycles "
                f"(~{best_e['time_to_target_s']/60:.1f} min), run-average "
                f"COP around {best_e['steady_cop']:.2f}.")
            aa1, aa2 = st.columns(2)
            with aa1:
                if st.button("✅ Yes, apply automatically", key="ae_apply",
                             type="primary", use_container_width=True):
                    new_active = dict(active)
                    new_active.update({
                        "material_key": best_c.material_key,
                        "strain_pct": best_c.strain_pct,
                        "phase_time_s": best_c.phase_time_s,
                        "exchanger_key": best_c.exchanger_key,
                        "ambient_c": best_c.ambient_c,
                        "target_c": best_c.target_c,
                        "bc_mode": "Fixed + Displacement",
                        "mode_compression": False,
                    })
                    st.session_state["_pending_apply_active"] = new_active
                    st.session_state["_pending_apply_reason"] = (
                        "Settings applied automatically by the AI "
                        "Engineering Assistant.")
                    st.session_state["_pending_settings"] = new_active
                    st.session_state["ae_ran"] = False
                    st.rerun()
            with aa2:
                if st.button("No, I'll set it myself", key="ae_skip",
                             use_container_width=True):
                    st.info("No problem — use User Defined Settings to "
                           "enter these values yourself.")

    hist = st.session_state["ai_ctx"].get("history", [])
    if hist:
        st.divider()
        st.markdown("#### Your recent recommendation requests")
        for h in hist[:8]:
            tag = "✅ found a fit" if h["found"] else "❌ no fit found"
            st.caption(f"{h['at'][:16].replace('T', ' ')} — "
                      f"{MATERIALS[h['material']].name} → {h['target_c']:.1f} °C "
                      f"@ {h['ambient_c']:.1f} °C ambient — {tag}")
