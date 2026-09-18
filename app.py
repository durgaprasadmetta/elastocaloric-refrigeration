import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time

# ============================================================
# NI-Ti ELASTOCALORIC REFRIGERATION
# PROFESSIONAL SCADA / HMI RESEARCH INTERFACE
# ============================================================

st.set_page_config(
    page_title="NiTi Elastocaloric SCADA",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# APPLICATION CONSTANTS
# ============================================================

APP_NAME = "NiTi ELASTOCALORIC REFRIGERATION"
APP_SUBTITLE = "Advanced Elastocaloric Refrigeration Using NiTi Alloy"

PHASES = {
    0: {
        "name": "SYSTEM READY",
        "short": "READY",
        "icon": "●",
        "description": "System initialized and waiting for operator command."
    },

    1: {
        "name": "MECHANICAL LOADING",
        "short": "LOADING",
        "icon": "⚙",
        "description": "NiTi element is mechanically loaded."
    },

    2: {
        "name": "HEAT REJECTION",
        "short": "HEAT REJECTION",
        "icon": "↗",
        "description": "Generated heat is rejected through forced air."
    },

    3: {
        "name": "MECHANICAL UNLOADING",
        "short": "UNLOADING",
        "icon": "❄",
        "description": "NiTi element is unloaded to generate cooling."
    },

    4: {
        "name": "COLD-SIDE RECOVERY",
        "short": "COLD RECOVERY",
        "icon": "◆",
        "description": "Cold-side chamber extracts useful cooling."
    }
}


# ============================================================
# SESSION STATE
# ============================================================

defaults = {

    "stage": 0,

    "cycle_count": 0,

    "auto_running": False,

    "target_reached": False,

    "chamber_temp": 25.0,

    "wire_temp": 25.0,

    "air_temp": 25.0,

    "chamber_max": 25.0,

    "chamber_min": 25.0,

    "wire_max": 25.0,

    "wire_min": 25.0,

    "air_max": 25.0,

    "air_min": 25.0,

    "stress": 0.0,

    "strain": 0.0,

    "cop": 0.0,

    "simulation_time": 0.0,

    "temperature_history": [25.0],

    "wire_history": [25.0],

    "air_history": [25.0],

    "stress_history": [0.0],

    "strain_history": [0.0],

    "time_history": [0.0],

    "event_log": [],

    "system_message": "SYSTEM READY",

    "last_update": "System initialized",

    "target_notification": False,

    "last_cycle_completed": False
}

for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# PROFESSIONAL CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #eef2f6;
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1600px;
    }

    /* HEADER */

    .top-header {

        background: linear-gradient(
            135deg,
            #071a2c 0%,
            #0d3555 55%,
            #12638a 100%
        );

        border-radius: 14px;

        padding: 22px 28px;

        color: white;

        box-shadow:
            0 8px 25px rgba(0,0,0,0.12);

        margin-bottom: 15px;
    }

    .top-header-title {

        font-size: 30px;

        font-weight: 800;

        letter-spacing: 0.5px;
    }

    .top-header-subtitle {

        font-size: 14px;

        opacity: 0.78;

        margin-top: 5px;
    }

    .online-pill {

        display: inline-block;

        padding: 5px 12px;

        border-radius: 20px;

        background: rgba(255,255,255,0.13);

        font-size: 12px;

        font-weight: 700;

        margin-top: 12px;
    }

    /* KPI */

    .kpi-card {

        background: white;

        border: 1px solid #dce3ea;

        border-radius: 12px;

        padding: 15px 16px;

        min-height: 105px;

        box-shadow:
            0 3px 10px rgba(0,0,0,0.045);
    }

    /* PHASE CARDS */

    .phase-card {

        border-radius: 12px;

        padding: 17px;

        min-height: 165px;

        border: 1px solid #d8dee6;

        background: white;

        position: relative;
    }

    .phase-running {

        border: 2px solid #f59e0b;

        background: linear-gradient(
            145deg,
            #fffaf0,
            #ffffff
        );

        box-shadow:
            0 0 0 3px rgba(245,158,11,0.10),
            0 8px 25px rgba(0,0,0,0.08);
    }

    .phase-complete {

        border: 2px solid #22c55e;

        background: linear-gradient(
            145deg,
            #f0fdf4,
            #ffffff
        );
    }

    .phase-pending {

        opacity: 0.58;

        background: #f8fafc;
    }

    .phase-number {

        font-size: 12px;

        font-weight: 800;

        color: #667085;

        letter-spacing: 1px;
    }

    .phase-title {

        font-size: 17px;

        font-weight: 800;

        margin-top: 8px;

        color: #102a43;
    }

    .phase-description {

        font-size: 12px;

        color: #667085;

        margin-top: 7px;

        line-height: 1.5;
    }

    .phase-status {

        margin-top: 12px;

        font-size: 11px;

        font-weight: 800;

        letter-spacing: 0.6px;
    }

    /* OPERATION PANEL */

    .operation-panel {

        background: #ffffff;

        border-radius: 13px;

        border: 1px solid #dce3ea;

        padding: 18px 20px;

        box-shadow:
            0 3px 12px rgba(0,0,0,0.045);
    }

    .operation-title {

        color: #102a43;

        font-size: 22px;

        font-weight: 800;
    }

    .operation-subtitle {

        color: #667085;

        font-size: 13px;

        margin-top: 4px;
    }

    /* MACHINE STATUS */

    .machine-box {

        background: white;

        border: 1px solid #dce3ea;

        border-radius: 12px;

        padding: 14px;

        min-height: 105px;
    }

    .machine-name {

        font-size: 12px;

        color: #667085;

        font-weight: 700;

        text-transform: uppercase;
    }

    .machine-status {

        font-size: 17px;

        font-weight: 800;

        margin-top: 8px;

        color: #102a43;
    }

    .machine-detail {

        font-size: 12px;

        color: #667085;

        margin-top: 5px;
    }

    /* TARGET ACHIEVEMENT */

    .target-achieved {

        background: linear-gradient(
            135deg,
            #064e3b,
            #047857
        );

        border-radius: 15px;

        padding: 25px;

        color: white;

        box-shadow:
            0 8px 30px rgba(4,120,87,0.25);

        margin-top: 15px;

        margin-bottom: 15px;
    }

    .target-achieved-title {

        font-size: 25px;

        font-weight: 900;
    }

    .target-achieved-text {

        font-size: 14px;

        opacity: 0.92;

        margin-top: 8px;
    }

    /* FOOTER */

    .footer {

        text-align: center;

        color: #98a2b3;

        font-size: 11px;

        padding-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# BACK-END SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ⚙️ ENGINEERING BACK END")

    st.caption(
        "Configuration parameters and control settings"
    )

    st.markdown("---")

    # MATERIAL

    st.markdown("### MATERIAL")

    material = st.selectbox(
        "Elastocaloric Material",
        [
            "NiTi Nitinol — Baseline",
            "Cu-Al-Ni Shape Memory Alloy",
            "Fe-Mn-Si Shape Memory Alloy",
            "Elastocaloric Polymer — Reference"
        ]
    )

    geometry = st.selectbox(
        "Core Geometry",
        [
            "NiTi Tube Bundle",
            "NiTi Wire Bundle",
            "Single NiTi Tube"
        ]
    )

    elements = st.number_input(
        "Active Elements",
        min_value=1,
        max_value=20,
        value=5,
        step=1
    )

    length = st.number_input(
        "Element Length (mm)",
        min_value=10.0,
        max_value=500.0,
        value=150.0,
        step=5.0
    )

    outer_diameter = st.number_input(
        "Outer Diameter (mm)",
        min_value=0.5,
        max_value=20.0,
        value=12.0,
        step=0.5
    )

    inner_diameter = st.number_input(
        "Inner Diameter (mm)",
        min_value=0.1,
        max_value=19.5,
        value=10.0,
        step=0.5
    )

    # MECHANICAL

    st.markdown("---")

    st.markdown("### MECHANICAL CONTROL")

    maximum_strain = st.slider(
        "Maximum Strain (%)",
        1.0,
        8.0,
        5.0,
        0.5
    )

    actuation = st.selectbox(
        "Actuation Mode",
        [
            "Uniaxial Tension",
            "Uniaxial Compression"
        ]
    )

    phase_duration = st.slider(
        "Cycle Phase Duration (s)",
        0.2,
        3.0,
        0.8,
        0.1
    )

    # THERMAL

    st.markdown("---")

    st.markdown("### THERMAL CONTROL")

    ambient_temperature = st.number_input(
        "Ambient Temperature (°C)",
        15.0,
        45.0,
        25.0,
        1.0
    )

    target_temperature = st.number_input(
        "Target Cold-Side Temperature (°C)",
        -30.0,
        20.0,
        -5.0,
        1.0
    )

    airflow = st.slider(
        "Forced-Air Flow (CFM)",
        10,
        100,
        50,
        5
    )

    # AUTOMATION

    st.markdown("---")

    st.markdown("### AUTOMATION")

    operating_mode = st.radio(
        "Control Mode",
        [
            "Manual",
            "Automatic"
        ]
    )

    st.info(
        "Automatic operation terminates immediately "
        "when the chamber reaches the target temperature."
    )


# ============================================================
# MATERIAL / THERMAL MODEL
# ============================================================

def material_factor():

    if "Cu-Al-Ni" in material:
        return 1.15

    if "Fe-Mn-Si" in material:
        return 0.85

    if "Polymer" in material:
        return 0.65

    return 1.0


def calculate_delta_temperature():

    return (
        12.5
        * material_factor()
        * (maximum_strain / 5.0)
        * np.sqrt(elements / 5.0)
    )


# ============================================================
# RESET
# ============================================================

def reset_system():

    st.session_state.stage = 0

    st.session_state.cycle_count = 0

    st.session_state.auto_running = False

    st.session_state.target_reached = False

    st.session_state.target_notification = False

    st.session_state.last_cycle_completed = False

    st.session_state.chamber_temp = ambient_temperature

    st.session_state.wire_temp = ambient_temperature

    st.session_state.air_temp = ambient_temperature

    st.session_state.chamber_max = ambient_temperature

    st.session_state.chamber_min = ambient_temperature

    st.session_state.wire_max = ambient_temperature

    st.session_state.wire_min = ambient_temperature

    st.session_state.air_max = ambient_temperature

    st.session_state.air_min = ambient_temperature

    st.session_state.stress = 0.0

    st.session_state.strain = 0.0

    st.session_state.cop = 0.0

    st.session_state.simulation_time = 0.0

    st.session_state.temperature_history = [
        ambient_temperature
    ]

    st.session_state.wire_history = [
        ambient_temperature
    ]

    st.session_state.air_history = [
        ambient_temperature
    ]

    st.session_state.stress_history = [0.0]

    st.session_state.strain_history = [0.0]

    st.session_state.time_history = [0.0]

    st.session_state.event_log = []

    st.session_state.system_message = (
        "SYSTEM READY"
    )

    st.session_state.last_update = (
        datetime.now().strftime("%H:%M:%S")
    )


# ============================================================
# EVENT LOG
# ============================================================

def log_event(message):

    timestamp = datetime.now().strftime("%H:%M:%S")

    st.session_state.event_log.insert(
        0,
        f"[{timestamp}] {message}"
    )

    if len(st.session_state.event_log) > 40:

        st.session_state.event_log.pop()


# ============================================================
# TEMPERATURE EXTREMES
# ============================================================

def update_temperature_limits():

    st.session_state.chamber_max = max(
        st.session_state.chamber_max,
        st.session_state.chamber_temp
    )

    st.session_state.chamber_min = min(
        st.session_state.chamber_min,
        st.session_state.chamber_temp
    )

    st.session_state.wire_max = max(
        st.session_state.wire_max,
        st.session_state.wire_temp
    )

    st.session_state.wire_min = min(
        st.session_state.wire_min,
        st.session_state.wire_temp
    )

    st.session_state.air_max = max(
        st.session_state.air_max,
        st.session_state.air_temp
    )

    st.session_state.air_min = min(
        st.session_state.air_min,
        st.session_state.air_temp
    )


# ============================================================
# TARGET CHECK
# ============================================================

def check_target_temperature():

    if (
        st.session_state.chamber_temp
        <= target_temperature
    ):

        # Clamp exactly to target
        st.session_state.chamber_temp = (
            target_temperature
        )

        st.session_state.target_reached = True

        # CRITICAL:
        # STOP AUTOMATION HERE
        st.session_state.auto_running = False

        # Notification flag
        st.session_state.target_notification = True

        st.session_state.system_message = (
            "TARGET TEMPERATURE ACHIEVED — "
            "AUTOMATION STOPPED"
        )

        log_event(
            "TARGET TEMPERATURE ACHIEVED."
        )

        log_event(
            "AUTOMATIC CYCLING STOPPED."
        )

        log_event(
            "SYSTEM ENTERED TARGET HOLD STATE."
        )

        return True

    return False


# ============================================================
# EXECUTE PHASE
# ============================================================

def execute_phase(phase):

    # Never execute another phase after target achievement
    if st.session_state.target_reached:

        return False

    delta_t = calculate_delta_temperature()

    # ========================================================
    # PHASE 1 — MECHANICAL LOADING
    # ========================================================

    if phase == 1:

        st.session_state.stage = 1

        st.session_state.strain = maximum_strain

        stress_value = (
            450
            + maximum_strain * 12
        )

        if actuation == "Uniaxial Compression":

            stress_value *= -1

        st.session_state.stress = stress_value

        st.session_state.wire_temp = (
            ambient_temperature + delta_t
        )

        st.session_state.air_temp = (
            ambient_temperature
            + delta_t * 0.25
        )

        st.session_state.system_message = (
            "MECHANICAL LOADING ACTIVE"
        )

        log_event(
            "PHASE 1 — Mechanical loading initiated."
        )

    # ========================================================
    # PHASE 2 — HEAT REJECTION
    # ========================================================

    elif phase == 2:

        st.session_state.stage = 2

        st.session_state.strain = maximum_strain

        if actuation == "Uniaxial Compression":

            st.session_state.stress = -410

        else:

            st.session_state.stress = 410

        heat_removal = (
            airflow / 100.0
        ) * 0.65

        st.session_state.wire_temp -= (
            st.session_state.wire_temp
            - ambient_temperature
        ) * heat_removal

        st.session_state.air_temp -= (
            st.session_state.air_temp
            - ambient_temperature
        ) * heat_removal

        st.session_state.system_message = (
            "HEAT REJECTION ACTIVE"
        )

        log_event(
            "PHASE 2 — Forced-air heat rejection active."
        )

    # ========================================================
    # PHASE 3 — MECHANICAL UNLOADING
    # ========================================================

    elif phase == 3:

        st.session_state.stage = 3

        st.session_state.strain = 0.0

        st.session_state.stress = 120.0

        st.session_state.wire_temp = (
            ambient_temperature - delta_t
        )

        st.session_state.air_temp = (
            ambient_temperature
            - delta_t * 0.40
        )

        st.session_state.system_message = (
            "ELASTOCALORIC COOLING ACTIVE"
        )

        log_event(
            "PHASE 3 — NiTi unloading / cooling event."
        )

    # ========================================================
    # PHASE 4 — COLD-SIDE RECOVERY
    # ========================================================

    elif phase == 4:

        st.session_state.stage = 4

        st.session_state.strain = 0.0

        st.session_state.stress = 0.0

        cooling_factor = (
            0.035
            * (airflow / 50.0)
            * np.sqrt(elements / 5.0)
        )

        temperature_difference = (
            st.session_state.chamber_temp
            - target_temperature
        )

        if temperature_difference > 0:

            st.session_state.chamber_temp -= (
                temperature_difference
                * cooling_factor
            )

        cooling_load = max(
            0.0,
            ambient_temperature
            - st.session_state.chamber_temp
        )

        mechanical_input = max(
            0.1,
            maximum_strain * 0.4
        )

        st.session_state.cop = (
            cooling_load
            / mechanical_input
        )

        st.session_state.system_message = (
            "COLD-SIDE RECOVERY ACTIVE"
        )

        # CRITICAL TARGET CHECK
        if check_target_temperature():

            return True

        log_event(
            "PHASE 4 — Cold-side recovery active."
        )

    # ========================================================
    # DATA LOGGING
    # ========================================================

    st.session_state.simulation_time += (
        phase_duration
    )

    st.session_state.temperature_history.append(
        st.session_state.chamber_temp
    )

    st.session_state.wire_history.append(
        st.session_state.wire_temp
    )

    st.session_state.air_history.append(
        st.session_state.air_temp
    )

    st.session_state.stress_history.append(
        st.session_state.stress
    )

    st.session_state.strain_history.append(
        st.session_state.strain
    )

    st.session_state.time_history.append(
        st.session_state.simulation_time
    )

    max_points = 150

    histories = [
        st.session_state.temperature_history,
        st.session_state.wire_history,
        st.session_state.air_history,
        st.session_state.stress_history,
        st.session_state.strain_history,
        st.session_state.time_history
    ]

    for history in histories:

        if len(history) > max_points:

            history.pop(0)

    update_temperature_limits()

    st.session_state.last_update = (
        datetime.now().strftime("%H:%M:%S")
    )

    return False


# ============================================================
# APPLICATION HEADER
# ============================================================

st.markdown(
    f"""
    <div class="top-header">

        <div class="top-header-title">
            ❄ {APP_NAME}
        </div>

        <div class="top-header-subtitle">
            {APP_SUBTITLE}
            &nbsp; | &nbsp;
            RESEARCH SCADA / HMI
        </div>

        <div class="online-pill">
            ● CONTROLLER ONLINE
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TOP STATUS BAR
# ============================================================

status1, status2, status3, status4 = st.columns(4)

with status1:

    if st.session_state.target_reached:

        st.success("🎯 TARGET ACHIEVED")

    elif st.session_state.auto_running:

        st.warning("🟠 AUTOMATIC RUNNING")

    else:

        st.info("🟢 SYSTEM READY")


with status2:

    st.metric(
        "COMPLETED CYCLES",
        st.session_state.cycle_count
    )


with status3:

    st.metric(
        "ACTIVE STAGE",
        PHASES[
            st.session_state.stage
        ]["short"]
    )


with status4:

    st.metric(
        "CONTROL TIME",
        f"{st.session_state.simulation_time:.1f} s"
    )


# ============================================================
# FRONT-END SELECTED CONFIGURATION
# ============================================================

st.markdown("---")

st.markdown(
    "### 🔎 ACTIVE SYSTEM CONFIGURATION"
)

c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:

    st.metric(
        "MATERIAL",
        "NiTi"
        if "NiTi" in material
        else material.split(" ")[0]
    )

with c2:

    st.metric(
        "GEOMETRY",
        geometry.replace("NiTi ", "")
    )

with c3:

    st.metric(
        "ELEMENTS",
        elements
    )

with c4:

    st.metric(
        "OD / ID",
        f"{outer_diameter:.1f} / "
        f"{inner_diameter:.1f} mm"
    )

with c5:

    st.metric(
        "LENGTH",
        f"{length:.0f} mm"
    )

with c6:

    st.metric(
        "MAX STRAIN",
        f"{maximum_strain:.1f}%"
    )


# ============================================================
# MASTER CONTROL
# ============================================================

st.markdown("---")

st.markdown("### 🎛 MASTER CONTROL")

mc1, mc2, mc3, mc4, mc5 = st.columns(5)

with mc1:

    if operating_mode == "Automatic":

        if (
            not st.session_state.auto_running
            and not st.session_state.target_reached
        ):

            if st.button(
                "▶ START AUTOMATION",
                type="primary",
                use_container_width=True
            ):

                st.session_state.auto_running = True

                st.session_state.target_notification = False

                log_event(
                    "Automatic cycling started by operator."
                )

                st.rerun()

        elif st.session_state.auto_running:

            if st.button(
                "■ STOP AUTOMATION",
                use_container_width=True
            ):

                st.session_state.auto_running = False

                log_event(
                    "Automatic cycling stopped manually."
                )

                st.rerun()

        else:

            st.success(
                "TARGET ACHIEVED — RESET REQUIRED"
            )

    else:

        st.info("MANUAL CONTROL MODE")


with mc2:

    if st.button(
        "↻ RESET CONTROLLER",
        use_container_width=True
    ):

        reset_system()

        st.rerun()


with mc3:

    st.metric(
        "TARGET",
        f"{target_temperature:.1f} °C"
    )


with mc4:

    difference = (
        st.session_state.chamber_temp
        - target_temperature
    )

    st.metric(
        "TARGET DIFFERENCE",
        f"{difference:.2f} °C"
    )


with mc5:

    if st.session_state.target_reached:

        st.success("HOLD STATE")

    elif st.session_state.auto_running:

        st.warning("AUTO ACTIVE")

    else:

        st.info("READY")


# ============================================================
# TARGET ACHIEVED NOTIFICATION
# ============================================================

if st.session_state.target_notification:

    st.markdown(
        f"""
        <div class="target-achieved">

            <div class="target-achieved-title">
                🎯 TARGET TEMPERATURE ACHIEVED
            </div>

            <div class="target-achieved-text">

                The enclosed chamber reached the configured
                target temperature of

                <b>{target_temperature:.2f} °C</b>.

                <br><br>

                Measured chamber temperature:
                <b>{st.session_state.chamber_temp:.2f} °C</b>

                <br><br>

                <b>
                AUTOMATIC CYCLING HAS STOPPED.
                </b>

                <br>

                The controller is now holding the system
                in the target-achieved state.

                <br><br>

                Press <b>RESET CONTROLLER</b> before
                starting another automatic run.

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# OPERATION SEQUENCE
# ============================================================

st.markdown("---")

st.markdown(
    "### 🚦 ELASTOCALORIC OPERATION SEQUENCE"
)

stage_columns = st.columns(4)

for index, phase_number in enumerate([1, 2, 3, 4]):

    phase = PHASES[phase_number]

    with stage_columns[index]:

        # RUNNING
        if (
            st.session_state.stage
            == phase_number
            and not st.session_state.target_reached
        ):

            st.markdown(
                f"""
                <div class="phase-card phase-running">

                    <div class="phase-number">
                        PHASE {phase_number}
                    </div>

                    <div class="phase-title">
                        {phase["icon"]}
                        {phase["name"]}
                    </div>

                    <div class="phase-description">
                        {phase["description"]}
                    </div>

                    <div class="phase-status"
                         style="color:#d97706;">

                        ● CURRENT OPERATION

                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        # COMPLETED
        elif (
            st.session_state.stage
            > phase_number
        ):

            st.markdown(
                f"""
                <div class="phase-card phase-complete">

                    <div class="phase-number">
                        PHASE {phase_number}
                    </div>

                    <div class="phase-title">
                        ✓ {phase["name"]}
                    </div>

                    <div class="phase-description">
                        {phase["description"]}
                    </div>

                    <div class="phase-status"
                         style="color:#16a34a;">

                        ✓ COMPLETED

                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        # TARGET HOLD
        elif (
            st.session_state.target_reached
            and phase_number
            == st.session_state.stage
        ):

            st.markdown(
                f"""
                <div class="phase-card phase-complete">

                    <div class="phase-number">
                        PHASE {phase_number}
                    </div>

                    <div class="phase-title">
                        ✓ {phase["name"]}
                    </div>

                    <div class="phase-description">
                        Target reached during this phase.
                    </div>

                    <div class="phase-status"
                         style="color:#047857;">

                        🎯 TARGET ACHIEVED

                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        # PENDING
        else:

            st.markdown(
                f"""
                <div class="phase-card phase-pending">

                    <div class="phase-number">
                        PHASE {phase_number}
                    </div>

                    <div class="phase-title">
                        {phase["name"]}
                    </div>

                    <div class="phase-description">
                        {phase["description"]}
                    </div>

                    <div class="phase-status"
                         style="color:#98a2b3;">

                        ○ PENDING

                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# CURRENT OPERATION
# ============================================================

st.markdown("---")

current_phase = PHASES[
    st.session_state.stage
]

st.markdown(
    f"""
    <div class="operation-panel">

        <div class="operation-title">

            CURRENT OPERATION:
            {current_phase["name"]}

        </div>

        <div class="operation-subtitle">

            {current_phase["description"]}

            &nbsp; | &nbsp;

            Last update:
            {st.session_state.last_update}

        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# THERMAL MONITORING
# ============================================================

st.markdown("---")

st.markdown(
    "### 🌡 LIVE THERMAL MONITORING"
)

t1, t2, t3, t4, t5, t6 = st.columns(6)

with t1:

    st.metric(
        "CHAMBER",
        f"{st.session_state.chamber_temp:.2f} °C"
    )

with t2:

    st.metric(
        "CHAMBER MAX",
        f"{st.session_state.chamber_max:.2f} °C"
    )

with t3:

    st.metric(
        "CHAMBER MIN",
        f"{st.session_state.chamber_min:.2f} °C"
    )

with t4:

    st.metric(
        "NiTi CURRENT",
        f"{st.session_state.wire_temp:.2f} °C"
    )

with t5:

    st.metric(
        "NiTi MAX",
        f"{st.session_state.wire_max:.2f} °C"
    )

with t6:

    st.metric(
        "NiTi MIN",
        f"{st.session_state.wire_min:.2f} °C"
    )


# ============================================================
# MECHANICAL CONDITION
# ============================================================

st.markdown("---")

st.markdown(
    "### ⚙ MECHANICAL CONDITION"
)

m1, m2, m3, m4 = st.columns(4)

with m1:

    st.metric(
        "STRAIN",
        f"{st.session_state.strain:.2f} %"
    )

with m2:

    st.metric(
        "STRESS",
        f"{st.session_state.stress:.1f} MPa"
    )

with m3:

    st.metric(
        "ESTIMATED COP",
        f"{st.session_state.cop:.3f}"
    )

with m4:

    st.metric(
        "MODEL ΔT",
        f"{calculate_delta_temperature():.2f} °C"
    )


# ============================================================
# PROCESS EQUIPMENT
# ============================================================

st.markdown("---")

st.markdown(
    "### 🔧 PROCESS EQUIPMENT STATUS"
)

e1, e2, e3, e4, e5 = st.columns(5)

with e1:

    actuator_state = (
        "HOLD"
        if st.session_state.target_reached
        else "ACTIVE"
    )

    st.markdown(
        f"""
        <div class="machine-box">

        <div class="machine-name">
        ACTUATOR
        </div>

        <div class="machine-status">
        {actuator_state}
        </div>

        <div class="machine-detail">
        Mechanical drive
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with e2:

    if st.session_state.stage == 2:

        valve_state = "OPEN"

        valve_detail = "Heat rejection"

    else:

        valve_state = "CLOSED"

        valve_detail = "Standby"

    st.markdown(
        f"""
        <div class="machine-box">

        <div class="machine-name">
        AIR VALVE
        </div>

        <div class="machine-status">
        {valve_state}
        </div>

        <div class="machine-detail">
        {valve_detail}
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with e3:

    if (
        st.session_state.stage == 2
        and not st.session_state.target_reached
    ):

        fan_state = "RUNNING"

    else:

        fan_state = "STANDBY"

    st.markdown(
        f"""
        <div class="machine-box">

        <div class="machine-name">
        COOLING FAN
        </div>

        <div class="machine-status">
        {fan_state}
        </div>

        <div class="machine-detail">
        {airflow} CFM
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with e4:

    if (
        st.session_state.stage == 3
        or st.session_state.stage == 4
    ):

        cold_state = "ACTIVE"

    else:

        cold_state = "STANDBY"

    st.markdown(
        f"""
        <div class="machine-box">

        <div class="machine-name">
        COLD-SIDE PATH
        </div>

        <div class="machine-status">
        {cold_state}
        </div>

        <div class="machine-detail">
        Thermal extraction
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with e5:

    if st.session_state.target_reached:

        controller_state = "TARGET HOLD"

    elif st.session_state.auto_running:

        controller_state = "AUTO RUN"

    else:

        controller_state = "READY"

    st.markdown(
        f"""
        <div class="machine-box">

        <div class="machine-name">
        CONTROLLER
        </div>

        <div class="machine-status">
        {controller_state}
        </div>

        <div class="machine-detail">
        Supervisory control
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MANUAL CONTROL
# ============================================================

if operating_mode == "Manual":

    st.markdown("---")

    st.markdown(
        "### 🕹 MANUAL PHASE CONTROL"
    )

    b1, b2, b3, b4 = st.columns(4)

    with b1:

        if st.button(
            "01  LOAD",
            use_container_width=True
        ):

            execute_phase(1)

            st.rerun()

    with b2:

        if st.button(
            "02  HEAT REJECT",
            use_container_width=True
        ):

            execute_phase(2)

            st.rerun()

    with b3:

        if st.button(
            "03  UNLOAD",
            use_container_width=True
        ):

            execute_phase(3)

            st.rerun()

    with b4:

        if st.button(
            "04  COLD RECOVERY",
            use_container_width=True
        ):

            target_hit = execute_phase(4)

            if not target_hit:

                st.session_state.cycle_count += 1

                log_event(
                    f"Manual cycle "
                    f"{st.session_state.cycle_count} completed."
                )

            st.rerun()


# ============================================================
# AUTOMATIC CYCLING ENGINE
# ============================================================

if (
    operating_mode == "Automatic"
    and st.session_state.auto_running
    and not st.session_state.target_reached
):

    target_hit = False

    # --------------------------------------------------------
    # EXECUTE ONE COMPLETE CYCLE
    # --------------------------------------------------------

    for phase in [1, 2, 3, 4]:

        # Stop immediately if operator stopped system
        if not st.session_state.auto_running:
            break

        # Stop immediately if target reached
        if st.session_state.target_reached:
            break

        # Execute phase
        target_hit = execute_phase(phase)

        # If target reached during this phase:
        # DO NOT CONTINUE TO NEXT PHASE.
        if target_hit:

            st.session_state.auto_running = False

            break

        # Phase delay
        time.sleep(phase_duration)

    # --------------------------------------------------------
    # TARGET WAS ACHIEVED
    # --------------------------------------------------------

    if st.session_state.target_reached:

        # IMPORTANT:
        # No new cycle is counted.
        # No new cycle is started.
        # Automation remains OFF.

        st.session_state.auto_running = False

        st.session_state.system_message = (
            "TARGET ACHIEVED — AUTOMATION STOPPED"
        )

        st.session_state.target_notification = True

        st.rerun()

    # --------------------------------------------------------
    # NORMAL CYCLE COMPLETED
    # --------------------------------------------------------

    elif st.session_state.auto_running:

        st.session_state.cycle_count += 1

        st.session_state.last_cycle_completed = True

        log_event(
            f"Automatic cycle "
            f"{st.session_state.cycle_count} completed."
        )

        st.rerun()


# ============================================================
# TARGET STATUS MESSAGE
# ============================================================

st.markdown("---")

if st.session_state.target_reached:

    st.markdown(
        f"""
        <div class="target-achieved">

            <div class="target-achieved-title">

                🎯 TARGET TEMPERATURE ACHIEVED

            </div>

            <div class="target-achieved-text">

                The enclosed chamber has reached the
                configured target temperature.

                <br><br>

                <b>Target:</b>
                {target_temperature:.2f} °C

                <br>

                <b>Measured:</b>
                {st.session_state.chamber_temp:.2f} °C

                <br><br>

                <b>
                ✓ AUTOMATIC CYCLING STOPPED
                </b>

                <br>

                ✓ Cycle counter frozen

                <br>

                ✓ Controller entered HOLD state

                <br>

                ✓ No further automatic cycles will start

                <br><br>

                Reset the controller to begin a new run.

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

elif st.session_state.auto_running:

    st.warning(
        f"""
        🔄 **AUTOMATIC CYCLING ACTIVE**

        Target:
        **{target_temperature:.2f} °C**

        Current chamber:
        **{st.session_state.chamber_temp:.2f} °C**

        The controller will automatically stop when
        the target temperature is reached.
        """
    )

else:

    st.info(
        "Controller ready. Select Manual or Automatic operation."
    )


# ============================================================
# LIVE GRAPHS
# ============================================================

st.markdown("---")

graph1, graph2 = st.columns(2)

# ------------------------------------------------------------
# THERMAL GRAPH
# ------------------------------------------------------------

with graph1:

    st.markdown(
        "### 📈 LIVE THERMAL RESPONSE"
    )

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    x = st.session_state.time_history

    ax.plot(
        x,
        st.session_state.temperature_history,
        linewidth=2,
        label="Chamber"
    )

    ax.plot(
        x,
        st.session_state.wire_history,
        linewidth=2,
        label="NiTi"
    )

    ax.plot(
        x,
        st.session_state.air_history,
        linewidth=2,
        label="Air"
    )

    ax.axhline(
        target_temperature,
        linestyle="--",
        linewidth=1.5,
        label="Target"
    )

    ax.axhline(
        ambient_temperature,
        linestyle=":",
        linewidth=1.2,
        label="Ambient"
    )

    ax.set_xlabel(
        "Time (s)"
    )

    ax.set_ylabel(
        "Temperature (°C)"
    )

    ax.grid(
        True,
        alpha=0.25
    )

    ax.legend()

    st.pyplot(fig)


# ------------------------------------------------------------
# STRESS / STRAIN GRAPH
# ------------------------------------------------------------

with graph2:

    st.markdown(
        "### 📊 MECHANICAL RESPONSE"
    )

    fig2, ax2 = plt.subplots(
        figsize=(8, 4)
    )

    ax2.plot(
        st.session_state.strain_history,
        st.session_state.stress_history,
        linewidth=2
    )

    ax2.scatter(
        [st.session_state.strain],
        [st.session_state.stress],
        s=70
    )

    ax2.set_xlabel(
        "Strain (%)"
    )

    ax2.set_ylabel(
        "Stress (MPa)"
    )

    ax2.grid(
        True,
        alpha=0.25
    )

    st.pyplot(fig2)


# ============================================================
# SENSOR EXTREMES
# ============================================================

st.markdown("---")

st.markdown(
    "### 🌡 SENSOR MEMORY — MAXIMUM / MINIMUM"
)

s1, s2, s3, s4, s5, s6 = st.columns(6)

with s1:

    st.metric(
        "CHAMBER CURRENT",
        f"{st.session_state.chamber_temp:.2f} °C"
    )

with s2:

    st.metric(
        "CHAMBER MAX",
        f"{st.session_state.chamber_max:.2f} °C"
    )

with s3:

    st.metric(
        "CHAMBER MIN",
        f"{st.session_state.chamber_min:.2f} °C"
    )

with s4:

    st.metric(
        "NiTi CURRENT",
        f"{st.session_state.wire_temp:.2f} °C"
    )

with s5:

    st.metric(
        "NiTi MAX",
        f"{st.session_state.wire_max:.2f} °C"
    )

with s6:

    st.metric(
        "NiTi MIN",
        f"{st.session_state.wire_min:.2f} °C"
    )


# ============================================================
# AIR SENSOR
# ============================================================

a1, a2 = st.columns(2)

with a1:

    st.metric(
        "AIR CURRENT",
        f"{st.session_state.air_temp:.2f} °C"
    )

with a2:

    st.metric(
        "AIR MAX / MIN",
        f"{st.session_state.air_max:.2f} / "
        f"{st.session_state.air_min:.2f} °C"
    )


# ============================================================
# SAFETY / INTERLOCK
# ============================================================

st.markdown("---")

st.markdown(
    "### 🚨 SAFETY & INTERLOCK MONITOR"
)

safe1, safe2, safe3, safe4 = st.columns(4)

strain_alarm = maximum_strain > 6.0

stress_alarm = (
    abs(st.session_state.stress)
    > 550
)

temperature_alarm = (
    st.session_state.chamber_temp
    < target_temperature - 3
)

with safe1:

    if strain_alarm:

        st.error(
            "⚠ STRAIN LIMIT"
        )

    else:

        st.success(
            "✓ STRAIN NORMAL"
        )


with safe2:

    if stress_alarm:

        st.error(
            "⚠ STRESS LIMIT"
        )

    else:

        st.success(
            "✓ STRESS NORMAL"
        )


with safe3:

    if temperature_alarm:

        st.warning(
            "TARGET OVERSHOOT"
        )

    else:

        st.success(
            "✓ THERMAL NORMAL"
        )


with safe4:

    if st.session_state.target_reached:

        st.success(
            "✓ TARGET HOLD"
        )

    elif st.session_state.auto_running:

        st.warning(
            "AUTO CONTROL"
        )

    else:

        st.success(
            "✓ CONTROLLER READY"
        )


# ============================================================
# EVENT LOG
# ============================================================

st.markdown("---")

log1, log2 = st.columns([2, 1])

with log1:

    st.markdown(
        "### 🧾 CONTROLLER EVENT LOG"
    )

    if st.session_state.event_log:

        st.code(
            "\n".join(
                st.session_state.event_log
            ),
            language="text"
        )

    else:

        st.info(
            "No events recorded."
        )


with log2:

    st.markdown(
        "### 📡 CONTROLLER STATUS"
    )

    st.write(
        f"**Mode:** {operating_mode}"
    )

    st.write(
        "**Controller:** ONLINE"
    )

    st.write(
        f"**Stage:** {current_phase['name']}"
    )

    st.write(
        f"**Target:** {target_temperature:.1f} °C"
    )

    st.write(
        f"**Chamber:** "
        f"{st.session_state.chamber_temp:.2f} °C"
    )

    st.write(
        f"**Cycles:** "
        f"{st.session_state.cycle_count}"
    )

    if st.session_state.target_reached:

        st.success(
            "TARGET HOLD"
        )

    elif st.session_state.auto_running:

        st.warning(
            "AUTOMATIC RUN"
        )

    else:

        st.info(
            "STANDBY"
        )


# ============================================================
# PERFORMANCE SUMMARY
# ============================================================

st.markdown("---")

st.markdown(
    "### 📋 SYSTEM PERFORMANCE SUMMARY"
)

performance_data = {

    "Parameter": [

        "Material",

        "Geometry",

        "Active Elements",

        "Element Length",

        "Outer Diameter",

        "Inner Diameter",

        "Maximum Strain",

        "Air Flow",

        "Ambient Temperature",

        "Target Temperature",

        "Current Chamber Temperature",

        "Chamber Maximum",

        "Chamber Minimum",

        "NiTi Maximum",

        "NiTi Minimum",

        "Cycles Completed",

        "Estimated COP"
    ],

    "Value": [

        material,

        geometry,

        f"{elements}",

        f"{length:.1f} mm",

        f"{outer_diameter:.1f} mm",

        f"{inner_diameter:.1f} mm",

        f"{maximum_strain:.1f} %",

        f"{airflow} CFM",

        f"{ambient_temperature:.1f} °C",

        f"{target_temperature:.1f} °C",

        f"{st.session_state.chamber_temp:.2f} °C",

        f"{st.session_state.chamber_max:.2f} °C",

        f"{st.session_state.chamber_min:.2f} °C",

        f"{st.session_state.wire_max:.2f} °C",

        f"{st.session_state.wire_min:.2f} °C",

        f"{st.session_state.cycle_count}",

        f"{st.session_state.cop:.3f}"
    ]
}

df = pd.DataFrame(
    performance_data
)

st.dataframe(
    df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# RESEARCH MODEL NOTICE
# ============================================================

st.markdown("---")

st.warning(
    """
    RESEARCH MODEL NOTICE

    This application is a supervisory SCADA/HMI prototype.
    Thermal and mechanical values are representative model
    outputs and should not be interpreted as experimentally
    validated NiTi material-property measurements.

    For experimental validation, measured NiTi transformation
    temperatures, stress-strain behaviour, hysteresis,
    heat-transfer coefficients, thermal mass, actuator
    dynamics, cycle frequency and fatigue characteristics
    should be incorporated.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

    NI-Ti ELASTOCALORIC REFRIGERATION SCADA |
    Academic Research Prototype |
    Supervisory Monitoring & Control Interface

    </div>
    """,
    unsafe_allow_html=True
)
