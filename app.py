import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NiTi Elastocaloric SCADA",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PROFESSIONAL UI STYLE
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #eef2f6;
    }

    .block-container {
        max-width: 1550px;
        padding-top: 1rem;
        padding-bottom: 3rem;
    }

    /* HEADER */

    .header {
        background: linear-gradient(
            135deg,
            #071a2c,
            #0b3554,
            #116b8c
        );

        padding: 25px 30px;
        border-radius: 16px;
        color: white;
        margin-bottom: 18px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }

    .header-title {
        font-size: 30px;
        font-weight: 800;
        letter-spacing: 0.5px;
    }

    .header-subtitle {
        margin-top: 6px;
        font-size: 14px;
        opacity: 0.82;
    }

    .online {
        display: inline-block;
        margin-top: 13px;
        padding: 5px 13px;
        border-radius: 20px;
        background: rgba(255,255,255,0.14);
        font-size: 12px;
        font-weight: 700;
    }

    /* SECTION */

    .section-title {
        font-size: 19px;
        font-weight: 800;
        color: #102a43;
        margin-top: 12px;
        margin-bottom: 10px;
    }

    /* OPERATION */

    .operation {
        background: white;
        border: 1px solid #dce3ea;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 3px 12px rgba(0,0,0,0.04);
    }

    .operation-name {
        font-size: 22px;
        font-weight: 800;
        color: #102a43;
    }

    .operation-description {
        color: #667085;
        font-size: 13px;
        margin-top: 6px;
    }

    /* TARGET */

    .target-box {
        background: linear-gradient(
            135deg,
            #064e3b,
            #047857
        );

        color: white;
        border-radius: 15px;
        padding: 23px;
        margin-top: 15px;
        margin-bottom: 15px;
        box-shadow: 0 8px 28px rgba(4,120,87,0.22);
    }

    .target-title {
        font-size: 25px;
        font-weight: 900;
    }

    .target-text {
        font-size: 14px;
        margin-top: 8px;
        line-height: 1.65;
    }

    /* PHASE */

    .phase {
        min-height: 155px;
        padding: 17px;
        border-radius: 13px;
        background: white;
        border: 1px solid #d8dee6;
    }

    .phase-current {
        border: 2px solid #f59e0b;
        background: #fffaf0;
        box-shadow: 0 0 0 3px rgba(245,158,11,0.08);
    }

    .phase-complete {
        border: 2px solid #22c55e;
        background: #f0fdf4;
    }

    .phase-pending {
        opacity: 0.55;
    }

    .phase-number {
        font-size: 11px;
        font-weight: 800;
        color: #667085;
        letter-spacing: 1px;
    }

    .phase-name {
        font-size: 16px;
        font-weight: 800;
        color: #102a43;
        margin-top: 8px;
    }

    .phase-description {
        font-size: 12px;
        color: #667085;
        margin-top: 7px;
        line-height: 1.45;
    }

    .phase-status {
        font-size: 11px;
        font-weight: 800;
        margin-top: 10px;
    }

    /* STATUS */

    .status-card {
        background: white;
        border: 1px solid #dce3ea;
        border-radius: 12px;
        padding: 15px;
        min-height: 100px;
    }

    .status-label {
        font-size: 11px;
        color: #667085;
        font-weight: 800;
        text-transform: uppercase;
    }

    .status-value {
        font-size: 19px;
        font-weight: 800;
        color: #102a43;
        margin-top: 7px;
    }

    .status-detail {
        font-size: 12px;
        color: #667085;
        margin-top: 5px;
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
# APPLICATION DATA
# ============================================================

PHASES = {
    1: {
        "name": "MECHANICAL LOADING",
        "description": "NiTi element is mechanically loaded to the selected strain.",
        "icon": "⚙️"
    },

    2: {
        "name": "HEAT REJECTION",
        "description": "Generated heat is rejected using forced-air heat transfer.",
        "icon": "↗️"
    },

    3: {
        "name": "MECHANICAL UNLOADING",
        "description": "NiTi is unloaded and the elastocaloric cooling effect is generated.",
        "icon": "❄️"
    },

    4: {
        "name": "COLD-SIDE RECOVERY",
        "description": "Cooling is transferred to the enclosed cold-side chamber.",
        "icon": "◆"
    }
}


# ============================================================
# SESSION STATE
# ============================================================

def initialize_state():

    initial = {

        "stage": 0,

        "cycle_count": 0,

        "automatic": False,

        "target_reached": False,

        "chamber_temp": 25.0,

        "niti_temp": 25.0,

        "air_temp": 25.0,

        "stress": 0.0,

        "strain": 0.0,

        "cop": 0.0,

        "time": 0.0,

        "last_phase": 0,

        "message": "SYSTEM READY",

        "events": [],

        "temperature_time": [0.0],

        "chamber_history": [25.0],

        "niti_history": [25.0],

        "air_history": [25.0],

        "stress_history": [0.0],

        "strain_history": [0.0],

        "notification_shown": False
    }

    for key, value in initial.items():

        if key not in st.session_state:

            st.session_state[key] = value


initialize_state()


# ============================================================
# SIDEBAR — ENGINEERING BACK END
# ============================================================

with st.sidebar:

    st.markdown("## ⚙️ ENGINEERING CONFIGURATION")

    st.caption(
        "Back-end parameters / supervisory control"
    )

    st.divider()

    st.markdown("### MATERIAL")

    material = st.selectbox(
        "Material",
        [
            "NiTi Nitinol",
            "Cu-Al-Ni SMA",
            "Fe-Mn-Si SMA",
            "Elastocaloric Polymer"
        ]
    )

    geometry = st.selectbox(
        "Geometry",
        [
            "NiTi Tube Bundle",
            "NiTi Wire Bundle",
            "Single NiTi Tube"
        ]
    )

    elements = st.number_input(
        "Number of Active Elements",
        min_value=1,
        max_value=20,
        value=5,
        step=1
    )

    length = st.number_input(
        "Element Length (mm)",
        min_value=20.0,
        max_value=500.0,
        value=150.0,
        step=5.0
    )

    outer_diameter = st.number_input(
        "Outer Diameter (mm)",
        min_value=0.5,
        max_value=30.0,
        value=12.0,
        step=0.5
    )

    inner_diameter = st.number_input(
        "Inner Diameter (mm)",
        min_value=0.1,
        max_value=29.0,
        value=10.0,
        step=0.5
    )

    st.divider()

    st.markdown("### MECHANICAL")

    max_strain = st.slider(
        "Maximum Strain (%)",
        1.0,
        8.0,
        5.0,
        0.5
    )

    actuation = st.selectbox(
        "Actuation",
        [
            "Uniaxial Tension",
            "Uniaxial Compression"
        ]
    )

    phase_time = st.slider(
        "Phase Time (s)",
        0.2,
        3.0,
        0.8,
        0.1
    )

    st.divider()

    st.markdown("### THERMAL")

    ambient = st.number_input(
        "Ambient Temperature (°C)",
        15.0,
        45.0,
        25.0,
        1.0
    )

    target = st.number_input(
        "Target Chamber Temperature (°C)",
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

    st.divider()

    st.markdown("### CONTROL MODE")

    mode = st.radio(
        "Operating Mode",
        ["Manual", "Automatic"]
    )

    st.info(
        "Automatic operation stops when the chamber "
        "reaches the target temperature."
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def add_event(message):

    stamp = datetime.now().strftime("%H:%M:%S")

    st.session_state.events.insert(
        0,
        f"[{stamp}] {message}"
    )

    st.session_state.events = (
        st.session_state.events[:30]
    )


def material_factor():

    if material == "Cu-Al-Ni SMA":
        return 1.1

    if material == "Fe-Mn-Si SMA":
        return 0.85

    if material == "Elastocaloric Polymer":
        return 0.65

    return 1.0


def model_delta_t():

    return (
        12.0
        * material_factor()
        * (max_strain / 5.0)
        * np.sqrt(elements / 5.0)
    )


def append_history():

    st.session_state.time += phase_time

    st.session_state.temperature_time.append(
        st.session_state.time
    )

    st.session_state.chamber_history.append(
        st.session_state.chamber_temp
    )

    st.session_state.niti_history.append(
        st.session_state.niti_temp
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

    # Keep every history exactly the same length.
    max_points = 120

    histories = [
        st.session_state.temperature_time,
        st.session_state.chamber_history,
        st.session_state.niti_history,
        st.session_state.air_history,
        st.session_state.stress_history,
        st.session_state.strain_history
    ]

    for history in histories:

        if len(history) > max_points:

            del history[0]


def reset_controller():

    st.session_state.stage = 0

    st.session_state.cycle_count = 0

    st.session_state.automatic = False

    st.session_state.target_reached = False

    st.session_state.notification_shown = False

    st.session_state.chamber_temp = ambient

    st.session_state.niti_temp = ambient

    st.session_state.air_temp = ambient

    st.session_state.stress = 0.0

    st.session_state.strain = 0.0

    st.session_state.cop = 0.0

    st.session_state.time = 0.0

    st.session_state.last_phase = 0

    st.session_state.message = "SYSTEM READY"

    st.session_state.events = []

    st.session_state.temperature_time = [0.0]

    st.session_state.chamber_history = [ambient]

    st.session_state.niti_history = [ambient]

    st.session_state.air_history = [ambient]

    st.session_state.stress_history = [0.0]

    st.session_state.strain_history = [0.0]

    add_event("Controller reset by operator.")


# ============================================================
# PHASE ENGINE
# ============================================================

def run_phase(phase):

    if st.session_state.target_reached:

        return True

    delta = model_delta_t()

    st.session_state.last_phase = phase
    st.session_state.stage = phase

    # --------------------------------------------------------
    # PHASE 1
    # --------------------------------------------------------

    if phase == 1:

        st.session_state.strain = max_strain

        stress = (
            450.0
            + max_strain * 12.0
        )

        if actuation == "Uniaxial Compression":

            stress = -stress

        st.session_state.stress = stress

        st.session_state.niti_temp = (
            ambient + delta
        )

        st.session_state.air_temp = (
            ambient + delta * 0.20
        )

        st.session_state.message = (
            "MECHANICAL LOADING ACTIVE"
        )

        add_event(
            "Phase 1 — Mechanical loading."
        )

    # --------------------------------------------------------
    # PHASE 2
    # --------------------------------------------------------

    elif phase == 2:

        st.session_state.strain = max_strain

        st.session_state.stress = (
            -410.0
            if actuation == "Uniaxial Compression"
            else 410.0
        )

        cooling = (
            airflow / 100.0
        ) * 0.65

        st.session_state.niti_temp -= (
            st.session_state.niti_temp - ambient
        ) * cooling

        st.session_state.air_temp -= (
            st.session_state.air_temp - ambient
        ) * cooling

        st.session_state.message = (
            "HEAT REJECTION ACTIVE"
        )

        add_event(
            "Phase 2 — Forced-air heat rejection."
        )

    # --------------------------------------------------------
    # PHASE 3
    # --------------------------------------------------------

    elif phase == 3:

        st.session_state.strain = 0.0

        st.session_state.stress = 120.0

        st.session_state.niti_temp = (
            ambient - delta
        )

        st.session_state.air_temp = (
            ambient - delta * 0.40
        )

        st.session_state.message = (
            "ELASTOCALORIC COOLING ACTIVE"
        )

        add_event(
            "Phase 3 — Mechanical unloading."
        )

    # --------------------------------------------------------
    # PHASE 4
    # --------------------------------------------------------

    elif phase == 4:

        st.session_state.strain = 0.0

        st.session_state.stress = 0.0

        factor = (
            0.035
            * (airflow / 50.0)
            * np.sqrt(elements / 5.0)
        )

        difference = (
            st.session_state.chamber_temp
            - target
        )

        if difference > 0:

            st.session_state.chamber_temp -= (
                difference * factor
            )

        cooling_load = max(
            0.0,
            ambient - st.session_state.chamber_temp
        )

        input_work = max(
            0.1,
            max_strain * 0.4
        )

        st.session_state.cop = (
            cooling_load / input_work
        )

        st.session_state.message = (
            "COLD-SIDE RECOVERY ACTIVE"
        )

        add_event(
            "Phase 4 — Cold-side recovery."
        )

    # --------------------------------------------------------
    # DATA RECORD
    # --------------------------------------------------------

    append_history()

    # --------------------------------------------------------
    # TARGET DETECTION
    # --------------------------------------------------------

    if (
        st.session_state.chamber_temp
        <= target
    ):

        st.session_state.chamber_temp = target

        st.session_state.target_reached = True

        st.session_state.automatic = False

        st.session_state.message = (
            "TARGET TEMPERATURE ACHIEVED"
        )

        add_event(
            "TARGET TEMPERATURE ACHIEVED."
        )

        add_event(
            "AUTOMATIC CYCLING STOPPED."
        )

        add_event(
            "SYSTEM ENTERED TARGET HOLD."
        )

        # Make sure the exact target value is
        # represented in the graph.
        if (
            st.session_state.chamber_history[-1]
            != target
        ):

            st.session_state.chamber_history[-1] = target

        return True

    return False


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="header">

        <div class="header-title">
            ❄ NiTi ELASTOCALORIC REFRIGERATION
        </div>

        <div class="header-subtitle">
            Advanced Elastocaloric Refrigeration Using NiTi Alloy
            &nbsp; | &nbsp;
            RESEARCH SCADA / HMI
        </div>

        <div class="online">
            ● CONTROLLER ONLINE
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TOP STATUS
# ============================================================

s1, s2, s3, s4 = st.columns(4)

with s1:

    if st.session_state.target_reached:

        st.success("🎯 TARGET ACHIEVED")

    elif st.session_state.automatic:

        st.warning("🟠 AUTOMATIC RUNNING")

    else:

        st.info("🟢 SYSTEM READY")


with s2:

    st.metric(
        "COMPLETED CYCLES",
        st.session_state.cycle_count
    )


with s3:

    if st.session_state.stage in PHASES:

        stage_name = PHASES[
            st.session_state.stage
        ]["name"]

    else:

        stage_name = "READY"

    st.metric(
        "ACTIVE STAGE",
        stage_name
    )


with s4:

    st.metric(
        "CHAMBER",
        f"{st.session_state.chamber_temp:.2f} °C"
    )


# ============================================================
# ACTIVE CONFIGURATION
# ============================================================

st.markdown(
    '<div class="section-title">🔎 ACTIVE SYSTEM CONFIGURATION</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    st.metric("MATERIAL", material.split()[0])

with c2:
    st.metric(
        "GEOMETRY",
        geometry.replace("NiTi ", "")
    )

with c3:
    st.metric("ELEMENTS", elements)

with c4:
    st.metric(
        "OD / ID",
        f"{outer_diameter:g} / {inner_diameter:g} mm"
    )

with c5:
    st.metric(
        "LENGTH",
        f"{length:g} mm"
    )

with c6:
    st.metric(
        "MAX STRAIN",
        f"{max_strain:.1f}%"
    )


# ============================================================
# MASTER CONTROLS
# ============================================================

st.markdown(
    '<div class="section-title">🎛 MASTER CONTROL</div>',
    unsafe_allow_html=True
)

b1, b2, b3, b4, b5 = st.columns(5)


with b1:

    if mode == "Automatic":

        if (
            not st.session_state.automatic
            and not st.session_state.target_reached
        ):

            if st.button(
                "▶ START AUTOMATION",
                type="primary",
                use_container_width=True
            ):

                st.session_state.automatic = True

                if st.session_state.stage == 0:

                    st.session_state.stage = 1

                add_event(
                    "Automatic operation started."
                )

                st.rerun()

        elif st.session_state.automatic:

            if st.button(
                "■ STOP AUTOMATION",
                use_container_width=True
            ):

                st.session_state.automatic = False

                add_event(
                    "Automatic operation stopped manually."
                )

                st.rerun()

        else:

            st.button(
                "🎯 TARGET HOLD",
                disabled=True,
                use_container_width=True
            )

    else:

        st.button(
            "MANUAL MODE",
            disabled=True,
            use_container_width=True
        )


with b2:

    if st.button(
        "↻ RESET CONTROLLER",
        use_container_width=True
    ):

        reset_controller()

        st.rerun()


with b3:

    st.metric(
        "TARGET",
        f"{target:.1f} °C"
    )


with b4:

    difference = (
        st.session_state.chamber_temp
        - target
    )

    st.metric(
        "TARGET GAP",
        f"{difference:.2f} °C"
    )


with b5:

    if st.session_state.target_reached:

        st.success("HOLD")

    elif st.session_state.automatic:

        st.warning("AUTO")

    else:

        st.info("READY")


# ============================================================
# TARGET ACHIEVED MESSAGE
# ============================================================

if st.session_state.target_reached:

    st.markdown(
        f"""
        <div class="target-box">

            <div class="target-title">
                🎯 TARGET TEMPERATURE ACHIEVED
            </div>

            <div class="target-text">

                The chamber has reached the configured
                target temperature.

                <br><br>

                <b>Target:</b>
                {target:.2f} °C

                <br>

                <b>Measured:</b>
                {st.session_state.chamber_temp:.2f} °C

                <br><br>

                <b>✓ AUTOMATIC CYCLING STOPPED</b>

                <br>
                ✓ No further automatic cycles will start

                <br>
                ✓ Cycle counter frozen

                <br>
                ✓ Controller is in TARGET HOLD

                <br><br>

                Press <b>RESET CONTROLLER</b> before starting
                another automatic run.

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# OPERATION SEQUENCE
# ============================================================

st.markdown(
    '<div class="section-title">🚦 ELASTOCALORIC OPERATION SEQUENCE</div>',
    unsafe_allow_html=True
)

phase_columns = st.columns(4)

for index in range(1, 5):

    phase = PHASES[index]

    with phase_columns[index - 1]:

        if st.session_state.target_reached:

            if index <= st.session_state.stage:

                css = "phase-complete"

                status = "✓ COMPLETED"

            else:

                css = "phase-pending"

                status = "○ NOT EXECUTED"

        elif index == st.session_state.stage:

            css = "phase-current"

            status = "● CURRENT OPERATION"

        elif index < st.session_state.stage:

            css = "phase-complete"

            status = "✓ COMPLETED"

        else:

            css = "phase-pending"

            status = "○ PENDING"

        st.markdown(
            f"""
            <div class="phase {css}">

                <div class="phase-number">
                    PHASE {index}
                </div>

                <div class="phase-name">
                    {phase["icon"]}
                    {phase["name"]}
                </div>

                <div class="phase-description">
                    {phase["description"]}
                </div>

                <div class="phase-status">
                    {status}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# CURRENT OPERATION
# ============================================================

st.markdown(
    '<div class="section-title">⚙️ CURRENT OPERATION</div>',
    unsafe_allow_html=True
)

if st.session_state.stage in PHASES:

    current = PHASES[
        st.session_state.stage
    ]

    operation_name = current["name"]

    operation_description = current["description"]

else:

    operation_name = "SYSTEM READY"

    operation_description = (
        "Waiting for operator command."
    )


st.markdown(
    f"""
    <div class="operation">

        <div class="operation-name">
            {operation_name}
        </div>

        <div class="operation-description">
            {operation_description}
        </div>

        <div class="operation-description">
            Controller message:
            <b>{st.session_state.message}</b>
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LIVE THERMAL MONITORING
# ============================================================

st.markdown(
    '<div class="section-title">🌡 LIVE THERMAL MONITORING</div>',
    unsafe_allow_html=True
)

t1, t2, t3, t4, t5, t6 = st.columns(6)

with t1:
    st.metric(
        "CHAMBER",
        f"{st.session_state.chamber_temp:.2f} °C"
    )

with t2:
    st.metric(
        "NiTi",
        f"{st.session_state.niti_temp:.2f} °C"
    )

with t3:
    st.metric(
        "AIR",
        f"{st.session_state.air_temp:.2f} °C"
    )

with t4:
    st.metric(
        "AMBIENT",
        f"{ambient:.2f} °C"
    )

with t5:
    st.metric(
        "TARGET",
        f"{target:.2f} °C"
    )

with t6:

    gap = (
        st.session_state.chamber_temp
        - target
    )

    st.metric(
        "TARGET GAP",
        f"{gap:.2f} °C"
    )


# ============================================================
# MECHANICAL CONDITION
# ============================================================

st.markdown(
    '<div class="section-title">⚙️ MECHANICAL CONDITION</div>',
    unsafe_allow_html=True
)

m1, m2, m3, m4 = st.columns(4)

with m1:

    st.metric(
        "STRAIN",
        f"{st.session_state.strain:.2f}%"
    )

with m2:

    st.metric(
        "STRESS",
        f"{st.session_state.stress:.1f} MPa"
    )

with m3:

    st.metric(
        "EST. COP",
        f"{st.session_state.cop:.3f}"
    )

with m4:

    st.metric(
        "MODEL ΔT",
        f"{model_delta_t():.2f} °C"
    )


# ============================================================
# EQUIPMENT STATUS
# ============================================================

st.markdown(
    '<div class="section-title">🔧 PROCESS EQUIPMENT STATUS</div>',
    unsafe_allow_html=True
)

e1, e2, e3, e4, e5 = st.columns(5)

equipment = [
    (
        "ACTUATOR",
        "HOLD"
        if st.session_state.target_reached
        else "READY"
        if not st.session_state.automatic
        else "ACTIVE",
        "Mechanical drive"
    ),

    (
        "AIR SYSTEM",
        "ACTIVE"
        if st.session_state.stage == 2
        else "STANDBY",
        f"{airflow} CFM"
    ),

    (
        "COLD-SIDE",
        "ACTIVE"
        if st.session_state.stage in [3, 4]
        else "STANDBY",
        "Thermal path"
    ),

    (
        "SENSORS",
        "ONLINE",
        "Temperature / force"
    ),

    (
        "CONTROLLER",
        "TARGET HOLD"
        if st.session_state.target_reached
        else "AUTO RUN"
        if st.session_state.automatic
        else "READY",
        "Supervisory control"
    )
]

for column, data in zip(
    [e1, e2, e3, e4, e5],
    equipment
):

    with column:

        st.markdown(
            f"""
            <div class="status-card">

                <div class="status-label">
                    {data[0]}
                </div>

                <div class="status-value">
                    {data[1]}
                </div>

                <div class="status-detail">
                    {data[2]}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# MANUAL CONTROLS
# ============================================================

if mode == "Manual":

    st.markdown(
        '<div class="section-title">🕹 MANUAL PHASE CONTROL</div>',
        unsafe_allow_html=True
    )

    p1, p2, p3, p4 = st.columns(4)

    with p1:

        if st.button(
            "01  LOAD",
            use_container_width=True
        ):

            run_phase(1)

            st.rerun()

    with p2:

        if st.button(
            "02  HEAT REJECT",
            use_container_width=True
        ):

            run_phase(2)

            st.rerun()

    with p3:

        if st.button(
            "03  UNLOAD",
            use_container_width=True
        ):

            run_phase(3)

            st.rerun()

    with p4:

        if st.button(
            "04  COLD RECOVERY",
            use_container_width=True
        ):

            target_hit = run_phase(4)

            if not target_hit:

                st.session_state.cycle_count += 1

                add_event(
                    f"Manual cycle "
                    f"{st.session_state.cycle_count} completed."
                )

            st.rerun()


# ============================================================
# AUTOMATIC CONTROLLER
#
# IMPORTANT:
# One phase is executed per Streamlit rerun.
# This prevents the page from freezing.
# ============================================================

if (
    mode == "Automatic"
    and st.session_state.automatic
    and not st.session_state.target_reached
):

    current_phase = st.session_state.stage

    if current_phase == 0:

        st.session_state.stage = 1

        st.rerun()

    else:

        target_hit = run_phase(
            current_phase
        )

        # ----------------------------------------------------
        # TARGET REACHED
        # ----------------------------------------------------

        if target_hit:

            st.session_state.automatic = False

            st.toast(
                "🎯 Target temperature achieved. "
                "Automatic cycling stopped.",
                icon="🎯"
            )

            st.rerun()

        # ----------------------------------------------------
        # MOVE TO NEXT PHASE
        # ----------------------------------------------------

        if current_phase < 4:

            st.session_state.stage = (
                current_phase + 1
            )

            st.rerun()

        # ----------------------------------------------------
        # COMPLETE ONE FULL CYCLE
        # ----------------------------------------------------

        else:

            st.session_state.cycle_count += 1

            add_event(
                f"Automatic cycle "
                f"{st.session_state.cycle_count} completed."
            )

            st.session_state.stage = 1

            st.rerun()


# ============================================================
# GRAPHS
# ============================================================

st.markdown(
    '<div class="section-title">📊 LIVE PROCESS TREND</div>',
    unsafe_allow_html=True
)

g1, g2 = st.columns(2)


# ============================================================
# THERMAL GRAPH
# ============================================================

with g1:

    st.markdown("#### 🌡 Thermal Response")

    # --------------------------------------------------------
    # SAFETY: force equal lengths
    # --------------------------------------------------------

    lengths = [
        len(st.session_state.temperature_time),
        len(st.session_state.chamber_history),
        len(st.session_state.niti_history),
        len(st.session_state.air_history)
    ]

    n = min(lengths)

    x = np.asarray(
        st.session_state.temperature_time[-n:]
    )

    chamber = np.asarray(
        st.session_state.chamber_history[-n:]
    )

    niti = np.asarray(
        st.session_state.niti_history[-n:]
    )

    air = np.asarray(
        st.session_state.air_history[-n:]
    )

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    ax.plot(
        x,
        chamber,
        linewidth=2,
        label="Chamber"
    )

    ax.plot(
        x,
        niti,
        linewidth=2,
        label="NiTi"
    )

    ax.plot(
        x,
        air,
        linewidth=2,
        label="Air"
    )

    ax.axhline(
        target,
        linestyle="--",
        linewidth=1.5,
        label="Target"
    )

    ax.axhline(
        ambient,
        linestyle=":",
        linewidth=1.2,
        label="Ambient"
    )

    ax.set_xlabel("Time (s)")

    ax.set_ylabel("Temperature (°C)")

    ax.grid(
        True,
        alpha=0.25
    )

    ax.legend()

    st.pyplot(
        fig,
        clear_figure=True
    )


# ============================================================
# STRESS-STRAIN GRAPH
# ============================================================

with g2:

    st.markdown("#### ⚙ Stress–Strain Response")

    lengths2 = [
        len(st.session_state.strain_history),
        len(st.session_state.stress_history)
    ]

    n2 = min(lengths2)

    strain_data = np.asarray(
        st.session_state.strain_history[-n2:]
    )

    stress_data = np.asarray(
        st.session_state.stress_history[-n2:]
    )

    fig2, ax2 = plt.subplots(
        figsize=(8, 4)
    )

    ax2.plot(
        strain_data,
        stress_data,
        linewidth=2
    )

    ax2.scatter(
        [st.session_state.strain],
        [st.session_state.stress],
        s=60
    )

    ax2.set_xlabel("Strain (%)")

    ax2.set_ylabel("Stress (MPa)")

    ax2.grid(
        True,
        alpha=0.25
    )

    st.pyplot(
        fig2,
        clear_figure=True
    )


# ============================================================
# EVENT LOG
# ============================================================

st.markdown(
    '<div class="section-title">🧾 CONTROLLER EVENT LOG</div>',
    unsafe_allow_html=True
)

log_col, info_col = st.columns([2, 1])

with log_col:

    if st.session_state.events:

        st.code(
            "\n".join(
                st.session_state.events
            ),
            language="text"
        )

    else:

        st.info(
            "No controller events recorded yet."
        )


with info_col:

    st.markdown("#### SYSTEM STATE")

    if st.session_state.target_reached:

        st.success(
            "🎯 TARGET HOLD"
        )

    elif st.session_state.automatic:

        st.warning(
            "🟠 AUTOMATIC OPERATION"
        )

    else:

        st.info(
            "🟢 READY"
        )

    st.write(
        f"**Mode:** {mode}"
    )

    st.write(
        f"**Cycles:** {st.session_state.cycle_count}"
    )

    st.write(
        f"**Stage:** {st.session_state.stage}"
    )

    st.write(
        f"**Target:** {target:.1f} °C"
    )

    st.write(
        f"**Chamber:** "
        f"{st.session_state.chamber_temp:.2f} °C"
    )


# ============================================================
# SYSTEM SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">📋 SYSTEM SUMMARY</div>',
    unsafe_allow_html=True
)

summary = pd.DataFrame(
    {
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
            "Current NiTi Temperature",
            "Current Air Temperature",
            "Stress",
            "Strain",
            "Completed Cycles",
            "Estimated COP",
            "Controller State"
        ],

        "Value": [
            material,
            geometry,
            elements,
            f"{length:.1f} mm",
            f"{outer_diameter:.1f} mm",
            f"{inner_diameter:.1f} mm",
            f"{max_strain:.1f} %",
            f"{airflow} CFM",
            f"{ambient:.1f} °C",
            f"{target:.1f} °C",
            f"{st.session_state.chamber_temp:.2f} °C",
            f"{st.session_state.niti_temp:.2f} °C",
            f"{st.session_state.air_temp:.2f} °C",
            f"{st.session_state.stress:.1f} MPa",
            f"{st.session_state.strain:.2f} %",
            st.session_state.cycle_count,
            f"{st.session_state.cop:.3f}",
            (
                "TARGET HOLD"
                if st.session_state.target_reached
                else "AUTOMATIC"
                if st.session_state.automatic
                else "READY"
            )
        ]
    }
)

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# RESEARCH NOTICE
# ============================================================

st.markdown("---")

st.warning(
    """
    RESEARCH PROTOTYPE NOTICE

    The thermal, mechanical and COP values displayed by this
    interface are simulated/modelled values intended for SCADA/HMI
    demonstration and system-development purposes. Experimental
    validation requires measured NiTi transformation temperatures,
    stress-strain behaviour, hysteresis, heat-transfer coefficients,
    thermal mass, actuator characteristics and experimentally
    measured cooling performance.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        NI-Ti ELASTOCALORIC REFRIGERATION SCADA
        &nbsp; | &nbsp;
        Advanced Elastocaloric Refrigeration Using NiTi Alloy
        &nbsp; | &nbsp;
        Academic Research Prototype

    </div>
    """,
    unsafe_allow_html=True
)
