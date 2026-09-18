import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NiTi Elastocaloric SCADA",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background: #eef2f6;
}

/* ================= HEADER ================= */

.header {
    background: linear-gradient(135deg, #071a2b, #123f63);
    padding: 28px 32px;
    border-radius: 16px;
    margin-bottom: 20px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.15);
}

.header-title {
    font-size: 30px;
    font-weight: 800;
    letter-spacing: 1px;
    color: white;
}

.header-subtitle {
    margin-top: 8px;
    font-size: 14px;
    color: #c8d8e8;
}

.online {
    display: inline-block;
    margin-top: 13px;
    padding: 6px 13px;
    border-radius: 20px;
    background: rgba(60,220,120,0.12);
    color: #58e38b;
    font-size: 13px;
    font-weight: 700;
}

/* ================= SECTIONS ================= */

.section-title {
    font-size: 18px;
    font-weight: 800;
    color: #16324a;
    margin-top: 18px;
    margin-bottom: 12px;
    padding-bottom: 7px;
    border-bottom: 2px solid #c7d2dc;
}

/* ================= CARDS ================= */

.status-card {
    background: white;
    border-radius: 12px;
    padding: 16px;
    min-height: 105px;
    border: 1px solid #d8e0e8;
    box-shadow: 0 3px 10px rgba(0,0,0,0.05);
}

.card-label {
    font-size: 11px;
    color: #6c7a89;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.card-value {
    font-size: 26px;
    font-weight: 800;
    color: #102a43;
    margin-top: 8px;
}

.card-unit {
    font-size: 12px;
    color: #718096;
}

/* ================= CURRENT OPERATION ================= */

.operation {
    background: white;
    padding: 20px 22px;
    border-radius: 13px;
    border: 1px solid #d9e1e8;
    box-shadow: 0 3px 12px rgba(0,0,0,0.05);
}

.operation h3 {
    color: #16324a;
    margin-top: 0;
    margin-bottom: 8px;
}

.operation p {
    color: #657786;
    margin-bottom: 14px;
}

/* ================= TARGET ================= */

.target-achieved {
    background: #e8f8ed;
    border-left: 6px solid #28a745;
    border-radius: 10px;
    padding: 18px;
    margin: 15px 0;
    color: #145c2a;
    font-weight: 800;
}

.target-box {
    background: #e9f4ff;
    border-left: 5px solid #2b78c5;
    border-radius: 10px;
    padding: 15px;
    margin-top: 18px;
}

/* ================= PHASES ================= */

.phase {
    padding: 16px;
    border-radius: 12px;
    min-height: 190px;
    border: 1px solid #d9e0e7;
    background: #f7f9fb;
}

.phase-current {
    background: #fff3df;
    border: 2px solid #f0a43b;
}

.phase-complete {
    background: #eaf8ef;
    border: 2px solid #55b978;
}

.phase-pending {
    opacity: 0.52;
}

.phase-icon {
    font-size: 25px;
    margin-bottom: 7px;
}

.phase-title {
    font-weight: 800;
    color: #183b56;
    font-size: 14px;
}

.phase-description {
    font-size: 12px;
    color: #657786;
    margin-top: 7px;
    line-height: 1.4;
}

.phase-status {
    margin-top: 12px;
    font-size: 12px;
    font-weight: 800;
}

/* ================= FOOTER ================= */

.footer {
    text-align: center;
    color: #718096;
    font-size: 12px;
    padding: 25px;
}

/* ================= SIDEBAR ================= */

section[data-testid="stSidebar"] {
    background: #f8fafc;
}

/* ================= BUTTONS ================= */

.stButton > button {
    border-radius: 9px;
    font-weight: 700;
    min-height: 42px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# PHASE DEFINITIONS
# ============================================================

PHASES = {

    1: {
        "name": "MECHANICAL LOADING",
        "description":
            "NiTi element is mechanically loaded to the selected strain.",
        "icon": "⚙️"
    },

    2: {
        "name": "HEAT REJECTION",
        "description":
            "Generated heat is rejected using forced-air heat transfer.",
        "icon": "↗️"
    },

    3: {
        "name": "MECHANICAL UNLOADING",
        "description":
            "NiTi is unloaded and the elastocaloric cooling effect is generated.",
        "icon": "❄️"
    },

    4: {
        "name": "COLD-SIDE RECOVERY",
        "description":
            "Cooling is transferred to the enclosed cold-side chamber.",
        "icon": "◆"
    }
}


# ============================================================
# SESSION STATE
# ============================================================

defaults = {

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

    "simulation_time": 0.0,

    "message": "SYSTEM READY",

    "events": [],

    "temperature_time": [0.0],
    "chamber_history": [25.0],
    "niti_history": [25.0],
    "air_history": [25.0],

    "stress_history": [0.0],
    "strain_history": [0.0]
}

for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ⚙️ SYSTEM SETTINGS")

    st.markdown("### Material")

    material = st.selectbox(
        "Active Material",
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

    elements = st.slider(
        "Active Elements",
        1,
        20,
        5
    )

    st.markdown("### Geometry")

    length = st.number_input(
        "Active Length (mm)",
        20,
        500,
        150
    )

    outer_diameter = st.number_input(
        "Outer Diameter (mm)",
        1.0,
        50.0,
        12.0
    )

    inner_diameter = st.number_input(
        "Inner Diameter (mm)",
        0.0,
        49.0,
        10.0
    )

    st.markdown("### Mechanical Parameters")

    max_strain = st.slider(
        "Maximum Strain (%)",
        1.0,
        8.0,
        5.0,
        step=0.5
    )

    actuation = st.selectbox(
        "Actuation",
        [
            "Uniaxial Tension",
            "Uniaxial Compression"
        ]
    )

    phase_time = st.slider(
        "Phase Duration (s)",
        0.2,
        3.0,
        0.8,
        step=0.1
    )

    st.markdown("### Thermal Parameters")

    ambient = st.slider(
        "Ambient Temperature (°C)",
        15.0,
        45.0,
        25.0,
        step=0.5
    )

    target = st.slider(
        "Target Chamber Temperature (°C)",
        -30.0,
        20.0,
        -5.0,
        step=0.5
    )

    airflow = st.slider(
        "Forced Airflow (CFM)",
        10,
        100,
        50
    )

    st.markdown("### Controller")

    mode = st.radio(
        "Operating Mode",
        ["Manual", "Automatic"]
    )

    st.divider()

    if st.button(
        "🔄 RESET CONTROLLER",
        use_container_width=True
    ):

        st.session_state.stage = 0
        st.session_state.cycle_count = 0
        st.session_state.automatic = False
        st.session_state.target_reached = False

        st.session_state.chamber_temp = ambient
        st.session_state.niti_temp = ambient
        st.session_state.air_temp = ambient

        st.session_state.stress = 0
        st.session_state.strain = 0
        st.session_state.cop = 0

        st.session_state.simulation_time = 0

        st.session_state.message = "SYSTEM READY"

        st.session_state.events = []

        st.session_state.temperature_time = [0]
        st.session_state.chamber_history = [ambient]
        st.session_state.niti_history = [ambient]
        st.session_state.air_history = [ambient]

        st.session_state.stress_history = [0]
        st.session_state.strain_history = [0]

        st.rerun()


# ============================================================
# FUNCTIONS
# ============================================================

def add_event(message):

    timestamp = datetime.now().strftime("%H:%M:%S")

    st.session_state.events.insert(
        0,
        f"{timestamp}  |  {message}"
    )

    st.session_state.events = (
        st.session_state.events[:30]
    )


def material_factor():

    factors = {

        "NiTi Nitinol": 1.10,

        "Cu-Al-Ni SMA": 0.85,

        "Fe-Mn-Si SMA": 0.65,

        "Elastocaloric Polymer": 1.00

    }

    return factors[material]


def calculate_delta_t():

    return (

        12.0
        * material_factor()
        * (max_strain / 5.0)
        * np.sqrt(elements / 5.0)

    )


def append_history():

    st.session_state.simulation_time += phase_time

    st.session_state.temperature_time.append(
        st.session_state.simulation_time
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

    max_points = 150

    for key in [

        "temperature_time",
        "chamber_history",
        "niti_history",
        "air_history",
        "stress_history",
        "strain_history"

    ]:

        st.session_state[key] = (
            st.session_state[key][-max_points:]
        )


# ============================================================
# RUN PHASE
# ============================================================

def run_phase(phase):

    if st.session_state.target_reached:

        return True

    delta_t = calculate_delta_t()


    # --------------------------------------------------------
    # PHASE 1
    # --------------------------------------------------------

    if phase == 1:

        st.session_state.strain = max_strain

        stress = (
            450
            + max_strain * 12
        )

        if actuation == "Uniaxial Compression":

            stress *= -1

        st.session_state.stress = stress

        st.session_state.niti_temp = (
            ambient + delta_t
        )

        st.session_state.air_temp = (
            ambient + delta_t * 0.20
        )

        st.session_state.message = (
            "Mechanical loading in progress"
        )


    # --------------------------------------------------------
    # PHASE 2
    # --------------------------------------------------------

    elif phase == 2:

        st.session_state.strain = max_strain

        stress = 410

        if actuation == "Uniaxial Compression":

            stress *= -1

        st.session_state.stress = stress

        airflow_factor = min(
            1.0,
            airflow / 50.0
        )

        st.session_state.niti_temp = (

            ambient
            + delta_t
            * (1 - 0.65 * airflow_factor)

        )

        st.session_state.air_temp = (

            ambient
            + delta_t
            * 0.12
            * airflow_factor

        )

        st.session_state.message = (
            "Heat rejection using forced airflow"
        )


    # --------------------------------------------------------
    # PHASE 3
    # --------------------------------------------------------

    elif phase == 3:

        st.session_state.strain = 0.0

        st.session_state.stress = 120

        st.session_state.niti_temp = (
            ambient - delta_t
        )

        st.session_state.air_temp = (
            ambient - delta_t * 0.40
        )

        st.session_state.message = (
            "Elastocaloric cooling generated"
        )


    # --------------------------------------------------------
    # PHASE 4
    # --------------------------------------------------------

    elif phase == 4:

        st.session_state.strain = 0.0

        st.session_state.stress = 0.0

        cooling_factor = (

            0.035
            * (airflow / 50.0)
            * np.sqrt(elements / 5.0)

        )

        current_temp = (
            st.session_state.chamber_temp
        )

        new_temp = (

            current_temp
            - (current_temp - target)
            * cooling_factor
            - 0.18
            * material_factor()
            * (max_strain / 5.0)

        )

        st.session_state.chamber_temp = new_temp

        st.session_state.niti_temp = (
            ambient - delta_t * 0.75
        )

        st.session_state.air_temp = (
            ambient - delta_t * 0.55
        )

        cooling_load = max(
            0.1,
            ambient - new_temp
        )

        input_work = max(
            1.0,
            abs(st.session_state.stress)
            * max_strain
            * length
            / 1000
        )

        st.session_state.cop = (
            cooling_load / input_work
        )

        st.session_state.message = (
            "Cold-side recovery in progress"
        )


    # --------------------------------------------------------
    # SAVE DATA
    # --------------------------------------------------------

    append_history()


    # --------------------------------------------------------
    # TARGET CHECK
    # --------------------------------------------------------

    if st.session_state.chamber_temp <= target:

        st.session_state.chamber_temp = target

        st.session_state.target_reached = True

        st.session_state.automatic = False

        st.session_state.message = (
            "TARGET TEMPERATURE ACHIEVED"
        )

        add_event(
            f"Target temperature "
            f"{target:.1f} °C achieved."
        )

        add_event(
            "Automatic cycling stopped."
        )

        add_event(
            "System entered TARGET HOLD."
        )

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
# GLOBAL STATUS
# ============================================================

if st.session_state.target_reached:

    st.success(
        f"🎯 TARGET ACHIEVED — Chamber temperature "
        f"{st.session_state.chamber_temp:.1f} °C | "
        f"Automatic cycling stopped."
    )

elif st.session_state.automatic:

    st.info(
        f"● AUTOMATIC RUNNING | "
        f"Target: {target:.1f} °C | "
        f"Chamber: "
        f"{st.session_state.chamber_temp:.1f} °C"
    )

else:

    st.info(
        f"● {mode.upper()} MODE | "
        f"Controller Ready"
    )


# ============================================================
# ACTIVE CONFIGURATION
# ============================================================

st.markdown(
    '<div class="section-title">ACTIVE SYSTEM CONFIGURATION</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4, c5, c6 = st.columns(6)


with c1:

    st.markdown(
        f"""
        <div class="status-card">

            <div class="card-label">
                Material
            </div>

            <div class="card-value"
                 style="font-size:19px;">
                {material}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with c2:

    st.markdown(
        f"""
        <div class="status-card">

            <div class="card-label">
                Geometry
            </div>

            <div class="card-value"
                 style="font-size:17px;">
                {geometry}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with c3:

    st.markdown(
        f"""
        <div class="status-card">

            <div class="card-label">
                Active Elements
            </div>

            <div class="card-value">
                {elements}
            </div>

            <div class="card-unit">
                ACTIVE
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with c4:

    st.markdown(
        f"""
        <div class="status-card">

            <div class="card-label">
                Diameter
            </div>

            <div class="card-value">
                {outer_diameter:g}
            </div>

            <div class="card-unit">
                mm OD
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with c5:

    st.markdown(
        f"""
        <div class="status-card">

            <div class="card-label">
                Length
            </div>

            <div class="card-value">
                {length}
            </div>

            <div class="card-unit">
                mm
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with c6:

    st.markdown(
        f"""
        <div class="status-card">

            <div class="card-label">
                Maximum Strain
            </div>

            <div class="card-value">
                {max_strain:g}
            </div>

            <div class="card-unit">
                %
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MASTER CONTROL
# ============================================================

st.markdown(
    '<div class="section-title">MASTER CONTROL</div>',
    unsafe_allow_html=True
)

b1, b2, b3, b4 = st.columns(4)


with b1:

    if mode == "Automatic":

        if not st.session_state.automatic:

            if not st.session_state.target_reached:

                if st.button(
                    "▶ START AUTOMATIC",
                    use_container_width=True
                ):

                    st.session_state.automatic = True

                    if st.session_state.stage == 0:

                        st.session_state.stage = 1

                    add_event(
                        "Automatic cycle started."
                    )

                    st.rerun()

        else:

            if st.button(
                "⏸ PAUSE AUTOMATIC",
                use_container_width=True
            ):

                st.session_state.automatic = False

                add_event(
                    "Automatic operation paused."
                )

                st.rerun()

    else:

        st.write("Manual operation")


with b2:

    if st.button(
        "⏹ STOP",
        use_container_width=True
    ):

        st.session_state.automatic = False

        add_event(
            "Controller stopped by operator."
        )

        st.rerun()


with b3:

    if st.button(
        "🔄 RESET",
        use_container_width=True
    ):

        st.session_state.stage = 0
        st.session_state.cycle_count = 0

        st.session_state.automatic = False
        st.session_state.target_reached = False

        st.session_state.chamber_temp = ambient
        st.session_state.niti_temp = ambient
        st.session_state.air_temp = ambient

        st.session_state.stress = 0
        st.session_state.strain = 0
        st.session_state.cop = 0

        st.session_state.simulation_time = 0

        st.session_state.message = "SYSTEM READY"

        st.session_state.events = []

        st.session_state.temperature_time = [0]
        st.session_state.chamber_history = [ambient]
        st.session_state.niti_history = [ambient]
        st.session_state.air_history = [ambient]

        st.session_state.stress_history = [0]
        st.session_state.strain_history = [0]

        st.rerun()


with b4:

    if st.session_state.target_reached:

        controller_state = "TARGET HOLD"

    elif st.session_state.automatic:

        controller_state = "RUNNING"

    else:

        controller_state = "READY"

    st.metric(
        "CONTROLLER STATE",
        controller_state
    )


# ============================================================
# TARGET ACHIEVED
# ============================================================

if st.session_state.target_reached:

    st.markdown(
        f"""
        <div class="target-achieved">

            🎯 TARGET TEMPERATURE ACHIEVED

            <br><br>

            Chamber Temperature:
            {st.session_state.chamber_temp:.1f} °C

            &nbsp;&nbsp; | &nbsp;&nbsp;

            Target:
            {target:.1f} °C

            <br><br>

            Automatic cycling has stopped.
            No new cycle will start until RESET.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# CURRENT OPERATION
# ============================================================

st.markdown(
    '<div class="section-title">CURRENT OPERATION</div>',
    unsafe_allow_html=True
)

if current_stage := st.session_state.stage:

    current_operation = PHASES[current_stage]["name"]
    current_icon = PHASES[current_stage]["icon"]
    current_description = PHASES[current_stage]["description"]

else:

    current_operation = "SYSTEM READY"
    current_icon = "●"
    current_description = (
        "Select Manual or Automatic operation to begin."
    )


st.markdown(
    f"""
    <div class="operation">

        <h3>
            {current_icon}
            &nbsp; {current_operation}
        </h3>

        <p>
            {current_description}
        </p>

        <strong>
            Controller Message:
        </strong>

        &nbsp;

        {st.session_state.message}

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# OPERATION SEQUENCE
# ============================================================

st.markdown(
    '<div class="section-title">ELASTOCALORIC OPERATION SEQUENCE</div>',
    unsafe_allow_html=True
)

phase_columns = st.columns(4)

current_stage = st.session_state.stage

for index, phase_number in enumerate([1, 2, 3, 4]):

    with phase_columns[index]:

        if phase_number == current_stage:

            css_class = "phase-current"
            status = "● CURRENT"

        elif phase_number < current_stage:

            css_class = "phase-complete"
            status = "✓ COMPLETE"

        else:

            css_class = "phase-pending"
            status = "○ PENDING"

        st.markdown(
            f"""
            <div class="phase {css_class}">

                <div class="phase-icon">
                    {PHASES[phase_number]["icon"]}
                </div>

                <div class="phase-title">
                    {phase_number}.
                    {PHASES[phase_number]["name"]}
                </div>

                <div class="phase-description">
                    {PHASES[phase_number]["description"]}
                </div>

                <div class="phase-status">
                    {status}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# LIVE THERMAL MONITORING
# ============================================================

st.markdown(
    '<div class="section-title">LIVE THERMAL MONITORING</div>',
    unsafe_allow_html=True
)

t1, t2, t3, t4 = st.columns(4)


with t1:

    st.metric(
        "CHAMBER TEMPERATURE",
        f"{st.session_state.chamber_temp:.2f} °C",
        f"Target {target:.1f} °C"
    )


with t2:

    st.metric(
        "NiTi TEMPERATURE",
        f"{st.session_state.niti_temp:.2f} °C"
    )


with t3:

    st.metric(
        "AIR TEMPERATURE",
        f"{st.session_state.air_temp:.2f} °C"
    )


with t4:

    st.metric(
        "AMBIENT",
        f"{ambient:.2f} °C"
    )


# ============================================================
# MECHANICAL CONDITION
# ============================================================

st.markdown(
    '<div class="section-title">MECHANICAL CONDITION</div>',
    unsafe_allow_html=True
)

m1, m2, m3, m4 = st.columns(4)


with m1:

    st.metric(
        "STRESS",
        f"{st.session_state.stress:.1f} MPa"
    )


with m2:

    st.metric(
        "STRAIN",
        f"{st.session_state.strain:.2f} %"
    )


with m3:

    st.metric(
        "CYCLES",
        str(st.session_state.cycle_count)
    )


with m4:

    st.metric(
        "EST. COP",
        f"{st.session_state.cop:.3f}"
    )


# ============================================================
# EQUIPMENT STATUS
# ============================================================

st.markdown(
    '<div class="section-title">EQUIPMENT STATUS</div>',
    unsafe_allow_html=True
)

equipment_columns = st.columns(5)

equipment = [

    ("LOAD ACTUATOR", "ONLINE"),

    ("LINEAR GUIDE", "ONLINE"),

    ("FORCED AIR", f"{airflow} CFM"),

    ("TEMPERATURE SENSOR", "ONLINE"),

    ("CHAMBER", "SEALED")

]


for col, (name, status) in zip(
    equipment_columns,
    equipment
):

    with col:

        st.markdown(
            f"""
            <div class="status-card">

                <div class="card-label">
                    {name}
                </div>

                <div style="
                    margin-top:10px;
                    font-weight:800;
                    color:#198754;
                ">

                    ● {status}

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# MANUAL CONTROL
# ============================================================

if mode == "Manual":

    st.markdown(
        '<div class="section-title">MANUAL PHASE CONTROL</div>',
        unsafe_allow_html=True
    )

    manual_columns = st.columns(4)

    for col, phase_number in zip(
        manual_columns,
        [1, 2, 3, 4]
    ):

        with col:

            if st.button(
                f"{PHASES[phase_number]['icon']} "
                f"{PHASES[phase_number]['name']}",
                use_container_width=True
            ):

                if not st.session_state.target_reached:

                    st.session_state.stage = phase_number

                    target_hit = run_phase(
                        phase_number
                    )

                    if target_hit:

                        st.toast(
                            "🎯 Target temperature achieved.",
                            icon="🎯"
                        )

                    st.rerun()


# ============================================================
# AUTOMATIC CONTROLLER
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

        if target_hit:

            st.session_state.automatic = False

            st.toast(
                "🎯 Target temperature achieved. "
                "Automatic cycling stopped.",
                icon="🎯"
            )

            st.rerun()

        if current_phase < 4:

            st.session_state.stage = (
                current_phase + 1
            )

            st.rerun()

        else:

            st.session_state.cycle_count += 1

            add_event(
                f"Automatic cycle "
                f"{st.session_state.cycle_count} completed."
            )

            st.session_state.stage = 1

            st.rerun()


# ============================================================
# THERMAL GRAPH
# ============================================================

st.markdown(
    '<div class="section-title">LIVE THERMAL TREND</div>',
    unsafe_allow_html=True
)

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
    figsize=(12, 4.5)
)

ax.plot(
    x,
    chamber,
    linewidth=2.5,
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

ax.set_xlabel(
    "Simulation Time (s)"
)

ax.set_ylabel(
    "Temperature (°C)"
)

ax.set_title(
    "Elastocaloric Thermal Response"
)

ax.grid(
    alpha=0.25
)

ax.legend()

st.pyplot(
    fig,
    use_container_width=True
)

plt.close(fig)


# ============================================================
# STRESS-STRAIN GRAPH
# ============================================================

st.markdown(
    '<div class="section-title">MECHANICAL RESPONSE</div>',
    unsafe_allow_html=True
)

mechanical_lengths = [

    len(st.session_state.stress_history),

    len(st.session_state.strain_history)

]

n_mechanical = min(
    mechanical_lengths
)

strain_data = np.asarray(
    st.session_state.strain_history[
        -n_mechanical:
    ]
)

stress_data = np.asarray(
    st.session_state.stress_history[
        -n_mechanical:
    ]
)


fig2, ax2 = plt.subplots(
    figsize=(12, 4.5)
)

ax2.plot(
    strain_data,
    stress_data,
    marker="o",
    linewidth=2
)

ax2.set_xlabel(
    "Strain (%)"
)

ax2.set_ylabel(
    "Stress (MPa)"
)

ax2.set_title(
    "NiTi Mechanical Response"
)

ax2.grid(
    alpha=0.25
)

st.pyplot(
    fig2,
    use_container_width=True
)

plt.close(fig2)


# ============================================================
# EVENT LOG
# ============================================================

st.markdown(
    '<div class="section-title">SYSTEM EVENT LOG</div>',
    unsafe_allow_html=True
)

if st.session_state.events:

    event_df = pd.DataFrame(
        {
            "EVENT":
                st.session_state.events
        }
    )

    st.dataframe(
        event_df,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No system events recorded."
    )


# ============================================================
# SYSTEM SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">SYSTEM SUMMARY</div>',
    unsafe_allow_html=True
)

if st.session_state.target_reached:

    controller_state = "TARGET HOLD"

elif st.session_state.automatic:

    controller_state = "AUTOMATIC RUNNING"

else:

    controller_state = "READY"


summary = pd.DataFrame({

    "PARAMETER": [

        "Material",
        "Geometry",
        "Active Elements",
        "Active Length",
        "Outer Diameter",
        "Inner Diameter",
        "Maximum Strain",
        "Actuation",
        "Forced Airflow",
        "Ambient Temperature",
        "Target Temperature",
        "Operating Mode",
        "Cycle Count",
        "Controller State"

    ],

    "VALUE": [

        material,
        geometry,
        elements,
        f"{length} mm",
        f"{outer_diameter:g} mm",
        f"{inner_diameter:g} mm",
        f"{max_strain:g} %",
        actuation,
        f"{airflow} CFM",
        f"{ambient:.1f} °C",
        f"{target:.1f} °C",
        mode,
        st.session_state.cycle_count,
        controller_state

    ]

})


st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# RESEARCH NOTICE
# ============================================================

st.markdown(
    """
    <div class="target-box">

        <strong>
            RESEARCH / SIMULATION NOTICE
        </strong>

        <br><br>

        This SCADA/HMI interface is a
        research-oriented simulation and visualization
        model for the project
        <strong>
            Advanced Elastocaloric Refrigeration Using NiTi Alloy
        </strong>.

        <br><br>

        Temperature, stress, strain and COP values
        shown by this interface are model-generated
        estimates and should not be interpreted as
        experimentally validated measurements.

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        NiTi Elastocaloric Refrigeration
        SCADA / HMI

        <br>

        Advanced Elastocaloric Refrigeration Using NiTi Alloy

        <br>

        Mechanical Engineering Research Interface

    </div>
    """,
    unsafe_allow_html=True
)
