import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NiTi Elastocaloric Refrigeration",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. SIMPLE, SAFE STYLING
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #eef2f6;
    }

    h1, h2, h3 {
        color: #16324a;
    }

    .main-header {
        background-color: #0b2942;
        padding: 24px;
        border-radius: 12px;
        margin-bottom: 20px;
    }

    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 30px;
    }

    .main-header p {
        color: #c9d9e8;
        margin-top: 8px;
        margin-bottom: 0;
    }

    .online-status {
        color: #65e18a;
        font-weight: bold;
        margin-top: 12px;
    }

    .ready-box {
        background-color: white;
        border-left: 5px solid #2b78c5;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }

    .target-box {
        background-color: #e8f7ec;
        border-left: 5px solid #28a745;
        padding: 18px;
        border-radius: 8px;
        margin: 15px 0;
    }

    .current-box {
        background-color: white;
        border: 1px solid #d5dde5;
        padding: 18px;
        border-radius: 10px;
        margin-bottom: 15px;
    }

    .phase-current {
        background-color: #fff1dc;
        border: 2px solid #e9a23b;
        padding: 15px;
        border-radius: 10px;
        min-height: 175px;
    }

    .phase-complete {
        background-color: #e9f7ee;
        border: 2px solid #55b879;
        padding: 15px;
        border-radius: 10px;
        min-height: 175px;
    }

    .phase-pending {
        background-color: #f5f6f8;
        border: 1px solid #d8dde3;
        padding: 15px;
        border-radius: 10px;
        min-height: 175px;
        opacity: 0.65;
    }

    .phase-title {
        font-weight: bold;
        color: #183b56;
        font-size: 14px;
    }

    .phase-description {
        color: #657786;
        font-size: 12px;
        margin-top: 8px;
    }

    .equipment-online {
        color: #198754;
        font-weight: bold;
    }

    .footer {
        text-align: center;
        color: #718096;
        padding: 25px;
        font-size: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. PROJECT PHASES
# ============================================================

PHASES = {
    1: {
        "name": "MECHANICAL LOADING",
        "icon": "⚙️",
        "description": "NiTi element is loaded to the selected strain."
    },
    2: {
        "name": "HEAT REJECTION",
        "icon": "🌬️",
        "description": "Heat is rejected using forced-air heat transfer."
    },
    3: {
        "name": "MECHANICAL UNLOADING",
        "icon": "❄️",
        "description": "Unloading produces the elastocaloric cooling effect."
    },
    4: {
        "name": "COLD-SIDE RECOVERY",
        "icon": "◆",
        "description": "Cooling is transferred to the enclosed chamber."
    }
}


# ============================================================
# 4. SESSION STATE
# ============================================================

if "stage" not in st.session_state:
    st.session_state.stage = 0

if "cycle_count" not in st.session_state:
    st.session_state.cycle_count = 0

if "automatic_running" not in st.session_state:
    st.session_state.automatic_running = False

if "target_reached" not in st.session_state:
    st.session_state.target_reached = False

if "chamber_temp" not in st.session_state:
    st.session_state.chamber_temp = 25.0

if "niti_temp" not in st.session_state:
    st.session_state.niti_temp = 25.0

if "air_temp" not in st.session_state:
    st.session_state.air_temp = 25.0

if "stress" not in st.session_state:
    st.session_state.stress = 0.0

if "strain" not in st.session_state:
    st.session_state.strain = 0.0

if "cop" not in st.session_state:
    st.session_state.cop = 0.0

if "sim_time" not in st.session_state:
    st.session_state.sim_time = 0.0

if "message" not in st.session_state:
    st.session_state.message = "SYSTEM READY"

if "events" not in st.session_state:
    st.session_state.events = []

if "time_history" not in st.session_state:
    st.session_state.time_history = [0.0]

if "chamber_history" not in st.session_state:
    st.session_state.chamber_history = [25.0]

if "niti_history" not in st.session_state:
    st.session_state.niti_history = [25.0]

if "air_history" not in st.session_state:
    st.session_state.air_history = [25.0]

if "stress_history" not in st.session_state:
    st.session_state.stress_history = [0.0]

if "strain_history" not in st.session_state:
    st.session_state.strain_history = [0.0]


# ============================================================
# 5. SIDEBAR
# ============================================================

st.sidebar.title("⚙️ SYSTEM SETTINGS")

st.sidebar.subheader("Material")

material = st.sidebar.selectbox(
    "Material",
    [
        "NiTi Nitinol",
        "Cu-Al-Ni SMA",
        "Fe-Mn-Si SMA",
        "Elastocaloric Polymer"
    ]
)

geometry = st.sidebar.selectbox(
    "Geometry",
    [
        "NiTi Tube Bundle",
        "NiTi Wire Bundle",
        "Single NiTi Tube"
    ]
)

elements = st.sidebar.slider(
    "Active Elements",
    1,
    20,
    5
)

st.sidebar.subheader("Geometry")

length = st.sidebar.number_input(
    "Active Length (mm)",
    min_value=20,
    max_value=500,
    value=150
)

outer_diameter = st.sidebar.number_input(
    "Outer Diameter (mm)",
    min_value=1.0,
    max_value=50.0,
    value=12.0
)

inner_diameter = st.sidebar.number_input(
    "Inner Diameter (mm)",
    min_value=0.0,
    max_value=49.0,
    value=10.0
)

st.sidebar.subheader("Mechanical")

max_strain = st.sidebar.slider(
    "Maximum Strain (%)",
    1.0,
    8.0,
    5.0,
    0.5
)

actuation = st.sidebar.selectbox(
    "Actuation",
    [
        "Uniaxial Tension",
        "Uniaxial Compression"
    ]
)

st.sidebar.subheader("Thermal")

ambient = st.sidebar.slider(
    "Ambient Temperature (°C)",
    15.0,
    45.0,
    25.0,
    0.5
)

target = st.sidebar.slider(
    "Target Chamber Temperature (°C)",
    -30.0,
    20.0,
    -5.0,
    0.5
)

airflow = st.sidebar.slider(
    "Forced Airflow (CFM)",
    10,
    100,
    50
)

st.sidebar.subheader("Controller")

mode = st.sidebar.radio(
    "Operating Mode",
    ["Manual", "Automatic"]
)

phase_duration = st.sidebar.slider(
    "Phase Duration",
    0.2,
    3.0,
    0.8,
    0.1
)


# ============================================================
# 6. FUNCTIONS
# ============================================================

def add_event(text):

    time_now = datetime.now().strftime("%H:%M:%S")

    st.session_state.events.insert(
        0,
        f"{time_now} | {text}"
    )

    st.session_state.events = (
        st.session_state.events[:30]
    )


def get_material_factor():

    if material == "NiTi Nitinol":
        return 1.10

    if material == "Cu-Al-Ni SMA":
        return 0.85

    if material == "Fe-Mn-Si SMA":
        return 0.65

    return 1.00


def get_delta_temperature():

    return (
        12
        * get_material_factor()
        * (max_strain / 5)
        * np.sqrt(elements / 5)
    )


def save_history():

    st.session_state.sim_time += phase_duration

    st.session_state.time_history.append(
        st.session_state.sim_time
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

    st.session_state.time_history = (
        st.session_state.time_history[-max_points:]
    )

    st.session_state.chamber_history = (
        st.session_state.chamber_history[-max_points:]
    )

    st.session_state.niti_history = (
        st.session_state.niti_history[-max_points:]
    )

    st.session_state.air_history = (
        st.session_state.air_history[-max_points:]
    )

    st.session_state.stress_history = (
        st.session_state.stress_history[-max_points:]
    )

    st.session_state.strain_history = (
        st.session_state.strain_history[-max_points:]
    )


def reset_system():

    st.session_state.stage = 0

    st.session_state.cycle_count = 0

    st.session_state.automatic_running = False

    st.session_state.target_reached = False

    st.session_state.chamber_temp = ambient

    st.session_state.niti_temp = ambient

    st.session_state.air_temp = ambient

    st.session_state.stress = 0.0

    st.session_state.strain = 0.0

    st.session_state.cop = 0.0

    st.session_state.sim_time = 0.0

    st.session_state.message = "SYSTEM READY"

    st.session_state.events = []

    st.session_state.time_history = [0.0]

    st.session_state.chamber_history = [ambient]

    st.session_state.niti_history = [ambient]

    st.session_state.air_history = [ambient]

    st.session_state.stress_history = [0.0]

    st.session_state.strain_history = [0.0]


def execute_phase(phase):

    if st.session_state.target_reached:
        return True

    delta_t = get_delta_temperature()


    # --------------------------------------------------------
    # 1. LOADING
    # --------------------------------------------------------

    if phase == 1:

        st.session_state.stage = 1

        st.session_state.strain = max_strain

        stress = 450 + max_strain * 12

        if actuation == "Uniaxial Compression":
            stress = -stress

        st.session_state.stress = stress

        st.session_state.niti_temp = (
            ambient + delta_t
        )

        st.session_state.air_temp = (
            ambient + delta_t * 0.20
        )

        st.session_state.message = (
            "MECHANICAL LOADING IN PROGRESS"
        )


    # --------------------------------------------------------
    # 2. HEAT REJECTION
    # --------------------------------------------------------

    elif phase == 2:

        st.session_state.stage = 2

        st.session_state.strain = max_strain

        stress = 410

        if actuation == "Uniaxial Compression":
            stress = -stress

        st.session_state.stress = stress

        airflow_factor = min(
            airflow / 50,
            1.0
        )

        st.session_state.niti_temp = (
            ambient
            + delta_t
            * (1 - 0.65 * airflow_factor)
        )

        st.session_state.air_temp = (
            ambient
            + delta_t * 0.12
            * airflow_factor
        )

        st.session_state.message = (
            "HEAT REJECTION USING FORCED AIR"
        )


    # --------------------------------------------------------
    # 3. UNLOADING
    # --------------------------------------------------------

    elif phase == 3:

        st.session_state.stage = 3

        st.session_state.strain = 0.0

        st.session_state.stress = 120

        st.session_state.niti_temp = (
            ambient - delta_t
        )

        st.session_state.air_temp = (
            ambient - delta_t * 0.40
        )

        st.session_state.message = (
            "ELASTOCALORIC COOLING GENERATED"
        )


    # --------------------------------------------------------
    # 4. COLD-SIDE RECOVERY
    # --------------------------------------------------------

    elif phase == 4:

        st.session_state.stage = 4

        st.session_state.strain = 0.0

        st.session_state.stress = 0.0

        cooling_factor = (
            0.035
            * (airflow / 50)
            * np.sqrt(elements / 5)
        )

        current = st.session_state.chamber_temp

        new_temperature = (
            current
            - (current - target)
            * cooling_factor
            - 0.18
            * get_material_factor()
            * (max_strain / 5)
        )

        st.session_state.chamber_temp = (
            new_temperature
        )

        st.session_state.niti_temp = (
            ambient - delta_t * 0.75
        )

        st.session_state.air_temp = (
            ambient - delta_t * 0.55
        )

        cooling_load = max(
            0.1,
            ambient - new_temperature
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
            "COLD-SIDE RECOVERY IN PROGRESS"
        )


    # --------------------------------------------------------
    # SAVE DATA
    # --------------------------------------------------------

    save_history()


    # --------------------------------------------------------
    # TARGET CHECK
    # --------------------------------------------------------

    if st.session_state.chamber_temp <= target:

        st.session_state.chamber_temp = target

        st.session_state.target_reached = True

        st.session_state.automatic_running = False

        st.session_state.message = (
            "TARGET TEMPERATURE ACHIEVED"
        )

        st.session_state.chamber_history[-1] = target

        add_event(
            f"TARGET {target:.1f} °C ACHIEVED"
        )

        add_event(
            "AUTOMATIC CYCLING STOPPED"
        )

        add_event(
            "SYSTEM ENTERED TARGET HOLD"
        )

        return True

    return False


# ============================================================
# 7. HEADER
# ============================================================

st.markdown(
    """
    <div class="main-header">

        <h1>
            ❄ NiTi ELASTOCALORIC REFRIGERATION
        </h1>

        <p>
            Advanced Elastocaloric Refrigeration Using NiTi Alloy
            &nbsp; | &nbsp;
            RESEARCH SCADA / HMI
        </p>

        <div class="online-status">
            ● CONTROLLER ONLINE
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 8. CONTROLLER STATUS
# ============================================================

if st.session_state.target_reached:

    st.success(
        f"🎯 TARGET TEMPERATURE ACHIEVED — "
        f"Chamber = {st.session_state.chamber_temp:.1f} °C | "
        f"Target = {target:.1f} °C | "
        f"Automatic cycling stopped."
    )

elif st.session_state.automatic_running:

    st.info(
        f"● AUTOMATIC OPERATION RUNNING | "
        f"Target = {target:.1f} °C | "
        f"Chamber = {st.session_state.chamber_temp:.1f} °C"
    )

else:

    st.info(
        f"● {mode.upper()} MODE | CONTROLLER READY"
    )


# ============================================================
# 9. MASTER CONTROLS
# ============================================================

st.subheader("MASTER CONTROL")

control1, control2, control3, control4 = st.columns(4)


with control1:

    if mode == "Automatic":

        if st.session_state.automatic_running:

            if st.button(
                "⏸ PAUSE AUTOMATIC",
                use_container_width=True
            ):

                st.session_state.automatic_running = False

                add_event(
                    "Automatic operation paused."
                )

                st.rerun()

        else:

            if not st.session_state.target_reached:

                if st.button(
                    "▶ START AUTOMATIC",
                    use_container_width=True
                ):

                    st.session_state.automatic_running = True

                    if st.session_state.stage == 0:
                        st.session_state.stage = 1

                    add_event(
                        "Automatic operation started."
                    )

                    st.rerun()


with control2:

    if st.button(
        "⏹ STOP",
        use_container_width=True
    ):

        st.session_state.automatic_running = False

        add_event(
            "Operation stopped by operator."
        )

        st.rerun()


with control3:

    if st.button(
        "🔄 RESET",
        use_container_width=True
    ):

        reset_system()

        st.rerun()


with control4:

    if st.session_state.target_reached:

        st.metric(
            "CONTROLLER STATE",
            "TARGET HOLD"
        )

    elif st.session_state.automatic_running:

        st.metric(
            "CONTROLLER STATE",
            "RUNNING"
        )

    else:

        st.metric(
            "CONTROLLER STATE",
            "READY"
        )


# ============================================================
# 10. CURRENT OPERATION
# ============================================================

st.subheader("CURRENT OPERATION")

if st.session_state.stage == 0:

    st.markdown(
        """
        <div class="current-box">

        <h3>● SYSTEM READY</h3>

        <p>
        Select Manual or Automatic operation to begin.
        </p>

        <strong>
        Controller Message:
        </strong>
        SYSTEM READY

        </div>
        """,
        unsafe_allow_html=True
    )

else:

    phase = PHASES[
        st.session_state.stage
    ]

    st.markdown(
        f"""
        <div class="current-box">

        <h3>
        {phase["icon"]}
        {phase["name"]}
        </h3>

        <p>
        {phase["description"]}
        </p>

        <strong>
        Controller Message:
        </strong>
        {st.session_state.message}

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 11. TARGET ACHIEVED
# ============================================================

if st.session_state.target_reached:

    st.markdown(
        f"""
        <div class="target-box">

        <h3>🎯 TARGET TEMPERATURE ACHIEVED</h3>

        <p>
        Chamber Temperature:
        <strong>{st.session_state.chamber_temp:.1f} °C</strong>
        </p>

        <p>
        Target:
        <strong>{target:.1f} °C</strong>
        </p>

        <p>
        Automatic cycling has stopped.
        The controller will remain in TARGET HOLD until RESET.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 12. ACTIVE CONFIGURATION
# ============================================================

st.subheader("ACTIVE SYSTEM CONFIGURATION")

config1, config2, config3, config4, config5, config6 = st.columns(6)

with config1:
    st.metric("MATERIAL", material)

with config2:
    st.metric("GEOMETRY", geometry)

with config3:
    st.metric("ELEMENTS", elements)

with config4:
    st.metric("OD", f"{outer_diameter:g} mm")

with config5:
    st.metric("LENGTH", f"{length} mm")

with config6:
    st.metric("STRAIN", f"{max_strain:g}%")


# ============================================================
# 13. OPERATION SEQUENCE
# ============================================================

st.subheader("ELASTOCALORIC OPERATION SEQUENCE")

phase_columns = st.columns(4)

for column, number in zip(
    phase_columns,
    [1, 2, 3, 4]
):

    with column:

        if number == st.session_state.stage:

            css = "phase-current"
            status = "● CURRENT"

        elif number < st.session_state.stage:

            css = "phase-complete"
            status = "✓ COMPLETE"

        else:

            css = "phase-pending"
            status = "○ PENDING"

        phase = PHASES[number]

        st.markdown(
            f"""
            <div class="{css}">

            <div style="font-size:25px;">
            {phase["icon"]}
            </div>

            <div class="phase-title">
            {number}. {phase["name"]}
            </div>

            <div class="phase-description">
            {phase["description"]}
            </div>

            <br>

            <strong>
            {status}
            </strong>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# 14. THERMAL MONITORING
# ============================================================

st.subheader("LIVE THERMAL MONITORING")

thermal1, thermal2, thermal3, thermal4 = st.columns(4)

with thermal1:
    st.metric(
        "CHAMBER",
        f"{st.session_state.chamber_temp:.2f} °C"
    )

with thermal2:
    st.metric(
        "NiTi",
        f"{st.session_state.niti_temp:.2f} °C"
    )

with thermal3:
    st.metric(
        "AIR",
        f"{st.session_state.air_temp:.2f} °C"
    )

with thermal4:
    st.metric(
        "AMBIENT",
        f"{ambient:.2f} °C"
    )


# ============================================================
# 15. MECHANICAL MONITORING
# ============================================================

st.subheader("MECHANICAL CONDITION")

mechanical1, mechanical2, mechanical3, mechanical4 = st.columns(4)

with mechanical1:

    st.metric(
        "STRESS",
        f"{st.session_state.stress:.1f} MPa"
    )

with mechanical2:

    st.metric(
        "STRAIN",
        f"{st.session_state.strain:.2f}%"
    )

with mechanical3:

    st.metric(
        "CYCLES",
        st.session_state.cycle_count
    )

with mechanical4:

    st.metric(
        "EST. COP",
        f"{st.session_state.cop:.3f}"
    )


# ============================================================
# 16. EQUIPMENT STATUS
# ============================================================

st.subheader("EQUIPMENT STATUS")

equipment = [
    ("LOAD ACTUATOR", "ONLINE"),
    ("LINEAR GUIDE", "ONLINE"),
    ("FORCED AIR", f"{airflow} CFM"),
    ("TEMPERATURE SENSOR", "ONLINE"),
    ("CHAMBER", "SEALED")
]

equipment_columns = st.columns(5)

for column, (name, status) in zip(
    equipment_columns,
    equipment
):

    with column:

        st.markdown(f"**{name}**")

        st.markdown(
            f"""
            <div class="equipment-online">
            ● {status}
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# 17. MANUAL CONTROL
# ============================================================

if mode == "Manual":

    st.subheader("MANUAL PHASE CONTROL")

    manual_columns = st.columns(4)

    for column, number in zip(
        manual_columns,
        [1, 2, 3, 4]
    ):

        with column:

            phase = PHASES[number]

            if st.button(
                f"{phase['icon']} {phase['name']}",
                use_container_width=True
            ):

                if not st.session_state.target_reached:

                    execute_phase(number)

                    st.rerun()


# ============================================================
# 18. AUTOMATIC CONTROLLER
# ============================================================

if (
    mode == "Automatic"
    and st.session_state.automatic_running
    and not st.session_state.target_reached
):

    current_phase = st.session_state.stage

    if current_phase == 0:

        st.session_state.stage = 1

        st.rerun()

    else:

        target_hit = execute_phase(
            current_phase
        )

        if target_hit:

            st.session_state.automatic_running = False

            st.toast(
                "🎯 Target temperature achieved.",
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
                f"Cycle "
                f"{st.session_state.cycle_count} completed."
            )

            st.session_state.stage = 1

            st.rerun()


# ============================================================
# 19. THERMAL GRAPH
# ============================================================

st.subheader("LIVE THERMAL TREND")

lengths = [
    len(st.session_state.time_history),
    len(st.session_state.chamber_history),
    len(st.session_state.niti_history),
    len(st.session_state.air_history)
]

n = min(lengths)

time_data = np.array(
    st.session_state.time_history[-n:]
)

chamber_data = np.array(
    st.session_state.chamber_history[-n:]
)

niti_data = np.array(
    st.session_state.niti_history[-n:]
)

air_data = np.array(
    st.session_state.air_history[-n:]
)

fig, ax = plt.subplots(
    figsize=(12, 4.5)
)

ax.plot(
    time_data,
    chamber_data,
    linewidth=2.5,
    label="Chamber"
)

ax.plot(
    time_data,
    niti_data,
    linewidth=2,
    label="NiTi"
)

ax.plot(
    time_data,
    air_data,
    linewidth=2,
    label="Air"
)

ax.axhline(
    target,
    linestyle="--",
    linewidth=1.5,
    label="Target"
)

ax.set_xlabel("Simulation Time (s)")
ax.set_ylabel("Temperature (°C)")
ax.set_title("Elastocaloric Thermal Response")

ax.grid(alpha=0.25)
ax.legend()

st.pyplot(
    fig,
    use_container_width=True
)

plt.close(fig)


# ============================================================
# 20. STRESS-STRAIN GRAPH
# ============================================================

st.subheader("MECHANICAL RESPONSE")

mechanical_lengths = [
    len(st.session_state.stress_history),
    len(st.session_state.strain_history)
]

m = min(mechanical_lengths)

strain_data = np.array(
    st.session_state.strain_history[-m:]
)

stress_data = np.array(
    st.session_state.stress_history[-m:]
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

ax2.set_xlabel("Strain (%)")
ax2.set_ylabel("Stress (MPa)")
ax2.set_title("NiTi Mechanical Response")

ax2.grid(alpha=0.25)

st.pyplot(
    fig2,
    use_container_width=True
)

plt.close(fig2)


# ============================================================
# 21. EVENT LOG
# ============================================================

st.subheader("SYSTEM EVENT LOG")

if len(st.session_state.events) > 0:

    event_table = pd.DataFrame({
        "EVENT": st.session_state.events
    })

    st.dataframe(
        event_table,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No events recorded yet."
    )


# ============================================================
# 22. SYSTEM SUMMARY
# ============================================================

st.subheader("SYSTEM SUMMARY")

if st.session_state.target_reached:

    controller_state = "TARGET HOLD"

elif st.session_state.automatic_running:

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
# 23. RESEARCH NOTICE
# ============================================================

st.warning(
    "RESEARCH / SIMULATION NOTICE: "
    "Temperature, stress, strain and COP values shown by "
    "this interface are model-generated estimates. "
    "They should not be interpreted as experimentally "
    "validated measurements."
)


# ============================================================
# 24. FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

    NiTi Elastocaloric Refrigeration SCADA / HMI
    <br>
    Advanced Elastocaloric Refrigeration Using NiTi Alloy
    <br>
    Mechanical Engineering Research Interface

    </div>
    """,
    unsafe_allow_html=True
)
