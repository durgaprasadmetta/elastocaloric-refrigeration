import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import time

# ============================================================
# INDUSTRIAL ELASTOCALORIC REFRIGERATION SCADA
# Research / Prototype Demonstration Interface
# ============================================================

st.set_page_config(
    page_title="NiTi Elastocaloric Refrigeration SCADA",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# GLOBAL CONSTANTS
# ============================================================

APP_TITLE = "Industrial NiTi Elastocaloric Refrigeration SCADA"
APP_SUBTITLE = "Advanced Elastocaloric Refrigeration Using NiTi Alloy"

PHASE_NAMES = {
    0: "SYSTEM READY",
    1: "LOADING / STRESSING",
    2: "HEAT REJECTION",
    3: "UNLOADING / COOLING",
    4: "COLD-SIDE RECOVERY"
}

PHASE_COLORS = {
    0: "Normal",
    1: "Warning",
    2: "Warning",
    3: "Normal",
    4: "Normal"
}

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

defaults = {
    "cycle_count": 0,
    "chamber_temp": 25.0,
    "temp_history": [25.0],
    "time_history": [0],
    "wire_temp": 25.0,
    "air_temp": 25.0,
    "cold_side_temp": 25.0,
    "hot_side_temp": 25.0,
    "stress": 0.0,
    "strain": 0.0,
    "cop": 0.0,
    "current_stage_idx": 0,
    "auto_running": False,
    "simulation_time": 0.0,
    "cycle_history": [],
    "alarm_active": False,
    "system_message": "System initialized. Waiting for operator command."
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f4f6f8;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    .scada-header {
        background: linear-gradient(90deg, #0b1f33, #12395a);
        padding: 20px 25px;
        border-radius: 10px;
        color: white;
        margin-bottom: 15px;
    }

    .scada-header h1 {
        margin: 0;
        font-size: 30px;
    }

    .scada-header p {
        margin: 5px 0 0 0;
        font-size: 15px;
        opacity: 0.85;
    }

    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #d9dee5;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }

    .status-green {
        color: #16803c;
        font-weight: bold;
    }

    .status-red {
        color: #c62828;
        font-weight: bold;
    }

    .status-orange {
        color: #d97706;
        font-weight: bold;
    }

    .small-label {
        font-size: 12px;
        color: #6b7280;
    }

    .phase-box {
        background: white;
        border-radius: 8px;
        border: 1px solid #d9dee5;
        padding: 10px;
        min-height: 90px;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    f"""
    <div class="scada-header">
        <h1>❄️ {APP_TITLE}</h1>
        <p>{APP_SUBTITLE} | Prototype Research Control & Monitoring Interface</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR — ENGINEERING CONFIGURATION
# ============================================================

st.sidebar.title("⚙️ Engineering Configuration")

st.sidebar.markdown("### Material Configuration")

material_selection = st.sidebar.selectbox(
    "Active Elastocaloric Material",
    [
        "NiTi Nitinol — Baseline",
        "Cu-Al-Ni Shape Memory Alloy",
        "Fe-Mn-Si Smart Alloy",
        "Elastocaloric Polymer — Reference"
    ]
)

element_type = st.sidebar.selectbox(
    "Core Element Geometry",
    [
        "NiTi Tube Bundle",
        "NiTi Wire Bundle",
        "Single NiTi Tube"
    ]
)

num_elements = st.sidebar.number_input(
    "Number of Active Elements",
    min_value=1,
    max_value=20,
    value=5,
    step=1
)

element_length = st.sidebar.number_input(
    "Active Element Length (mm)",
    min_value=10.0,
    max_value=500.0,
    value=150.0,
    step=5.0
)

outer_diameter = st.sidebar.number_input(
    "Outer Diameter (mm)",
    min_value=0.5,
    max_value=20.0,
    value=12.0,
    step=0.5
)

if element_type == "NiTi Tube Bundle" or element_type == "Single NiTi Tube":

    inner_diameter = st.sidebar.number_input(
        "Inner Diameter (mm)",
        min_value=0.1,
        max_value=max(0.2, outer_diameter - 0.1),
        value=min(10.0, max(0.2, outer_diameter - 0.5)),
        step=0.5
    )

else:
    inner_diameter = 0.0


st.sidebar.markdown("---")
st.sidebar.markdown("### Mechanical Parameters")

strain_limit = st.sidebar.slider(
    "Maximum Mechanical Strain (%)",
    min_value=1.0,
    max_value=8.0,
    value=5.0,
    step=0.5
)

force_mode = st.sidebar.selectbox(
    "Actuation Mode",
    [
        "Uniaxial Tension",
        "Uniaxial Compression"
    ]
)

cycle_speed = st.sidebar.slider(
    "Phase Duration (s)",
    min_value=0.3,
    max_value=3.0,
    value=0.8,
    step=0.1
)


st.sidebar.markdown("---")
st.sidebar.markdown("### Thermal Boundary Conditions")

ambient_temp = st.sidebar.number_input(
    "Ambient Temperature (°C)",
    min_value=15.0,
    max_value=45.0,
    value=25.0,
    step=1.0
)

target_temp = st.sidebar.number_input(
    "Cold-Side Target Temperature (°C)",
    min_value=-30.0,
    max_value=20.0,
    value=-5.0,
    step=1.0
)

fan_flow = st.sidebar.slider(
    "Air Flow Rate (CFM)",
    min_value=10,
    max_value=100,
    value=50,
    step=5
)


st.sidebar.markdown("---")
st.sidebar.markdown("### Control Architecture")

mode = st.sidebar.radio(
    "Operating Mode",
    [
        "Manual Diagnostics",
        "Automated Cycling"
    ]
)


# ============================================================
# PHYSICS / SIMULATION MODEL
# ============================================================

def calculate_material_factor():
    if "Cu-Al-Ni" in material_selection:
        return 1.15

    if "Fe-Mn-Si" in material_selection:
        return 0.85

    if "Polymer" in material_selection:
        return 0.65

    return 1.0


def calculate_thermal_effect():
    material_factor = calculate_material_factor()

    geometry_factor = np.sqrt(
        max(num_elements, 1) / 5
    )

    strain_factor = strain_limit / 5.0

    base_delta = 12.5

    delta_T = (
        base_delta
        * material_factor
        * strain_factor
        * geometry_factor
    )

    return delta_T


def execute_step_physics(stage_idx):

    st.session_state.current_stage_idx = stage_idx

    delta_T = calculate_thermal_effect()

    # --------------------------------------------------------
    # PHASE 1 — LOADING / STRESSING
    # --------------------------------------------------------

    if stage_idx == 1:

        st.session_state.strain = strain_limit

        base_stress = 450 + (strain_limit * 12)

        if force_mode == "Uniaxial Compression":
            st.session_state.stress = -base_stress
        else:
            st.session_state.stress = base_stress

        st.session_state.wire_temp = (
            ambient_temp + delta_T
        )

        st.session_state.hot_side_temp = (
            ambient_temp + delta_T * 0.75
        )

        st.session_state.air_temp = (
            ambient_temp + delta_T * 0.30
        )

        st.session_state.system_message = (
            "Mechanical loading initiated. "
            "Elastocaloric heat generation detected."
        )


    # --------------------------------------------------------
    # PHASE 2 — HEAT REJECTION
    # --------------------------------------------------------

    elif stage_idx == 2:

        st.session_state.strain = strain_limit

        if force_mode == "Uniaxial Compression":
            st.session_state.stress = -410
        else:
            st.session_state.stress = 410

        cooling_factor = fan_flow / 100.0

        st.session_state.wire_temp -= (
            st.session_state.wire_temp - ambient_temp
        ) * cooling_factor * 0.65

        st.session_state.hot_side_temp -= (
            st.session_state.hot_side_temp - ambient_temp
        ) * cooling_factor * 0.75

        st.session_state.air_temp -= (
            st.session_state.air_temp - ambient_temp
        ) * cooling_factor * 0.80

        st.session_state.system_message = (
            "Heat rejection phase active. "
            "Forced-air circulation removing generated heat."
        )


    # --------------------------------------------------------
    # PHASE 3 — UNLOADING / ELASTOCALORIC COOLING
    # --------------------------------------------------------

    elif stage_idx == 3:

        st.session_state.strain = 0.0
        st.session_state.stress = 120

        st.session_state.wire_temp = (
            ambient_temp - delta_T
        )

        st.session_state.cold_side_temp = (
            ambient_temp - delta_T * 0.70
        )

        st.session_state.air_temp = (
            ambient_temp - delta_T * 0.40
        )

        st.session_state.system_message = (
            "Unloading initiated. "
            "Elastocaloric temperature drop generated."
        )


    # --------------------------------------------------------
    # PHASE 4 — COLD-SIDE RECOVERY
    # --------------------------------------------------------

    elif stage_idx == 4:

        st.session_state.strain = 0.0
        st.session_state.stress = 0.0

        recovery_factor = (
            0.035
            * (fan_flow / 50.0)
            * np.sqrt(num_elements / 5)
        )

        if st.session_state.chamber_temp > target_temp:

            st.session_state.chamber_temp -= (
                st.session_state.chamber_temp - target_temp
            ) * recovery_factor

        else:

            st.session_state.chamber_temp += (
                ambient_temp - st.session_state.chamber_temp
            ) * 0.01

        st.session_state.cold_side_temp = min(
            st.session_state.cold_side_temp,
            st.session_state.chamber_temp
        )

        st.session_state.temp_history.append(
            st.session_state.chamber_temp
        )

        st.session_state.simulation_time += cycle_speed

        st.session_state.time_history.append(
            st.session_state.simulation_time
        )

        if len(st.session_state.temp_history) > 80:
            st.session_state.temp_history.pop(0)

        if len(st.session_state.time_history) > 80:
            st.session_state.time_history.pop(0)

        cooling_load = max(
            0,
            ambient_temp - st.session_state.chamber_temp
        )

        mechanical_input = max(
            0.1,
            strain_limit * 0.4
        )

        st.session_state.cop = (
            cooling_load / mechanical_input
        )

        st.session_state.system_message = (
            "Cold-side recovery active. "
            "System preparing for next elastocaloric cycle."
        )


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🖥️ SCADA LIVE MONITOR",
        "⚙️ ENGINEERING PARAMETERS",
        "📊 PERFORMANCE & DATA"
    ]
)


# ============================================================
# TAB 1 — LIVE SCADA
# ============================================================

with tab1:

    # --------------------------------------------------------
    # CONTROL PANEL
    # --------------------------------------------------------

    st.subheader("🎛️ Master Control Panel")

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        if mode == "Automated Cycling":

            if not st.session_state.auto_running:

                if st.button(
                    "🚀 START AUTO CYCLE",
                    type="primary",
                    use_container_width=True
                ):
                    st.session_state.auto_running = True
                    st.session_state.system_message = (
                        "Automatic cycling sequence started."
                    )
                    st.rerun()

            else:

                if st.button(
                    "🛑 STOP AUTO CYCLE",
                    use_container_width=True
                ):
                    st.session_state.auto_running = False
                    st.session_state.system_message = (
                        "Automatic cycling stopped by operator."
                    )
                    st.rerun()

        else:

            st.info("Manual diagnostic mode")


    with c2:

        if st.button(
            "🔄 RESET SYSTEM",
            use_container_width=True
        ):

            st.session_state.cycle_count = 0
            st.session_state.chamber_temp = ambient_temp
            st.session_state.temp_history = [ambient_temp]
            st.session_state.time_history = [0]
            st.session_state.wire_temp = ambient_temp
            st.session_state.air_temp = ambient_temp
            st.session_state.cold_side_temp = ambient_temp
            st.session_state.hot_side_temp = ambient_temp
            st.session_state.stress = 0
            st.session_state.strain = 0
            st.session_state.cop = 0
            st.session_state.current_stage_idx = 0
            st.session_state.simulation_time = 0
            st.session_state.auto_running = False
            st.session_state.cycle_history = []

            st.session_state.system_message = (
                "System reset completed."
            )

            st.rerun()


    with c3:

        st.metric(
            "Completed Cycles",
            st.session_state.cycle_count
        )


    with c4:

        st.metric(
            "Current Phase",
            PHASE_NAMES[
                st.session_state.current_stage_idx
            ]
        )


    st.markdown("---")


    # --------------------------------------------------------
    # LIVE KPIs
    # --------------------------------------------------------

    st.subheader("📡 Live Engineering Measurements")

    k1, k2, k3, k4, k5, k6 = st.columns(6)

    with k1:
        st.metric(
            "Cold Chamber",
            f"{st.session_state.chamber_temp:.2f} °C"
        )

    with k2:
        st.metric(
            "NiTi Element",
            f"{st.session_state.wire_temp:.2f} °C"
        )

    with k3:
        st.metric(
            "Air Temperature",
            f"{st.session_state.air_temp:.2f} °C"
        )

    with k4:
        st.metric(
            "Stress",
            f"{st.session_state.stress:.1f} MPa"
        )

    with k5:
        st.metric(
            "Strain",
            f"{st.session_state.strain:.2f} %"
        )

    with k6:
        st.metric(
            "Estimated COP",
            f"{st.session_state.cop:.3f}"
        )


    st.markdown("---")


    # --------------------------------------------------------
    # PHASE SEQUENCER
    # --------------------------------------------------------

    st.subheader("🚦 Elastocaloric Cycle Sequencer")

    p1, p2, p3, p4 = st.columns(4)

    phase_data = [
        (
            p1,
            1,
            "🔥",
            "LOADING",
            "Mechanical loading / stress application"
        ),
        (
            p2,
            2,
            "💨",
            "HEAT REJECTION",
            "Forced-air heat removal"
        ),
        (
            p3,
            3,
            "❄️",
            "UNLOADING",
            "Elastocaloric cooling generation"
        ),
        (
            p4,
            4,
            "🧊",
            "COLD-SIDE RECOVERY",
            "Cold-side heat absorption"
        )
    ]

    for col, phase_number, icon, title, description in phase_data:

        with col:

            if (
                st.session_state.current_stage_idx
                == phase_number
            ):

                st.success(
                    f"{icon} PHASE {phase_number}\n\n"
                    f"**{title}**\n\n"
                    f"{description}"
                )

            else:

                st.info(
                    f"{icon} PHASE {phase_number}\n\n"
                    f"**{title}**\n\n"
                    f"{description}"
                )


    # --------------------------------------------------------
    # MANUAL PHASE CONTROL
    # --------------------------------------------------------

    if mode == "Manual Diagnostics":

        st.markdown("---")
        st.subheader("🕹️ Manual Phase Command")

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            if st.button(
                "🔥 Load",
                use_container_width=True
            ):
                execute_step_physics(1)
                st.rerun()

        with m2:
            if st.button(
                "💨 Reject Heat",
                use_container_width=True
            ):
                execute_step_physics(2)
                st.rerun()

        with m3:
            if st.button(
                "❄️ Unload",
                use_container_width=True
            ):
                execute_step_physics(3)
                st.rerun()

        with m4:
            if st.button(
                "🧊 Recover Cold",
                use_container_width=True
            ):
                execute_step_physics(4)
                st.session_state.cycle_count += 1

                st.session_state.cycle_history.append(
                    {
                        "Cycle": st.session_state.cycle_count,
                        "Cold Temp (°C)": round(
                            st.session_state.chamber_temp,
                            2
                        ),
                        "Maximum Stress (MPa)": round(
                            abs(st.session_state.stress),
                            2
                        ),
                        "Maximum Strain (%)": strain_limit,
                        "COP": round(
                            st.session_state.cop,
                            3
                        )
                    }
                )

                st.rerun()


    # --------------------------------------------------------
    # PROCESS FLOW
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader("🔧 Process Flow & Valve Status")

    f1, f2, f3, f4, f5 = st.columns(5)

    current = st.session_state.current_stage_idx

    with f1:
        st.markdown("### 🟦")
        st.markdown("**NiTi Core**")
        st.caption(
            f"{num_elements} elements | "
            f"{element_length:.0f} mm"
        )

    with f2:
        st.markdown("### ⚙️")
        st.markdown("**Actuator**")
        st.caption(
            f"{st.session_state.strain:.2f}% strain"
        )

    with f3:

        if current == 2:
            st.markdown("### 🟢")
            st.markdown("**Air Valve**")
            st.caption("OPEN — HEAT REJECTION")
        else:
            st.markdown("### 🔴")
            st.markdown("**Air Valve**")
            st.caption("STANDBY")

    with f4:

        if current == 3:
            st.markdown("### 🟢")
            st.markdown("**Cold-Side Path**")
            st.caption("ACTIVE")
        else:
            st.markdown("### 🔴")
            st.markdown("**Cold-Side Path**")
            st.caption("STANDBY")

    with f5:

        st.markdown("### 🟢")
        st.markdown("**Controller**")

        if st.session_state.auto_running:
            st.caption("AUTO RUNNING")
        else:
            st.caption("OPERATOR CONTROL")


    # --------------------------------------------------------
    # TEMPERATURE GRAPH
    # --------------------------------------------------------

    st.markdown("---")

    graph1, graph2 = st.columns(2)

    with graph1:

        st.subheader("🌡️ Live Thermal Response")

        fig, ax = plt.subplots(figsize=(8, 4))

        if len(st.session_state.temp_history) > 1:

            x = np.arange(
                len(st.session_state.temp_history)
            )

            ax.plot(
                x,
                st.session_state.temp_history,
                linewidth=2
            )

        ax.axhline(
            target_temp,
            linestyle="--",
            linewidth=1.5
        )

        ax.axhline(
            ambient_temp,
            linestyle=":",
            linewidth=1.2
        )

        ax.set_xlabel("Simulation Sample")
        ax.set_ylabel("Temperature (°C)")
        ax.set_title(
            "Cold Chamber Temperature"
        )

        ax.grid(True, alpha=0.25)

        st.pyplot(fig)


    # --------------------------------------------------------
    # STRESS STRAIN GRAPH
    # --------------------------------------------------------

    with graph2:

        st.subheader("⚙️ Mechanical Operating State")

        strain_points = np.array(
            [
                0,
                strain_limit * 0.25,
                strain_limit * 0.50,
                strain_limit * 0.75,
                strain_limit
            ]
        )

        stress_points = (
            450
            + strain_points * 12
        )

        if force_mode == "Uniaxial Compression":
            stress_points = -stress_points

        fig2, ax2 = plt.subplots(figsize=(8, 4))

        ax2.plot(
            strain_points,
            stress_points,
            marker="o",
            linewidth=2
        )

        ax2.scatter(
            [st.session_state.strain],
            [st.session_state.stress],
            s=80
        )

        ax2.set_xlabel("Strain (%)")
        ax2.set_ylabel("Stress (MPa)")
        ax2.set_title(
            "Representative Mechanical State"
        )

        ax2.grid(True, alpha=0.25)

        st.pyplot(fig2)


    # --------------------------------------------------------
    # ALARM / INTERLOCK PANEL
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader("🚨 Safety & Interlock Monitoring")

    alarm1, alarm2, alarm3, alarm4 = st.columns(4)

    temperature_alarm = (
        st.session_state.chamber_temp
        < target_temp - 5
    )

    strain_alarm = (
        st.session_state.strain
        > 6.0
    )

    stress_alarm = (
        abs(st.session_state.stress)
        > 550
    )

    if temperature_alarm:
        st.session_state.alarm_active = True

    if strain_alarm or stress_alarm:
        st.session_state.alarm_active = True

    with alarm1:

        if temperature_alarm:
            st.error("⚠️ COLD-SIDE LIMIT")
        else:
            st.success("✓ Temperature OK")

    with alarm2:

        if strain_alarm:
            st.error("⚠️ STRAIN LIMIT")
        else:
            st.success("✓ Strain OK")

    with alarm3:

        if stress_alarm:
            st.error("⚠️ STRESS LIMIT")
        else:
            st.success("✓ Stress OK")

    with alarm4:

        if st.session_state.auto_running:
            st.warning("AUTO SEQUENCE ACTIVE")
        else:
            st.success("SYSTEM SAFE / STANDBY")


    st.info(
        f"**Controller Message:** "
        f"{st.session_state.system_message}"
    )


# ============================================================
# TAB 2 — ENGINEERING PARAMETERS
# ============================================================

with tab2:

    st.header("⚙️ Engineering Parameter Database")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Material & Geometry")

        st.write(
            f"**Material:** {material_selection}"
        )

        st.write(
            f"**Geometry:** {element_type}"
        )

        st.write(
            f"**Active Elements:** {num_elements}"
        )

        st.write(
            f"**Element Length:** "
            f"{element_length:.1f} mm"
        )

        st.write(
            f"**Outer Diameter:** "
            f"{outer_diameter:.1f} mm"
        )

        if inner_diameter > 0:
            st.write(
                f"**Inner Diameter:** "
                f"{inner_diameter:.1f} mm"
            )


    with col2:

        st.subheader("Operating Conditions")

        st.write(
            f"**Maximum Strain:** "
            f"{strain_limit:.2f}%"
        )

        st.write(
            f"**Actuation:** {force_mode}"
        )

        st.write(
            f"**Ambient Temperature:** "
            f"{ambient_temp:.1f} °C"
        )

        st.write(
            f"**Target Cold Temperature:** "
            f"{target_temp:.1f} °C"
        )

        st.write(
            f"**Air Flow:** "
            f"{fan_flow} CFM"
        )

        st.write(
            f"**Phase Duration:** "
            f"{cycle_speed:.1f} s"
        )


    st.markdown("---")

    st.subheader("📐 Derived Engineering Parameters")

    delta_T = calculate_thermal_effect()

    d1, d2, d3, d4 = st.columns(4)

    with d1:
        st.metric(
            "Estimated ΔT Potential",
            f"{delta_T:.2f} °C"
        )

    with d2:

        active_length = (
            num_elements
            * element_length
        )

        st.metric(
            "Total Active Length",
            f"{active_length:.0f} mm"
        )

    with d3:

        if inner_diameter > 0:

            area = (
                np.pi / 4
                * (
                    outer_diameter**2
                    - inner_diameter**2
                )
            )

        else:

            area = (
                np.pi
                * outer_diameter**2
                / 4
            )

        st.metric(
            "Cross-Sectional Area",
            f"{area:.2f} mm²"
        )

    with d4:

        estimated_force = (
            area
            * abs(st.session_state.stress)
        )

        st.metric(
            "Representative Force",
            f"{estimated_force / 1000:.2f} kN"
        )


    st.warning(
        "Engineering note: The thermal and mechanical "
        "values in this application are representative "
        "simulation values for SCADA demonstration. "
        "They should not be treated as experimentally "
        "validated NiTi material-property data."
    )


# ============================================================
# TAB 3 — PERFORMANCE & DATA
# ============================================================

with tab3:

    st.header("📊 Cycle Performance & Data Logging")

    if st.session_state.cycle_history:

        import pandas as pd

        df = pd.DataFrame(
            st.session_state.cycle_history
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        r1, r2, r3 = st.columns(3)

        with r1:

            st.metric(
                "Cycles Logged",
                len(df)
            )

        with r2:

            st.metric(
                "Lowest Chamber Temperature",
                f"{df['Cold Temp (°C)'].min():.2f} °C"
            )

        with r3:

            st.metric(
                "Maximum Recorded COP",
                f"{df['COP'].max():.3f}"
            )

    else:

        st.info(
            "No completed cycles have been logged yet. "
            "Run at least one complete cycle to populate "
            "the performance database."
        )


    st.markdown("---")

    st.subheader("🧪 Research Interpretation")

    st.markdown(
        """
        **Operating principle represented in this prototype:**

        1. **Mechanical Loading** — NiTi elements are mechanically
           stressed, producing elastocaloric heating.

        2. **Heat Rejection** — The generated heat is rejected to
           the surroundings using forced air circulation.

        3. **Mechanical Unloading** — The NiTi elements are unloaded,
           producing a temperature reduction through the
           elastocaloric effect.

        4. **Cold-Side Recovery** — The cooled NiTi elements interact
           with the cold-side chamber and provide a cooling effect.

        5. **Cycle Repetition** — The sequence can be repeated to
           investigate cyclic thermal response and control behavior.
        """
    )


    st.markdown("---")

    st.caption(
        "Prototype SCADA interface for academic research and "
        "concept validation — not a certified industrial control system."
    )


# ============================================================
# AUTOMATED CYCLING ENGINE
# ============================================================

if (
    mode == "Automated Cycling"
    and st.session_state.auto_running
):

    for phase in [1, 2, 3, 4]:

        if not st.session_state.auto_running:
            break

        execute_step_physics(phase)

        time.sleep(cycle_speed)

    if st.session_state.auto_running:

        st.session_state.cycle_count += 1

        st.session_state.cycle_history.append(
            {
                "Cycle": st.session_state.cycle_count,
                "Cold Temp (°C)": round(
                    st.session_state.chamber_temp,
                    2
                ),
                "Maximum Stress (MPa)": round(
                    abs(st.session_state.stress),
                    2
                ),
                "Maximum Strain (%)": strain_limit,
                "COP": round(
                    st.session_state.cop,
                    3
                )
            }
        )

        st.rerun()
