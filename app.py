from __future__ import annotations

from dataclasses import replace
from io import BytesIO

import pandas as pd
import plotly.express as px
import streamlit as st

from niti_elastocaloric.io import compare_frames, csv_bytes, excel_bytes, import_experiment
from niti_elastocaloric.model import (
    MATERIALS,
    PHASES,
    Config,
    Geometry,
    PhysicsEngine,
    active_alarms,
    build_config,
    config_frame,
    geometry_table,
    run_parameter_sweep,
    run_simulation,
    summarize_cycles,
)


st.set_page_config(page_title="NiTi Elastocaloric Refrigeration", page_icon="❄️", layout="wide")


def init_session() -> None:
    defaults = {
        "simulation": None,
        "experiment": pd.DataFrame(),
        "live_records": [],
        "live_state": None,
        "live_config_key": None,
        "running": False,
        "logging": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def config_signature(config: Config) -> str:
    return repr(config.as_dict())


def sidebar_config() -> Config:
    st.sidebar.header("Operating mode")
    mode = st.sidebar.radio("Data source", ["SIMULATION", "EXPERIMENT"], horizontal=True)
    st.session_state["mode"] = mode
    st.sidebar.divider()
    st.sidebar.header("Configuration")
    material_key = st.sidebar.selectbox("Material", list(MATERIALS), index=0)
    material = MATERIALS[material_key]
    form = st.sidebar.selectbox("Element form", ["Tube", "Wire"])
    n_elements = st.sidebar.number_input("Number of elements", min_value=1, max_value=100, value=5, step=1)
    od_mm = st.sidebar.number_input("Outer diameter / wire diameter (mm)", min_value=0.1, value=12.0, step=0.5)
    id_mm = 0.0 if form == "Wire" else st.sidebar.number_input("Inner diameter (mm)", min_value=0.0, value=10.0, step=0.5)
    length_mm = st.sidebar.number_input("Element length (mm)", min_value=1.0, value=150.0, step=5.0)
    max_strain_pct = st.sidebar.number_input("Maximum strain (%)", min_value=0.0, value=material.transformation_strain * 100, step=0.1)
    phase_time_s = st.sidebar.number_input("Phase time (s)", min_value=0.1, value=5.0, step=0.5)
    ambient_c = st.sidebar.number_input("Ambient temperature (°C)", value=25.0, step=1.0)
    target_c = st.sidebar.number_input("Target temperature (°C)", value=10.0, step=1.0)
    chamber_capacity = st.sidebar.number_input("Chamber thermal capacity (J/K)", min_value=1.0, value=800.0, step=50.0)
    ua = st.sidebar.number_input("Insulation UA (W/K)", min_value=0.0, value=0.35, step=0.05)
    exchanger = st.sidebar.selectbox("Heat exchanger", ["Finned forced air", "Bare tube forced air"])
    h = 620.0 if exchanger.startswith("Finned") else 90.0
    airflow = st.sidebar.number_input("Air flow (CFM)", min_value=0.1, value=50.0, step=5.0)
    cycles = st.sidebar.number_input("Number of cycles", min_value=1, max_value=500, value=40, step=1)
    dt = st.sidebar.selectbox("Recording interval (s)", [0.1, 0.2, 0.4, 0.5, 1.0], index=3)
    try:
        return build_config(
            material_key=material_key,
            form=form,
            n_elements=int(n_elements),
            od_mm=od_mm,
            id_mm=id_mm,
            length_mm=length_mm,
            max_strain=max_strain_pct / 100,
            phase_time_s=phase_time_s,
            ambient_c=ambient_c,
            target_c=target_c,
            chamber_capacity_j_k=chamber_capacity,
            insulation_ua_w_k=ua,
            exchanger=exchanger,
            h_w_m2k=h,
            air_flow_cfm=airflow,
            n_cycles=int(cycles),
            dt_s=dt,
        )
    except ValueError as exc:
        st.sidebar.error(str(exc))
        return Config()


def header(config: Config) -> None:
    st.title("NiTi ELASTOCALORIC REFRIGERATION")
    st.caption("Advanced Elastocaloric Refrigeration Using NiTi Alloy · RESEARCH SCADA / HMI")
    mode = st.session_state.get("mode", "SIMULATION")
    st.success(f"● CONTROLLER ONLINE · {mode} DATA · IDEALIZED / LUMPED-PARAMETER MODEL")
    if not config.fixed_target_is_valid:
        st.warning("Target must be below ambient for cooling mode.")


def status_cards(config: Config, frame: pd.DataFrame, state) -> None:
    last = frame.iloc[-1].to_dict() if not frame.empty else {}
    values = [
        ("CHAMBER", f"{last.get('chamber_c', config.ambient_c):.2f} °C"),
        ("NiTi ELEMENT", f"{last.get('element_c', config.ambient_c):.2f} °C"),
        ("AIR IN / OUT", f"{last.get('air_in_c', config.ambient_c):.2f} / {last.get('air_out_c', config.ambient_c):.2f} °C"),
        ("TARGET", f"{config.target_c:.2f} °C"),
        ("TIME TO TARGET", f"{state.target_time_s:.1f} s" if state and state.target_time_s is not None else "—"),
        ("CYCLE / PHASE", f"{state.cycle if state else 0} · {state.phase.value if state else 'LOAD'}"),
        ("STRAIN / STRESS", f"{last.get('strain_pct', 0):.2f}% / {last.get('stress_mpa', 0):.1f} MPa"),
        ("COOLING POWER", f"{last.get('cooling_w', 0):.2f} W"),
        ("COP", f"{last.get('cop', 0):.3f}"),
    ]
    cols = st.columns(3)
    for index, (label, value) in enumerate(values):
        cols[index % 3].metric(label, value)


def current_frame() -> pd.DataFrame:
    if st.session_state.get("mode") == "EXPERIMENT" and not st.session_state.experiment.empty:
        return st.session_state.experiment
    sim = st.session_state.get("simulation")
    return sim.readings if sim is not None else pd.DataFrame(st.session_state.get("live_records", []))


def dashboard_page(config: Config) -> None:
    frame = current_frame()
    state = st.session_state.get("live_state") or (st.session_state.simulation.state if st.session_state.simulation else None)
    status_cards(config, frame, state)
    alarms = active_alarms(config, state)
    if alarms:
        for level, message in alarms:
            (st.error if level == "ALARM" else st.warning)(f"{level} — {message}")
    else:
        st.info("Safety interlocks clear at the current operating point.")
    if frame.empty:
        st.info("No readings yet. Use Run complete simulation or the Control page to advance the solver.")
        return
    temperature_columns = [column for column in ["chamber_c", "element_c", "air_in_c", "air_out_c"] if column in frame]
    plot = px.line(frame.tail(500), x="time_s", y=temperature_columns, title="Temperature vs Time")
    if "chamber_c" in frame:
        plot.add_hline(y=config.target_c, line_dash="dash", annotation_text="Target")
    st.plotly_chart(plot, use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.subheader("Engineering properties")
        st.dataframe(geometry_table(config), hide_index=True, use_container_width=True)
    with right:
        st.subheader("Target status")
        reached = state.target_reached if state else False
        st.metric("Status", "REACHED" if reached else "NOT REACHED")
        st.write(f"Current chamber: {frame.iloc[-1]['chamber_c']:.2f} °C")
        st.write(f"Temperature difference: {frame.iloc[-1]['chamber_c'] - config.target_c:.2f} °C")
        st.write(f"Cycle to target: {state.target_cycle if state and state.target_cycle is not None else '—'}")
        st.write(f"Phase: {state.target_phase or '—' if state else '—'}")


def control_page(config: Config) -> None:
    st.header("Control")
    st.caption("Interactive controls advance the same physics engine used for the batch simulation.")
    steps_per_press = st.number_input("Simulation speed / steps per START press", min_value=1, max_value=500, value=10, step=1)
    cols = st.columns(6)
    if cols[0].button("START", use_container_width=True):
        st.session_state.running = True
        if st.session_state.live_state is None:
            st.session_state.live_state = PhysicsEngine(config).state
        engine = PhysicsEngine(config, st.session_state.live_state)
        for _ in range(int(steps_per_press)):
            st.session_state.live_records.append(engine.step())
        st.session_state.live_state = engine.state
        st.rerun()
    if cols[1].button("PAUSE", use_container_width=True):
        st.session_state.running = False
    if cols[2].button("STOP", use_container_width=True):
        st.session_state.running = False
    if cols[3].button("RESET", use_container_width=True):
        st.session_state.live_records = []
        st.session_state.live_state = None
        st.session_state.simulation = None
        st.session_state.running = False
        st.rerun()
    if cols[4].button("SINGLE CYCLE", use_container_width=True):
        if st.session_state.live_state is None:
            st.session_state.live_state = PhysicsEngine(config).state
        engine = PhysicsEngine(config, st.session_state.live_state)
        start_cycle = engine.state.cycle
        while engine.state.cycle == start_cycle and len(st.session_state.live_records) < 10_000:
            st.session_state.live_records.append(engine.step())
        st.session_state.live_state = engine.state
    if cols[5].button("STEP PHASE", use_container_width=True):
        if st.session_state.live_state is None:
            st.session_state.live_state = PhysicsEngine(config).state
        engine = PhysicsEngine(config, st.session_state.live_state)
        start_phase = engine.state.phase_index
        while engine.state.phase_index == start_phase and len(st.session_state.live_records) < 10_000:
            st.session_state.live_records.append(engine.step())
        st.session_state.live_state = engine.state
    log_cols = st.columns(3)
    if log_cols[0].button("START LOGGING", use_container_width=True):
        st.session_state.logging = True
    if log_cols[1].button("STOP LOGGING", use_container_width=True):
        st.session_state.logging = False
    if log_cols[2].button("RESET TARGET TIMER", use_container_width=True):
        state = st.session_state.live_state
        if state is not None:
            state.target_reached = False
            state.target_time_s = None
            state.target_cycle = None
            state.target_phase = None
            for record in st.session_state.live_records:
                record["target_reached"] = False
        if st.session_state.simulation is not None:
            state = st.session_state.simulation.state
            state.target_reached = False
            state.target_time_s = None
            state.target_cycle = None
            state.target_phase = None
    st.write(f"Controller state: **{'RUNNING' if st.session_state.running else 'PAUSED / READY'}**")
    st.write(f"Data logging: **{'ACTIVE' if st.session_state.logging else 'STOPPED'}**")
    if st.button("RUN COMPLETE SIMULATION", type="primary"):
        with st.spinner("Running coupled mechanical, thermal, chamber and energy model..."):
            st.session_state.simulation = run_simulation(config)
            st.session_state.live_records = st.session_state.simulation.readings.to_dict("records")
            st.session_state.live_state = st.session_state.simulation.state
        st.success("Simulation complete. All displayed points originate from the physics engine.")
    frame = current_frame()
    if not frame.empty:
        status_cards(config, frame, st.session_state.live_state)
        st.dataframe(frame.tail(20), hide_index=True, use_container_width=True)


def configuration_page(config: Config) -> None:
    st.header("Configuration & Engineering Model")
    st.warning("Material properties are reference values for an idealized model, not guaranteed laboratory properties.")
    st.subheader("Calculated configuration")
    st.dataframe(config_frame(config), hide_index=True, use_container_width=True)
    with st.expander("Equations used by the solver", expanded=True):
        st.markdown(
            """
            **Geometry**  
            `A = π/4 (OD² − ID²)` · `V = A_total × L` · `m = ρV` · `C = m Cp`

            **Mechanics**  
            `ΔL = εL` · `F = σA` · `W_mech = ∫ F dx` using timestep integration

            **Elastocaloric effect**  
            `ξ = clamp(ε / ε_tr, 0, 1)` · `ΔT_ad ≈ T ΔS ξ / Cp`

            **Air-side heat transfer**  
            `V̇ = CFM × 0.00047194745` · `ṁ = ρV̇` · `Q̇_air = ṁ Cp (T_in − T_out)`

            **Chamber and performance**  
            `C_ch dT_ch/dt = UA(T_amb − T_ch) − Q_useful` · `COP = Q_cooling / W_input`
            """
        )
    st.subheader("Assumptions and limitations")
    st.write(
        "This is an idealized lumped-parameter reference model. It does not claim exact laboratory prediction, "
        "material certification, fatigue validation, or experimental validation. Detailed hysteresis, contact "
        "resistance, fluid distribution, actuator dynamics, and sensor uncertainty are not resolved."
    )


def trends_page(config: Config) -> None:
    st.header("Live Trends")
    frame = current_frame()
    if frame.empty:
        st.info("Run a simulation or advance the controller first.")
        return
    charts = [
        (["stress_mpa"], "Stress vs Time"),
        (["strain_pct"], "Strain vs Time"),
        (["cooling_w"], "Cooling Power vs Time"),
        (["force_n", "displacement_mm"], "Force and Displacement vs Time"),
    ]
    for columns, title in charts:
        available = [column for column in columns if column in frame]
        if available:
            st.plotly_chart(px.line(frame, x="time_s", y=available, title=title), use_container_width=True)
    if {"strain_pct", "stress_mpa"}.issubset(frame.columns):
        st.plotly_chart(px.scatter(frame, x="strain_pct", y="stress_mpa", color="phase" if "phase" in frame else None, title="Stress vs Strain"), use_container_width=True)


def cycle_analysis_page(config: Config) -> None:
    st.header("Cycle Analysis")
    simulation = st.session_state.simulation
    if simulation is None:
        st.info("Run the complete simulation to generate cycle summaries.")
        return
    summary = simulation.cycle_summary
    st.dataframe(summary, hide_index=True, use_container_width=True)
    if not summary.empty:
        a, b = st.columns(2)
        a.plotly_chart(px.line(summary, x="cycle", y="cop", title="COP vs Cycle"), use_container_width=True)
        b.plotly_chart(px.line(summary, x="cycle", y="chamber_c", title="Chamber Temperature vs Cycle"), use_container_width=True)
        st.plotly_chart(px.line(summary, x="cycle", y="cooling_energy_j", title="Cooling Energy vs Cycle"), use_container_width=True)


def experiment_page(config: Config) -> None:
    st.header("Experiment")
    st.info("Measured data is kept separate from calculated simulation data. No experimental results are generated by this app.")
    upload = st.file_uploader("Import measured CSV or XLSX", type=["csv", "xlsx", "xls"])
    if upload is not None:
        try:
            st.session_state.experiment = import_experiment(upload)
            st.success(f"Loaded {len(st.session_state.experiment):,} measured readings.")
        except (ValueError, ImportError) as exc:
            st.error(str(exc))
    if not st.session_state.experiment.empty:
        st.dataframe(st.session_state.experiment.head(100), hide_index=True, use_container_width=True)
        st.plotly_chart(px.line(st.session_state.experiment, x="time_s", y=["chamber_c", "element_c"], title="Experimental temperatures"), use_container_width=True)


def comparison_page(config: Config) -> None:
    st.header("Simulation vs Experiment")
    simulation = current_frame()
    experiment = st.session_state.experiment
    if simulation.empty or experiment.empty:
        st.info("Run a simulation and import measured data on the Experiment page.")
        return
    metrics = st.multiselect("Compare signals", ["chamber_c", "element_c", "air_out_c", "stress_mpa", "strain_pct", "cooling_w"], default=["chamber_c", "element_c", "air_out_c"])
    joined, summary = compare_frames(simulation, experiment, metrics)
    st.dataframe(summary, hide_index=True, use_container_width=True)
    if not joined.empty:
        for metric in metrics:
            columns = [f"{metric}_simulation", f"{metric}_experimental"]
            if all(column in joined for column in columns):
                st.plotly_chart(px.line(joined, x="time_s", y=columns, title=f"{metric}: calculated vs measured"), use_container_width=True)


def data_page(config: Config) -> None:
    st.header("Data & Export")
    frame = current_frame()
    if frame.empty:
        st.info("No simulation data available yet.")
        return
    st.download_button("EXPORT CSV", csv_bytes(frame), "niti_time_history.csv", "text/csv")
    sim = st.session_state.simulation
    if sim is not None:
        st.download_button("EXPORT EXCEL", excel_bytes(sim), "niti_research_run.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.info("Excel export with configuration and target sheets becomes available after a simulation run.")
    st.dataframe(frame, hide_index=True, use_container_width=True)


def target_analysis_page(config: Config) -> None:
    st.header("Target Analysis")
    frame = current_frame()
    state = st.session_state.live_state or (st.session_state.simulation.state if st.session_state.simulation else None)
    reached = bool(state and state.target_reached)
    if reached:
        st.success("TARGET REACHED")
    else:
        st.warning("TARGET NOT REACHED WITH CURRENT PARAMETERS")
    cols = st.columns(5)
    cols[0].metric("Current chamber", f"{frame.iloc[-1]['chamber_c']:.2f} °C" if not frame.empty and "chamber_c" in frame else "—")
    cols[1].metric("Target", f"{config.target_c:.2f} °C")
    cols[2].metric("Temperature difference", f"{frame.iloc[-1]['chamber_c'] - config.target_c:.2f} °C" if not frame.empty and "chamber_c" in frame else "—")
    cols[3].metric("Time to target", f"{state.target_time_s:.1f} s" if state and state.target_time_s is not None else "—")
    cols[4].metric("Cycle to target", str(state.target_cycle) if state and state.target_cycle is not None else "—")
    if state:
        st.write(f"Phase at target: **{state.target_phase or '—'}**")
    if not frame.empty and "chamber_c" in frame:
        chart = px.line(frame, x="time_s", y="chamber_c", title="Chamber temperature vs target")
        chart.add_hline(y=config.target_c, line_dash="dash", annotation_text="Target")
        st.plotly_chart(chart, use_container_width=True)


def analysis_page(config: Config) -> None:
    st.header("Parameter Study & Recommendation")
    tab1, tab2 = st.tabs(["Parameter sweep", "Candidate configurations"])
    with tab1:
        parameter = st.selectbox("Sweep parameter", ["Material", "Number of elements", "Strain (%)", "Air flow (CFM)", "Heat transfer coefficient", "Phase time (s)", "Target (°C)"])
        defaults = {
            "Material": "NiTi,Cu-Al-Ni,Fe-Mn-Si,Natural Rubber",
            "Number of elements": "3,5,8",
            "Strain (%)": "3,5.5,7",
            "Air flow (CFM)": "25,50,75",
            "Heat transfer coefficient": "90,620",
            "Phase time (s)": "3,5,8",
            "Target (°C)": "8,10,12",
        }
        values_text = st.text_input("Values, comma separated", defaults[parameter])
        if st.button("RUN PARAMETER SWEEP"):
            values = [item.strip() for item in values_text.split(",") if item.strip()]
            result = run_parameter_sweep(config, parameter, values)
            st.session_state["sweep"] = result
        if "sweep" in st.session_state:
            st.dataframe(st.session_state.sweep, hide_index=True, use_container_width=True)
            st.download_button("EXPORT SWEEP CSV", csv_bytes(st.session_state.sweep), "niti_parameter_sweep.csv", "text/csv")
    with tab2:
        st.write("Candidates are sorted only by the engineering criterion selected by the user.")
        criterion = st.selectbox("Sort criterion", ["COP", "Cooling power (W)", "Minimum chamber °C", "Time to target (s)"])
        if st.button("FIND CANDIDATES"):
            values = [3, 5.5, 7]
            sweep = run_parameter_sweep(config, "Strain (%)", values)
            ascending = criterion in {"Minimum chamber °C", "Time to target (s)"}
            st.dataframe(sweep.sort_values(criterion, ascending=ascending, na_position="last"), hide_index=True, use_container_width=True)


def safety_page(config: Config) -> None:
    st.header("Safety & Interlocks")
    state = st.session_state.live_state or (st.session_state.simulation.state if st.session_state.simulation else None)
    alarms = active_alarms(config, state)
    if not alarms:
        st.success("ALL INTERLOCKS CLEAR")
    else:
        for level, message in alarms:
            (st.error if level == "ALARM" else st.warning)(f"{level} — {message}")
    st.dataframe(
        pd.DataFrame(
            [
                ("Stress limit", config.material.stress_limit_mpa, "MPa"),
                ("Transformation strain", config.material.transformation_strain * 100, "%"),
                ("Safe temperature range", f"{config.safe_min_c} to {config.safe_max_c}", "°C"),
                ("Indicative fatigue life", config.material.fatigue_cycles, "cycles"),
                ("Wall thickness", config.geometry.wall_thickness_mm, "mm"),
                ("Target vs Af", f"{config.target_c} vs {config.material.af_c}", "°C"),
            ],
            columns=["Interlock", "Value", "Unit"],
        ),
        hide_index=True,
        use_container_width=True,
    )


def main() -> None:
    init_session()
    config = sidebar_config()
    header(config)
    pages = ["Dashboard", "Control", "Configuration", "Live Trends", "Cycle Analysis", "Target Analysis", "Experiment", "Simulation vs Experiment", "Safety", "Data", "Parameter Study"]
    page = st.sidebar.radio("Navigate", pages)
    if page == "Dashboard":
        dashboard_page(config)
    elif page == "Control":
        control_page(config)
    elif page == "Configuration":
        configuration_page(config)
    elif page == "Live Trends":
        trends_page(config)
    elif page == "Cycle Analysis":
        cycle_analysis_page(config)
    elif page == "Target Analysis":
        target_analysis_page(config)
    elif page == "Experiment":
        experiment_page(config)
    elif page == "Simulation vs Experiment":
        comparison_page(config)
    elif page == "Safety":
        safety_page(config)
    elif page == "Data":
        data_page(config)
    else:
        analysis_page(config)


if __name__ == "__main__":
    main()
