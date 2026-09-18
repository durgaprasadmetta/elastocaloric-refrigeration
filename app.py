# ============================================================
# EXECUTE PHASE — CORRECTED VERSION
# ============================================================

def execute_phase(phase):

    # Never execute another phase after target achievement
    if st.session_state.target_reached:
        return False

    delta_t = calculate_delta_temperature()

    target_hit = False

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

        log_event(
            "PHASE 4 — Cold-side recovery active."
        )

        # ----------------------------------------------------
        # TARGET CHECK
        # ----------------------------------------------------

        if (
            st.session_state.chamber_temp
            <= target_temperature
        ):

            st.session_state.chamber_temp = (
                target_temperature
            )

            st.session_state.target_reached = True

            st.session_state.auto_running = False

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

            target_hit = True

    # ========================================================
    # IMPORTANT:
    # ALWAYS LOG DATA BEFORE RETURNING
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

    # --------------------------------------------------------
    # KEEP ALL HISTORIES THE SAME LENGTH
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # UPDATE SENSOR LIMITS
    # --------------------------------------------------------

    update_temperature_limits()

    st.session_state.last_update = (
        datetime.now().strftime("%H:%M:%S")
    )

    return target_hit
