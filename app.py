with st.expander("🧑‍🔬 Ask the material expert", expanded=False):
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
                    st.session_state["w_material"] = best_c.material_key
                    st.session_state["w_strain"] = best_c.strain_pct
                    st.session_state["w_phase_time"] = best_c.phase_time_s
                    st.session_state["w_hx"] = best_c.exchanger_key
                    st.session_state["w_ambient"] = best_c.ambient_c
                    st.session_state["w_target"] = best_c.target_c
                    st.session_state["ae_ran"] = False
                    st.session_state["_pending_full_reset"] = False
                    event("Settings applied automatically by the material "
                          "expert.")
                    reset("Run cleared — new expert-recommended settings "
                          "loaded.")
                    st.rerun()
            with aa2:
                if st.button("No, I'll set it myself", key="ae_skip",
                             use_container_width=True):
                    st.info("No problem — use the values listed above.")
