    if st.session_state.chamber_temp <= 0:
        box_color = "#0B3C5D"
        text_color = "white"
    else:
        box_color = "#1E1E1E"
        text_color = "#FFA500"
        
    st.markdown(f"<div style='background-color: {box_color}; padding: 20px; border-radius: 10px; text-align: center; color: {text_color};'><h3>Isolated Sub-Zero Vault</h3><h1>{st.session_state.chamber_temp:.1f}°C</h1><p>Target Goal: -18.0°C</p></div>", unsafe_allowed_html=True)
