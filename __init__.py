"""
niti_elastocaloric
===================
Reusable physics/model + I/O package backing the NiTi Elastocaloric
Refrigeration SCADA/HMI Streamlit app (app.py).

Submodules:
    model — Material/Geometry/Config dataclasses, the PhysicsEngine lumped-
            parameter solver, parameter sweeps, alarms, and summary tables.
    io    — CSV/Excel export, measured-data import, and simulation-vs-
            experiment comparison helpers.
"""

__all__ = ["model", "io"]
