"""
niti_elastocaloric.io
=======================
Export/import helpers used by the Experiment, Simulation vs Experiment,
and Data pages: CSV/Excel export, measured-data import with column
auto-detection, and a simulation-vs-experiment comparison.
"""

from __future__ import annotations

from io import BytesIO
from typing import Sequence, Tuple

import pandas as pd

__all__ = ["csv_bytes", "excel_bytes", "import_experiment", "compare_frames"]


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def excel_bytes(simulation) -> bytes:
    """`simulation` is a niti_elastocaloric.model.Simulation — time
    history, cycle summary, configuration and target-result sheets."""
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        simulation.readings.to_excel(writer, sheet_name="Time history", index=False)
        if simulation.cycle_summary is not None and not simulation.cycle_summary.empty:
            simulation.cycle_summary.to_excel(writer, sheet_name="Cycle summary", index=False)
        pd.DataFrame([simulation.config.as_dict()]).to_excel(
            writer, sheet_name="Configuration", index=False)
        pd.DataFrame({
            "Target reached": [simulation.state.target_reached],
            "Time to target (s)": [simulation.state.target_time_s],
            "Cycle to target": [simulation.state.target_cycle],
            "Phase at target": [simulation.state.target_phase],
        }).to_excel(writer, sheet_name="Target summary", index=False)
    return buffer.getvalue()


_TIME_COLUMN_CANDIDATES = ("time_s", "time", "t", "elapsed_s", "time (s)")
_COLUMN_ALIASES = {
    "time_s": _TIME_COLUMN_CANDIDATES,
    "chamber_c": ("chamber_c", "chamber", "chamber_temp_c", "chamber (°c)", "chamber (c)"),
    "element_c": ("element_c", "element", "wire_c", "element (°c)", "element (c)"),
    "air_in_c": ("air_in_c", "air_in", "air in (°c)", "air in (c)"),
    "air_out_c": ("air_out_c", "air_out", "air out (°c)", "air out (c)"),
    "strain_pct": ("strain_pct", "strain", "strain (%)"),
    "stress_mpa": ("stress_mpa", "stress", "stress (mpa)"),
    "cooling_w": ("cooling_w", "cooling", "cooling_power_w", "cooling (w)"),
    "cop": ("cop",),
}


def import_experiment(upload) -> pd.DataFrame:
    """Reads an uploaded CSV/XLSX of measured data and maps its columns
    onto the app's canonical names (time_s, chamber_c, element_c, ...)
    by matching against common header variants, case-insensitively.
    Raises ValueError for unreadable/empty files or a missing time
    column, and ImportError if the file format needs a package that
    isn't installed (e.g. openpyxl for .xlsx)."""
    name = (getattr(upload, "name", "") or "").lower()
    try:
        if name.endswith((".xlsx", ".xls")):
            frame = pd.read_excel(upload)
        else:
            frame = pd.read_csv(upload)
    except ImportError as exc:
        raise ImportError(
            "Reading this Excel file needs an engine package that isn't "
            "installed (openpyxl for .xlsx, xlrd for legacy .xls). Install "
            "it, or upload a CSV instead."
        ) from exc
    except Exception as exc:  # noqa: BLE001 — surfaced to the user as-is
        raise ValueError(f"Could not read the uploaded file: {exc}") from exc

    if frame.empty:
        raise ValueError("The uploaded file contains no rows.")

    lower_lookup = {str(c).strip().lower(): c for c in frame.columns}
    renamed = {}
    for canonical, candidates in _COLUMN_ALIASES.items():
        for candidate in candidates:
            match = lower_lookup.get(candidate)
            if match is not None:
                renamed[match] = canonical
                break
    frame = frame.rename(columns=renamed)

    if "time_s" not in frame.columns:
        raise ValueError(
            "Could not find a time column. Expected a header like one of: "
            + ", ".join(_TIME_COLUMN_CANDIDATES)
        )

    for column in frame.columns:
        if column in _COLUMN_ALIASES:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    # Force a consistent float dtype on the join key: an integer-looking
    # time column (e.g. "0,1,2,3") parses as int64, and pd.merge_asof in
    # compare_frames() requires both sides' join key to share a dtype —
    # without this cast, a perfectly normal uploaded CSV can crash the
    # Simulation vs Experiment page with a MergeError.
    frame["time_s"] = frame["time_s"].astype("float64")

    frame = frame.dropna(subset=["time_s"]).sort_values("time_s").reset_index(drop=True)
    return frame


def compare_frames(simulation: pd.DataFrame, experiment: pd.DataFrame,
                   metrics: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Time-aligns measured data onto the nearest simulated sample (by
    time_s) and reports RMSE/MAE/bias per metric. Returns (joined,
    summary); joined carries `{metric}_simulation` / `{metric}_experimental`
    columns for whichever metrics exist in both frames."""
    empty_summary = pd.DataFrame(columns=["metric", "rmse", "mae", "bias", "n_points"])
    if simulation is None or experiment is None or simulation.empty or experiment.empty or not metrics:
        return pd.DataFrame(), empty_summary
    if "time_s" not in simulation.columns or "time_s" not in experiment.columns:
        return pd.DataFrame(), empty_summary

    shared = [m for m in metrics if m in simulation.columns and m in experiment.columns]
    sim = simulation[["time_s", *shared]].dropna(subset=["time_s"]).sort_values("time_s")
    exp = experiment[["time_s", *shared]].dropna(subset=["time_s"]).sort_values("time_s")
    if sim.empty or exp.empty or not shared:
        return pd.DataFrame(), empty_summary
    # Both sides must share a dtype on the merge key regardless of how
    # each frame originated (simulated readings vs. an uploaded file).
    sim = sim.astype({"time_s": "float64"})
    exp = exp.astype({"time_s": "float64"})

    joined = pd.merge_asof(exp, sim, on="time_s", direction="nearest",
                           suffixes=("_experimental", "_simulation"))

    summary_rows = []
    for metric in shared:
        sim_col, exp_col = f"{metric}_simulation", f"{metric}_experimental"
        if sim_col not in joined.columns or exp_col not in joined.columns:
            continue
        diff = (joined[sim_col] - joined[exp_col]).dropna()
        if diff.empty:
            continue
        summary_rows.append({
            "metric": metric,
            "rmse": float((diff ** 2).mean() ** 0.5),
            "mae": float(diff.abs().mean()),
            "bias": float(diff.mean()),
            "n_points": int(diff.shape[0]),
        })
    summary = pd.DataFrame(summary_rows) if summary_rows else empty_summary
    return joined, summary
