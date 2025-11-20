"""
Dual‑Sweep FET Analyzer Webapp
==============================

This Streamlit application extends the single‑sweep FET analyzer to handle
datasets obtained from dual gate‑voltage sweeps.  The uploaded `.xlsx`
file must contain at least the following numeric columns:

* `drainI` — drain current (ID)
* `drainV` — drain–source voltage (VDS)
* `gateI`  — gate leakage current (IG)
* `gateV`  — gate–source voltage (VGS)

Based on guidelines from *How to report and benchmark emerging field‑effect
transistors*【741723062978635†L266-L269】【741723062978635†L352-L357】, this app performs the following tasks:

1. **Separate the dual sweep into forward and reverse segments.**  The
   separation point is detected by finding the first index where the
   gate voltage decreases (sign change of the derivative), assuming the
   first sweep is monotonically increasing in VGS and the second is
   decreasing.
2. **Allow the user to select which sweep (forward or reverse) to
   analyze.**
3. **Subthreshold swing (SS) extraction**: Display `log10(drainI)` vs
   `gateV` and allow the user to select the voltage range for linear
   regression.  The slope of this fit is used to compute the subthreshold
   swing in mV/decade【741723062978635†L358-L361】.
4. **Transconductance (gm) extraction**: Display `drainI` vs `gateV` and
   allow the user to choose the voltage range for linear regression.  The
   slope yields gm【741723062978635†L352-L357】.
5. **Mobility calculation**: Use gm and user‑provided device parameters
   (channel length, channel width, insulator capacitance and VDS) to
   compute the field‑effect mobility μFE via μFE = L·gm/(W·Cins·VDS)
   【741723062978635†L266-L269】.
6. **On/off current ratio**: Display `drainI` vs `gateV` (with log scale
   on the y‑axis) and ask the user to define a current threshold.  Values
   above the threshold are treated as the on‑state; values below, the
   off‑state.  The maximum on‑state current and minimum off‑state current
   are used to compute Ion/Ioff【741723062978635†L363-L367】.

The app preserves the original features for interactive tables and saving
regression results.  To run it, install dependencies from `requirements.txt`
and execute `streamlit run fet_dual_sweep_app.py`.
"""

import io
import time
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def load_excel(file) -> pd.DataFrame:
    """Load the first sheet from an uploaded Excel file into a DataFrame."""
    try:
        return pd.read_excel(file)
    except Exception as exc:
        raise ValueError(f"Could not read the Excel file: {exc}")


def numeric_columns(df: pd.DataFrame) -> List[str]:
    """Return a list of numeric column names from the DataFrame."""
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def dataframe_to_bytes(df: pd.DataFrame, filetype: str = "csv") -> bytes:
    """Serialize a DataFrame to CSV or Excel bytes for downloading."""
    if filetype == "csv":
        return df.to_csv(index=False).encode("utf-8")
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="results")
    return output.getvalue()


def perform_linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Perform a simple linear regression on the provided arrays and return
    slope, intercept, R² and RMSE.
    """
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    r2 = float(r2_score(y, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    return slope, intercept, r2, rmse


def compute_subthreshold_swing(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the subthreshold swing (SS) in mV/decade from gate voltage (VGS)
    and drain current (ID) samples.  Assumes x and y are already filtered to
    the desired subthreshold region【741723062978635†L358-L361】.
    """
    # Filter out non‑positive currents to avoid log10 issues
    mask = y > 0
    if mask.sum() < 2:
        return float("nan")
    x_valid = x[mask]
    log_id = np.log10(y[mask])
    X = x_valid.reshape(-1, 1)
    slope, _, _, _ = perform_linear_regression(X, log_id)
    if slope == 0:
        return float("nan")
    return 1000.0 / slope


def compute_mobility(gm: float, channel_length: float, channel_width: float, cins: float, vds: float) -> float:
    """
    Compute the field‑effect mobility μFE in cm²/Vs using μFE = L·gm/(W·Cins·VDS)【741723062978635†L266-L269】.
    Inputs must be in SI units (L and W in m, Cins in F/m², gm in A/V, VDS in V).  Returns μFE in cm²/Vs.
    """
    denom = channel_width * cins * vds
    if denom == 0:
        return float("nan")
    mu_m2 = (channel_length * gm) / denom
    return mu_m2 * 1e4


# def compute_ion_ioff(currents: np.ndarray, threshold: float) -> Tuple[float, float, float]:
#     """
#     Compute Ion (max ID above a threshold), Ioff (min ID below the threshold) and their ratio Ion/Ioff.
#     This allows the user to define a current threshold to separate on/off states【741723062978635†L363-L367】.
#     """
#     if currents.size == 0:
#         return float("nan"), float("nan"), float("nan")
#     # Identify on/off states
#     on_values = currents[currents >= threshold]
#     off_values = currents[currents < threshold]
#     ion = float(np.nanmax(on_values)) if on_values.size > 0 else float("nan")
#     ioff = float(np.nanmin(off_values)) if off_values.size > 0 else float("nan")
#     if ioff is None or ioff <= 0 or np.isnan(ion) or np.isnan(ioff):
#         ratio = float("nan")
#     else:
#         ratio = ion / ioff
#     return ion, ioff, ratio


def separate_sweeps(df: pd.DataFrame, gate_col: str = "gateV") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separate a dual sweep into forward and reverse segments based on
    the gate voltage column.  The function identifies the first index
    where the derivative of gate voltage becomes negative, assuming
    the forward sweep is increasing.  Returns two DataFrames: (forward,
    reverse).  If no sign change is detected, the entire dataset is
    treated as forward and the reverse DataFrame is empty.
    """
    if gate_col not in df.columns or len(df) < 2:
        return df.copy(), df.iloc[0:0].copy()
    gate = df[gate_col].to_numpy()
    dV = np.diff(gate)
    sign = np.sign(dV)
    idx_rev_start: Optional[int] = None
    for i, s in enumerate(sign):
        # # look for a transition from non‑negative to negative
        # if i==0:
        #     flag = s < 0
        # if flag and s > 0:
        #     idx_rev_start = i + 1
        #     break
        # elif not flag and s < 0:
        #     idx_rev_start = i + 1
        #     break
        if s == 0:
            idx_rev_start = i + 1
            break
    if idx_rev_start is None:
        return df.copy(), df.iloc[0:0].copy()
    return df.iloc[:idx_rev_start].copy(), df.iloc[idx_rev_start:].copy()


def create_results_table() -> pd.DataFrame:
    """Initialize an empty results table for storing regression parameters."""
    columns = [
        "timestamp",
        "source_file",
        "sweep",
        "parameter",
        "fit_min",
        "fit_max",
        "slope",
        "intercept",
        "r2",
        "rmse",
        "n_points",
    ]
    return pd.DataFrame(columns=columns)


def main() -> None:
    """Run the dual‑sweep FET analyzer Streamlit app."""
    st.set_page_config(page_title="Dual Sweep FET Analyzer", layout="wide")
    if "results_table" not in st.session_state:
        st.session_state.results_table = create_results_table()

    st.title("🔄 Dual Sweep FET Analyzer")

    left, right = st.columns(2)
    with left:
        uploaded = st.file_uploader(
            "Upload a dual‑sweep Excel file (drainI, drainV, gateI, gateV)", type=["xlsx"]
        )

    if uploaded:
        try:
            df = load_excel(uploaded)
        except ValueError as err:
            st.error(str(err))
            return
        with left:
            st.success(f"Loaded: **{uploaded.name}** — {df.shape[0]} rows × {df.shape[1]} cols")

            # Normalize column names to lower-case for consistent processing.
            # This ensures that columns such as 'GateV' or 'DrainI' are handled even if
            # the user spreadsheet uses different capitalization.
            df.columns = [c.lower() for c in df.columns]

            # Separate sweeps. Use lower-case column name 'gatev' after normalization.
            fwd_df, rev_df = separate_sweeps(df, gate_col="gatev")
            st.write(
                f"Forward sweep points: {len(fwd_df)}, Reverse sweep points: {len(rev_df)}"
            )
            sweep_option = "Forward" if len(rev_df) == 0 else st.radio(
                "Select sweep for analysis:", options=["Forward", "Reverse"], index=0
            )
            selected_df = fwd_df if sweep_option == "Forward" else rev_df
            
            
            with st.expander("Data Preview (interactive table)", expanded=True):
                st.dataframe(selected_df, use_container_width=True)

            num_cols = numeric_columns(selected_df)
            # require gateV and drainI to compute metrics; but allow other x/y for scatter
            if len(num_cols) < 2:
                st.warning("At least two numeric columns are required for plotting and regression.")
                return

        with right:
            st.subheader(f"Sweep: {sweep_option}")
            # Select X and Y axes.  Suggest gatev/draini if present.  Use case-insensitive search.
            def find_index(options: List[str], target: str) -> int:
                for i, opt in enumerate(options):
                    if opt.lower() == target:
                        return i
                return 0

            x_col = st.selectbox(
                "X axis (choose gatev for metrics)",
                options=num_cols,
                index=find_index(num_cols, "gatev") if num_cols else 0,
            )
            y_options = [c for c in num_cols if c != x_col]
            y_col = st.selectbox(
                "Y axis (choose draini for metrics)",
                options=y_options,
                index=find_index(y_options, "draini") if y_options else 0,
            )

            plot_df = selected_df[[x_col, y_col]].dropna().copy()
            if plot_df.empty:
                st.warning("No valid data points available after removing NaNs.")
                return

            # base scatter plot using selected columns
            fig = px.scatter(
                plot_df,
                x=x_col,
                y=y_col,
                title=f"Scatter: {y_col} vs {x_col}",
                opacity=0.6,
            )
            st.plotly_chart(fig, use_container_width=True)

            # FET metrics if proper columns selected
            metrics_available = (x_col.lower() == "gatev" and y_col.lower() == "draini")
            if metrics_available:
                st.markdown("### Parameter extraction")
                param_option = st.selectbox(
                    "Select parameter to extract:",
                    options=["Subthreshold swing (SS)", "Transconductance (gm)", "On/off ratio"],
                    index=0,
                )
                if param_option == "Subthreshold swing (SS)":
                    # Show log10 plot using selected y-col
                    fig_ss = px.scatter(
                        plot_df,
                        x=x_col,
                        y=y_col,
                        log_y=True,
                        title="log10(ID) vs VGS for SS extraction",
                        opacity=0.6,
                    )
                    st.plotly_chart(fig_ss, use_container_width=True)
                    # slider for gate voltage range
                    x_min_all = float(plot_df[x_col].min())
                    x_max_all = float(plot_df[x_col].max())
                    st.caption("Select gate‑voltage range for SS fit:")
                    ss_min, ss_max = st.slider(
                        "VGS range (subthreshold region)",
                        min_value=x_min_all,
                        max_value=x_max_all,
                        value=(x_min_all, x_max_all),
                        step=(x_max_all - x_min_all) / 100 if x_max_all > x_min_all else 1.0,
                        format="%.4f",
                    )
                    mask = (plot_df[x_col] >= ss_min) & (plot_df[x_col] <= ss_max)
                    sel_x = plot_df.loc[mask, x_col].to_numpy()
                    sel_y = plot_df.loc[mask, y_col].to_numpy()
                    ss_value = compute_subthreshold_swing(sel_x, sel_y)
                    st.write(
                        f"Computed SS: {ss_value:.2f} mV/decade" if not np.isnan(ss_value) else "Cannot compute SS with selected range."
                    )
                    # option to save results
                    if st.button("Save SS result"):
                        new_row = {
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "source_file": uploaded.name,
                            "sweep": sweep_option,
                            "parameter": "SS",
                            "fit_min": ss_min,
                            "fit_max": ss_max,
                            "slope": None,
                            "intercept": None,
                            "r2": None,
                            "rmse": None,
                            "n_points": int(mask.sum()),
                        }
                        st.session_state.results_table = pd.concat(
                            [st.session_state.results_table, pd.DataFrame([new_row])],
                            ignore_index=True,
                        )
                        st.success("SS result saved to results table.")
                elif param_option == "Transconductance (gm)":
                    # linear plot for gm using selected columns
                    fig_gm = px.scatter(
                        plot_df,
                        x=x_col,
                        y=y_col,
                        title="ID vs VGS for gm extraction",
                        opacity=0.6,
                    )
                    st.plotly_chart(fig_gm, use_container_width=True)
                    x_min_all = float(plot_df[x_col].min())
                    x_max_all = float(plot_df[x_col].max())
                    st.caption("Select gate‑voltage range for gm fit:")
                    gm_min, gm_max = st.slider(
                        "VGS range (gm)",
                        min_value=x_min_all,
                        max_value=x_max_all,
                        value=(x_min_all, x_max_all),
                        step=(x_max_all - x_min_all) / 100 if x_max_all > x_min_all else 1.0,
                        format="%.4f",
                    )
                    mask = (plot_df[x_col] >= gm_min) & (plot_df[x_col] <= gm_max)
                    sel_x = plot_df.loc[mask, x_col].to_numpy()
                    sel_y = plot_df.loc[mask, y_col].to_numpy()
                    if len(sel_x) >= 2:
                        gm_slope, gm_intercept, gm_r2, gm_rmse = perform_linear_regression(
                            sel_x.reshape(-1, 1), sel_y
                        )
                        st.write(
                            f"Computed gm (slope): {gm_slope:.3e} A/V\nR²: {gm_r2:.4f}, RMSE: {gm_rmse:.3e}, Points used: {len(sel_x)}"
                        )
                        # mobility calculation
                        st.markdown("#### Mobility calculation parameters")
                        c1, c2 = st.columns(2)
                        with c1:
                            L_input = st.number_input(
                                "Channel length L (μm)", min_value=0.0, value=1.0, step=0.1, format="%.3f"
                            )
                            W_input = st.number_input(
                                "Channel width W (μm)", min_value=0.0, value=1.0, step=0.1, format="%.3f"
                            )
                        with c2:
                            Cins_input = st.number_input(
                                "Insulator capacitance Cins (nF/cm²) \n for 285nm SiO2 is 121162 nF/m^2", min_value=0.0, value=121162.0, step=1.0, format="%.3f"
                            )
                            VDS_input = st.number_input(
                                "VDS for gm (V)",
                                min_value=0.0,
                                value=float(np.nanmean(selected_df.get("drainv", pd.Series([1])))),
                                step=0.1,
                                format="%.3f",
                            )
                        # convert units
                        L_m = L_input * 1e-6
                        W_m = W_input * 1e-6
                        Cins_F_m2 = Cins_input * 1e-5  # nF/cm² to F/m²
                        mu = compute_mobility(gm_slope, L_m, W_m, Cins_F_m2, VDS_input)
                        st.write(
                            f"Estimated μFE: {mu:.3f} cm²/Vs" if not np.isnan(mu) else "Cannot compute mobility (check inputs)."
                        )
                        if st.button("Save gm result"):
                            new_row = {
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "source_file": uploaded.name,
                                "sweep": sweep_option,
                                "parameter": "gm",
                                "fit_min": gm_min,
                                "fit_max": gm_max,
                                "slope": gm_slope,
                                "intercept": gm_intercept,
                                "r2": gm_r2,
                                "rmse": gm_rmse,
                                "n_points": len(sel_x),
                            }
                            st.session_state.results_table = pd.concat(
                                [st.session_state.results_table, pd.DataFrame([new_row])],
                                ignore_index=True,
                            )
                            st.success("gm result saved to results table.")
                    else:
                        st.info("Not enough points for regression.")
                elif param_option == "On/off ratio":
                    # log plot for Ion/Ioff using selected columns
                    fig_io = px.scatter(
                        plot_df,
                        x=x_col,
                        y=y_col,
                        log_y=True,
                        title="ID vs VGS (log scale) for Ion/Ioff extraction",
                        opacity=0.6,
                    )
                    st.plotly_chart(fig_io, use_container_width=True)
                    # threshold input for current splitting
                    y_min, y_max = float(plot_df[y_col].min()), float(plot_df[y_col].max())
                    st.caption("Define current threshold separating on/off states:")
                    thresh = st.slider(
                        "Current threshold (A)",
                        min_value=y_min,
                        max_value=y_max,
                        value=(y_min + y_max) / 2.0,
                        step=(y_max - y_min) / 100 if y_max > y_min else 1.0,
                        format="%.3e",
                    )
                    ion, ioff, ratio = compute_ion_ioff(plot_df[y_col].to_numpy(), thresh)
                    st.write(
                        f"Ion: {ion:.3e} A, Ioff: {ioff:.3e} A, Ion/Ioff: {ratio:.3e}" if not np.isnan(ratio) else "Cannot compute Ion/Ioff (check threshold)."
                    )
                    if st.button("Save Ion/Ioff result"):
                        new_row = {
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "source_file": uploaded.name,
                            "sweep": sweep_option,
                            "parameter": "Ion/Ioff",
                            "fit_min": None,
                            "fit_max": None,
                            "slope": None,
                            "intercept": None,
                            "r2": None,
                            "rmse": None,
                            "n_points": len(plot_df),
                        }
                        st.session_state.results_table = pd.concat(
                            [st.session_state.results_table, pd.DataFrame([new_row])],
                            ignore_index=True,
                        )
                        st.success("Ion/Ioff result saved to results table.")

            else:
                st.info(
                    "FET metrics extraction requires selecting gateV for the X axis and drainI for the Y axis."
                )

        with left:
            # results table and downloads
            st.markdown("---")
            st.subheader("📑 Results Table")
            st.caption("Saved parameter extraction results.")
            res_df = st.session_state.results_table
            st.dataframe(res_df, use_container_width=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.download_button(
                    "Download CSV",
                    data=dataframe_to_bytes(res_df, "csv"),
                    file_name="fet_parameter_results.csv",
                    mime="text/csv",
                    disabled=res_df.empty,
                )
            with c2:
                st.download_button(
                    "Download Excel",
                    data=dataframe_to_bytes(res_df, "xlsx"),
                    file_name="fet_parameter_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    disabled=res_df.empty,
                )
            with c3:
                if st.button("Clear results table"):
                    st.session_state.results_table = create_results_table()
                    st.success("Results table cleared.")
    else:
        st.info("Upload a dual‑sweep Excel file to begin.")


if __name__ == "__main__":
    main()