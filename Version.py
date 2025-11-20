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
transistors*  , this app performs the following tasks:

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
   swing in mV/decade .
4. **Transconductance (gm) extraction**: Display `drainI` vs `gateV` and
   allow the user to choose the voltage range for linear regression.  The
   slope yields gm .
5. **Mobility calculation**: Use gm and user‑provided device parameters
   (channel length, channel width, insulator capacitance and VDS) to
   compute the field‑effect mobility μFE via μFE = L·gm/(W·Cins·VDS)
    .
6. **On/off current ratio**: Display `drainI` vs `gateV` (with log scale
   on the y‑axis) and ask the user to define a current threshold.  Values
   above the threshold are treated as the on‑state; values below, the
   off‑state.  The maximum on‑state current and minimum off‑state current
   are used to compute Ion/Ioff .

The app preserves the original features for interactive tables and saving
regression results.  To run it, install dependencies from `requirements.txt`
and execute `streamlit run fet_dual_sweep_app.py`.
"""
# ...existing code...

import io
import time
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
# removed Plotly imports — use Seaborn + Matplotlib for plotting
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# --- added: plot editor helpers ---
def _init_edit_flag(key: str) -> None:
    flag = f"edit_{key}"
    if flag not in st.session_state:
        st.session_state[flag] = False

def plot_options_ui(plot_key: str, default_title: str, default_xlabel: str, default_ylabel: str,
                    default_figsize: tuple = (6.0, 4.0), default_alpha: float = 0.6,
                    default_yscale: str = "linear"):
    """
    Creates an Edit Plot button + options expander. Returns a dict with plotting options.
    The button toggles an edit flag in Streamlit session_state so the options persist.
    """
    _init_edit_flag(plot_key)
    btn_key = f"btn_{plot_key}"
    if st.button("Edit plot", key=btn_key):
        st.session_state[f"edit_{plot_key}"] = not st.session_state[f"edit_{plot_key}"]

    if st.session_state[f"edit_{plot_key}"]:
        with st.expander("Plot options", expanded=True):
            title = st.text_input("Title", value=default_title, key=f"title_{plot_key}")
            xlabel = st.text_input("X axis label", value=default_xlabel, key=f"xlabel_{plot_key}")
            ylabel = st.text_input("Y axis label", value=default_ylabel, key=f"ylabel_{plot_key}")
            w = st.number_input("Figure width (inches)", min_value=1.0, value=float(default_figsize[0]), key=f"w_{plot_key}")
            h = st.number_input("Figure height (inches)", min_value=1.0, value=float(default_figsize[1]), key=f"h_{plot_key}")
            alpha = st.slider("Point alpha", 0.0, 1.0, value=float(default_alpha), key=f"alpha_{plot_key}")
            ms = st.slider("Marker size", 1, 200, value=40, key=f"ms_{plot_key}")
            yscale = st.selectbox("Y scale", options=["linear", "log"], index=0 if default_yscale == "linear" else 1, key=f"ys_{plot_key}")
            show_grid = st.checkbox("Show grid", value=True, key=f"grid_{plot_key}")
        return {
            "title": title,
            "xlabel": xlabel,
            "ylabel": ylabel,
            "figsize": (w, h),
            "alpha": alpha,
            "markersize": ms,
            "yscale": yscale,
            "grid": show_grid,
        }
    # default options when editor not open
    return {
        "title": default_title,
        "xlabel": default_xlabel,
        "ylabel": default_ylabel,
        "figsize": default_figsize,
        "alpha": default_alpha,
        "markersize": 40,
        "yscale": default_yscale,
        "grid": True,
    }

def render_scatter_with_editor(df: pd.DataFrame, x_col: str, y_col: str, plot_key: str,
                               default_title: str, default_yscale: str = "linear",
                               overlay_line: Optional[dict] = None):
    """
    Render a Seaborn scatter with an Edit plot button. overlay_line optionally contains
    {'x': np.ndarray, 'y': np.ndarray, 'label': str, 'color': str}.
    """
    opts = plot_options_ui(plot_key, default_title, x_col, y_col, default_figsize=(6.0,4.0), default_alpha=0.6, default_yscale=default_yscale)
    fig, ax = plt.subplots(figsize=opts["figsize"])
    # seaborn scatter: use 's' argument for marker size
    sns.scatterplot(data=df, x=x_col, y=y_col, alpha=opts["alpha"], s=opts["markersize"], ax=ax, edgecolor="k", linewidth=0.2)
    ax.set_title(opts["title"])
    ax.set_xlabel(opts["xlabel"])
    ax.set_ylabel(opts["ylabel"])
    if opts["grid"]:
        ax.grid(True, linestyle="--", alpha=0.4)
    if opts["yscale"] == "log":
        ax.set_yscale("log")
    if overlay_line is not None:
        ax.plot(overlay_line["x"], overlay_line["y"], color=overlay_line.get("color", "red"),
                linewidth=1.5, label=overlay_line.get("label", "fit"))
        ax.legend()
    st.pyplot(fig)
    plt.close(fig)
# --- end added helpers ---

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
    the desired subthreshold region .
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
    Compute the field‑effect mobility μFE in cm²/Vs using μFE = L·gm/(W·Cins·VDS) .
    Inputs must be in SI units (L and W in m, Cins in F/m², gm in A/V, VDS in V).  Returns μFE in cm²/Vs.
    """
    denom = channel_width * cins * vds
    if denom == 0:
        return float("nan")
    mu_m2 = (channel_length * gm) / denom
    return mu_m2 * 1e4


def compute_ion_ioff(currents: np.ndarray, threshold: float) -> Tuple[float, float, float]:
    """
    Compute Ion (max ID above a threshold), Ioff (min ID below the threshold) and their ratio Ion/Ioff.
    This allows the user to define a current threshold to separate on/off states .
    """
    if currents.size == 0:
        return float("nan"), float("nan"), float("nan")
    # Identify on/off states
    on_values = currents[currents >= threshold]
    off_values = currents[currents < threshold]
    ion = float(np.nanmax(on_values)) if on_values.size > 0 else float("nan")
    ioff = float(np.nanmin(off_values)) if off_values.size > 0 else float("nan")
    if ioff is None or ioff <= 0 or np.isnan(ion) or np.isnan(ioff):
        ratio = float("nan")
    else:
        ratio = ion / ioff
    return ion, ioff, ratio


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
        "parameter value",
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
            plot_option = st.radio(
                "Select plot option:", options=["Linear", "Log"], index=0
            )
            plot_df = selected_df[[x_col, y_col]].dropna().copy()
            if plot_df.empty:
                st.warning("No valid data points available after removing NaNs.")
                return
            # base scatter plot using selected columns (Seaborn + Matplotlib)
            default_title = f"Scatter: {y_col} vs {x_col}"
            # allow user to edit this base scatter
            render_scatter_with_editor(
                plot_df, x_col, y_col, plot_key="base",
                default_title=default_title,
                default_yscale="log" if (plot_option == "Log") else "linear",
                overlay_line=None,
            )

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
                    # Prepare data and show log10(ID) vs VGS using Seaborn
                    plot_df_plot = plot_df.copy()
                    plot_df_plot["log_id"] = np.log10(plot_df_plot[y_col].clip(lower=1e-30))
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

                    # plot scatter of log10(ID) and overlay linear fit in log-space if possible
                    fig_ss, ax_ss = plt.subplots()
                    sns.scatterplot(data=plot_df_plot, x=x_col, y="log_id", alpha=0.6, ax=ax_ss)
                    ax_ss.set_title("log10(ID) vs VGS for SS extraction")
                    ax_ss.set_xlabel(x_col)
                    ax_ss.set_ylabel("log10(ID)")
                    log_mask = sel_y > 0
                    if log_mask.sum() >= 2:
                        X_fit = sel_x[log_mask].reshape(-1, 1)
                        Y_fit = np.log10(sel_y[log_mask])
                        slope, intercept, r2, rmse = perform_linear_regression(X_fit, Y_fit)
                        x_line = np.linspace(ss_min, ss_max, 100)
                        y_line = slope * x_line + intercept
                        ax_ss.plot(x_line, y_line, color="red", linewidth=1.5, label="SS fit")
                        ax_ss.legend()
                    st.pyplot(fig_ss)
                    plt.close(fig_ss)

                    st.write(
                        f"Computed SS: {ss_value:.2f} mV/decade" if not np.isnan(ss_value) else "Cannot compute SS with selected range."
                    )
                    # option to save results
                    if st.button("Save SS result"):
                        new_row = {
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "source_file": uploaded.name,
                            "sweep": sweep_option,
                            "parameter": "SS (mV/decade)",
                            "parameter value": ss_value,
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
                    # linear plot for gm using selected columns (Seaborn)
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
                        # plot scatter and overlay linear fit
                        fig_gm, ax_gm = plt.subplots()
                        sns.scatterplot(data=plot_df, x=x_col, y=y_col, alpha=0.6, ax=ax_gm)
                        x_line = np.linspace(gm_min, gm_max, 100)
                        y_line = gm_slope * x_line + gm_intercept
                        ax_gm.plot(x_line, y_line, color="red", linewidth=1.5, label="gm fit")
                        ax_gm.set_title("ID vs VGS for gm extraction")
                        ax_gm.set_xlabel(x_col)
                        ax_gm.set_ylabel(y_col)
                        ax_gm.legend()
                        st.pyplot(fig_gm)
                        plt.close(fig_gm)

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
                                "parameter": "mobility (cm²/Vs)",
                                "parameter value": mu,
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
                    # log plot for Ion/Ioff using Seaborn (set log y-axis)
                    fig_io, ax_io = plt.subplots()
                    sns.scatterplot(data=plot_df, x=x_col, y=y_col, alpha=0.6, ax=ax_io)
                    ax_io.set_yscale("log")
                    ax_io.set_title("ID vs VGS (log scale) for Ion/Ioff extraction")
                    ax_io.set_xlabel(x_col)
                    ax_io.set_ylabel(y_col)
                    st.pyplot(fig_io)
                    plt.close(fig_io)

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
                            "parameter value": ratio,
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