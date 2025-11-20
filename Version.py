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

# --- added: dataframe scientific formatting helper ---
def style_dataframe_sci(df: pd.DataFrame,
                        small_thresh: float = 1e-3,
                        large_thresh: float = 1e6,
                        sig_figs: int = 3) -> "pd.io.formats.style.Styler":
    """
    Return a pandas Styler that formats numeric cells in scientific notation
    when abs(value) < small_thresh or abs(value) >= large_thresh.
    Non-numeric values are left unchanged.
    """
    def _fmt(x):
        try:
            if pd.isnull(x):
                return ""
            if isinstance(x, (int, np.integer)):
                v = int(x)
                if v != 0 and (abs(v) < small_thresh or abs(v) >= large_thresh):
                    return f"{v:.{sig_figs}e}"
                return str(v)
            if isinstance(x, (float, np.floating)):
                v = float(x)
                if v != 0 and (abs(v) < small_thresh or abs(v) >= large_thresh):
                    return f"{v:.{sig_figs}e}"
                # default fixed format for regular-sized floats
                return f"{v:.6f}"
        except Exception:
            pass
        return x
    return df.style.format(_fmt)
# --- end added helper ---

# --- added: plot editor helpers ---
def _init_edit_flag(key: str) -> None:
    flag = f"edit_{key}"
    if flag not in st.session_state:
        st.session_state[flag] = False

def plot_options_ui(plot_key: str, default_title: str, default_xlabel: str, default_ylabel: str,
                    default_figsize: tuple = (6.0, 4.0), default_alpha: float = 1,
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
            ms = st.slider("Line Width", 1, 5, value=10, key=f"ms_{plot_key}")
            yscale = st.selectbox("Y scale", options=["linear", "log"], index=0 if default_yscale == "linear" else 1, key=f"ys_{plot_key}")
            show_grid = st.checkbox("Show grid", value=True, key=f"grid_{plot_key}")
        return {
            "title": title,
            "xlabel": xlabel,
            "ylabel": ylabel,
            "figsize": (w, h),
            "alpha": alpha,
            "linewidth": ms,
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
        "linewidth": 1.5,
        "yscale": default_yscale,
        "grid": True,
    }

def render_scatter_with_editor(df: pd.DataFrame, x_col: str, y_col: str, plot_key: str,
                               default_title: str, default_yscale: str = "linear",
                               overlay_line: Optional[dict] = None):
    """
    Render a Seaborn scatter with an Edit plot button. overlay_line optionally contains
    {'x': np.ndarray, 'y': np.ndarray, 'label': str, 'color': str}.
    Robust to non-positive y values when log y-scale is requested.
    """
    opts = plot_options_ui(plot_key, default_title, x_col, y_col, default_figsize=(6.0,4.0), default_alpha=1, default_yscale=default_yscale)

    # If user requested log scale but there are non-positive y values, filter them out.
    plot_df = df.copy()
    if opts["yscale"] == "log":
        pos_mask = plot_df[y_col] > 0
        if not pos_mask.any():
            st.warning("No positive y values available for log scale plotting — showing linear plot instead.")
            opts["yscale"] = "linear"
        else:
            plot_df = plot_df.loc[pos_mask].copy()

    fig, ax = plt.subplots(figsize=opts["figsize"])
    # seaborn lineplot for continuity + scatter-like view (alpha controls visibility)
    sns.lineplot(data=plot_df, x=x_col, y=y_col, alpha=opts["alpha"],
                 linewidth=opts["linewidth"], ax=ax)
    ax.set_title(opts["title"])
    ax.set_xlabel(opts["xlabel"])
    ax.set_ylabel(opts["ylabel"])
    if opts["grid"]:
        ax.grid(True, linestyle="--", alpha=0.4)
    if opts["yscale"] == "log":
        ax.set_yscale("log")
    if overlay_line is not None:
        # Overlay line may contain values outside filtered set; plot anyway but avoid invalid log plotting issues
        ax.plot(overlay_line["x"], overlay_line["y"], color=overlay_line.get("color", "red"),
                linewidth=1, label=overlay_line.get("label", "fit"), linestyle="--")
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
    and drain current (ID) samples.  Handles non-positive and very small currents
    by replacing them with an adaptive small epsilon before taking log10.
    """
    if x.size == 0 or y.size == 0:
        return float("nan")
    # Determine adaptive epsilon from smallest positive current (if present)
    pos = y[y > 0]
    eps = float(pos.min() / 10.0) if pos.size > 0 else 1e-30
    # Avoid eps being zero or subnormal zero
    if not np.isfinite(eps) or eps <= 0:
        eps = 1e-30
    # Replace non-positive values with eps for log computation
    y_safe = np.where(y > 0, y, eps)
    log_id = np.log10(y_safe)
    X = x.reshape(-1, 1)
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


def separate_sweeps(df: pd.DataFrame, gate_col: str = "gatev") -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        "x_intercept",
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
    if "file_storage" not in st.session_state:
        st.session_state.file_storage = {}
    if "current_file" not in st.session_state:
        st.session_state.current_file = None

    st.title("🔄 Dual Sweep FET Analyzer (Version4)")

    # --- File Manager Section ---
    st.markdown("### 📁 File Manager")
    file_mgr_col1, file_mgr_col2, file_mgr_col3 = st.columns([2, 1, 1])
    
    with file_mgr_col1:
        uploaded_files = st.file_uploader(
            "Upload Excel files (drainI, drainV, gateI, gateV)",
            type=["xlsx"],
            accept_multiple_files=True,
            key="multi_uploader",
        )
        if uploaded_files:
            for uploaded in uploaded_files:
                if uploaded.name not in st.session_state.file_storage:
                    try:
                        df = load_excel(uploaded)
                        st.session_state.file_storage[uploaded.name] = {
                            "data": df,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        st.success(f"✅ Loaded: {uploaded.name}")
                    except ValueError as err:
                        st.error(f"❌ {uploaded.name}: {err}")
    
    with file_mgr_col2:
        if st.session_state.file_storage:
            st.write(f"**Files loaded:** {len(st.session_state.file_storage)}")
    
    with file_mgr_col3:
        if st.button("🗑️ Clear all files"):
            st.session_state.file_storage = {}
            st.session_state.current_file = None
            st.success("All files cleared.")

    
    left, right = st.columns(2)
    with left:
        # --- File List & Selection ---
        if st.session_state.file_storage:
            st.markdown("#### Select a file to analyze:")
            file_names = list(st.session_state.file_storage.keys())
            selected_file = st.selectbox("Available files:", options=file_names, key="file_selector")
            st.session_state.current_file = selected_file
            
            # Display file info
            file_info = st.session_state.file_storage[selected_file]
            st.caption(f"Loaded: {file_info['timestamp']} | Rows: {len(file_info['data'])} | Cols: {len(file_info['data'].columns)}")
        else:
            st.info("📤 Upload one or more Excel files to begin.")
            return

        # --- Analysis Section (once a file is selected) ---
        if st.session_state.current_file is None:
            return
        
        uploaded = st.session_state.current_file
        file_info = st.session_state.file_storage[uploaded]
        df = file_info["data"].copy()
        st.success(f"Analyzing: **{uploaded}**")

        # Normalize column names to lower-case for consistent processing.
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
            st.dataframe(style_dataframe_sci(selected_df), use_container_width=True)

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

        # base scatter plot using selected columns (Seaborn + Matplotlib)
        default_title = f"Scatter: {y_col} vs {x_col}"
        render_scatter_with_editor(
            plot_df, x_col, y_col, plot_key="base",
            default_title=default_title,
            default_yscale="linear",
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
                # compute adaptive epsilon from positive values to avoid arbitrary fixed clip
                pos_values = plot_df_plot[y_col][plot_df_plot[y_col] > 0]
                eps = float(pos_values.min() / 10.0) if pos_values.size > 0 else 1e-30
                if not np.isfinite(eps) or eps <= 0:
                    eps = 1e-30
                plot_df_plot["log_id"] = np.log10(plot_df_plot[y_col].clip(lower=eps))
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
                default_title = "log10(ID) vs VGS for SS extraction"
                overlay = None
                ss_slope = None
                ss_intercept = None
                ss_r2 = None
                ss_rmse = None
                ss_n_points = int(mask.sum())
                ss_x_intercept = None

                log_mask = sel_y > 0
                if log_mask.sum() >= 2:
                    X_fit = sel_x[log_mask].reshape(-1, 1)
                    Y_fit = np.log10(sel_y[log_mask])
                    ss_slope, ss_intercept, ss_r2, ss_rmse = perform_linear_regression(X_fit, Y_fit)
                    x_line = np.linspace(ss_min, ss_max, 100)
                    y_line = ss_slope * x_line + ss_intercept
                    overlay = {"x": x_line, "y": y_line, "label": "SS fit", "color": "red"}
                    # compute x-intercept where log10(ID) == 0 -> x = -intercept / slope
                    try:
                        if ss_slope != 0 and np.isfinite(ss_slope) and np.isfinite(ss_intercept):
                            ss_x_intercept = float(-ss_intercept / ss_slope)
                    except Exception:
                        ss_x_intercept = None

                render_scatter_with_editor(plot_df_plot, x_col, "log_id", plot_key="ss", default_title=default_title, default_yscale="linear", overlay_line=overlay)

                st.write(
                    f"Computed SS: {ss_value:.2f} mV/decade" if not np.isnan(ss_value) else "Cannot compute SS with selected range."
                )
                # option to save results
                if st.button("Save SS result"):
                    new_row = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "source_file": uploaded,
                        "sweep": sweep_option,
                        "parameter": "SS",
                        "fit_min": ss_min,
                        "fit_max": ss_max,
                        "slope": ss_slope,
                        "intercept": ss_intercept,
                        "x_intercept": ss_x_intercept,
                        "r2": ss_r2,
                        "rmse": ss_rmse,
                        "n_points": ss_n_points,
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
                    # compute x-intercept for linear fit (where ID == 0)
                    gm_x_intercept = None
                    try:
                        if gm_slope != 0 and np.isfinite(gm_slope) and np.isfinite(gm_intercept):
                            gm_x_intercept = float(-gm_intercept / gm_slope)
                    except Exception:
                        gm_x_intercept = None

                    st.write(
                        f"Computed gm (slope): {gm_slope:.3e} A/V\nR²: {gm_r2:.4f}, RMSE: {gm_rmse:.3e}, Points used: {len(sel_x)}"
                    )
                    # plot scatter and overlay linear fit
                    default_title = "ID vs VGS for gm extraction"
                    x_line = np.linspace(gm_min, gm_max, 100)
                    y_line = gm_slope * x_line + gm_intercept
                    overlay = {"x": x_line, "y": y_line, "label": "gm fit", "color": "red"}
                    render_scatter_with_editor(plot_df, x_col, y_col, plot_key="gm", default_title=default_title, default_yscale="linear", overlay_line=overlay)

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
                            "source_file": uploaded,
                            "sweep": sweep_option,
                            "parameter": "gm",
                            "fit_min": gm_min,
                            "fit_max": gm_max,
                            "slope": gm_slope,
                            "intercept": gm_intercept,
                            "x_intercept": gm_x_intercept,
                            "r2": gm_r2,
                            "rmse": gm_rmse,
                            "n_points": len(sel_x),
                        }
                        st.session_state.results_table = pd.concat(
                            [st.session_state.results_table, pd.DataFrame([new_row])],
                            ignore_index=True,
                        )
                        st.success("gm result saved to results table.")
            elif param_option == "On/off ratio":
                # log plot for Ion/Ioff using Seaborn (set log y-axis)
                default_title = "ID vs VGS (log scale) for Ion/Ioff extraction"
                render_scatter_with_editor(plot_df, x_col, y_col, plot_key="io", default_title=default_title, default_yscale="log", overlay_line=None)

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
                        "source_file": uploaded,
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
        st.dataframe(style_dataframe_sci(res_df), use_container_width=True)
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


if __name__ == "__main__":
    main()