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
    Creates an Edit Plot button + organized options expander with intuitive controls.
    Returns a dict with plotting options. The button toggles an edit flag in 
    Streamlit session_state so the options persist across reruns.
    """
    _init_edit_flag(plot_key)
    btn_key = f"btn_{plot_key}"
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("✏️ Edit plot", key=btn_key, use_container_width=True):
            st.session_state[f"edit_{plot_key}"] = not st.session_state[f"edit_{plot_key}"]

    # Reserve middle column for spacing / future controls
    with col2:
        st.write("")

    with col3:
        if st.session_state[f"edit_{plot_key}"]:
            if st.button("↺ Reset defaults", key=f"reset_{plot_key}", use_container_width=True):
                # Clear session state for this plot to reset to defaults
                for key in list(st.session_state.keys()):
                    if key.endswith(f"_{plot_key}"):
                        del st.session_state[key]
                st.rerun()

    if st.session_state[f"edit_{plot_key}"]:
        with st.expander("Plot Customization", expanded=True):
            
            # --- Labels & Title Section ---
            st.markdown("**📝 Labels & Title**")
            title = st.text_input(
                "Plot title", 
                value=default_title, 
                key=f"title_{plot_key}",
                help="Main title displayed at the top of the plot"
            )
            
            col_label1, col_label2 = st.columns(2)
            with col_label1:
                xlabel = st.text_input(
                    "X-axis label", 
                    value=default_xlabel, 
                    key=f"xlabel_{plot_key}",
                    help="Label for the horizontal axis"
                )
            with col_label2:
                ylabel = st.text_input(
                    "Y-axis label", 
                    value=default_ylabel, 
                    key=f"ylabel_{plot_key}",
                    help="Label for the vertical axis"
                )
            
            st.divider()
            
            # --- Size & Scale Section ---
            st.markdown("**📐 Size & Scale**")
            col_size1, col_size2 = st.columns(2)
            with col_size1:
                w = st.number_input(
                    "Figure width (inches)", 
                    min_value=1.0, 
                    max_value=20.0,
                    value=float(default_figsize[0]), 
                    step=0.5,
                    key=f"w_{plot_key}",
                    help="Width of the plot in inches"
                )
            with col_size2:
                h = st.number_input(
                    "Figure height (inches)", 
                    min_value=1.0, 
                    max_value=20.0,
                    value=float(default_figsize[1]), 
                    step=0.5,
                    key=f"h_{plot_key}",
                    help="Height of the plot in inches"
                )
            
            col_scale1, col_scale2 = st.columns(2)
            with col_scale1:
                yscale = st.selectbox(
                    "Y-axis scale", 
                    options=["linear", "log"], 
                    index=0 if default_yscale == "linear" else 1, 
                    key=f"ys_{plot_key}",
                    help="Use log scale for wide dynamic ranges (exponential data)"
                )
            with col_scale2:
                xscale = st.selectbox(
                    "X-axis scale",
                    options=["linear", "log"],
                    index=0,
                    key=f"xs_{plot_key}",
                    help="Use log scale for wide dynamic ranges"
                )
            
            st.divider()
            
            # --- Line & Marker Style Section ---
            st.markdown("**🎨 Line & Marker Style**")
            col_line1, col_line2, col_line3 = st.columns(3)
            
            with col_line1:
                linewidth = st.slider(
                    "Line width", 
                    min_value=0.5, 
                    max_value=5.0,
                    value=2.0, 
                    step=0.5,
                    key=f"lw_{plot_key}",
                    help="Thickness of plot lines"
                )
            
            with col_line2:
                alpha = st.slider(
                    "Point transparency", 
                    min_value=0.0, 
                    max_value=1.0,
                    value=float(default_alpha), 
                    step=0.1,
                    key=f"alpha_{plot_key}",
                    help="0 = fully transparent, 1 = fully opaque"
                )
            
            with col_line3:
                marker_size = st.slider(
                    "Marker size",
                    min_value=1,
                    max_value=15,
                    value=6,
                    step=1,
                    key=f"ms_{plot_key}",
                    help="Size of data point markers"
                )

            # Toggle to show/hide markers
            show_markers = st.checkbox(
                "Show markers",
                value=False,
                key=f"markers_{plot_key}",
                help="Toggle display of point markers on top of the line"
            )

            # Line style selector (solid/dashed/dotted/dashdot)
            line_style_choice = st.selectbox(
                "Line style",
                options=["solid", "dashed", "dotted", "dashdot"],
                index=0,
                key=f"ls_{plot_key}",
                help="Visual style of the plotted line"
            )
            _ls_map = {"solid": "-", "dashed": "--", "dotted": ":", "dashdot": "-."}
            line_style = _ls_map.get(line_style_choice, "-")
            
            st.divider()
            
            # --- Grid & Legend Section ---
            st.markdown("**📊 Grid & Legend**")
            col_grid1, col_grid2, col_grid3 = st.columns(3)
            
            with col_grid1:
                show_grid = st.checkbox(
                    "Show grid", 
                    value=True, 
                    key=f"grid_{plot_key}",
                    help="Display background grid lines"
                )
            
            with col_grid2:
                if show_grid:
                    grid_alpha = st.slider(
                        "Grid opacity",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.4,
                        step=0.1,
                        key=f"grid_alpha_{plot_key}",
                        help="Grid line transparency"
                    )
                else:
                    grid_alpha = 0.4
            
            with col_grid3:
                grid_style = st.selectbox(
                    "Grid style",
                    options=["solid", "dashed", "dotted"],
                    index=1,
                    key=f"grid_style_{plot_key}",
                    help="Visual style of grid lines"
                ) if show_grid else "dashed"
            
            st.divider()
            
            # --- Font & Colors Section ---
            st.markdown("**🔤 Fonts & Colors**")
            col_font1, col_font2 = st.columns(2)
            
            with col_font1:
                title_size = st.slider(
                    "Title font size",
                    min_value=8,
                    max_value=24,
                    value=14,
                    step=1,
                    key=f"title_size_{plot_key}",
                    help="Size of the plot title text"
                )
            
            with col_font2:
                label_size = st.slider(
                    "Axis label font size",
                    min_value=8,
                    max_value=20,
                    value=12,
                    step=1,
                    key=f"label_size_{plot_key}",
                    help="Size of axis label text"
                )
            
            st.divider()
            
            # --- Advanced Options ---
            with st.expander("Advanced Options", expanded=False):
                st.markdown("**Additional Settings**")
                
                col_adv1, col_adv2 = st.columns(2)
                with col_adv1:
                    tight_layout = st.checkbox(
                        "Auto-tight layout",
                        value=True,
                        key=f"tight_{plot_key}",
                        help="Automatically adjust spacing to fit labels"
                    )
                
                with col_adv2:
                    show_legend = st.checkbox(
                        "Show legend",
                        value=True,
                        key=f"legend_{plot_key}",
                        help="Display plot legend if available"
                    )
                # Legend placement and style
                col_leg1, col_leg2 = st.columns(2)
                with col_leg1:
                    legend_loc = st.selectbox(
                        "Legend location",
                        options=[
                            "best",
                            "upper right",
                            "upper left",
                            "lower left",
                            "lower right",
                            "right",
                            "center left",
                            "center right",
                            "lower center",
                            "upper center",
                            "center",
                        ],
                        index=0,
                        key=f"legend_loc_{plot_key}",
                        help="Location of the legend on the plot",
                    )
                with col_leg2:
                    legend_outside = st.checkbox(
                        "Place legend outside",
                        value=False,
                        key=f"legend_outside_{plot_key}",
                        help="If checked, legend will be placed outside the axes to the right",
                    )

                col_leg3, col_leg4 = st.columns(2)
                with col_leg3:
                    legend_size = st.number_input(
                        "Legend font size",
                        min_value=6,
                        max_value=24,
                        value=10,
                        step=1,
                        key=f"legend_size_{plot_key}",
                    )
                with col_leg4:
                    legend_ncol = st.slider(
                        "Legend columns",
                        min_value=1,
                        max_value=4,
                        value=1,
                        step=1,
                        key=f"legend_ncol_{plot_key}",
                    )

                dpi = st.number_input(
                    "Export DPI (dots per inch)",
                    min_value=72,
                    max_value=600,
                    value=150,
                    step=50,
                    key=f"dpi_{plot_key}",
                    help="Resolution for exported images (higher = better quality)"
                )
        
        return {
            "title": title,
            "xlabel": xlabel,
            "ylabel": ylabel,
            "figsize": (w, h),
            "alpha": alpha,
            "linewidth": linewidth,
                "line_style": line_style,
            "yscale": yscale,
            "xscale": xscale,
            "grid": show_grid,
            "grid_alpha": grid_alpha if show_grid else 0,
            "grid_style": grid_style if show_grid else "solid",
            "marker_size": marker_size,
            "show_markers": show_markers,
            "title_size": title_size,
            "label_size": label_size,
            "tight_layout": tight_layout,
            "show_legend": show_legend,
            "legend_loc": legend_loc,
            "legend_outside": legend_outside,
            "legend_size": legend_size,
            "legend_ncol": legend_ncol,
            "dpi": dpi,
        }
    
    # default options when editor not open (minimal set)
    return {
        "title": default_title,
        "xlabel": default_xlabel,
        "ylabel": default_ylabel,
        "figsize": default_figsize,
        "alpha": default_alpha,
        "linewidth": 2.0,
        "line_style": "-",
        "yscale": default_yscale,
        "xscale": "linear",
        "grid": True,
        "grid_alpha": 0.4,
        "grid_style": "dashed",
        "marker_size": 6,
        "show_markers": False,
        "title_size": 14,
        "label_size": 12,
        "tight_layout": True,
        "show_legend": True,
        "legend_loc": "best",
        "legend_outside": False,
        "legend_size": 10,
        "legend_ncol": 1,
        "dpi": 150,
    }

def render_scatter_with_editor(df: pd.DataFrame, x_col: str, y_col: str, plot_key: str,
                               default_title: str, default_yscale: str = "linear",
                               overlay_line: Optional[dict] = None,
                               opts_override: Optional[dict] = None):
    """
    Render a Seaborn scatter with an Edit plot button. overlay_line optionally contains
    {'x': np.ndarray, 'y': np.ndarray, 'label': str, 'color': str}.
    Robust to non-positive y values when log y-scale is requested.
    """
    # If caller provides options (from an external editor placed elsewhere), use them.
    if opts_override is not None:
        opts = opts_override
    else:
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

    fig, ax = plt.subplots(figsize=opts["figsize"], dpi=opts.get("dpi", 150))

    # Draw continuous line using matplotlib so we can control linestyle
    ax.plot(
        plot_df[x_col],
        plot_df[y_col],
        linestyle=opts.get("line_style", "-"),
        linewidth=opts.get("linewidth", 2.0),
        alpha=opts.get("alpha", 1.0),
    )

    # Overlay scatter markers to reflect marker size and transparency (optional)
    if opts.get("show_markers", False):
        ax.scatter(
            plot_df[x_col],
            plot_df[y_col],
            s=(opts.get("marker_size", 6) ** 2),
            alpha=opts.get("alpha", 1.0),
            edgecolors="none",
        )

    # Titles and labels with font sizes from editor
    ax.set_title(opts.get("title", ""), fontsize=opts.get("title_size", 14))
    ax.set_xlabel(opts.get("xlabel", x_col), fontsize=opts.get("label_size", 12))
    ax.set_ylabel(opts.get("ylabel", y_col), fontsize=opts.get("label_size", 12))
    ax.tick_params(axis="both", labelsize=opts.get("label_size", 12))

    # Grid styling
    if opts.get("grid", True):
        style_map = {"solid": "-", "dashed": "--", "dotted": ":"}
        style = style_map.get(opts.get("grid_style", "dashed"), "--")
        ax.grid(True, linestyle=style, alpha=opts.get("grid_alpha", 0.4))

    # Axis scales
    if opts.get("yscale", "linear") == "log":
        ax.set_yscale("log")
    if opts.get("xscale", "linear") == "log":
        ax.set_xscale("log")

    # Overlay fit/line if provided
    if overlay_line is not None:
        ax.plot(
            overlay_line["x"],
            overlay_line["y"],
            color=overlay_line.get("color", "red"),
            linewidth=1,
            label=overlay_line.get("label", "fit"),
            linestyle="--",
        )

    # Legend handling using editor options
    if opts.get("show_legend", True):
        loc = opts.get("legend_loc", "best")
        fontsize = opts.get("legend_size", 10)
        ncol = opts.get("legend_ncol", 1)
        if opts.get("legend_outside", False):
            # place outside to the right
            ax.legend(bbox_to_anchor=(1.05, 1), loc=loc, fontsize=fontsize, ncol=ncol)
        else:
            ax.legend(loc=loc, fontsize=fontsize, ncol=ncol)

    if opts.get("tight_layout", True):
        plt.tight_layout()

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


def render_multifile_comparison() -> None:
    """
    Render a section for comparing data across multiple uploaded files.
    Allows users to select files and plot the same metric (e.g., ID vs VGS) 
    from each file on a single axis for easy comparison.
    """
    st.markdown("---")
    st.subheader("📊 Multi-File Comparison Plot")
    
    if not st.session_state.file_storage:
        st.info("Upload multiple files to enable comparison plotting.")
        return
    
    file_names = list(st.session_state.file_storage.keys())

    # Provide a checkbox table so users can pick files to compare (select-all available)
    st.caption("Select files to compare — check the boxes for files to include")
    select_all_key = "multifile_select_all"
    select_all = st.checkbox("Select all files", value=(len(file_names) <= 2), key=select_all_key)
    # synchronize individual checkboxes when Select all is toggled
    prev_key = f"{select_all_key}_prev"
    prev_val = st.session_state.get(prev_key, None)
    # initialize previous value if not present
    if prev_val is None:
        st.session_state[prev_key] = select_all
        prev_val = select_all
    # if the Select all checkbox changed since last render, update per-file checkboxes
    if select_all != prev_val:
        for i, fname in enumerate(file_names):
            cb_key = f"multifile_select_{i}"
            st.session_state[cb_key] = select_all
        st.session_state[prev_key] = select_all

    # header row
    c0, c1, c2 = st.columns([0.05, 0.6, 0.35])
    with c1:
        st.markdown("**File**")
    with c2:
        st.markdown("**Info**")

    selected_files = []
    for i, fname in enumerate(file_names):
        cb_key = f"multifile_select_{i}"
        # default checked when Select all enabled, otherwise preserve previous state if present
        default = select_all if cb_key not in st.session_state else st.session_state.get(cb_key, False)
        cb_col, name_col, info_col = st.columns([0.05, 0.6, 0.35])
        with cb_col:
            checked = st.checkbox("", value=default, key=cb_key)
        with name_col:
            st.write(fname)
        with info_col:
            fi = st.session_state.file_storage.get(fname, {})
            ts = fi.get("timestamp", "")
            rows = len(fi.get("data", pd.DataFrame())) if fi.get("data") is not None else 0
            st.write(f"{ts} | Rows: {rows}")

        if st.session_state.get(cb_key):
            selected_files.append(fname)
    
    if not selected_files:
        st.warning("Select at least one file to plot.")
        return
    
    # Load and normalize data from selected files
    data_dict = {}
    for fname in selected_files:
        file_info = st.session_state.file_storage[fname]
        df = file_info["data"].copy()
        df.columns = [c.lower() for c in df.columns]
        data_dict[fname] = df
    
    # Get available numeric columns (intersection across all selected files)
    all_cols = [set(numeric_columns(data_dict[f])) for f in selected_files]
    common_cols = list(set.intersection(*all_cols)) if all_cols else []
    
    if len(common_cols) < 2:
        st.warning("Selected files do not have at least 2 common numeric columns.")
        return
    
    # Select axes for comparison
    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox(
            "X axis (common column):",
            options=common_cols,
            index=0 if "gatev" not in common_cols else common_cols.index("gatev")
        )
    with col2:
        y_options = [c for c in common_cols if c != x_col]
        y_col = st.selectbox(
            "Y axis (common column):",
            options=y_options,
            index=0 if "draini" not in y_options else y_options.index("draini")
        )
    
    # Show plot options on the left and the comparison plot on the right
    default_title = f"Multi-File Comparison: {y_col} vs {x_col}"
    opt_col, plot_col = st.columns([1, 1])
    with opt_col:
        mf_opts = plot_options_ui(
            plot_key="multifile",
            default_title=default_title,
            default_xlabel=x_col,
            default_ylabel=y_col,
            default_figsize=(10.0, 6.0),
            default_alpha=0.7,
            default_yscale="linear",
        )
        # Allow renaming legend entries for selected files
        st.markdown("**🖊️ Legend names**")
        legend_names = {}
        for i, fname in enumerate(selected_files):
            key = f"legend_name_multifile_{i}"
            # preserve previous custom name if present
            default_name = st.session_state.get(key, fname)
            label = st.text_input(f"Label for: {fname}", value=default_name, key=key)
            legend_names[fname] = label
        # attach to options so plotting code can use the custom labels
        mf_opts["legend_names"] = legend_names

    with plot_col:
        # Create comparison plot using options from editor
        fig, ax = plt.subplots(figsize=mf_opts.get("figsize", (10.0, 6.0)), dpi=mf_opts.get("dpi", 150))
        colors = sns.color_palette("husl", len(selected_files))

        for idx, fname in enumerate(selected_files):
            df = data_dict[fname]
            plot_df = df[[x_col, y_col]].dropna().copy()
            if plot_df.empty:
                st.warning(f"⚠️ {fname}: No valid data points for {x_col} vs {y_col}")
                continue

            if mf_opts.get("yscale", "linear") == "log":
                pos_mask = plot_df[y_col] > 0
                if not pos_mask.any():
                    st.warning(f"⚠️ {fname}: No positive y values for log scale")
                    continue
                plot_df = plot_df.loc[pos_mask].copy()

            # use custom legend name when provided
            legend_names = mf_opts.get("legend_names", {}) if isinstance(mf_opts, dict) else {}
            label_name = legend_names.get(fname, fname)
            plot_kwargs = {
                "linewidth": mf_opts.get("linewidth", 2.0),
                "label": label_name,
                "color": colors[idx],
                "alpha": mf_opts.get("alpha", 0.7),
                "linestyle": mf_opts.get("line_style", "-"),
            }
            if mf_opts.get("show_markers", False):
                ax.plot(plot_df[x_col], plot_df[y_col], marker="o", markersize=mf_opts.get("marker_size", 6), **plot_kwargs)
            else:
                ax.plot(plot_df[x_col], plot_df[y_col], **plot_kwargs)

        ax.set_title(mf_opts.get("title", default_title), fontsize=mf_opts.get("title_size", 14), fontweight="bold")
        ax.set_xlabel(mf_opts.get("xlabel", x_col), fontsize=mf_opts.get("label_size", 12))
        ax.set_ylabel(mf_opts.get("ylabel", y_col), fontsize=mf_opts.get("label_size", 12))

        if mf_opts.get("yscale", "linear") == "log":
            ax.set_yscale("log")
        if mf_opts.get("xscale", "linear") == "log":
            ax.set_xscale("log")

        if mf_opts.get("grid", True):
            style = mf_opts.get("grid_style", "dashed")
            style_map = {"solid": "-", "dashed": "--", "dotted": ":"}
            ax.grid(True, linestyle=style_map.get(style, "--"), alpha=mf_opts.get("grid_alpha", 0.4))

        if mf_opts.get("show_legend", True):
            loc = mf_opts.get("legend_loc", "upper left")
            fontsize = mf_opts.get("legend_size", 10)
            ncol = mf_opts.get("legend_ncol", 1)
            if mf_opts.get("legend_outside", True):
                ax.legend(bbox_to_anchor=(1.05, 1), loc=loc, fontsize=fontsize, ncol=ncol)
            else:
                ax.legend(loc=loc, fontsize=fontsize, ncol=ncol)

        if mf_opts.get("tight_layout", True):
            plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        # Export comparison plot
        if st.button("📥 Download comparison plot as PNG"):
            img_bytes = io.BytesIO()
            fig.savefig(img_bytes, format="png", dpi=300, bbox_inches="tight")
            img_bytes.seek(0)
            st.download_button(
                "Click to download",
                data=img_bytes.getvalue(),
                file_name=f"multifile_comparison_{time.strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png"
            )


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

    # --- Tab navigation ---
    tab1, tab2 = st.tabs(["📈 Single File Analysis", "📊 Multi-File Comparison"])
    
    with tab1:
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
        
        # Data preview & axis selection
        with st.expander("Data Preview (interactive table)", expanded=True):
            st.dataframe(style_dataframe_sci(selected_df), use_container_width=True)

        num_cols = numeric_columns(selected_df)
        # require gateV and drainI to compute metrics; but allow other x/y for scatter
        if len(num_cols) < 2:
            st.warning("At least two numeric columns are required for plotting and regression.")
            return

        # Select axes in a compact row
        sel_c1, sel_c2 = st.columns(2)
        def find_index(options: List[str], target: str) -> int:
            for i, opt in enumerate(options):
                if opt.lower() == target:
                    return i
            return 0

        with sel_c1:
            x_col = st.selectbox(
                "X axis (choose gatev for metrics)",
                options=num_cols,
                index=find_index(num_cols, "gatev") if num_cols else 0,
            )
        with sel_c2:
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

        # Split the view: left = plot editor, right = main plot (half/half)
        plot_opt_col, plot_display_col = st.columns([1, 1])
        default_title = f"Scatter: {y_col} vs {x_col}"
        with plot_opt_col:
            base_opts = plot_options_ui(
                plot_key="base",
                default_title=default_title,
                default_xlabel=x_col,
                default_ylabel=y_col,
                default_figsize=(6.0, 4.0),
                default_alpha=1.0,
                default_yscale="linear",
            )

        with plot_display_col:
            st.subheader(f"Sweep: {sweep_option}")
            render_scatter_with_editor(
                plot_df,
                x_col,
                y_col,
                plot_key="base",
                default_title=default_title,
                default_yscale="linear",
                overlay_line=None,
                opts_override=base_opts,
            )

        # Parameter extraction section placed below the plot (options left, plot right)
        st.markdown("---")
        metrics_available = (x_col.lower() == "gatev" and y_col.lower() == "draini")
        st.subheader("Parameter extraction")
        if metrics_available:
            param_option = st.selectbox(
                "Select parameter to extract:",
                options=["Subthreshold swing (SS)", "Transconductance (gm)", "Regression"],
                index=0,
            )

            param_opt_col, param_plot_col = st.columns([1, 1])
            with param_opt_col:
                # dedicated editor for parameter plots — keep separate keys so parameter plots can have different options
                if param_option.startswith("Subthreshold"):
                    pkey = "param_ss"
                    default_yscale = "linear"
                elif param_option.startswith("Transconductance"):
                    pkey = "param_gm"
                    default_yscale = "linear"
                else:  # Regression
                    pkey = "param_reg"
                    default_yscale = "linear"
                
                param_opts = plot_options_ui(
                    plot_key=pkey,
                    default_title="Parameter plot",
                    default_xlabel=x_col,
                    default_ylabel=y_col,
                    default_figsize=(6.0, 4.0),
                    default_alpha=1.0,
                    default_yscale=default_yscale,
                )

            with param_plot_col:
                if param_option == "Subthreshold swing (SS)":
                    plot_df_plot = plot_df.copy()
                    pos_values = plot_df_plot[y_col][plot_df_plot[y_col] > 0]
                    eps = float(pos_values.min() / 10.0) if pos_values.size > 0 else 1e-30
                    if not np.isfinite(eps) or eps <= 0:
                        eps = 1e-30
                    plot_df_plot["log_id"] = np.log10(plot_df_plot[y_col].clip(lower=eps))
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

                    overlay = None
                    log_mask = sel_y > 0
                    if log_mask.sum() >= 2:
                        X_fit = sel_x[log_mask].reshape(-1, 1)
                        Y_fit = np.log10(sel_y[log_mask])
                        ss_slope, ss_intercept, ss_r2, ss_rmse = perform_linear_regression(X_fit, Y_fit)
                        x_line = np.linspace(ss_min, ss_max, 100)
                        y_line = ss_slope * x_line + ss_intercept
                        overlay = {"x": x_line, "y": y_line, "label": "SS fit", "color": "red"}

                    render_scatter_with_editor(plot_df_plot, x_col, "log_id", plot_key="ss_param", default_title="log10(ID) vs VGS (SS)", default_yscale="linear", overlay_line=overlay, opts_override=param_opts)
                    st.write(f"Computed SS: {ss_value:.2f} mV/decade" if not np.isnan(ss_value) else "Cannot compute SS with selected range.")
                    if st.button("Save SS result"):
                        new_row = {
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "source_file": uploaded,
                            "sweep": sweep_option,
                            "parameter": "SS",
                            "fit_min": ss_min,
                            "fit_max": ss_max,
                            "slope": (ss_slope if 'ss_slope' in locals() else None),
                            "intercept": (ss_intercept if 'ss_intercept' in locals() else None),
                            "x_intercept": (float(-ss_intercept / ss_slope) if 'ss_slope' in locals() and ss_slope and np.isfinite(ss_slope) and np.isfinite(ss_intercept) else None),
                            "r2": (ss_r2 if 'ss_r2' in locals() else None),
                            "rmse": (ss_rmse if 'ss_rmse' in locals() else None),
                            "n_points": int(mask.sum()),
                        }
                        st.session_state.results_table = pd.concat([st.session_state.results_table, pd.DataFrame([new_row])], ignore_index=True)
                        st.success("SS result saved to results table.")

                elif param_option == "Transconductance (gm)":
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
                        gm_slope, gm_intercept, gm_r2, gm_rmse = perform_linear_regression(sel_x.reshape(-1, 1), sel_y)
                        x_line = np.linspace(gm_min, gm_max, 100)
                        y_line = gm_slope * x_line + gm_intercept
                        overlay = {"x": x_line, "y": y_line, "label": "gm fit", "color": "red"}
                        render_scatter_with_editor(plot_df, x_col, y_col, plot_key="gm_param", default_title="ID vs VGS (gm)", default_yscale="linear", overlay_line=overlay, opts_override=param_opts)
                        st.write(f"Computed gm (slope): {gm_slope:.3e} A/V — R²: {gm_r2:.4f}")
                        
                        # Mobility calculation option
                        with st.expander("Calculate field-effect mobility (μFE) from gm"):
                            st.caption("Provide device parameters to calculate mobility: μFE = L·gm/(W·Cins·VDS)")
                            mob_col1, mob_col2, mob_col3 = st.columns(3)
                            
                            with mob_col1:
                                L_m = st.number_input(
                                    "Channel length L (m):",
                                    value=1.0e-6,
                                    min_value=1e-9,
                                    step=1e-7,
                                    format="%.3e",
                                    key="gm_L_input"
                                )
                            
                            with mob_col2:
                                W_m = st.number_input(
                                    "Channel width W (m):",
                                    value=1.0e-5,
                                    min_value=1e-9,
                                    step=1e-6,
                                    format="%.3e",
                                    key="gm_W_input"
                                )
                            
                            with mob_col3:
                                Cins_f = st.number_input(
                                    "Insulator capacitance Cins (F/m²):",
                                    value=1.15e-2,
                                    min_value=1e-4,
                                    step=1e-3,
                                    format="%.3e",
                                    key="gm_Cins_input"
                                )
                            
                            VDS_v = st.number_input(
                                "Drain-source voltage VDS (V):",
                                value=0.05,
                                min_value=1e-3,
                                step=0.01,
                                format="%.4f",
                                key="gm_VDS_input"
                            )
                            
                            if st.button("Calculate mobility", key="calc_mobility_btn"):
                                mu_fe = compute_mobility(gm_slope, L_m, W_m, Cins_f, VDS_v)
                                if np.isfinite(mu_fe):
                                    st.success(f"Field-effect mobility μFE = {mu_fe:.3f} cm²/Vs")
                                    if st.button("Save mobility result", key="save_mobility_btn"):
                                        new_row = {
                                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                            "source_file": uploaded,
                                            "sweep": sweep_option,
                                            "parameter": "μFE (mobility)",
                                            "fit_min": gm_min,
                                            "fit_max": gm_max,
                                            "slope": mu_fe,
                                            "intercept": None,
                                            "x_intercept": None,
                                            "r2": None,
                                            "rmse": None,
                                            "n_points": len(sel_x),
                                        }
                                        st.session_state.results_table = pd.concat([st.session_state.results_table, pd.DataFrame([new_row])], ignore_index=True)
                                        st.success("Mobility result saved to results table.")
                                else:
                                    st.error("Cannot calculate mobility. Check device parameters and ensure VDS ≠ 0.")
                        
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
                                "x_intercept": (float(-gm_intercept / gm_slope) if gm_slope and np.isfinite(gm_slope) and np.isfinite(gm_intercept) else None),
                                "r2": gm_r2,
                                "rmse": gm_rmse,
                                "n_points": len(sel_x),
                            }
                            st.session_state.results_table = pd.concat([st.session_state.results_table, pd.DataFrame([new_row])], ignore_index=True)
                            st.success("gm result saved to results table.")

                else:  # Regression
                    x_min_all = float(plot_df[x_col].min())
                    x_max_all = float(plot_df[x_col].max())
                    st.caption("Select data range for regression:")
                    reg_min, reg_max = st.slider(
                        "Data range",
                        min_value=x_min_all,
                        max_value=x_max_all,
                        value=(x_min_all, x_max_all),
                        step=(x_max_all - x_min_all) / 100 if x_max_all > x_min_all else 1.0,
                        format="%.4f",
                        key="reg_range_slider"
                    )
                    
                    # Regression type selection
                    reg_type = st.radio(
                        "Regression type:",
                        options=["Normal linear (y vs x)", "Logarithmic (log(y) vs x)"],
                        index=0,
                        horizontal=True,
                        key="reg_type_radio"
                    )
                    
                    mask = (plot_df[x_col] >= reg_min) & (plot_df[x_col] <= reg_max)
                    sel_x = plot_df.loc[mask, x_col].to_numpy()
                    sel_y = plot_df.loc[mask, y_col].to_numpy()
                    if len(sel_x) >= 2:
                        if reg_type == "Normal linear (y vs x)":
                            # Normal linear regression
                            reg_slope, reg_intercept, reg_r2, reg_rmse = perform_linear_regression(sel_x.reshape(-1, 1), sel_y)
                            x_line = np.linspace(reg_min, reg_max, 100)
                            y_line = reg_slope * x_line + reg_intercept
                            overlay = {"x": x_line, "y": y_line, "label": "Linear fit", "color": "red"}
                            render_scatter_with_editor(plot_df, x_col, y_col, plot_key="reg_param", default_title=f"{y_col} vs {x_col} (Linear)", default_yscale="linear", overlay_line=overlay, opts_override=param_opts)
                            st.write(f"Linear fit: slope = {reg_slope:.3e}, intercept = {reg_intercept:.3e}, R² = {reg_r2:.4f}")
                            reg_label = "Linear"
                        else:
                            # Logarithmic regression: log(y) vs x
                            log_mask = sel_y > 0
                            if log_mask.sum() >= 2:
                                X_fit = sel_x[log_mask].reshape(-1, 1)
                                Y_fit = np.log10(sel_y[log_mask])
                                reg_slope, reg_intercept, reg_r2, reg_rmse = perform_linear_regression(X_fit, Y_fit)
                                x_line = np.linspace(reg_min, reg_max, 100)
                                y_line = reg_slope * x_line + reg_intercept
                                overlay = {"x": x_line, "y": y_line, "label": "Log fit", "color": "red"}
                                
                                # Create plot_df_plot with log_y for visualization
                                plot_df_plot = plot_df.copy()
                                pos_values = plot_df_plot[y_col][plot_df_plot[y_col] > 0]
                                eps = float(pos_values.min() / 10.0) if pos_values.size > 0 else 1e-30
                                if not np.isfinite(eps) or eps <= 0:
                                    eps = 1e-30
                                plot_df_plot["log_y"] = np.log10(plot_df_plot[y_col].clip(lower=eps))
                                
                                render_scatter_with_editor(plot_df_plot, x_col, "log_y", plot_key="reg_param", default_title=f"log10({y_col}) vs {x_col} (Log)", default_yscale="linear", overlay_line=overlay, opts_override=param_opts)
                                st.write(f"Log fit: slope = {reg_slope:.3e}, intercept = {reg_intercept:.3e}, R² = {reg_r2:.4f}")
                                reg_label = "Logarithmic"
                            else:
                                st.warning("Cannot perform logarithmic regression: insufficient positive y values.")
                                reg_slope = reg_intercept = reg_r2 = reg_rmse = None
                                reg_label = "Logarithmic (failed)"
                        
                        if reg_slope is not None and st.button("Save regression result", key="save_reg_result"):
                            new_row = {
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "source_file": uploaded,
                                "sweep": sweep_option,
                                "parameter": f"Regression ({reg_label})",
                                "fit_min": reg_min,
                                "fit_max": reg_max,
                                "slope": reg_slope,
                                "intercept": reg_intercept,
                                "x_intercept": (float(-reg_intercept / reg_slope) if reg_slope and np.isfinite(reg_slope) and np.isfinite(reg_intercept) else None),
                                "r2": reg_r2,
                                "rmse": reg_rmse,
                                "n_points": len(sel_x) if reg_type == "Normal linear (y vs x)" else log_mask.sum(),
                            }
                            st.session_state.results_table = pd.concat([st.session_state.results_table, pd.DataFrame([new_row])], ignore_index=True)
                            st.success(f"Regression ({reg_label}) result saved to results table.")
        else:
            st.info("FET metrics extraction requires selecting gateV for the X axis and drainI for the Y axis.")

        # Results table (full width below analysis)
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
    
    with tab2:
        # Multi-file comparison section
        render_multifile_comparison()


if __name__ == "__main__":
    main()