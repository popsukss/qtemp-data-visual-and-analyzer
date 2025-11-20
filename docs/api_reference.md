# API Reference

## Functions

### separate_sweeps(df, gate_col="gatev")
Separates dual-sweep data.

### perform_linear_regression(x, y)
Returns (slope, intercept, r2, rmse)

### compute_subthreshold_swing(x, y)
Returns SS in mV/decade

### compute_mobility(gm, L, W, Cins, VDS)
Returns mobility in cm²/Vs

### compute_ion_ioff(currents, threshold)
Returns (Ion, Ioff, ratio)
