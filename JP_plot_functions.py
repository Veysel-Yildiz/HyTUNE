
"""
plot_functions.py

plotting utilities for HyTUNE.

Public functions:
 - DecisonTree_Plot(P, out_path="hydropower.svg", colordict=colors)
 - EL_scenario_plots(sen_elevations, date, scenarios, show=True)
 - combined_yearly_energy(xCal_DailyPowers, xSim_DailyPowers, Sim_Replacement_year,
                          Sim_head_all, Qdesign, date, show=True)
 - drought_indices(res_levels, Lcrit)
 - parallel_coord_map(..., ax=None, show=True, force_update_cbar=False)
 - draw_bar_panel(ax, values, labels, ...)
"""


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from ptreeopt.plotting import *
import os, subprocess
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from ptreeopt.plotting import graphviz_export  # used by DecisonTree_Plot

# ----------------------------------------
# Color mapping for discrete actions
# Keep user's color mapping but moved here for clarity.
# ----------------------------------------
colors = {
    'No Replacement': 'lightgray', 'Replace Existing': 'darkgray',
    'H+2.5': 'skyblue', 'H+5': 'lightblue', 'H+7.5': 'cornflowerblue', 'H+10': 'blue',
    'H-2.5': 'lightsalmon', 'H-5': 'lightcoral', 'H-7.5': 'indianred', 'H-10': 'red',
    'D+5': 'lightgreen', 'D+10': 'green', 'D+15': 'darkgreen', 'D+20': 'forestgreen',
    'D-5': 'lightgoldenrodyellow', 'D-10': 'gold', 'D-15': 'darkgoldenrod', 'D-20': 'orange',
    'H+2.5 & D+5': 'paleturquoise', 'H+2.5 & D+10': 'aquamarine', 'H+2.5 & D+15': 'mediumaquamarine',
    'H+2.5 & D+20': 'turquoise', 'H+2.5 & D-5': 'peachpuff', 'H+2.5 & D-10': 'moccasin',
    'H+2.5 & D-15': 'bisque', 'H+2.5 & D-20': 'navajowhite',
    'H-2.5 & D+5': 'mistyrose', 'H-2.5 & D+10': 'pink', 'H-2.5 & D+15': 'hotpink',
    'H-2.5 & D+20': 'deeppink', 'H-2.5 & D-5': 'lavenderblush', 'H-2.5 & D-10': 'plum',
    'H-2.5 & D-15': 'violet', 'H-2.5 & D-20': 'orchid',
    'H+5 & D+5': 'deepskyblue', 'H+5 & D+10': 'dodgerblue', 'H+5 & D+15': 'mediumblue',
    'H+5 & D+20': 'coral', 'H+5 & D-5': 'salmon', 'H+5 & D-10': 'coral',
    'H+5 & D-15': 'tomato', 'H+5 & D-20': 'orangered',
    'H-5 & D+5': 'sandybrown', 'H-5 & D+10': 'peru', 'H-5 & D+15': 'chocolate',
    'H-5 & D+20': 'peru', 'H-5 & D-5': 'orangered', 'H-5 & D-10': 'firebrick',
    'H-5 & D-15': 'brown', 'H-5 & D-20': 'peru',
    'H+7.5 & D+5': 'steelblue', 'H+7.5 & D+10': 'royalblue', 'H+7.5 & D+15': 'mediumslateblue',
    'H+7.5 & D+20': 'slateblue', 'H+7.5 & D-5': 'lightsteelblue', 'H+7.5 & D-10': 'cadetblue',
    'H+7.5 & D-15': 'powderblue', 'H+7.5 & D-20': 'aliceblue',
    'H-7.5 & D+5': 'lightseagreen', 'H-7.5 & D+10': 'mediumseagreen', 'H-7.5 & D+15': 'seagreen',
    'H-7.5 & D+20': 'darkseagreen', 'H-7.5 & D-5': 'indianred', 'H-7.5 & D-10': 'crimson',
    'H-7.5 & D-15': 'firebrick', 'H-7.5 & D-20': 'darkred',
    'H+10 & D+5': 'midnightblue', 'H+10 & D+10': 'royalblue', 'H+10 & D+15': 'blueviolet',
    'H+10 & D+20': 'mediumpurple', 'H+10 & D-5': 'lightcoral', 'H+10 & D-10': 'red',
    'H+10 & D-15': 'darkred', 'H+10 & D-20': 'maroon',
    'H-10 & D+5': 'darkorange', 'H-10 & D+10': 'orangered', 'H-10 & D+15': 'red',
    'H-10 & D+20': 'firebrick', 'H-10 & D-5': 'crimson', 'H-10 & D-10': 'brown',
    'H-10 & D-15': 'sienna', 'H-10 & D-20': 'maroon'
}


# %%

def DecisonTree_Plot(P):
    """
    Export decision-tree visualization to SVG using ptreeopt.plotting.graphviz_export.

    Parameters
    ----------
    P : decision tree object (as produced by optimizer)
    out_path : str
        Path to write the SVG.
    colordict : dict
        Mapping action_name -> color. Falls back to module `colors`.
    """
    
    graphviz_export(P, 'hydropower.svg', colordict=colors) # creates one SVG
    

# %%

# -----------------------
# Scenario elevation plotting
# -----------------------

def EL_scenario_plots(sen_elevations, date, scenarios):
    """
    Plot reservoir elevation scenarios with 10-year moving averages.

    Parameters
    ----------
    sen_elevations : array-like
        2D array of scenario time series (rows = days, cols = scenarios).
    date : array-like
        Array of datetime-like values.
    scenarios : dict
        Dictionary {scenario_index: "label"} to highlight specific scenarios.
        Example: {0: "Dry Scenario", 20: "Wet Scenario"}

    Plot includes:
      - 5–95% band (gray)
      - Ensemble mean (navy)
      - Selected scenarios (colored lines with labels)
    """

    # --- Build DataFrame ---
    df_sims = pd.DataFrame(
        sen_elevations, columns=[f"sen_{i}" for i in range(sen_elevations.shape[1])]
    )
    date = pd.to_datetime(date)
    df_sims["date"] = date

    # filter from 1985 onwards
    df_sims = df_sims[df_sims["date"] >= "1985-01-01"].reset_index(drop=True)

    # set date as index
    df_sims = df_sims.set_index("date")

    # --- 10-year moving average ---
    df_smooth = df_sims.rolling(window=3650, min_periods=1).mean()

    # --- Percentiles for band ---
    q05 = df_smooth.quantile(0.025, axis=1)
    q95 = df_smooth.quantile(0.975, axis=1)
    mean = df_smooth.mean(axis=1)
    
    # --- Set up figure ---
    plt.rcParams.update({
      "axes.labelsize": 30,
      "xtick.labelsize": 28,
      "ytick.labelsize": 28,
      "legend.fontsize": 28})

    plt.figure(figsize=(20, 7))

    # 1) Shaded band = uncertainty across scenarios
    plt.fill_between(df_smooth.index, q05, q95,
                     color="lightgray", alpha=0.5,
                     label="95% confidence interval")

    # 2) Mean line
    plt.plot(df_smooth.index, mean,
             color="dimgray", linestyle="--", linewidth=4,
             label="Ensemble Mean")

    # 3) Highlighted scenarios
    colors = ["#FF7F0E", "#9467BD", "#2CA02C", "#D62728"]  # cycle of distinct colors
    for (idx, label), color in zip(scenarios.items(), colors):
        col = f"sen_{idx}"
        if col in df_smooth.columns:
            plt.plot(df_smooth.index, df_smooth[col],
                     color=color, linewidth=4, alpha=0.99,
                     label=label)
        else:
            print(f"Warning: {col} not found in DataFrame")
    

    
    # --- Labels and style ---
    plt.xlabel("Time (daily)")
    plt.ylabel("Reservoir Level (m)")
    plt.grid(True, alpha=0.3)

    # Custom ticks
    start, end = df_smooth.index.min(), df_smooth.index.max()
    ticks = pd.date_range(start=start, end=end, periods=7)
    plt.xticks(ticks, [d.strftime("%Y") for d in ticks])

    # Legend
    plt.legend(bbox_to_anchor=(0.10, 0.03), loc="lower left",ncol=2)
    plt.tight_layout()
    plt.show()

# %%

def combined_yearly_energy(xCal_DailyPowers, xSim_DailyPowers,
                           Sim_Replacement_year, Sim_head_all, Qdesign, date):
    """
    Three-row figure:
      top: annual energy (reactive vs optimized) with replacement year markers
      middle: fleet head composition scatter
      bottom: design discharge capacity over time

    Returns fig, axes tuple.
    """

    # --- Helper: derive replacement counts ---
    def derive_replacement_counts_simple(rep_type):
        rep_type = np.asarray(rep_type)
        nz = np.flatnonzero(rep_type != 0)
        counts = np.zeros_like(rep_type, dtype=int)
        counts[nz] = 6
        for k in (2, 5, 8):  # 3rd & 6th replacement → 5 instead of 6
            if k < len(nz):
                counts[nz[k]] = 5
        return counts
     

    rep_years = np.zeros_like(Sim_Replacement_year, dtype=float)
    for i, flag in enumerate(Sim_Replacement_year):
       if flag >= 1:
           rep_years[i] = 1985 + 5 * i

    time = date[4*365:]

# Build DataFrame
    df = pd.DataFrame({ "reactive": xCal_DailyPowers, "optimized": xSim_DailyPowers, "time": time})
    df["diff"] = df["optimized"] - df["reactive"] 
    df["date"] = pd.to_datetime(df["time"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    
     # --- Annual average (in GWh) ---
    annual_avg = df.groupby("year")[["reactive", "optimized"]].sum().mean() / 1e6
    avg_reactive = annual_avg["reactive"]
    avg_optimized = annual_avg["optimized"]

# --- Yearly aggregation ---
    yearly_data = df.groupby("year")[["reactive", "optimized"]].sum().reset_index()
    yearly_data["date"] = pd.to_datetime(yearly_data["year"].astype(str) + "-01-01")


    # --- Set up figure ---
    plt.rcParams.update({
      "axes.labelsize": 20,
      "xtick.labelsize": 18,
      "ytick.labelsize": 18,
      "legend.fontsize": 18})
    
    
    fig = plt.figure(figsize=(11, 10))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2.5, 1.4, 1.4], hspace=0.7)

    ax1 = plt.subplot(gs[0])
    ax1.plot(yearly_data["date"], yearly_data["optimized"] / 1e6,
         label=f"Optimized (AAE: {avg_optimized:.1f} GWh)",
         color="red", linewidth=3, linestyle="--")  # dashed red
    ax1.plot(yearly_data["date"], yearly_data["reactive"] / 1e6,
         label=f"Reactive (AAE: {avg_reactive:.1f} GWh)",
         color="dimgray", linewidth=3)

    #ax1.set_xlabel("Year")
    ax1.set_ylabel("Annual Energy (GWh)")
    ax1.grid(True)

# --- Custom ticks: first, last, and 5 evenly spaced in between ---
    start = yearly_data["date"].min()
    end = yearly_data["date"].max()
    ticks = pd.date_range(start=start, end=end, periods=7)
    ax1.set_xticks(ticks)
    ax1.set_xticklabels([d.strftime("%Y") for d in ticks])
    
# Replacement year markers
    first = True
    for ry in rep_years:
       if ry:
           ry_date = pd.to_datetime(f"{int(ry)}-01-01")
           if start <= ry_date <= end:
               ax1.axvline(ry_date, linestyle="--", color="black")
               first = False
    
    # Reactive replacement year marker (blue dashed at 2040)
    #reactive_year = pd.to_datetime("2040-01-01")
    #ax1.axvline(reactive_year, linestyle="--", color="#007FFF", linewidth=2)
   
    ax1.legend(bbox_to_anchor=(0.44, 1.35), loc="upper left")
    
    # --- Customize x-axis ticks: only start, end, and replacement years ---
    tick_years = [start, end]

# Add all replacement marker years that fall in the range
    for ry in rep_years:
      if ry:
        ry_date = pd.to_datetime(f"{int(ry)}-01-01")
        if start <= ry_date <= end:
            tick_years.append(ry_date)

# Add the 2040 reactive replacement year if within range
    # if start <= reactive_year <= end:
    #   tick_years.append(reactive_year)

# Sort and apply ticks
    tick_years = sorted(list(set(tick_years)))
    ax1.set_xticks(tick_years)
    ax1.set_xticklabels([d.strftime("%Y") for d in tick_years], rotation=45)


    # === MIDLE PANEL: fleet design head scatter ===

    years = tick_years  # same as x-axis of top plot

    # Clean Sim_head_all
    mask = ~np.isnan(Sim_head_all).all(axis=0)
    data_cleaned = Sim_head_all[:, mask]
    head_all = np.where(np.isnan(data_cleaned), 150, data_cleaned)
    head_initial = np.full(head_all.shape[1], 150)

    Sim_head_extended = head_all
    Sim_head_extended  = np.vstack([Sim_head_extended, Sim_head_extended[-1, :]])
   
    # Replacement counts
    rep_counts = derive_replacement_counts_simple(Sim_Replacement_year)
    rep_counts_ext = np.concatenate([np.zeros(4, dtype=int), rep_counts, [0]])

    # Track fleet composition
    total_turbines = 17
    turbine_num = np.ones((1, 6), dtype=int)*6
    turbine_num[:, [2, 5]] = 5
    num_cols = Sim_head_extended.shape[1]
    turbine_numx = turbine_num[:,0:num_cols]
    counts = turbine_numx.T
    # Ensure same number of steps
    steps = Sim_head_extended.shape[0]

    # Collect results
    records = []

    for t in range(steps):
        
        heads = Sim_head_extended[t, :]

        # Expand into counts per head
        for head, count in zip(heads, counts):
             records.extend([{"Step": t, "Head": head}] * int(count.item()))

    df = pd.DataFrame(records)
    df_pivot = df.groupby(["Step", "Head"]).size().reset_index(name="Count")
    
    dt = pd.to_datetime(years)
    color_order = ["blue", "green", "orangered", "purple"]
    
# Extract years
    years_rep = np.array(dt.year.tolist())
    rep_steps = np.round((years_rep -1985) /5);
    numyears = np.arange(0, 116, 5)
    # Scatter plot 
    ax2 = plt.subplot(gs[1])
    
    unique_heads = sorted(df_pivot["Head"].unique())  # consistent order

    for i, head in enumerate(unique_heads):
        subset = df_pivot[(df_pivot["Head"] == head) & (df_pivot["Step"].isin(rep_steps))]
        x = np.array([numyears[step] for step in subset["Step"]])
        y = [head] * len(subset)
        sizes = subset["Count"] * 30  # bubble size scaling
        xa = x + 1985  # convert to actual years
        if xa[-1] == 100:
            xa[-1] = 99
        color = color_order[i % len(color_order)]  # wrap around if more than 4
        ax2.scatter(xa, y, s=sizes, alpha=0.6, color=color, label=f"{head} m")
    
    
    #ax2.set_xlabel("Year")
    ax2.set_ylabel("Head (m)")
    ax2.set_ylim(130, 165)
    ax2.set_xticks(years_rep)
    ax2.set_xticklabels(years_rep.astype(str), rotation=45)
    # ax2.set_xticks(years)
    # ax2.set_xticklabels([d.strftime("%Y") for d in years], rotation=45)
    ax2.legend()
    ax2.grid(True)

    # Combine legends: heads + turbine sizes
    handles, labels = ax2.get_legend_handles_labels()

    legend_handles = [
      plt.scatter([], [], s=100, color=h.get_facecolors()[0], alpha=0.6, label=lab)
      for h, lab in zip(handles, labels)]
    
    ax2.legend(legend_handles, labels,  bbox_to_anchor=(0.74, 1.5), loc="upper left", fontsize=18)
    
    pos = ax2.get_position()
    ax2.set_position([pos.x0, pos.y0 - 0.02, pos.width, pos.height])  # shift down by 0.03


# %

    ##############################
    # === BOTTOM PANEL: design discharge capacity ===
    Qdesign  = np.vstack([Qdesign, Qdesign[-1, :]])
    ax3 = plt.subplot(gs[2])
    
    
    realyears = numyears +1985
    realyears[-1] = 2099
    
    idx = np.where(np.isin(realyears, years_rep))[0]
    # Plot gray baseline line
    ax3.plot(realyears[idx], Qdesign[idx], color="tan", linewidth=2, linestyle="--", zorder=1)

    # Plot red thick circle markers
    ax3.scatter(realyears[idx], Qdesign[idx],
                s=100,                 # size of circles
                facecolors="none",     # hollow circle
                edgecolors="black",      # red edges
                linewidths=3,          # thick edges
                zorder=2)

    ax3.set_xlabel("Time (Years)")
    ax3.set_ylabel("Discharge (m³/s)")
    ax3.set_ylim(1200, 1999) 
    ax3.set_xticks(years_rep)
    ax3.set_xticklabels(years_rep.astype(str), rotation=45)
    ax3.grid(True,  alpha=0.6)


# %%

# -----------------------
# drought indices (robust)
# -----------------------
def drought_indices(res_levels, Lcrit):
    """
    Compute drought indices for each scenario.
    
    Parameters
    ----------
    res_levels : ndarray (n_time, n_scenarios)
        Reservoir level time series.
    Lcrit : float
        Critical threshold for drought.
        
    Returns
    -------
    drought_freq : array (n_scenarios,)
        Fraction of time below threshold.
    drought_duration : array (n_scenarios,)
        Longest continuous spell below threshold.
    drought_intensity : array (n_scenarios,)
        Average depth below threshold.
    """
    n_time, n_scenarios = res_levels.shape
    drought_freq = np.zeros(n_scenarios)
    drought_duration = np.zeros(n_scenarios)
    drought_intensity = np.zeros(n_scenarios)
    
    for j in range(n_scenarios):
        series = res_levels[:, j]
        
        # Frequency
        drought_freq[j] = np.mean(series < Lcrit)
        
        # Duration
        below = (series < Lcrit).astype(int)
        if below.any():
            # lengths of consecutive drought spells
            spells = np.diff(np.where(np.concatenate(([below[0]],
                                                     below[:-1] != below[1:],
                                                     [True])))[0])[::2]
            drought_duration[j] = spells.max()
        else:
            drought_duration[j] = 0
        
        # Intensity
        drought_intensity[j] = np.mean(np.maximum(0, Lcrit - series))
    
    return drought_freq, drought_duration, drought_intensity

# %%

# -----------------------
# parallel coordinate plotting (shared colorbar)
# -----------------------

def parallel_coord_map(
    RL_medians, RL_cv, D_means, drought_freq, dE_mean, firm_gain, dif_NPV, RP, Peak_power, nloc,
    ax=None, cmap='spring_r', figsize=(20, 5), show=True, force_update_cbar=False):
    """
    Parallel coordinates with a single shared horizontal colorbar for Delta NPV.

    Inputs are 1D arrays (one value per scenario). Function returns the axis used.
    """

    # --- Build DataFrame ---
    df = pd.DataFrame({
       r"$\mathrm{Q_{mean}}$": D_means,
       r"$\mathrm{RL_{10}}$": RL_medians,
       r"$\mathrm{RL_{CV}}$": RL_cv,
       r"$\mathrm{Fr_{drought}}$": drought_freq,
       r"$\mathrm{\Delta AE}$": dE_mean.ravel(),
       r"$\mathrm{FPG}$": firm_gain,
       r"$\mathrm{\Delta PP}$": Peak_power,
       r"$\mathrm{EG_{IP}}$": RP,
       r"$\mathrm{\Delta NPV}$": dif_NPV,
    })

    # --- Baseline row ---
    n_cols = len(df.columns)
    base_values = []
    for j, col in enumerate(df.columns):
        if j < 4:
            base_values.append(df[col].mean())
        elif j == 7:
            base_values.append(1)
        else:
            base_values.append(0)
    base_row = pd.DataFrame([base_values], columns=df.columns)

    df_with_base = pd.concat([df, base_row], ignore_index=True)
    minmax_combined = {col: (df_with_base[col].min(), df_with_base[col].max())
                       for col in df_with_base.columns}
    df_norm = df_with_base.copy()
    for col in df_with_base.columns:
        lo, hi = minmax_combined[col]
        df_norm[col] = (df_with_base[col] - lo) / (hi - lo) if hi > lo else 0.5

    # --- Figure/axis ---
    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(figsize=figsize)
    fig = ax.figure

    # --- Colormap input ---
    if isinstance(cmap, (list, tuple, np.ndarray)):
        cmap_obj = ListedColormap(cmap)
        cmap_name = None
    elif isinstance(cmap, str):
        cmap_obj = plt.get_cmap(cmap)
        cmap_name = cmap
    else:
        cmap_obj = cmap
        cmap_name = getattr(cmap, 'name', None)

    # --- Shared vmin/vmax and cmap tracking ---
    cmin, cmax = float(df["$\mathrm{\\Delta NPV}$"].min()), float(df["$\mathrm{\\Delta NPV}$"].max())
    if not hasattr(fig, "_shared_vrange"):
        fig._shared_vrange = (cmin, cmax)
    else:
        old_min, old_max = fig._shared_vrange
        fig._shared_vrange = (min(old_min, cmin), max(old_max, cmax))

    if not hasattr(fig, "_shared_cmap_obj") or force_update_cbar:
        fig._shared_cmap_obj = cmap_obj

    cmap_shared = fig._shared_cmap_obj
    vmin, vmax = fig._shared_vrange
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # --- Plot lines ---
    n_rows = len(df_norm)
    for i in range(n_rows - 1):
        val = float(df["$\mathrm{\\Delta NPV}$"].iloc[i])
        color = cmap_shared(norm(val))
        ax.plot(df_norm.columns, df_norm.iloc[i, :],
                color=color, alpha=0.85, linewidth=3, zorder=2)

    # --- Baseline partial black line ---
    j_start = 4
    x_subset = df_norm.columns[j_start:]
    y_subset = df_norm.iloc[-1, j_start:]
    ax.plot(x_subset, y_subset, color='r', linewidth=4, zorder=10, label='Baseline')

    # --- Remove old shared colorbar axes ---
    for a in list(fig.axes):
        if getattr(a, "get_gid", lambda: None)() == "shared_deltaNPV_cbar":
            try: a.remove()
            except Exception: pass


    # --- Anchor shared colorbar above topmost axes ---
    anchor_ax = max([a for a in fig.axes if hasattr(a, "get_position")],
                    key=lambda aa: aa.get_position().y1, default=ax)
    pos = anchor_ax.get_position()
    pad, height = 0.05, 0.02
    
    # make colorbar 70% width and centered
    cbar_width = pos.width * 0.9
    left_offset = pos.x0 + pos.width * 0.01
    cax = fig.add_axes([left_offset, pos.y1 + pad, cbar_width, height])
    cax.set_gid("shared_deltaNPV_cbar")

# --- Shared top colorbar ---
    sm = mpl.cm.ScalarMappable(cmap=cmap_shared, norm=norm)
    sm.set_array([])
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap_shared, norm=norm, orientation='horizontal')
 
    cax.xaxis.set_label_position('top')
    cax.xaxis.tick_top()

# Remove default label
    cbar.set_label("", fontsize=30)

# Add custom label at the left start of the colorbar
    cax.text(-0.2, 0.5, r"$\mathrm{\Delta NPV} \, (\mathrm{M} \$)$",
         ha='left', va='bottom', fontsize=32, transform=cax.transAxes)
    cax.tick_params(labelsize=30)


    # --- Grid & formatting ---
    ax.grid(True, alpha=0.5)
    ax.set_yticks([])
    for y in [0.25, 0.5, 0.75]:
        ax.axhline(y, color="lightgray", linestyle="--", linewidth=0.8, zorder=1)

    # --- Annotate top & bottom values ---
    orig_minmax = {col: (df_with_base[col].min(), df_with_base[col].max()) for col in df_with_base.columns}
    for j, col in enumerate(df_with_base.columns):
        lo, hi = orig_minmax[col]
        ax.text(j, -0.099, f"{lo:.2f}", ha="center", va="top", fontsize=30, color="dimgray")
        ax.text(j, 1.03, f"{hi:.2f}", ha="center", va="bottom", fontsize=30, color="dimgray")

    # --- nloc == 1 ---
    if nloc == 1:
        for j, col in enumerate(df.columns):
            if j > 3:
                idx_min = df[col].idxmin()
                raw_min = df[col].iloc[idx_min]
                y_norm = df_norm[col].iloc[idx_min]
                y_text = y_norm - 0.04
                ax.text(j, y_text, f"{raw_min:.2f}", ha="center", va='top',
                        fontsize=29, color="black",
                        bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.85))
                ax.plot(j, y_norm, 'o', color='black', markersize=8, zorder=12)

    # --- nloc == 2 ---
    if nloc == 2:
        for j, col in enumerate(df.columns):
            if j > 3:
                if j == 4:
                    idx_val = df[col].idxmax()
                    raw_val = df[col].iloc[idx_val]
                    y_norm = df_norm[col].iloc[idx_val]
                elif j == 5:
                    raw_val = 0
                    lo, hi = minmax_combined[col]
                    y_norm = 0 if hi == lo else (raw_val - lo) / (hi - lo)
                else:
                    idx_val = df[col].idxmin()
                    raw_val = df[col].iloc[idx_val]
                    y_norm = df_norm[col].iloc[idx_val]
                y_text = y_norm - 0.025
                ax.text(j, y_text, f"{raw_val:.2f}", ha="center", va='top',
                        fontsize=29, color="black",
                        bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.85))
                ax.plot(j, y_norm, 'o', color='black', markersize=8, zorder=12)

    # --- Axes cleanup ---
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="x", length=0, pad=75)
    plt.setp(ax.get_xticklabels(), fontsize=34)
    for i, label in enumerate(ax.get_xticklabels()):
        label.set_color("mediumblue" if i < 4 else "black")


    return ax

# %%

# Helper: draw a small left-side bar panel
def draw_bar_panel(ax, values, labels, ylim=(0, 41), ylabel="Turbines Replaced",
                   bar_colors=("green", "dodgerblue"), fontsize=18, dashed_lines=(17, 25)):
    """
    Draw a compact vertical bar chart on `ax` and return the x-range of the second bar
    (used to draw short horizontal dashed lines above it).
    """
    ax.bar(labels, values, width=0.4, color=bar_colors, alpha=0.7,
           edgecolor="black", linewidth=1.2)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right")

    # Get geometric extents for the second bar (HyTUNE)
    hy_bar = ax.patches[1]
    x_left = hy_bar.get_x() - 0.1
    x_right = hy_bar.get_x() + hy_bar.get_width() + 0.1

    # Draw dashed horizontal lines only across the HyTUNE bar region
    for y in dashed_lines:
        ax.hlines(y, x_left, x_right, color="navy", linewidth=4, linestyles="dashed")

    # Return x span so other panels can reuse it if needed
    return x_left, x_right

