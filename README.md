
HyTUNE (Hydropower Turbine Upgrade and Next-generation Planning) is a dynamic decision-support tool that integrates basin hydrology, plant hydraulics, and adaptive optimization to guide turbine replacement timing and configuration.
The methods and code in this repository support the analysis presented in the paper "Adaptive Turbine Replacement Improves Hydropower Flexibility in a Changing Climate (V. Yildiz et al.)", submitted to Water Resources Research.


**Repository overview

This repository contains two primary Jupyter notebooks and a set of Python modules used to optimize and simulate adaptive turbine replacement policies.

    HyTUNE_opt_JN — HyTUNE decision-tree optimization.
Runs a parallelized evolutionary search (PTreeOpt) over discrete replacement actions and continuous system state features to find decision-tree policies.
Output: snapshots_optimal.pkl (pickled optimization snapshots for later simulation and analysis).

    HyTUNE_sim_JN — Simulation with optimal decision tree.
Loads the optimized decision tree, runs hydropower simulations across all scenario realizations, computes summary metrics, and visualizes results.
Also identifies representative reservoir elevation scenarios (driest and wettest) and prepares inputs for parallel-coordinate visualizations of performance and climate metrics.

** Main functions and modules

Below are the main functions and what they do. These appear in the notebooks or the plot_functions.py / model modules.

  # Optimization

    PTreeOpt wrapper (HyTUNE_opt_JN)
Builds the optimization problem: feature bounds, discrete action set, population and genetic parameters.
Evaluates candidate decision trees by calling the hydropower model evaluator model_V.f.
Runs the search in parallel and stores periodic snapshots for later use.

    energy_model (from mRun_HydropowerModel.mRun_ResSimModel)
Encapsulates the plant hydraulic model, efficiency, and economic calculations.
Provides a .f(policy, mode=...) method that either evaluates objective metrics during optimization or runs full scenario simulations when given a policy.
Returns time series of power, hydraulic states, replacement years, efficiency traces, installed capacity additions, simulated NPV and other performance outputs.

  # Simulation / postprocessing

Load and apply optimal policy
Load snapshots_optimal.pkl (or snapshots.pkl) and select the final decision tree policy.

Call model_V.f(policy, mode='simulation') to generate Sim_DailyPowers, Sim_PeakPowers, IC_added, Sim_of (objective/NPV) and hydraulic traces.

Scenario selection & preprocessing
Read scenario CSVs (Daily_predict_discharges.csv, Daily_predict_elevations.csv)

  # Plotting and diagnostics (plot_functions.py)

-DecisonTree_Plot
Exports a static visualization (SVG) of the decision tree with action node colors.

-EL_scenario_plots
Plots reservoir elevation ensembles with a long-window (10-yr) moving average, 95% band, ensemble mean, and highlighted scenario traces.

-combined_yearly_energy
Produces a three-panel figure: annual energy comparison with replacement markers, fleet design head composition scatter, and time series of design discharge capacity.

-parallel_coord_map
Builds parallel-coordinate visualizations to compare climate and performance metrics across scenarios. It normalizes metrics, appends a baseline row, and places a shared horizontal colorbar for ΔNPV.

-draw_bar_panel
Small helper to draw the left-side bar panels used alongside parallel plots. Returns geometry used to draw short dashed markers.

-drought_indices
Computes drought metrics per scenario: frequency (fraction of time below threshold), longest continuous drought duration, and mean drought intensity.


**License
This project is licensed under the MIT License.

**Contact
For questions or help, contact: [veysel.yildiz@duke.edu].