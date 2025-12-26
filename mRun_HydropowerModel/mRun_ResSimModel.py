import numpy as np
from scipy.interpolate import interp1d

#############################################################################hydraulics##
#from hydraulic_functions import calculate_hydraulics

def calculate_hydraulics(input_flows, num_turbine_op):
    """
    Calculate the hydraulic head loss in a piping system.

    Parameters:
    input_flows (numpy.ndarray): Array of input flow rates (m³/s).
    num_turbine_op (numpy.ndarray): Array indicating the number of turbines operating.

    Returns:
    numpy.ndarray: Array of head losses (m).
    """
    # Hydraulic system parameters
    L1 = 360  # Pipe 1 length in meters
    D1 = 9.15  # Pipe 1 diameter in meters
    L2 = 100  # Pipe 2 length in meters
    D2 = 3.97  # Pipe 2 diameter in meters
    kminor1 = 0.605  # Minor loss coefficient for Pipe 1
    kminor2 = 0.553  # Minor loss coefficient for Pipe 2
    
    # Constants
    ve = 1.004e-6  # Kinematic viscosity of water (m²/s)
    e_roughness = 6.36e-4  # Pipe roughness (m)

    # Relative roughness
    ed1 = e_roughness / D1
    ed2 = e_roughness / D2

    # Equivalent diameters based on operating units
    D1p = D1 * np.minimum(4, np.ceil(num_turbine_op / 4))**(2/5)
    D2p = D2 * num_turbine_op**(2/5)

    # Reynolds numbers
    Re_d1 = 4 * input_flows / (np.pi * D1p * ve)
    Re_d2 = 4 * input_flows / (np.pi * D2p * ve)

    # Friction factors using the Colebrook-White equation
    f_d1 = moody(ed1, Re_d1)
    f_d2 = moody(ed2, Re_d2)

    # Flow velocities
    V_d1 = 4 * input_flows / (np.pi * D1p**2)
    V_d2 = 4 * input_flows / (np.pi * D2p**2)

    # Frictional head loss calculations using the Darcy-Weisbach equation
    hloss = ((f_d1 * L1 / D1 * V_d1**2) + kminor1 * V_d1**2) / 19.62 + \
            ((f_d2 * L2 / D2 * V_d2**2) + kminor2 * V_d2**2) / 19.62

    return hloss

################################################## MOODY ########################
def moody(ed, Re):
    """Return ff, friction factor
    --------------------------------------
      Inputs:

        HP : structure with variables used in calculation
        ed : the relative roughness: epsilon / diameter.
        Re : the Reynolds number

    """
    ff = np.zeros_like(Re)

    # Find the indices for Laminar, Transitional and Turbulent flow regimes

    LamR = np.where((0 < Re) & (Re < 2000))
    LamT = np.where(Re > 4000)
    LamTrans = np.where((2000 < Re) & (Re < 4000))

    ff[LamR] = 64 / Re[LamR]

    # Calculate friction factor for Turbulent flow using the Colebrook-White approximation
    ff[LamT] = 1.325 / (np.log(ed / 3.7 + 5.74 / (Re[LamT] ** 0.9)) ** 2)

    Y3 = -0.86859 * np.log(ed / 3.7 + 5.74 / (4000**0.9))
    Y2 = ed / 3.7 + 5.74 / (Re[LamTrans] ** 0.9)
    FA = Y3 ** (-2)
    FB = FA * (2 - 0.00514215 / (Y2 * Y3))
    R = Re[LamTrans] / 2000
    X1 = 7 * FA - FB
    X2 = 0.128 - 17 * FA + 2.5 * FB
    X3 = -0.128 + 13 * FA - 2 * FB
    X4 = R * (0.032 - 3 * FA + 0.5 * FB)
    ff[LamTrans] = X1 + R * (X2 + R * (X3 + X4))

    return ff

################################################## MOODY ########################
def Period_Flows(input_flows, Q_design):
    """
    Return flow for each period: peak, mid, low demand
    """

    qsize = input_flows.shape[0]

    # Mask for low flow days
    low_flow_mask = (input_flows.flatten() < 57)

    # Initialize outputs
    peak_flows = np.zeros((qsize, 1))
    mid_flows = np.zeros((qsize, 1))
    offpeak_flows = np.zeros((qsize, 1))

    # --- Case 1: low flows (<57)
    peak_flows[low_flow_mask] = 4 * input_flows[low_flow_mask]

    # --- Case 2: normal/high flows
    if np.any(~low_flow_mask):
        inflow_high = input_flows[~low_flow_mask]

        # Constant off-peak (6 hours × 57)
        OFFpeak_flows = 57 * np.ones((inflow_high.shape[0], 1))

        # Peak (6 hours at demand)
        peak_h = np.minimum(4 * inflow_high - 3 * OFFpeak_flows, Q_design)

        # Mid flows (12 hours)
        Midflows = np.maximum(inflow_high - peak_h/4 - OFFpeak_flows/4, 0) * 2
        mid_h = np.minimum(Midflows, Q_design)

        # Remaining flows → push to off-peak
        remain_mid = np.maximum(Midflows - mid_h, 0)
        off_h = np.minimum(OFFpeak_flows + remain_mid * 2, Q_design)

        # Assign back
        peak_flows[~low_flow_mask] = peak_h
        mid_flows[~low_flow_mask] = mid_h
        offpeak_flows[~low_flow_mask] = off_h

    # Final stacked array
    periods_flows = np.hstack((peak_flows, mid_flows, offpeak_flows))
    return periods_flows

################################################## efficiency_periods ########################
def efficiency_periods(hnet, nk, idx, H_designs, efficiency_vector,Effcurve):
    
    """Return efficiency for each period: peak, mid days, low demand
    --------------------------------------
      Inputs:


    """

   # Initialize the shifted_efficiency array
    shifted_efficiency = np.ones((len(hnet), nk))
    
    # Expand the flow values to match the number of periods      
    hnet_duplicated = np.repeat(hnet, nk, axis=1)
    
    
    # Populate the shifted_efficiency array based on start_idx
    for i in range(nk):
        
        shifted_efficiency[:, i] = efficiency_vector[int(idx[i,0]): int(idx[i,1]) ]
       
    # Repeat the shifted_efficiency array to match the required dimensions
    shifted_efficiency = np.tile(shifted_efficiency, (1, 3))

    # Calculate the efficiency adjustment
    efficiency_interpolator = interp1d(Effcurve[:, 0], Effcurve[:, 1], kind='linear', fill_value='extrapolate')
    
    eff = efficiency_interpolator(hnet_duplicated / H_designs * 100) * 0.01
           
    # Adjust the efficiency with the shifted values
    eff_adjusted = eff * shifted_efficiency    

    return eff_adjusted, shifted_efficiency[-1,0:nk]

################################################## outage_function ########################
# Define the outage function F(x)
def outage_function(x):
    
    """The curve steepens over time due to 
    the exponential increase in failures and the rising probability of occurrence.
    --------------------------------------
      Inputs: turbine age as x

    """   
    f_x = 0.0162 * np.exp(0.04186 * x)  # Exponential outage increase
    
    #sigmoid functionProbability of occurrence
    P_x = 1 / (1 + np.exp(-0.1 * (x - 40)))  #it exhibits an S-shaped curve, smoothly transitioning from 0 to 1 as x increases.
    return f_x * P_x
    

def pairwise_replace(arr):
    """
    For each row, if col(i) < col(i+1), set col(i) = col(i+1).
    Works for all complete pairs (0,1), (2,3), (4,5), ...
    """
    arr = arr.copy()
    n_cols = arr.shape[1]

    # Loop only until the second-to-last column if even, or ignore last if odd
    for i in range(0, n_cols - 1, 2):
        mask = arr[:, i] < arr[:, i+1]
        arr[mask, i] = arr[mask, i+1]

    return arr


##################################################  ########################
def term_NPV(DailyRevenue, r, per_start, per_end, Cost):
    """
    Calculates the total discounted net revenue from daily revenue data.

    Parameters:
    -----------
    DailyRevenue :Array of daily revenue values over multiple years.
    r :  Annual discount rate (e.g., 0.05 for 5%).
    nperiod : the planning or decision period (each period is 5 years after year 20).
    cost : One-time investment or replacement cost in millions.

    Returns:
    --------
    discounted_revenue : Net present value (NPV) of the revenue minus cost, fully discounted to year 0.
    """
  
    num_years = per_end - per_start
    
    total_discounted = 0.0
    # Adjust discount rate for very long-term planning
    # if Ycurrent > 50:        r = 0.01
    # Aggregate daily revenue into annual totals and discount each year's revenue
    for t in range(num_years):
        start = t * 365
        end = start + 365
        annual_revenue = np.sum(DailyRevenue[start:end]) / 1e6  # Convert to millions
        discounted = annual_revenue / ((1 + r) ** (t + 1))       # Discount to start of each year
        total_discounted += discounted
    
        # Determine number of years from initial time to start of simulation
        Ycurrent = per_start
        # Discount total revenue (minus cost) back to year 0
        discounted_revenue = (total_discounted - Cost) / ((1 + r) ** (Ycurrent + 1))

    return discounted_revenue

# %% MAIN
class energy_model():
    """
 Optimizes turbine replacement planning and energy generation.
    Inputs:
        D_discharge: Initial discharge capacities (vector)
        head_range: Range of turbine head values
        Q: Flow data (vector)
        h_el: Elevation head data (vector)
        Effcurve: Efficiency curve data (matrix: head-efficiency pairs)
        e_price: Energy price for peak time, mid day and off peak time
        i_rate: interest rate
        X: Optimization parameters [num_periods, replacement times, turbines replaced, new heads, discharges]

    Outputs:
        OF: Objective function value (cost-effectiveness measure)
    """
    def __init__(self, Effcurve,  e_price, i_rate,sen_discharges,sen_elevations):
        
        
        
        Q_sen = sen_discharges
        H_sen =  sen_elevations
        s_length = Q_sen.shape[1]
        
        Nyear = Q_sen.shape[0]/365
        self.discount_factors = 1 / (1 + i_rate) ** Nyear 
        self.CRF = i_rate * (1 + i_rate) ** Nyear / ((1 + i_rate) ** Nyear - 1)
        self.e_price = e_price
        
        self.Effcurve = Effcurve
        
        self.i_rate = i_rate
        self.sen_discharges = sen_discharges
        self.sen_elevations = sen_elevations
        
        # Create a two-column dataset: head, discharge, age
        # self.X_exs = np.column_stack((head_range[2] * np.ones(D_discharge.shape), D_discharge, np.ones(D_discharge.shape)*20)) 
        
        D_discharge = np.array([45] + [96.3] * 15 + [50])
        X_1985 = np.column_stack((150 * np.ones(D_discharge.shape), D_discharge, np.ones(D_discharge.shape)*0)) 
        #X_1980[:,2] = 45
        
        # D_discharge3 = np.array([61] + [132] * 15 + [68])
        # X_exs3 = np.column_stack((150 * np.ones(D_discharge3.shape), D_discharge2, np.ones(D_discharge3.shape)*25)) 
        # X_exs3[0:5,0] = 135.65
        # X_exs3[1:5,1] = 149
        # X_exs3[0:5,2] = 0
        # X_now = X_exs3
        # X_now[:,2] += 5
        # flipped = np.flipud(X_now)
        
        #from calibration
        eff_drop = 1 - 0.2
        period_days = np.arange(1, 54750)  # 1 to 31390 inclusive
        efficiency_vector = 1 + ((eff_drop - 1) / 54750) * period_days
        self.efficiency_vector = efficiency_vector
        
        self.X_exs =X_1985.copy()
        
        intervals = [(0, 4)] + [(i, i+5) for i in range(4, 110, 5)]
        
        intervals_10 = [(0, 4), (0, 9)] + [(i, i+9) for i in range(5, 110, 5)]
        
        
        r_length = len(intervals)

        data_years = sen_discharges.shape[0] // 365   # integer division
        self.nyears = data_years
        
        ResEl_5r = np.zeros((r_length,s_length))
        ResEl_10r = np.zeros((r_length,s_length))
        
        ResEl_maxr = np.zeros((r_length,s_length))
        ResEl_minr = np.zeros((r_length,s_length))
        
        Q_5r = np.zeros((r_length,s_length))
        Q_10r = np.zeros((r_length,s_length))
        
        for j in range(s_length):   
            
            h_sel = H_sen[:,j]
            q_sel = Q_sen[:,j]
            
            h_el_yearly = h_sel.reshape(data_years, 365)
            q_yearly = q_sel.reshape(data_years, 365)
            
            ResEl_5r[:,j] = np.array([h_el_yearly[start:end].mean() for start, end in intervals])
            ResEl_10r[:,j] = np.array([h_el_yearly[start:end].mean() for start, end in intervals_10])

            ResEl_maxr[:,j] = np.array([h_el_yearly[start:end].max() for start, end in intervals])
            ResEl_minr[:,j] = np.array([h_el_yearly[start:end].min() for start, end in intervals_10])
            
            Q_5r[:,j] = np.array([q_yearly[start:end].mean() for start, end in intervals])
            Q_10r[:,j] = np.array([q_yearly[start:end].mean() for start, end in intervals_10])        
            
            
        ResEl_5x = ResEl_5r.T 
        ResEl_10x = ResEl_10r.T 
        ResEl_5x = ResEl_5x.flatten() 
        ResEl_10x = ResEl_10x.flatten() 
                
        ResEl_max = ResEl_maxr.T  
        ResEl_max = ResEl_max.flatten()    

        ResEl_min = ResEl_minr.T  
        ResEl_min = ResEl_min.flatten()  

        Q_5x = Q_5r.T 
        Q_10x = Q_10r.T 
        Q_5x = Q_5x.flatten() 
        Q_10x = Q_10x.flatten() 
        
        self.ResEl_5x = ResEl_5x # Compute the average for each interval
        self.ResEl_10x = ResEl_10x # Compute the average for each interval 
        
        self.ResEl_max = ResEl_max
        self.ResEl_min = ResEl_min
        
        self.Q_5x = Q_5x # Compute the average for each interval
        self.Q_10x = Q_10x # Compute the average for each interval 
        
        self.nt_change = 6 # six turbine to be chnaged each period, yet for pchange=2 and 5 it will be 5. 
       
        # Initialize the first element
        
        self.max_head = 150
        self.min_head = 150
        self.Av_efficiency = 89.7  # Initialize the first element
        self.max_efficiency = 90.82
        self.min_efficiency = 90.82 
        self.Q_High_eff = 1327.5
        self.Q_Low_eff = 1327.5

        self.s_length = s_length 
        self.Q_sen = Q_sen
        self.H_sen = H_sen
        
        self.ResEl_threshold_5 = 150.0
        # penalty magnitude used in optimization mode for soft penalty approach
        self.optim_penalty = 1e9
        
    def f(self, P, mode='optimization'):
        
        # Create a fresh copy for each run
        Xpars = np.copy(self.X_exs)
        Xorig = np.copy(self.X_exs)
        ResEl_5 = self.ResEl_5x
        ResEl_10 = self.ResEl_10x
        ResEl_max = self.ResEl_max
        ResEl_min = self.ResEl_min
        
        Q_5 = self.Q_5x
        Q_10 = self.Q_10x
        
        sen_length = self.s_length
        H_sen = self.H_sen
        Q_sen = self.Q_sen
        nyears = self.nyears
        target = np.zeros((ResEl_5.size,3))
        
        r_length = int(ResEl_5.size/sen_length)
        Revenue = np.zeros((r_length,sen_length))
        
        Replacement_year = np.zeros((r_length, sen_length))
        IC_added = np.zeros((r_length, sen_length))
  
        Sim_revenue = np.zeros((r_length,sen_length))
        Sim_cost    = np.zeros((len(ResEl_5), 1))
        Sim_eff     = np.zeros((len(ResEl_5), 3))
        Sim_head    = np.zeros((len(ResEl_5), 2))
        Qdesign     = np.zeros((len(ResEl_5), 1))
        Sim_head_all= np.zeros((len(ResEl_5), 4))
        Sim_Q       = np.zeros((len(ResEl_5), 2))
        
        # Create an empty DataFrame
        DailyPowers_all = np.zeros(((nyears-4)*365,sen_length))
        DailyPowers_peak = np.zeros(((nyears-4)*365,sen_length))
        
        # Store initial values
        initial_Xpars = np.copy(self.X_exs)
        initial_max_head = self.max_head
        initial_min_head = self.min_head
        initial_Av_efficiency = self.Av_efficiency
        initial_max_efficiency = self.max_efficiency
        initial_min_efficiency = self.min_efficiency
        initial_Q_High_eff = self.Q_High_eff
        initial_Q_Low_eff = self.Q_Low_eff

        previous_j = None  # Track the previous j value
        
        intervals_per_scenario = len(ResEl_5) // sen_length

        for t in range(len(ResEl_5)):
            
            j = t // intervals_per_scenario        # scenario index
            adjusted_t = t % intervals_per_scenario  # interval index
 
            # Reset states if j has changed
            if j != previous_j:
                Xpars = initial_Xpars
                max_head = initial_max_head
                min_head = initial_min_head
                Av_efficiency = initial_Av_efficiency
                max_efficiency = initial_max_efficiency
                min_efficiency = initial_min_efficiency
                Q_High_eff = initial_Q_High_eff
                Q_Low_eff = initial_Q_Low_eff
                pchange = 0

                
            previous_j = j  # Update tracking variable

            policy, rules = P.evaluate([ ResEl_5[t], ResEl_10[t], max_head, min_head, Av_efficiency, max_efficiency,
                                          Q_5[t], Q_10[t], ResEl_max[t], ResEl_min[t] ])

            # Debug: check what the tree is doing
            # if t % 23 == 20 and mode == "simulation":
            #      print(f"Scenario {j}, interval={adjusted_t}, policy={policy}")
            
            # print(f"Scenario {j}, interval={adjusted_t}, policy={policy}")
            
            if policy == 'No Replacement':      target[t] = np.array([0, 0, 0]) # in order of change(yes/no), head (m), discharge(%)      
            elif policy == 'Replace Existing':  target[t] = np.array([1, 0, 0])
            # Single Head (H) and Discharge (D) adjustments  
            elif policy == 'H+2.5':         target[t] = np.array([2,  2.5,  0])
            elif policy == 'H+5':           target[t] = np.array([2,  5,    0])   
            elif policy == 'H+7.5':         target[t] = np.array([2,  7.5,  0])
            elif policy == 'H+10':          target[t] = np.array([2,  10,   0])  
            elif policy == 'H-2.5':         target[t] = np.array([2, -2.5, 0])        
            elif policy == 'H-5':           target[t] = np.array([2, -5,   0])        
            elif policy == 'H-7.5':         target[t] = np.array([2, -7.5, 0])    
            elif policy == 'H-10':          target[t] = np.array([2, -10,  0]) 
            elif policy == 'D+5':           target[t] = np.array([2,  0,  5])                  
            elif policy == 'D+10':          target[t] = np.array([2,  0,  10])             
            elif policy == 'D+15':          target[t] = np.array([2,  0,  15])             
            elif policy == 'D+20':          target[t] = np.array([2,  0,  20])                
            elif policy == 'D-5':           target[t] = np.array([2,  0, -5])               
            elif policy == 'D-10':          target[t] = np.array([2,  0, -10])            
            elif policy == 'D-15':          target[t] = np.array([2,  0, -15])     
            elif policy == 'D-20':          target[t] = np.array([2,  0, -20])   
            # Combined policies with H+-2.5
            elif policy == 'H+2.5 & D+5':    target[t] = np.array([2,  2.5,  5])     
            elif policy == 'H+2.5 & D+10':   target[t] = np.array([2,  2.5,  10])    
            elif policy == 'H+2.5 & D+15':   target[t] = np.array([2,  2.5,  15])    
            elif policy == 'H+2.5 & D+20':   target[t] = np.array([2,  2.5,  20])    
            elif policy == 'H+2.5 & D-5':    target[t] = np.array([2,  2.5, -5])    
            elif policy == 'H+2.5 & D-10':   target[t] = np.array([2,  2.5, -10])    
            elif policy == 'H+2.5 & D-15':   target[t] = np.array([2,  2.5, -15])    
            elif policy == 'H+2.5 & D-20':   target[t] = np.array([2,  2.5, -20]) 
            elif policy == 'H-2.5 & D+5':    target[t] = np.array([2, -2.5,  5])     
            elif policy == 'H-2.5 & D+10':   target[t] = np.array([2, -2.5,  10])    
            elif policy == 'H-2.5 & D+15':   target[t] = np.array([2, -2.5,  15])    
            elif policy == 'H-2.5 & D+20':   target[t] = np.array([2, -2.5,  20])    
            elif policy == 'H-2.5 & D-5':    target[t] = np.array([2, -2.5, -5])    
            elif policy == 'H-2.5 & D-10':   target[t] = np.array([2, -2.5, -10])    
            elif policy == 'H-2.5 & D-15':   target[t] = np.array([2, -2.5, -15])    
            elif policy == 'H-2.5 & D-20':   target[t] = np.array([2, -2.5, -20])  
            # Combined policies with H+-5
            elif policy == 'H+5 & D+5':      target[t] = np.array([2,  5,  5])     
            elif policy == 'H+5 & D+10':     target[t] = np.array([2,  5,  10])    
            elif policy == 'H+5 & D+15':     target[t] = np.array([2,  5,  15])    
            elif policy == 'H+5 & D+20':     target[t] = np.array([2,  5,  20])    
            elif policy == 'H+5 & D-5':      target[t] = np.array([2,  5, -5])    
            elif policy == 'H+5 & D-10':     target[t] = np.array([2,  5, -10])    
            elif policy == 'H+5 & D-15':     target[t] = np.array([2,  5, -15])    
            elif policy == 'H+5 & D-20':     target[t] = np.array([2,  5, -20]) 
            elif policy == 'H-5 & D+5':      target[t] = np.array([2, -5,  5])     
            elif policy == 'H-5 & D+10':     target[t] = np.array([2, -5,  10])    
            elif policy == 'H-5 & D+15':     target[t] = np.array([2, -5,  15])    
            elif policy == 'H-5 & D+20':     target[t] = np.array([2, -5,  20])    
            elif policy == 'H-5 & D-5':      target[t] = np.array([2, -5, -5])    
            elif policy == 'H-5 & D-10':     target[t] = np.array([2, -5, -10])    
            elif policy == 'H-5 & D-15':     target[t] = np.array([2, -5, -15])    
            elif policy == 'H-5 & D-20':     target[t] = np.array([2, -5, -20]) 
            # Combined policies with H+-7.5           
            elif policy == 'H+7.5 & D+5':    target[t] = np.array([2,  7.5,  5])     
            elif policy == 'H+7.5 & D+10':   target[t] = np.array([2,  7.5,  10])    
            elif policy == 'H+7.5 & D+15':   target[t] = np.array([2,  7.5,  15])    
            elif policy == 'H+7.5 & D+20':   target[t] = np.array([2,  7.5,  20])    
            elif policy == 'H+7.5 & D-5':    target[t] = np.array([2,  7.5, -5])    
            elif policy == 'H+7.5 & D-10':   target[t] = np.array([2,  7.5, -10])    
            elif policy == 'H+7.5 & D-15':   target[t] = np.array([2,  7.5, -15])    
            elif policy == 'H+7.5 & D-20':   target[t] = np.array([2,  7.5, -20]) 
            elif policy == 'H-7.5 & D+5':    target[t] = np.array([2, -7.5,  5])     
            elif policy == 'H-7.5 & D+10':   target[t] = np.array([2, -7.5,  10])    
            elif policy == 'H-7.5 & D+15':   target[t] = np.array([2, -7.5,  15])    
            elif policy == 'H-7.5 & D+20':   target[t] = np.array([2, -7.5,  20])    
            elif policy == 'H-7.5 & D-5':    target[t] = np.array([2, -7.5, -5])    
            elif policy == 'H-7.5 & D-10':   target[t] = np.array([2, -7.5, -10])    
            elif policy == 'H-7.5 & D-15':   target[t] = np.array([2, -7.5, -15])    
            elif policy == 'H-7.5 & D-20':   target[t] = np.array([2, -7.5, -20])  
            # Combined policies with H+-10
            elif policy == 'H+10 & D+5':     target[t] = np.array([2,  10,   5])     
            elif policy == 'H+10 & D+10':    target[t] = np.array([2,  10,  10])    
            elif policy == 'H+10 & D+15':    target[t] = np.array([2,  10,  15])    
            elif policy == 'H+10 & D+20':    target[t] = np.array([2,  10,  20])    
            elif policy == 'H+10 & D-5':     target[t] = np.array([2,  10,  -5])    
            elif policy == 'H+10 & D-10':    target[t] = np.array([2,  10,  -10])    
            elif policy == 'H+10 & D-15':    target[t] = np.array([2,  10,  -15])    
            elif policy == 'H+10 & D-20':    target[t] = np.array([2,  10,  -20]) 
            elif policy == 'H-10 & D+5':     target[t] = np.array([2, -10,   5])     
            elif policy == 'H-10 & D+10':    target[t] = np.array([2, -10,   10])    
            elif policy == 'H-10 & D+15':    target[t] = np.array([2, -10,   15])    
            elif policy == 'H-10 & D+20':    target[t] = np.array([2, -10,   20])    
            elif policy == 'H-10 & D-5':     target[t] = np.array([2, -10,  -5])    
            elif policy == 'H-10 & D-10':    target[t] = np.array([2, -10,  -10])    
            elif policy == 'H-10 & D-15':    target[t] = np.array([2, -10,  -15])    
            elif policy == 'H-10 & D-20':    target[t] = np.array([2, -10,  -20]) 
            # 
            

  
            # --- Now run simulation only when appropriate ---
            if mode == 'simulation':
                Sim_revenue[adjusted_t,j], DPowers, Sim_cost[t], AvEff, X_new, all_heads, Q_design, \
                    head_high, head_low, max_eff, min_eff, eff_Q_high, eff_Q_low, orig_pchange, addedIC, peak_energy = \
                    self.energy_sim(adjusted_t, target[t], pchange, Xpars, Q_sen[:, j], H_sen[:, j], Xorig)

                id_start = adjusted_t * 5 * 365
                id_end = id_start + 5 * 365
                DailyPowers_all[id_start:id_end, j] = DPowers
                DailyPowers_peak[id_start:id_end, j] = peak_energy

                Sim_eff[t, :] = np.array([AvEff, max_eff, min_eff])
                Sim_head[t, :] = np.array([head_high, head_low])
                Sim_Q[t, :] = np.array([eff_Q_high, eff_Q_low])
                Qdesign[t] = Q_design
                IC_added[adjusted_t, j] = addedIC
                head_row = np.full(4, np.nan)
                head_row[:len(all_heads)] = all_heads
                Sim_head_all[t, :] = head_row

                idchange = target[t]
                if idchange[0] == 1:
                    Replacement_year[adjusted_t, j] = 1
                elif idchange[0] == 2:
                    Replacement_year[adjusted_t, j] = 2
                        
            else:

                Revenue[adjusted_t,j], _, _, AvEff, X_new, _,_, head_high, head_low, max_eff, min_eff, eff_Q_high, eff_Q_low, orig_pchange,_ ,_= \
                           self.energy_sim(adjusted_t, target[t], pchange, Xpars, Q_sen[:,j], H_sen[:,j], Xorig) 
   
            # update pchange and state as before
            pchange = orig_pchange

            Xpars = X_new
            max_head = head_high
            min_head = head_low
            Av_efficiency = AvEff * 100
            max_efficiency = max_eff * 100
            min_efficiency = min_eff * 100
            Q_High_eff = eff_Q_high
            Q_Low_eff = eff_Q_low
            
     
        # Objective function: sum of revenues
        
        OF =  np.sum(Revenue)
          
        # Return simulation details if in simulation mode.
        if mode == 'simulation':
            OFsim =  np.sum(Sim_revenue, axis=0)
            return OFsim,  DailyPowers_all, Xpars, Replacement_year, Sim_eff, Sim_head, Sim_Q, Sim_head_all, \
                   Qdesign, IC_added, DailyPowers_peak,
        else:
            return float(-OF)

    
    def energy_sim(self, nperiod, target,  pchange, Xpars, Q, h_el, Xorig):
         
       Effcurve = self.Effcurve 
       efficiency_vector =  self.efficiency_vector 
       
       e_price = self.e_price 
       #discount_factors = self.discount_factors
       # CRF = self.CRF
       i_rate  = self.i_rate 
       nt_change = self.nt_change
       
       #unpack original setup parameters
       orig_head = Xorig[:, 0]
       orig_discharge = Xorig[:, 1]

       
       #unpack modified parameters
       X_head = Xpars[:, 0]
       X_discharge = Xpars[:, 1]
       X_age = Xpars[:, 2]
       
       flow_start = (4 + nperiod *5)*365
       flow_end = flow_start + 5*365
       
       year_start = nperiod *5
       year_end = nperiod *5 + 5
       # if nperiod== 12: 
       #          flow_end = flow_start + 6*365
                
       hf_idx = np.arange(flow_start,flow_end, dtype=int)

       partial_flows = Q[hf_idx] # flow for this period
          
       partial_h_el  = h_el[hf_idx] # elevations for this period
          
       input_flows = partial_flows.reshape(1, -1)
       input_flows = input_flows.T
  
       Rev_D_discharge = X_discharge  #Rev_D_discharge, Rev_D_head, Rev_D_age = (np.zeros((17, 1)) for _ in range(3))
       Rev_D_head = X_head
       Rev_D_age = X_age + 5 # Initialize the array with the default value   
       
       orig_pchange = pchange # Store the original pchange value for later checking
       
       
       if target[0] == 0: # no replacement
       
          added_IC = 0
          orig_pchange = pchange # Store the original pchange value for later checking
          Costper = 0.00000001
          
           # Adjust pchange if it is 4 or greater using a 3-unit decrement rule
          if pchange >= 4:
              pchange -= 3 * ((pchange - 1) // 3)       
          
       else: # nperiod >= 1    
          
          pchange += 1
          
          orig_pchange = pchange # Store the original pchange value for later checking
          
           # For pchange values >  4 and  subtract 3 times the integer quotient of (pchange - 1) divided by 3 
           #to adjust its value in 3-unit increments.
          if pchange >= 4:
              pchange -= 3 * ((pchange - 1) // 3)     
    
          s_idx = nt_change * (pchange - 1) - max(0, pchange - 2)
          e_idx = s_idx + nt_change
       
          if pchange in {3, 6, 9, 12}:
             s_idx = 12
             nt_change = 5
             e_idx = s_idx + nt_change
 
          Rev_D_age[s_idx:e_idx] =  5 # Assign the specific value to the desired interval    
           
         ############ Assign parameters, update the setup 
          
          if target[0] == 1: # existing replacement

             new_Q = np.sum(X_discharge[s_idx:e_idx])
             new_head = X_head[s_idx] 
             added_IC = new_head*new_Q* 9.81 
            
          elif target[0] == 2: # diversified replacement
       
             delta_H = target[1]/100
             delta_Q = target[2]/100
           
             new_head = orig_head[s_idx]*(1 + delta_H)
              
             Rev_D_head[s_idx:e_idx] = new_head # Rev_D_head[s_idx:e_idx] = X_head[s_idx:e_idx] + delta_H
             
             Rev_D_discharge[s_idx:e_idx] = orig_discharge[s_idx:e_idx]*(1 + delta_Q)

             new_Q = np.sum( Rev_D_discharge[s_idx:e_idx])
           
             added_IC = new_head*new_Q* 9.81 
             
          #print(new_head)   
        # RUN ######
          Costper = 1.23 * 1.4762 * np.exp(11.4931 + 0.4161 * np.log(added_IC / 1000) - 0.3752 * np.log(new_head * 3.281))/1000
          
       Xnext = (np.concatenate(([Rev_D_head],[Rev_D_discharge],  [Rev_D_age]))).T # data for the next step
          
       Q_design = np.sum(Rev_D_discharge) # Combine initial and updated capacities
  
          # Get unique elements and their indices
       unique_elements, uidx = np.unique(Rev_D_age, return_inverse=True)
          
       nk = len(unique_elements)      
       
       
       outage = np.zeros((len(hf_idx), nk))
       for i in range(nk):
          start = (unique_elements[i] -5 )* 365
          stop  = unique_elements[i] * 365
          datax = np.arange(start, stop)
          outage[:,i] = np.round(datax / 365)*0.00333
          
          
       # Compute the sum of each group
       new_Q_capacity = np.bincount(uidx, weights=Rev_D_discharge)

       #unique_ordered = Rev_D_age[np.sort(np.unique(Rev_D_age, return_index=True)[1])]
          
       # Define fixed flows for peak, off-peak, and mid periods     
       periods_flows = Period_Flows(input_flows, Q_design)
          
       # Calculate the number of turbines operating, ensuring values are between 1 and 17
       num_turbine_op = np.clip(np.round(periods_flows / 75), 1, 17).astype(int)
       #print(num_turbine_op)
       # Calculate head loss using the 'calculate_hydraulics' function
       hloss = calculate_hydraulics(periods_flows, num_turbine_op)

       # Calculate net head by subtracting head loss from elevation head
       h_el_replicated = np.tile(partial_h_el[:, np.newaxis], (1,  3))
       hnet = h_el_replicated - hloss
  
       indices = [np.where(Rev_D_age == element)[0][0] for element in unique_elements]
       
       all_heads = Rev_D_head[indices]
      # print(all_heads)

       # Combine design heads into a single array
       H_designs = np.tile(all_heads, 3)
          
       high_head = max(H_designs)
       low_head  = min(H_designs)
       
       
       replacement_times = unique_elements  - 5 # to get initital state
    
       day_years = len(hnet)
       
        # Calculate Eff indices for periods
       start_idx = np.concatenate(np.array([replacement_times * 365]))
       end_idx = start_idx + day_years # Add the increment to each element of start_idx
                        
       idx = np.column_stack((start_idx, end_idx))  # Equivalent to [start_idx; end_idx]'  
       
       #Efficiency
       xeff_adjusted, max_min_eff = efficiency_periods(hnet, nk, idx, H_designs, efficiency_vector, Effcurve)
       #nk = len(unique_elements)  
        
       eff_adjusted = pairwise_replace(xeff_adjusted)
       
       
       AvEff = np.sum(eff_adjusted[:,0:nk] * new_Q_capacity) / Q_design/day_years  
         # print(AvEff)  
          
       high_eff_Q = new_Q_capacity[0]
       low_eff_Q  = new_Q_capacity[-1]
          
       max_eff   = np.round(0.92*max(max_min_eff), 4)
       min_eff   = np.round(0.92*min(max_min_eff), 4)
       
#####################################       # Initialize arrays
       DailyPowers = np.zeros(len(hnet))
       DailyRevenue = np.zeros(len(hnet))
       
       DailyPowers_peak = np.zeros(len(hnet))
       
        # --- Main simulation ---
       timetowait = 1  # years
       n_days = timetowait * 365
       
       n_days = 2
       
       if target[0]  > 0:
         # Index ranges
         idx1 = np.arange(0, n_days)                 # first 3 years
         idx2 = np.arange(n_days, len(hnet))         # rest of the horizon
         outagex1 =  outage[0:n_days,:]
         outagex2 =  outage[n_days:,:]

          # Split dataset
         nfirst_flows = periods_flows[:n_days, :]  # first 3 years
         nrest_flows    = periods_flows[n_days:, :]  # remaining years
         
         # --- First period (skip first setup) ---
         flow_peak, flow_mid, flow_off = ( nfirst_flows[:, 0], nfirst_flows[:, 1], nfirst_flows[:, 2]) 
         
         n_cols = eff_adjusted.shape[1]
         meff_adjusted = eff_adjusted.copy()
         if n_cols > 3:
             meff_adjusted[:, 0] = meff_adjusted[:, 1]  # col1 = col2
             meff_adjusted[:, 2] = meff_adjusted[:, 3]  # col3 = col4
             meff_adjusted[:, 4] = meff_adjusted[:, 5]  # col5 = col6
     

         for i in range(nk):  
       #  for i in range(nk): 
            
             flow_peak, flow_mid, flow_off,DailyPowers, DailyRevenue, DailyPowers_peak = compute_power_and_revenue(
                 flow_peak, flow_mid, flow_off, new_Q_capacity, meff_adjusted, hnet,
                 e_price, outagex1, idx1, nk,
                 DailyPowers, DailyRevenue, DailyPowers_peak, i  )


          # --- Second period (remaining years) ---       
         flow_peak, flow_mid, flow_off = ( nrest_flows[:, 0], nrest_flows[:, 1], nrest_flows[:, 2]) 
         
         for i in range(nk):
            flow_peak, flow_mid, flow_off, DailyPowers, DailyRevenue, DailyPowers_peak = compute_power_and_revenue(
               flow_peak, flow_mid, flow_off, 
               new_Q_capacity, eff_adjusted, hnet,
               e_price, outagex2, idx2, nk,
               DailyPowers, DailyRevenue,DailyPowers_peak, i )

       else:
         # --- No change: whole period ---
         idx_all = np.arange(len(hnet))
         
         flow_peak, flow_mid, flow_off = ( periods_flows[:, 0], periods_flows[:, 1], periods_flows[:, 2])          
         for i in range(nk):
             flow_peak, flow_mid, flow_off, DailyPowers, DailyRevenue, DailyPowers_peak = compute_power_and_revenue(
               flow_peak, flow_mid, flow_off, 
               new_Q_capacity, eff_adjusted,hnet,
               e_price, outage, idx_all, nk,
               DailyPowers, DailyRevenue, DailyPowers_peak, i )
    
 #########################################################     
       # Constraint 1: Apply penalty if average efficiency is below 0.80
       if (high_head < 135) or (low_head < 133) or (np.max(unique_elements) >75.1):
       #if (AvEff < 0.75) or (high_head < 135) or (low_head < 135) or (np.max(unique_elements) >44):
              Costper *= 1e9
          
       # if orig_pchange >4.1 : Costper *= (orig_pchange)**5
       #  #print(f"pchange: {pchange}, orig_pchange: {orig_pchange}, nperiod: {nperiod}")

       Cost = Costper #np.sum(Costper * discount_factors)
       
       discounted_revenue = term_NPV(DailyRevenue, i_rate, year_start, year_end, Cost )  
      # print(discounted_revenue, Cost)
         #AAE = np.mean(DailyPowers) * 365 / 1e6
       
       #return discounted_revenue, AvEff, Xnext, high_head, low_head, max_eff, min_eff, high_eff_Q, low_eff_Q, orig_pchange
       return discounted_revenue, DailyPowers, Cost, AvEff, Xnext, all_heads,Q_design, high_head, low_head, \
               max_eff, min_eff, high_eff_Q, low_eff_Q, orig_pchange, added_IC, DailyPowers_peak

      # return discounted_revenue, DailyPowers, Cost, AvEff, Xnext, all_heads,Q_design, high_head, low_head, max_eff, orig_pchange, added_IC


# --- Helper function ---
def compute_power_and_revenue(flow_peak, flow_mid, flow_off, new_Q_capacity, eff_adjusted,
                              hnet, e_price, outage, idx, nk,
                              DailyPowers, DailyRevenue,DailyPowers_peak, i):
    """
    Compute power and revenue for turbine i, update flows and totals.
    """
  
    hnet_peak = hnet[:, 0]
    hnet_mid = hnet[:, 1]
    hnet_off = hnet[:, 2]
           
    # Turbine-specific flows (capped at available flow)
    newT_peak = np.minimum(new_Q_capacity[i], flow_peak)
    newT_mid  = np.minimum(new_Q_capacity[i], flow_mid)
    newT_off  = np.minimum(new_Q_capacity[i], flow_off)

    # Efficiency values
    eff_peak = eff_adjusted[:, i]
    eff_mid  = eff_adjusted[:, i + nk]
    eff_off  = eff_adjusted[:, i + 2 * nk]
    
    # Update power
    DailyPowers[idx] += (
        newT_peak * eff_peak[idx] * hnet_peak[idx] * 6 +
        newT_mid  * eff_mid[idx]  * hnet_mid[idx]  * 12 +
        newT_off  * eff_off[idx]  * hnet_off[idx]  * 6
    ) * 9.6138 * (1 - outage[:,i])

    # Update revenue
    DailyRevenue[idx] += (
        newT_peak * eff_peak[idx] * hnet_peak[idx] * 6 * e_price[0] +
        newT_mid  * eff_mid[idx]  * hnet_mid[idx]  * 12 * e_price[1] +
        newT_off  * eff_off[idx]  * hnet_off[idx]  * 6 * e_price[2]
    ) * 9.6138 * (1 - outage[:,i])
    
    DailyPowers_peak[idx] += (newT_peak * eff_peak[idx] * hnet_peak[idx] * 6) * 9.6138 * (1 - outage[:,i])
    # Update remaining flows
    flow_peak = np.maximum(0, flow_peak - newT_peak)
    flow_mid  = np.maximum(0, flow_mid  - newT_mid)
    flow_off  = np.maximum(0, flow_off  - newT_off)

    return flow_peak, flow_mid, flow_off, DailyPowers, DailyRevenue, DailyPowers_peak
    

