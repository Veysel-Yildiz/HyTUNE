
import numpy as np

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
    kminor1 = 0.75  # Minor loss coefficient for Pipe 1
    kminor2 = 0.5  # Minor loss coefficient for Pipe 2
    
    # Constants
    ve = 1.004e-6  # Kinematic viscosity of water (m²/s)
    e_roughness = 0.45e-4  # Pipe roughness (m)

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
    hloss = ((f_d1 * L1 / D1p * V_d1**2) + kminor1 * V_d1**2) / 19.62 + \
            ((f_d2 * L2 / D2p * V_d2**2) + kminor2 * V_d2**2) / 19.62

    return hloss


################################################## MOODY ########################


def moody(ed, Re):
    """Return f, friction factor
    --------------------------------------
      Inputs:

        HP : structure with variables used in calculation
        ed : the relative roughness: epsilon / diameter.
        Re : the Reynolds number

    """
    f = np.zeros_like(Re)

    # Find the indices for Laminar, Transitional and Turbulent flow regimes

    LamR = np.where((0 < Re) & (Re < 2000))
    LamT = np.where(Re > 4000)
    LamTrans = np.where((2000 < Re) & (Re < 4000))

    f[LamR] = 64 / Re[LamR]

    # Calculate friction factor for Turbulent flow using the Colebrook-White approximation
    f[LamT] = 1.325 / (np.log(ed / 3.7 + 5.74 / (Re[LamT] ** 0.9)) ** 2)

    Y3 = -0.86859 * np.log(ed / 3.7 + 5.74 / (4000**0.9))
    Y2 = ed / 3.7 + 5.74 / (Re[LamTrans] ** 0.9)
    FA = Y3 ** (-2)
    FB = FA * (2 - 0.00514215 / (Y2 * Y3))
    R = Re[LamTrans] / 2000
    X1 = 7 * FA - FB
    X2 = 0.128 - 17 * FA + 2.5 * FB
    X3 = -0.128 + 13 * FA - 2 * FB
    X4 = R * (0.032 - 3 * FA + 0.5 * FB)
    f[LamTrans] = X1 + R * (X2 + R * (X3 + X4))

    return f
