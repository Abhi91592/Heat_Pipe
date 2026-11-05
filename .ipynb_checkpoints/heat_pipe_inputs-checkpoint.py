import numpy as np
# ============================================================================
# HEAT-PIPE OPERATING CONDITIONS
# ============================================================================

T_min = 1000.0       # Minimum operating temperature [K]
T_max = 1600.0      # Maximum operating temperature [K]
T_step = 10.0       # Temperature increment [K]

n_wick_layers = 7   # Number of wire-mesh wick layers
T_w = 0.0000381     # Thickness of a single mesh layer [m] (0.0015 in)

Q = 2000 # Heat added to the heat pipe 

def temperature_range():
    """
    Return the heat-pipe operating-temperature array.

    Returns
    -------
    numpy.ndarray
        Temperature range in kelvin.
    """
    return np.arange(T_min, T_max + T_step, T_step)

#-----------------------------------------
# Here we will declare all the geometry of the heat pipe and the wick sturcture 
#-----------------------------------------
do = 0.02          # Outer diameter of the heat pipe all length in meters
di = 0.016         # Inner diameter of the heat pipe 
Lt = 0.2           # total length
Le = 0.05          # evaporator length
La = 0.1           # adiabitc length
Lc = 0.05          # condesnor length

# ----------------------------------------------------------------------
# ORIENTATION: vertical heat pipe, evaporator at the BOTTOM, condenser
# at the TOP. Liquid returns downward from condenser to evaporator, i.e.
# gravity ASSISTS liquid return (gravity-aided orientation).
#
# psi is measured from the horizontal, per S. W. Chi's convention
# (Eq. 2-17 and Fig. 2-10). psi = 0 is horizontal; psi = pi/2 (90 deg) is
# fully vertical. A fully vertical, gravity-aided pipe is the case
# psi = +pi/2 in the sign convention adopted downstream (see the
# capillary-limit and liquid-pressure-drop kernels for how the sign of
# the rho_l*g*sin(psi) term is applied).
# ----------------------------------------------------------------------
psi = np.pi / 2.0   # heat pipe inclination/angle [rad]: 90 deg = vertical

g = 9.81           # gravity

S = 1.05           # crimping constant from literature do not chnage
N = 3937           # wires per meter
dw = 0.0000254     # wire diameter