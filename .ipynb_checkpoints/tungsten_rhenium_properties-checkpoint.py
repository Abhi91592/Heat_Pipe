# ============================================================================
# tungsten_rhenium_properties.py
# Tungsten-Rhenium (W-Re) alloy thermal properties
# (thermal conductivity, specific heat, density)
#
# Same interface and units as tungsten_properties.py, so the two modules are
# drop-in interchangeable:
#     thermal_conductivity(T) -> W/(m*K)
#     specific_heat(T)        -> J/(g*K)     (table is J/(kg*K); converted)
#     density(T)              -> kg/m^3
#
# Unlike the pure-W module (closed-form correlations), these properties come
# from TABULATED data and are evaluated by shape-preserving (PCHIP)
# interpolation between the tabulated points. No extrapolation is done:
# requests outside the tabulated range raise ValueError.
#
# Sources:
#   Specific heat (1500-3600 K):
#     Cezairliyan, A. & Miiller, A. P. (1985), Eq. Cp1,
#     DOI 10.1007/BF00500031
#   Density (293.15-873.15 K):
#     [1] Toenshoff, D. A., Zysk, E. D., & Fleischner, P. L. (1972).
#         High temperature thermocouple development program, Part A and
#         Part B. NASA CR-120898.   (293.15 K value reported directly)
#     [2] Aldana, S. L., & Tamashiro, R. N. (1991). High-efficiency 20 GHz
#         traveling wave tube development for space communications.
#         NASA CR-187160.           (other values derived via Eq. D1 of [2])
#   Thermal conductivity (373.15-1353.15 K):
#     Fukuda, M., Hasegawa, A. & Nogami, S. (2018). Thermal properties of
#     pure tungsten and its alloys for fusion applications. Fusion Eng. Des.
#     Fig. 5, DOI 10.1016/j.fusengdes.2018.04.117.
#     NOTE: digitized from a published figure -> approx. values (~ +/-2-3%).
# ============================================================================
import numpy as np
from scipy.interpolate import PchipInterpolator


def check_temperature(T, Tmin, Tmax, property_name="property"):
    """
    Validate temperature is within the data's supported range.
    Returns T as a numpy array for vectorized downstream use.
    """
    T = np.asarray(T, dtype=float)
    if np.any(T < Tmin) or np.any(T > Tmax):
        raise ValueError(
            f"Temperature out of range for W-Re {property_name}: "
            f"valid range is {Tmin} K to {Tmax} K."
        )
    return T


# ----------------------------------------------------------------------------
# TABULATED DATA
# ----------------------------------------------------------------------------
# Specific heat: T [K], cp [J/(kg*K)]  -- Cezairliyan & Miiller (1985)
_CP_T = np.array([1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300,
                  2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200,
                  3300, 3400, 3500, 3600], dtype=float)
_CP_VAL = np.array([159.5, 161.0, 163.1, 165.6, 168.5, 171.7, 175.3, 179.1,
                    183.2, 187.7, 192.4, 197.6, 203.2, 209.3, 216.0, 223.4,
                    231.7, 241.0, 251.4, 263.1, 276.3, 291.2])

# Density: T [K], rho [kg/m^3]  -- NASA CR-120898 + NASA CR-187160
_RHO_T = np.array([293.15, 298.15, 473.15, 673.15, 873.15])
_RHO_VAL = np.array([19400.0, 19399.0, 19353.0, 19299.0, 19242.0])

# Thermal conductivity: T [K], k [W/(m*K)]  -- Fukuda et al. (2018), Fig. 5
_K_T = np.array([373.15, 473.15, 573.15, 673.15, 773.15, 873.15, 973.15,
                 1073.15, 1173.15, 1273.15, 1353.15])
_K_VAL = np.array([111.7, 108.4, 106.3, 108.1, 108.1, 108.1, 107.8,
                   108.7, 107.8, 112.0, 115.2])

# PCHIP: passes through every data point, no overshoot between points
# (important for the noisy digitized k data).
_cp_interp = PchipInterpolator(_CP_T, _CP_VAL, extrapolate=False)
_rho_interp = PchipInterpolator(_RHO_T, _RHO_VAL, extrapolate=False)
_k_interp = PchipInterpolator(_K_T, _K_VAL, extrapolate=False)


# ----------------------------------------------------------------------------
# THERMAL CONDUCTIVITY [Fukuda, Hasegawa & Nogami 2018, digitized]
# Valid range: 373.15 K to 1353.15 K
# Units: W/(m*K)
# ----------------------------------------------------------------------------
def thermal_conductivity(T):
    T = check_temperature(T, Tmin=_K_T[0], Tmax=_K_T[-1],
                          property_name="thermal conductivity")
    return _k_interp(T)


# ----------------------------------------------------------------------------
# SPECIFIC HEAT [Cezairliyan & Miiller 1985]
# Valid range: 1500 K to 3600 K
# Units: J/(g*K)  (same as tungsten_properties.specific_heat)
# ----------------------------------------------------------------------------
def specific_heat(T):
    T = check_temperature(T, Tmin=_CP_T[0], Tmax=_CP_T[-1],
                          property_name="specific heat")
    return _cp_interp(T) / 1000.0  # J/(kg*K) -> J/(g*K)


# ----------------------------------------------------------------------------
# DENSITY [NASA CR-120898 + NASA CR-187160]
# Valid range: 293.15 K to 873.15 K
# Units: kg/m^3
# ----------------------------------------------------------------------------
def density(T):
    T = check_temperature(T, Tmin=_RHO_T[0], Tmax=_RHO_T[-1],
                          property_name="density")
    return _rho_interp(T)