# ============================================================================
# tungsten_properties.py
# Tungsten thermal properties (thermal conductivity, specific heat, density)
# Source: MOOSE Framework TungstenThermalPropertiesMaterial, based on
# Milner, J. L., Karkos, P., & Bowers, J. J. (2024). Space Nuclear
# Propulsion (SNP) Material Property Handbook (SNP-HDBK-0008). NASA.
# https://ntrs.nasa.gov/citations/20240004217
# ============================================================================
import numpy as np


def check_temperature(T, Tmin, Tmax, property_name="property"):
    """
    Validate temperature is within the correlation's supported range.
    Returns T as a numpy array for vectorized downstream use.
    """
    T = np.asarray(T, dtype=float)
    if np.any(T < Tmin) or np.any(T > Tmax):
        raise ValueError(
            f"Temperature out of range for {property_name}: "
            f"valid range is {Tmin} K to {Tmax} K."
        )
    return T


# ----------------------------------------------------------------------------
# THERMAL CONDUCTIVITY [MOOSE TungstenThermalPropertiesMaterial]
# Valid range: 1 K to 3653 K (two-piece correlation)
# Units: W/(m*K)
# ----------------------------------------------------------------------------
def thermal_conductivity(T):
    T = check_temperature(T, Tmin=1.0, Tmax=3653.0, property_name="thermal conductivity")
    t = T / 1000.0

    # Low-T branch: 1 <= T < 55 K
    k_low = (7.348e5 * t**0.874) / (
        1 + 25.44 * t - 8.304e3 * t**2 + 1.18e6 * t**3
    )

    # High-T branch: 55 <= T <= 3653 K
    k_high = (-3.679 + 1.181e2 * t + 58.79 * t**2 + 2.867 * t**3) / (
        -2.052e-2 + 4.741e-1 * t + t**2
    )

    return np.where(T < 55.0, k_low, k_high)


# ----------------------------------------------------------------------------
# SPECIFIC HEAT [MOOSE TungstenThermalPropertiesMaterial]
# Valid range: 11 K to 3700 K (two-piece correlation)
# Units: J/(g*K)
# ----------------------------------------------------------------------------
def specific_heat(T):
    T = check_temperature(T, Tmin=11.0, Tmax=3700.0, property_name="specific heat")
    t = T / 1000.0

    # Low-T branch: 11 <= T < 293 K
    cp_low = (3.103e2 * t**3.03) / (
        1 - 8.815 * t + 1.295e2 * t**2 + 1.874e3 * t**3
    )

    # High-T branch: 293 <= T < 3700 K
    cp_high = (
        1.301e-1
        + 2.225e-2 * t
        - 7.224e-3 * t**2
        + 3.539e-3 * t**3
        - 3.061e-4 / t**2
    )

    return np.where(T < 293.0, cp_low, cp_high)


# ----------------------------------------------------------------------------
# DENSITY [MOOSE TungstenThermalPropertiesMaterial]
# Valid range: 5 K to 3600 K (via thermal expansion, two-piece correlation)
# Units: kg/m^3
# ----------------------------------------------------------------------------
def density(T):
    T = check_temperature(T, Tmin=5.0, Tmax=3600.0, property_name="density")
    t = T / 1000.0
    rho_0 = 19250.0  # room-temperature density [kg/m^3]

    # Low-T branch: 5 <= T < 294 K
    dL_L0_low = -8.529e-2 - 9.915e-2 * t + 2.257 * t**2 - 3.157 * t**3

    # High-T branch: 294 <= T <= 3600 K
    dL_L0_high = -1.4e-1 + 4.869e-1 * t - 3.056e-2 * t**2 + 2.234e-2 * t**3

    dL_L0 = np.where(T < 294.0, dL_L0_low, dL_L0_high)  # [%]

    return rho_0 / (1 + dL_L0 / 100.0) ** 3



# ============================================================================
#W-Re Properties
# ============================================================================

