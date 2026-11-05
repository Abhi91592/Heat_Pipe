# ============================================================================
# LITHIUM THERMOPHYSICAL PROPERTIES: LIQUID + VAPOR
# ============================================================================
# All temperatures are entered in kelvin [K].
# All pressures are entered in pascals [Pa].
# Returned properties use SI units.
# PRIMARY REFERENCES
# ------------------
# [1] H. W. Davison,
#     "Compilation of Thermophysical Properties of Liquid Lithium,"
#     NASA TN D-4650, 1968.
#     https://ntrs.nasa.gov/api/citations/19680018893/downloads/19680018893.pdf
#
# [2] D. W. Jeppson, J. L. Ballif, W. W. Yuan, and B. E. Chou,
#     "Lithium Literature Review: Lithium's Properties and Interactions,"
#     HEDL-TME 78-15, Hanford Engineering Development Laboratory, 1978.
#     https://www.osti.gov/servlets/purl/6885395/.pdf
#
# [3] M. W. Chase, Jr.,
#     "NIST-JANAF Thermochemical Tables, Fourth Edition,"
#     J. Phys. Chem. Ref. Data, Monograph 9, 1998.
#     NIST Chemistry WebBook, SRD 69, lithium gas and condensed-phase data.
#     https://webbook.nist.gov/cgi/cbook.cgi?ID=C7439932&Mask=1
#     https://webbook.nist.gov/cgi/cbook.cgi?ID=C7439932&Mask=2
#
# [4] M. Bouledroua,
#     "Second Virial Coefficients and Viscosity Property of Monatomic
#     Alkali-Metal Gases," J. Plasma Fusion Res. Series, Vol. 7,
#     pp. 217-219, 2006.
#     https://www.jspf.or.jp/JPFRS/PDF/Vol7/jpfr2006_07-217.pdf
#
# [5] Standard ideal-gas and transport-property definitions:
#     rho = P/(R*T), Cv = Cp - R, gamma = Cp/Cv,
#     a = sqrt(gamma*R*T), nu = mu/rho,
#     Pr = Cp*mu/k, alpha = k/(rho*Cp).
#
# PROPERTY SELECTION
# ------------------
# Liquid density              : HEDL Eq. (13), converted to K and kg/m^3 [2]
# Surface tension             : NASA Eq. (6) [1]
# Liquid thermal conductivity : NASA polynomial approximation to Eq. (8) [1]
# Saturation vapor pressure   : NASA Eq. (9) [1]
# Liquid dynamic viscosity    : NASA Eq. (10) [1]
# Vapor viscosity             : Bouledroua monatomic-gas power laws [4]
# Vapor/liquid thermochemistry: NIST-JANAF Shomate data [3]
# ============================================================================

import warnings
import numpy as np
# ============================================================================
# REFERENCE CONSTANTS
# ============================================================================

R_universal = 8.31446261815324       # Universal gas constant [J/(mol*K)]
M_lithium = 6.941e-3                 # Lithium molar mass [kg/mol] [3]

R_lithium_vapor = R_universal / M_lithium
# 1197.8768 J/(kg*K), calculated from R_universal/M_lithium

T_melting = 453.69                   # Melting temperature [K] [3]
T_boiling_normal = 1608.0            # Normal boiling temperature [K] [1]
P_atmospheric = 1.013e5              # Atmospheric pressure used by NASA [Pa]

latent_heat_fusion = 4.55e5          # Latent heat of fusion [J/kg] [1]


# ============================================================================
# INPUT VALIDATION
# ============================================================================
def check_temperature(
    T,
    Tmin=None,
    Tmax=None,
    property_name="property"
):
    T = np.asarray(T, dtype=float)

    if np.any(~np.isfinite(T)):
        raise ValueError("Temperature must contain only finite values.")

    if np.any(T <= 0.0):
        raise ValueError("Temperature must be greater than 0 K.")

    outside = np.zeros_like(T, dtype=bool)

    if Tmin is not None:
        outside |= T < Tmin

    if Tmax is not None:
        outside |= T > Tmax

    if np.any(outside):
        lower = "-infinity" if Tmin is None else f"{Tmin:.2f}"
        upper = "+infinity" if Tmax is None else f"{Tmax:.2f}"

        warnings.warn(
            f"{property_name}: temperature is outside the recommended "
            f"range {lower} K to {upper} K; extrapolation is being used.",
            RuntimeWarning,
            stacklevel=2,
        )

    return T


def check_pressure(P, property_name="pressure"):
    P = np.asarray(P, dtype=float)

    if np.any(~np.isfinite(P)):
        raise ValueError(
            f"{property_name} must contain only finite values."
        )

    if np.any(P <= 0.0):
        raise ValueError(
            f"{property_name} must be greater than 0 Pa."
        )

    return P


# ============================================================================
# NIST-JANAF SHOMATE COEFFICIENTS [3]
# ============================================================================

LI_LIQUID_SHOMATE_LOW = {
    "A": 32.46972,
    "B": -2.635975,
    "C": -6.327128,
    "D": 4.230359,
    "E": 0.005686,
    "F": -7.117319,
    "G": 74.29361,
    "H": 2.380002,
}
# Valid range: 453.69 K to 700 K [3]


LI_LIQUID_SHOMATE_HIGH = {
    "A": 26.00896,
    "B": 5.632375,
    "C": -4.013227,
    "D": 0.873686,
    "E": 0.344150,
    "F": -4.199690,
    "G": 66.36284,
    "H": 2.380002,
}
# Valid range: 700 K to 1620.12 K [3]


LI_GAS_SHOMATE = {
    "A": 23.33408,
    "B": -2.772423,
    "C": 0.767421,
    "D": -0.003595,
    "E": -0.035246,
    "F": 151.5035,
    "G": 166.1885,
    "H": 159.3004,
}
# Valid range: 1620.12 K to 6000 K [3]


def _shomate_cp_molar(T, coefficients):
    T = np.asarray(T, dtype=float)
    t = T / 1000.0

    return (
        coefficients["A"]
        + coefficients["B"] * t
        + coefficients["C"] * t**2
        + coefficients["D"] * t**3
        + coefficients["E"] / t**2
    )


def _shomate_standard_enthalpy_molar(T, coefficients):
    T = np.asarray(T, dtype=float)
    t = T / 1000.0

    enthalpy_increment = (
        coefficients["A"] * t
        + coefficients["B"] * t**2 / 2.0
        + coefficients["C"] * t**3 / 3.0
        + coefficients["D"] * t**4 / 4.0
        - coefficients["E"] / t
        + coefficients["F"]
        - coefficients["H"]
    )

    return coefficients["H"] + enthalpy_increment


# ============================================================================
# LIQUID-LITHIUM PROPERTIES
# ============================================================================
# ----------------------------------------------------------------------------
# 1. LIQUID DENSITY [2]
# HEDL Eq. (13), original form:
# ----------------------------------------------------------------------------
def liquid_density(T):
    T = check_temperature(
        T,
        Tmin=473.15,
        Tmax=1873.15,
        property_name="liquid density",
    )
    rho_l = 562.78815 - 0.101 * T
    if np.any(rho_l <= 0.0):
        warnings.warn(
            "liquid density became non-positive; the HEDL correlation is "
            "being extrapolated far outside its usable range.",
            RuntimeWarning,
            stacklevel=2,
        )
    return rho_l  # kg/m^3

# ----------------------------------------------------------------------------
# 5. SURFACE TENSION [1]
# NASA Eq. (6):
# ----------------------------------------------------------------------------
def surface_tension(T):
    T = check_temperature(
        T,
        Tmin=460.0,
        Tmax=1300.0,
        property_name="surface tension",
    )
    sigma = 0.447 - 1.07e-4 * T - 1.351e-8 * T**2
    if np.any(sigma <= 0.0):
        warnings.warn(
            "surface tension became non-positive; the NASA correlation is "
            "being extrapolated beyond a physically usable range.",
            RuntimeWarning,
            stacklevel=2,
        )
    return sigma  # N/m

# ----------------------------------------------------------------------------
# 6. LIQUID THERMAL CONDUCTIVITY [1]
# NASA polynomial approximation to the modified-Ewing Eq. (8):
# ----------------------------------------------------------------------------
def liquid_thermal_conductivity(T):
    T = check_temperature(
        T,
        Tmin=T_melting,
        Tmax=1600.0,
        property_name="liquid thermal conductivity",
    )
    return (
        21.874
        + 0.056255 * T
        - 1.8325e-5 * T**2
    )


# ----------------------------------------------------------------------------
# 7. LIQUID DYNAMIC VISCOSITY [1]
# NASA Eq. (10):
# ---------------------------------------------------------------------------
def liquid_dynamic_viscosity(T):
    T = check_temperature(
        T,
        Tmin=T_melting,
        Tmax=1600.0,
        property_name="liquid dynamic viscosity",
    )
    log10_mu_l = (
        -3.080
        + 57.63 / T
        - 5.172e-4 * T
    )
    return 10.0**log10_mu_l  # Pa*s


# ----------------------------------------------------------------------------
# 12. NIST LIQUID SPECIFIC ENTHALPY [3]
# Standard enthalpy on the NIST reference basis.
# Used with vapor enthalpy to calculate h_fg.
# ----------------------------------------------------------------------------
def liquid_specific_enthalpy_nist(T):
    T = check_temperature(
        T,
        Tmin=453.69,
        Tmax=1620.12,
        property_name="NIST liquid specific enthalpy",
    )
    h_low = _shomate_standard_enthalpy_molar(
        T,
        LI_LIQUID_SHOMATE_LOW,
    )
    h_high = _shomate_standard_enthalpy_molar(
        T,
        LI_LIQUID_SHOMATE_HIGH,
    )
    h_molar = np.where(
        T < 700.0,
        h_low,
        h_high
    )
    return h_molar * 1000.0 / M_lithium  # J/kg


# ============================================================================
# SATURATION / PHASE-EQUILIBRIUM PROPERTIES
# ============================================================================
# ----------------------------------------------------------------------------
# 13. SATURATION VAPOR PRESSURE [1]
# NASA Eq. (9):
# log10(P_sat) = 10.015 - 8064.5/T
# P_sat is in Pa and T is in K.
# Experimental data range used by NASA: approximately 800 K to 1800 K
# NASA standard deviation: approximately 3.38%
# NASA maximum deviation: -32.6% at a very low pressure near 6 Pa
# ----------------------------------------------------------------------------
def vapor_pressure(T):
    T = check_temperature(
        T,
        Tmin=800.0,
        Tmax=1800.0,
        property_name="saturation vapor pressure",
    )
    log10_P_sat = 10.015 - 8064.5 / T
    return 10.0**log10_P_sat  # Pa

# ============================================================================
# MONATOMIC LITHIUM-VAPOR PROPERTIES
# ============================================================================
# ----------------------------------------------------------------------------
# 16. VAPOR SPECIFIC GAS CONSTANT [3,5, calculated]
# ----------------------------------------------------------------------------
def vapor_specific_gas_constant():
    return R_lithium_vapor


# ----------------------------------------------------------------------------
# 17. VAPOR SPECIFIC HEAT AT CONSTANT PRESSURE [3,5]
# From 1620.12 K to 6000 K:
#   NIST-JANAF Li(g) Shomate heat-capacity equation.
# ----------------------------------------------------------------------------
def vapor_specific_heat_cp(T):
    T = check_temperature(
        T,
        Tmin=100.0,
        Tmax=6000.0,
        property_name="vapor specific heat Cp",
    )
    cp_monatomic_molar = np.full_like(
        T,
        2.5 * R_universal,
        dtype=float,
    )
    cp_nist_molar = _shomate_cp_molar(
        T,
        LI_GAS_SHOMATE
    )
    cp_molar = np.where(
        T < 1620.12,
        cp_monatomic_molar,
        cp_nist_molar,
    )
    return cp_molar / M_lithium  # J/(kg*K)


# ----------------------------------------------------------------------------
# 18. VAPOR SPECIFIC HEAT AT CONSTANT VOLUME [3,5, calculated]
# ----------------------------------------------------------------------------
def vapor_specific_heat_cv(T):
    return (
        vapor_specific_heat_cp(T)
        - R_lithium_vapor
    )


# ----------------------------------------------------------------------------
# 19. VAPOR SPECIFIC-HEAT RATIO [3,5, calculated]
# gamma_v = Cp,v/Cv,v
# Dimensionless
# ----------------------------------------------------------------------------
def vapor_specific_heat_ratio(T):
    cp_v = vapor_specific_heat_cp(T)
    cv_v = vapor_specific_heat_cv(T)
    return cp_v / cv_v


# ----------------------------------------------------------------------------
# 20. MONATOMIC VAPOR SPECIFIC ENTHALPY [3,5]
# Below 1620.12 K:
#   h_v = DeltaHf_gas(298.15) + (5/2)*R_u*(T - 298.15)
# From 1620.12 K to 6000 K:
#   NIST-JANAF Li(g) Shomate enthalpy equation.
# NIST standard gas formation enthalpy:
# DeltaHf_gas(298.15) = 159.3004 kJ/mol.
# This is atomic Li(g), not an equilibrium Li/Li2 mixture.
# Units: J/kg
# ----------------------------------------------------------------------------
def vapor_specific_enthalpy(T):
    T = check_temperature(
        T,
        Tmin=100.0,
        Tmax=6000.0,
        property_name="vapor specific enthalpy",
    )
    h_monatomic_molar = (
        LI_GAS_SHOMATE["H"]
        + 2.5
        * R_universal
        * (T - 298.15)
        / 1000.0
    )
    h_nist_molar = _shomate_standard_enthalpy_molar(
        T,
        LI_GAS_SHOMATE,
    )
    h_molar = np.where(
        T < 1620.12,
        h_monatomic_molar,
        h_nist_molar,
    )
    return h_molar * 1000.0 / M_lithium  # J/kg


# ----------------------------------------------------------------------------
# 21. LATENT HEAT OF VAPORIZATION [3, calculated]
# h_fg(T) = h_v(T) - h_l(T)
# h_v uses atomic Li(g) thermochemistry.
# h_l uses NIST liquid-lithium Shomate thermochemistry.
# Directly supported overlap: 453.69 K to 1620.12 K.
# Above 1620.12 K, the liquid Shomate correlation is extrapolated.
# Units: J/kg
# ----------------------------------------------------------------------------

def latent_heat_vaporization(T):
    T = check_temperature(
        T,
        Tmin=453.69,
        Tmax=1620.12,
        property_name="latent heat of vaporization",
    )
    h_v = vapor_specific_enthalpy(T)
    h_l = liquid_specific_enthalpy_nist(T)
    return h_v - h_l  # J/kg


# ----------------------------------------------------------------------------
# 22. SATURATED VAPOR DENSITY [1,3,5, calculated]
# Monatomic ideal-gas approximation:
# rho_v = P/(R_v*T)
# If P is omitted, NASA saturation pressure P_sat(T) is used.
# This is not an experimental vapor-density correlation and neglects Li2.
# Units: kg/m^3
# ----------------------------------------------------------------------------

def vapor_density(T, P=None):
    T = check_temperature(
        T,
        Tmin=800.0,
        Tmax=2200.0,
        property_name="vapor density",
    )
    if P is None:
        P_used = vapor_pressure(T)
    else:
        P_used = check_pressure(
            P,
            property_name="vapor pressure"
        )
    return (
        P_used
        / (R_lithium_vapor * T)
    )  # kg/m^3


# ----------------------------------------------------------------------------
# 24. VAPOR DYNAMIC VISCOSITY [4]
# Bouledroua monatomic-lithium-gas fit:
# eta/(1e-7 Pa*s) = 0.314*T^0.863
# SI form:
# mu_v = 0.314e-7*T^0.863
# Valid theoretical fitting range: 100 K to 3000 K
# No formal uncertainty was reported for the power-law fit. The paper states
# that its lithium viscosity values differ from one comparison source by no
# more than approximately 5.4%.
# Units: Pa*s
# ----------------------------------------------------------------------------
def vapor_dynamic_viscosity(T):
    T = check_temperature(
        T,
        Tmin=100.0,
        Tmax=3000.0,
        property_name="vapor dynamic viscosity",
    )
    return 0.314e-7 * T**0.863  # Pa*s