# ============================================================================
# SILVER THERMOPHYSICAL PROPERTIES: LIQUID + VAPOR
# ============================================================================
# Same structure and function names as lithium_properties.py, so it can be
# imported in place of it:   import silver_properties as li   (or as ag)
#
# All temperatures are entered in kelvin [K].
# All pressures are entered in pascals [Pa].
# Returned properties use SI units.
#
# PRIMARY REFERENCES
# ------------------
# [1] CRC Handbook of Chemistry and Physics, "Density of Molten Elements and
#     Representative Salts": Ag, rho_m = 9.320 g/cm^3 at t_m = 961.78 C,
#     k = 0.0009 g/(cm^3*C), t_max = 1500 C.
#     https://moltensalt.org/references/static/downloads/pdf/element-salt-densities.pdf
#
# [2] "Density and thermal expansion of silver in the solid and liquid
#     states" (2020): rho = 9327.1 - 0.9956*(T - 1234.93) kg/m^3.
#     Used as a cross-check on [1].
#     https://www.researchgate.net/publication/347350917
#
# [3] "Surface tension and adsorption in liquid silver-oxygen alloys,"
#     Acta Metallurgica (1982), DOI 10.1016/0001-6160(82)90180-8.
#     Pure Ag: sigma = 890 mN/m at the melting point,
#     surface entropy (d sigma/dT) = -0.16 mN/(m*K).
#     https://www.sciencedirect.com/science/article/abs/pii/0001616082901808
#
# [4] Yaws Handbook of Vapor Pressure, Ag correlation as listed by Cheméo:
#     ln(P/kPa) = A + B/(T + C),  A = 16.6364, B = -28295.0, C = -81.22,
#     valid 1283 K to 2453.15 K.
#     https://www.chemeo.com/cid/18-740-0/silver.pdf
#     Checked here against the Wikipedia/CRC vapor-pressure table
#     (1 Pa @ 1283 K ... 100 kPa @ 2433 K): agreement better than 0.2 %.
#
# [5] ESPI Metals, Silver technical data: liquid specific heat
#     0.310 kJ/(kg*K) from 961 C to 2227 C.
#     https://www.espimetals.com/index.php/technical-data/194-Silver
#
# [6] Burcat & Ruscic, "Third Millennium Ideal Gas and Condensed Phase
#     Thermochemical Database," ANL-05/20 (2005, updates to 2023):
#     Ag(g): DeltaHf(298) = 284.9 kJ/mol, Cp(298) = 20.786 J/(mol*K)
#     (= 5/2 R, monatomic), S(298) = 172.998 J/(mol*K); M = 107.8682 g/mol.
#     https://respecth.elte.hu/burcat/hf2023.docx
#
# [7] Constants: T_melt = 1234.93 K (ITS-90 fixed point), T_boil = 2435 K,
#     DeltaH_fus = 11.28 kJ/mol, DeltaH_vap(T_b) ~ 254 kJ/mol.
#     https://en.wikipedia.org/wiki/Silver
#     MatWeb, Silver: heat of vaporization 2332 J/g (cross-check for h_fg).
#
# [8] Vibrating-finger viscometer data for liquid Ag, Int. J. Thermophys.
#     (2016), DOI 10.1007/s10765-016-2104-7, as listed by Cheméo:
#     3.520 mPa*s @ 1273 K, 2.904 @ 1323 K, 2.498 @ 1523 K, 2.214 @ 1673 K.
#
# [9] Standard ideal-gas / kinetic-theory relations:
#     rho = P/(R*T), Cv = Cp - R, gamma = Cp/Cv, a = sqrt(gamma*R*T),
#     Clausius-Clapeyron h_fg = R_v*T^2*dln(P)/dT (ideal vapor, v_l << v_g),
#     Chapman-Enskog rigid-sphere viscosity,
#     monatomic-gas Eucken relation k = (15/4)*(R_u/M)*mu,
#     Wiedemann-Franz law k = L0*T/rho_e, L0 = 2.44e-8 W*Ohm/K^2.
#
# PROPERTY SELECTION
# ------------------
# Liquid density              : CRC linear correlation [1], checked vs [2]
# Liquid specific heat        : constant 310 J/(kg*K) [5]
# Liquid enthalpy             : cp*(T - T_melt), relative to liquid at T_melt
# Surface tension             : linear, 890 mN/m at T_melt, -0.16 mN/(m*K) [3]
# Saturation vapor pressure   : Yaws correlation [4]
# Latent heat of vaporization : Clausius-Clapeyron from [4]; checked vs [7]
# Vapor Cp, Cv, gamma         : monatomic ideal gas, Cp = 5/2 R [6]
# Vapor enthalpy              : DeltaHf(298) + 5/2 R (T - 298.15) [6]
#
# !! PROVISIONAL PROPERTIES - REPLACE WHEN THE REFERENCE PAPERS ARE IN HAND !!
# Liquid dynamic viscosity    : Arrhenius fit (Assael form) to the 4 points
#                               of [8]; fit scatter ~ +/-7 %.
#                               -> Replace a1, a2 with the reference values
#                                  of Assael et al., High Temp.-High Press.
#                                  41, 161 (2012), valid 1235-1600 K.
# Liquid thermal conductivity : Wiedemann-Franz with an ESTIMATED liquid
#                               resistivity (not taken from a source found
#                               in this session).
#                               -> Replace RHO_E_MELT and DRHO_E_DT with
#                                  liquid-Ag resistivity data (e.g. Iida &
#                                  Guthrie, Thermophysical Properties of
#                                  Metallic Liquids, OUP 2015, Ch. 9; or the
#                                  TU Graz pulse-heating compilation).
# Vapor viscosity/conductivity: Chapman-Enskog RIGID-SPHERE estimate with
#                               sigma = 2 x van der Waals radius (1.72 A).
#                               Rigid spheres give mu ~ T^0.5; real metal
#                               vapors are closer to T^0.7-0.9.
#                               -> Replace with Biolsi & Holland, Int. J.
#                                  Thermophys. (2007),
#                                  DOI 10.1007/s10765-007-0217-8.
# Each provisional function issues a one-time RuntimeWarning when called.
# ============================================================================

import warnings
import numpy as np

# ============================================================================
# REFERENCE CONSTANTS
# ============================================================================
R_universal = 8.31446261815324       # Universal gas constant [J/(mol*K)]
M_silver = 107.8682e-3               # Silver molar mass [kg/mol] [6]

R_silver_vapor = R_universal / M_silver
# 77.0798 J/(kg*K), calculated from R_universal/M_silver

T_melting = 1234.93                  # Melting temperature [K] [7]
T_boiling_normal = 2435.0            # Normal boiling temperature [K] [7]
P_atmospheric = 1.01325e5            # Standard atmosphere [Pa]

latent_heat_fusion = 11.28e3 / M_silver   # [J/kg] [7], = 1.0457e5 J/kg

DELTA_HF_GAS_298 = 284.9e3           # Ag(g) formation enthalpy [J/mol] [6]

# Yaws vapor-pressure coefficients [4]  (P in kPa, T in K)
YAWS_A = 16.6364
YAWS_B = -28295.0
YAWS_C = -81.22
T_VP_MIN, T_VP_MAX = 1283.0, 2453.15

# ---- PROVISIONAL coefficients (see header) --------------------------------
# Liquid viscosity, Assael form: log10(mu / mPa*s) = a1 + a2/T
# Least-squares fit to the 4 data points of [8].
VISC_A1 = -0.23483
VISC_A2 = 962.770                    # [K]

# Liquid electrical resistivity (ESTIMATE, not from a retrieved source):
RHO_E_MELT = 17.2e-8                 # [Ohm*m] at T_melting
DRHO_E_DT = 0.0090e-8                # [Ohm*m/K]
LORENZ_L0 = 2.44e-8                  # Sommerfeld Lorenz number [W*Ohm/K^2]

# Vapor rigid-sphere collision diameter (2 x van der Waals radius 172 pm)
SIGMA_HS_ANGSTROM = 3.44             # [Angstrom]

_provisional_warned = set()


def _warn_provisional(name, message):
    if name not in _provisional_warned:
        _provisional_warned.add(name)
        warnings.warn(f"{name}: PROVISIONAL value. {message}",
                      RuntimeWarning, stacklevel=3)


# ============================================================================
# INPUT VALIDATION (identical behaviour to lithium_properties.py)
# ============================================================================
def check_temperature(T, Tmin=None, Tmax=None, property_name="property"):
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
        raise ValueError(f"{property_name} must contain only finite values.")
    if np.any(P <= 0.0):
        raise ValueError(f"{property_name} must be greater than 0 Pa.")
    return P


# ============================================================================
# LIQUID-SILVER PROPERTIES
# ============================================================================
# ----------------------------------------------------------------------------
# 1. LIQUID DENSITY [1]
# CRC: rho = rho_m - k*(t - t_m), rho_m = 9.320 g/cm^3, k = 0.0009 g/(cm^3*C)
# SI form: rho_l = 9320 - 0.9*(T - 1234.93)          [kg/m^3]
# Valid: T_melt to 1773.15 K (t_max = 1500 C)
# Cross-check [2]: 9327.1 - 0.9956*(T - 1234.93); differs < 0.6 % to 1773 K.
# ----------------------------------------------------------------------------
def liquid_density(T):
    T = check_temperature(T, Tmin=T_melting, Tmax=1773.15,
                          property_name="liquid density")
    rho_l = 9320.0 - 0.9 * (T - T_melting)
    return rho_l  # kg/m^3


# ----------------------------------------------------------------------------
# 2. LIQUID ELECTRICAL RESISTIVITY  [PROVISIONAL - see header]
# rho_e = RHO_E_MELT + DRHO_E_DT*(T - T_melt)
# Units: Ohm*m
# ----------------------------------------------------------------------------
def liquid_electrical_resistivity(T):
    T = check_temperature(T, Tmin=T_melting, Tmax=T_boiling_normal,
                          property_name="liquid electrical resistivity")
    _warn_provisional(
        "liquid_electrical_resistivity",
        "Resistivity coefficients are estimates; replace RHO_E_MELT and "
        "DRHO_E_DT with sourced liquid-Ag data.")
    return RHO_E_MELT + DRHO_E_DT * (T - T_melting)


# ----------------------------------------------------------------------------
# 3. LIQUID ENTHALPY RELATIVE TO LIQUID SILVER AT T_melt [5]
# h_l = cp_l*(T - T_melt)
# Units: J/kg
# ----------------------------------------------------------------------------
def liquid_enthalpy_relative_melting(T):
    T = check_temperature(T, Tmin=T_melting, Tmax=2500.15,
                          property_name="liquid enthalpy relative to T_melt")
    return 310.0 * (T - T_melting)  # J/kg


# ----------------------------------------------------------------------------
# 4. LIQUID SPECIFIC HEAT CAPACITY [5]
# 0.310 kJ/(kg*K), 961 C to 2227 C  (= 33.4 J/(mol*K))
# ----------------------------------------------------------------------------
def liquid_specific_heat(T):
    T = check_temperature(T, Tmin=T_melting, Tmax=2500.15,
                          property_name="liquid specific heat")
    return np.full_like(T, 310.0, dtype=float)  # J/(kg*K)


# ----------------------------------------------------------------------------
# 5. SURFACE TENSION [3]
# sigma = 0.890 - 0.16e-3*(T - T_melt)               [N/m]
# Measured for oxygen-free Ag; oxygen strongly LOWERS sigma.
# Recommended range: T_melt to 1573.15 K (range of the sessile-drop data
# used for cross-checking); beyond that the linear law is extrapolated.
# ----------------------------------------------------------------------------
def surface_tension(T):
    T = check_temperature(T, Tmin=T_melting, Tmax=1573.15,
                          property_name="surface tension")
    sigma = 0.890 - 0.16e-3 * (T - T_melting)
    if np.any(sigma <= 0.0):
        warnings.warn(
            "surface tension became non-positive; the linear correlation is "
            "being extrapolated beyond a physically usable range.",
            RuntimeWarning, stacklevel=2)
    return sigma  # N/m


# ----------------------------------------------------------------------------
# 6. LIQUID THERMAL CONDUCTIVITY  [PROVISIONAL - see header]
# Wiedemann-Franz: k_l = L0*T/rho_e
# Units: W/(m*K)
# ----------------------------------------------------------------------------
def liquid_thermal_conductivity(T):
    T = check_temperature(T, Tmin=T_melting, Tmax=T_boiling_normal,
                          property_name="liquid thermal conductivity")
    _warn_provisional(
        "liquid_thermal_conductivity",
        "Wiedemann-Franz estimate using an unsourced resistivity.")
    return LORENZ_L0 * T / liquid_electrical_resistivity(T)


# ----------------------------------------------------------------------------
# 7. LIQUID DYNAMIC VISCOSITY  [PROVISIONAL - see header]
# Assael form: log10(mu/(mPa*s)) = a1 + a2/T, fitted to the data of [8].
# Data range: 1273 K to 1673 K.
# Units: Pa*s
# ----------------------------------------------------------------------------
def liquid_dynamic_viscosity(T):
    T = check_temperature(T, Tmin=T_melting, Tmax=1673.0,
                          property_name="liquid dynamic viscosity")
    _warn_provisional(
        "liquid_dynamic_viscosity",
        "4-point Arrhenius fit (+/-7 %); replace a1, a2 with Assael et al. "
        "(2012) reference coefficients.")
    return 10.0 ** (VISC_A1 + VISC_A2 / T) * 1e-3  # Pa*s


# ----------------------------------------------------------------------------
# 8. LIQUID KINEMATIC VISCOSITY [calculated]
# ----------------------------------------------------------------------------
def liquid_kinematic_viscosity(T):
    return liquid_dynamic_viscosity(T) / liquid_density(T)


# ----------------------------------------------------------------------------
# 9. LIQUID PRANDTL NUMBER [calculated]
# Pr_l = cp_l*mu_l/k_l
# ----------------------------------------------------------------------------
def liquid_prandtl_number(T):
    return (liquid_specific_heat(T) * liquid_dynamic_viscosity(T)
            / liquid_thermal_conductivity(T))


# ----------------------------------------------------------------------------
# 10. LIQUID THERMAL DIFFUSIVITY [calculated]
# alpha_l = k_l/(rho_l*cp_l)        Units: m^2/s
# ----------------------------------------------------------------------------
def liquid_thermal_diffusivity(T):
    return (liquid_thermal_conductivity(T)
            / (liquid_density(T) * liquid_specific_heat(T)))


# ============================================================================
# SATURATION / PHASE-EQUILIBRIUM PROPERTIES
# ============================================================================
# ----------------------------------------------------------------------------
# 11. SATURATION VAPOR PRESSURE [4]
# ln(P_sat/kPa) = 16.6364 - 28295.0/(T - 81.22)
# Valid 1283 K to 2453.15 K
# ----------------------------------------------------------------------------
def vapor_pressure(T):
    T = check_temperature(T, Tmin=T_VP_MIN, Tmax=T_VP_MAX,
                          property_name="saturation vapor pressure")
    return np.exp(YAWS_A + YAWS_B / (T + YAWS_C)) * 1e3  # Pa


# ----------------------------------------------------------------------------
# 12. SATURATION TEMPERATURE FROM PRESSURE [4, calculated]
# Algebraic inverse of the Yaws equation.
# ----------------------------------------------------------------------------
def saturation_temperature(P):
    P = check_pressure(P, property_name="saturation pressure")
    denominator = np.log(P / 1e3) - YAWS_A
    if np.any(denominator >= 0.0):
        raise ValueError("Pressure is outside the mathematical range of the "
                         "Yaws vapor-pressure correlation.")
    T_sat = YAWS_B / denominator - YAWS_C
    check_temperature(T_sat, Tmin=T_VP_MIN, Tmax=T_VP_MAX,
                      property_name="saturation temperature")
    return T_sat  # K


# ----------------------------------------------------------------------------
# 13. SATURATION-PRESSURE DERIVATIVE [4, calculated]
# dP/dT = P * (-B)/(T + C)^2
# ----------------------------------------------------------------------------
def saturation_pressure_derivative(T):
    T = check_temperature(T, Tmin=T_VP_MIN, Tmax=T_VP_MAX,
                          property_name="saturation-pressure derivative")
    return vapor_pressure(T) * (-YAWS_B) / (T + YAWS_C) ** 2  # Pa/K


# ============================================================================
# MONATOMIC SILVER-VAPOR PROPERTIES
# ============================================================================
# ----------------------------------------------------------------------------
# 14. VAPOR SPECIFIC GAS CONSTANT [6,9]
# ----------------------------------------------------------------------------
def vapor_specific_gas_constant():
    return R_silver_vapor


# ----------------------------------------------------------------------------
# 15. VAPOR SPECIFIC HEAT AT CONSTANT PRESSURE [6]
# Monatomic Ag(g): Cp = 5/2 R_u = 20.786 J/(mol*K). The first electronic
# excitation of Ag lies near 3.7 eV, so electronic contributions are
# negligible below ~3000 K.
# ----------------------------------------------------------------------------
def vapor_specific_heat_cp(T):
    T = check_temperature(T, Tmin=100.0, Tmax=3000.0,
                          property_name="vapor specific heat Cp")
    return np.full_like(T, 2.5 * R_universal / M_silver, dtype=float)


# ----------------------------------------------------------------------------
# 16. VAPOR SPECIFIC HEAT AT CONSTANT VOLUME [9]
# ----------------------------------------------------------------------------
def vapor_specific_heat_cv(T):
    return vapor_specific_heat_cp(T) - R_silver_vapor


# ----------------------------------------------------------------------------
# 17. VAPOR SPECIFIC-HEAT RATIO [9]   (= 5/3)
# ----------------------------------------------------------------------------
def vapor_specific_heat_ratio(T):
    return vapor_specific_heat_cp(T) / vapor_specific_heat_cv(T)


# ----------------------------------------------------------------------------
# 18. MONATOMIC VAPOR SPECIFIC ENTHALPY [6]
# h_v = DeltaHf_gas(298.15) + (5/2)*R_u*(T - 298.15), on the standard
# (elements-at-298 K) basis.  Units: J/kg
# ----------------------------------------------------------------------------
def vapor_specific_enthalpy(T):
    T = check_temperature(T, Tmin=100.0, Tmax=3000.0,
                          property_name="vapor specific enthalpy")
    h_molar = DELTA_HF_GAS_298 + 2.5 * R_universal * (T - 298.15)
    return h_molar / M_silver


# ----------------------------------------------------------------------------
# 19. LATENT HEAT OF VAPORIZATION [4,9, calculated]
# Clausius-Clapeyron with ideal vapor and v_l << v_g:
# h_fg = R_v*T^2*dln(P)/dT = R_v*T^2*(-B)/(T + C)^2
# Check: 2334 J/g (251.8 kJ/mol) at 2435 K vs 2332 J/g (MatWeb) and
#        ~254 kJ/mol (Wikipedia) [7].
# Units: J/kg
# ----------------------------------------------------------------------------
def latent_heat_vaporization(T):
    T = check_temperature(T, Tmin=T_VP_MIN, Tmax=T_VP_MAX,
                          property_name="latent heat of vaporization")
    return R_silver_vapor * T**2 * (-YAWS_B) / (T + YAWS_C) ** 2


# ----------------------------------------------------------------------------
# 20. SATURATED VAPOR DENSITY [4,9, calculated]
# Monatomic ideal gas: rho_v = P/(R_v*T); P defaults to P_sat(T).
# Neglects Ag2 (minor species).   Units: kg/m^3
# ----------------------------------------------------------------------------
def vapor_density(T, P=None):
    T = check_temperature(T, Tmin=T_VP_MIN, Tmax=3000.0,
                          property_name="vapor density")
    P_used = vapor_pressure(T) if P is None else check_pressure(
        P, property_name="vapor pressure")
    return P_used / (R_silver_vapor * T)


# ----------------------------------------------------------------------------
# 21. VAPOR SPEED OF SOUND [9]
# a_v = sqrt(gamma_v*R_v*T)          Units: m/s
# ----------------------------------------------------------------------------
def vapor_speed_of_sound(T):
    T = check_temperature(T, Tmin=T_VP_MIN, Tmax=3000.0,
                          property_name="vapor speed of sound")
    return np.sqrt(vapor_specific_heat_ratio(T) * R_silver_vapor * T)


# ----------------------------------------------------------------------------
# 22. VAPOR DYNAMIC VISCOSITY  [PROVISIONAL - see header]
# Chapman-Enskog, rigid spheres:
# mu = 2.6693e-6 * sqrt(M[g/mol]*T) / sigma[A]^2          [Pa*s]
# ----------------------------------------------------------------------------
def vapor_dynamic_viscosity(T):
    T = check_temperature(T, Tmin=100.0, Tmax=3000.0,
                          property_name="vapor dynamic viscosity")
    _warn_provisional(
        "vapor_dynamic_viscosity",
        "Rigid-sphere kinetic-theory estimate; replace with Biolsi & "
        "Holland (2007) silver-vapor values.")
    return 2.6693e-6 * np.sqrt(M_silver * 1e3 * T) / SIGMA_HS_ANGSTROM**2


# ----------------------------------------------------------------------------
# 23. VAPOR KINEMATIC VISCOSITY [calculated]
# ----------------------------------------------------------------------------
def vapor_kinematic_viscosity(T, P=None):
    return vapor_dynamic_viscosity(T) / vapor_density(T, P=P)


# ----------------------------------------------------------------------------
# 24. VAPOR THERMAL CONDUCTIVITY  [PROVISIONAL - inherits from item 22]
# Monatomic gas (Eucken, exact for atoms): k_v = (15/4)*(R_u/M)*mu_v
# ----------------------------------------------------------------------------
def vapor_thermal_conductivity(T):
    return 3.75 * R_silver_vapor * vapor_dynamic_viscosity(T)


# ----------------------------------------------------------------------------
# 25. VAPOR PRANDTL NUMBER [calculated]   (= 2/3 for a monatomic gas)
# ----------------------------------------------------------------------------
def vapor_prandtl_number(T):
    return (vapor_specific_heat_cp(T) * vapor_dynamic_viscosity(T)
            / vapor_thermal_conductivity(T))


# ----------------------------------------------------------------------------
# 26. VAPOR THERMAL DIFFUSIVITY [calculated]
# ----------------------------------------------------------------------------
def vapor_thermal_diffusivity(T, P=None):
    return (vapor_thermal_conductivity(T)
            / (vapor_density(T, P=P) * vapor_specific_heat_cp(T)))


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================
def liquid_properties(T):
    return {
        "temperature_K": np.asarray(T, dtype=float),
        "density_kg_m3": liquid_density(T),
        "electrical_resistivity_ohm_m": liquid_electrical_resistivity(T),
        "enthalpy_relative_melting_J_kg": liquid_enthalpy_relative_melting(T),
        "specific_heat_J_kgK": liquid_specific_heat(T),
        "surface_tension_N_m": surface_tension(T),
        "thermal_conductivity_W_mK": liquid_thermal_conductivity(T),
        "dynamic_viscosity_Pa_s": liquid_dynamic_viscosity(T),
        "kinematic_viscosity_m2_s": liquid_kinematic_viscosity(T),
        "prandtl_number": liquid_prandtl_number(T),
        "thermal_diffusivity_m2_s": liquid_thermal_diffusivity(T),
    }


def vapor_properties(T, P=None):
    P_used = vapor_pressure(T) if P is None else check_pressure(P)
    return {
        "temperature_K": np.asarray(T, dtype=float),
        "pressure_Pa": P_used,
        "specific_gas_constant_J_kgK": vapor_specific_gas_constant(),
        "density_kg_m3": vapor_density(T, P=P_used),
        "specific_heat_cp_J_kgK": vapor_specific_heat_cp(T),
        "specific_heat_cv_J_kgK": vapor_specific_heat_cv(T),
        "specific_heat_ratio": vapor_specific_heat_ratio(T),
        "specific_enthalpy_J_kg": vapor_specific_enthalpy(T),
        "speed_of_sound_m_s": vapor_speed_of_sound(T),
        "dynamic_viscosity_Pa_s": vapor_dynamic_viscosity(T),
        "kinematic_viscosity_m2_s": vapor_kinematic_viscosity(T, P=P_used),
        "thermal_conductivity_W_mK": vapor_thermal_conductivity(T),
        "prandtl_number": vapor_prandtl_number(T),
        "thermal_diffusivity_m2_s": vapor_thermal_diffusivity(T, P=P_used),
    }


def silver_properties(T, P=None):
    return {
        "liquid": liquid_properties(T),
        "saturation": {
            "pressure_Pa": vapor_pressure(T),
            "dP_dT_Pa_K": saturation_pressure_derivative(T),
            "latent_heat_vaporization_J_kg": latent_heat_vaporization(T),
        },
        "vapor": vapor_properties(T, P=P),
    }


# ----------------------------------------------------------------------------
# PROPERTY TABLE
# ----------------------------------------------------------------------------
def property_table(T_start=1300.0, T_end=2400.0, T_step=100.0, csv_path=None):
    """Print (and optionally save as CSV) a table of the heat-pipe-relevant
    silver properties. Out-of-range warnings are suppressed here; the table
    marks extrapolated cells with '*' in the printed output."""
    T = np.arange(T_start, T_end + 0.5 * T_step, T_step)
    cols = [
        ("T [K]",          lambda t: t,                                   None),
        ("P_sat [Pa]",     vapor_pressure,                                (T_VP_MIN, T_VP_MAX)),
        ("rho_l [kg/m3]",  liquid_density,                                (T_melting, 1773.15)),
        ("sigma [N/m]",    surface_tension,                               (T_melting, 1573.15)),
        ("mu_l [Pa s]",    liquid_dynamic_viscosity,                      (T_melting, 1673.0)),
        ("k_l [W/mK]",     liquid_thermal_conductivity,                   (T_melting, T_boiling_normal)),
        ("h_fg [J/kg]",    latent_heat_vaporization,                      (T_VP_MIN, T_VP_MAX)),
        ("rho_v [kg/m3]",  vapor_density,                                 (T_VP_MIN, 3000.0)),
        ("mu_v [Pa s]",    vapor_dynamic_viscosity,                       (100.0, 3000.0)),
        ("gamma_v [-]",    vapor_specific_heat_ratio,                     (100.0, 3000.0)),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = np.column_stack([f(T) for _, f, _ in cols])

    head = "".join(f"{h:>15}" for h, _, _ in cols)
    print(head)
    print("-" * len(head))
    for i, t in enumerate(T):
        line = ""
        for j, (_, _, rng) in enumerate(cols):
            v = data[i, j]
            star = "*" if rng and not (rng[0] <= t <= rng[1]) else " "
            line += f"{v:14.4e}{star}" if j else f"{v:14.0f} "
        print(line)
    print("* = correlation extrapolated outside its recommended range")
    print("mu_l, k_l, mu_v (and k_v) are PROVISIONAL - see file header.")

    if csv_path:
        np.savetxt(csv_path, data, delimiter=",",
                   header=",".join(h for h, _, _ in cols), comments="")
    return T, data


if __name__ == "__main__":
    property_table(csv_path="silver_property_table.csv")