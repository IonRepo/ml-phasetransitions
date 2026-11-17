"""Smooth temperature dependence of the free energy and transition-temperature solver.

The free energy of a polymorph (without the ground-state energy) is smoothed with

    F(T) = alpha + beta * 1e-5 * T^2 + gamma * 1e-10 * T^4          (meV / atom)

Coefficients are stored *scaled* (alpha, beta, gamma), i.e. exactly the numbers that multiply 1, 1e-5 T^2 and 1e-10 T^4.
Unlike the previous implementation, beta and gamma are *signed* and found by linear least squares, so there is no
convergence failure and no sign constraint. The constraint was wrong once an electronic term is included: for a
metal, F_el ~ -T^2 but its next Sommerfeld term is positive in T^4.
"""

import numpy as np

BETA_SCALE = 1e-5
GAMMA_SCALE = 1e-10

# meV / (atom K)  ->  J / (kg K):  1 meV = 1.602176634e-22 J,  N_A = 6.02214076e23,  mass in g / mol
ENTROPY_CONVERSION = 1.602176634e-22 * 6.02214076e23 * 1e3  # = 9.6485e4 / mass_per_atom


def design_matrix(temperatures):
    t = np.asarray(temperatures, dtype=float)
    return np.stack([np.ones_like(t), BETA_SCALE * t**2, GAMMA_SCALE * t**4], axis=-1)


def helmholtz(temperatures, alpha, beta, gamma):
    """F(T) = alpha + beta*1e-5*T^2 + gamma*1e-10*T^4 (coefficients in scaled units)."""
    return design_matrix(temperatures) @ np.array([alpha, beta, gamma])


def helmholtz_derivative(temperatures, alpha, beta, gamma):
    """dF/dT in meV / (atom K)."""
    t = np.asarray(temperatures, dtype=float)
    return 2 * BETA_SCALE * beta * t + 4 * GAMMA_SCALE * gamma * t**3


def fit_helmholtz(temperatures, free_energies):
    """Least-squares fit of the three scaled coefficients. Returns NaNs if the data are not finite."""
    temperatures = np.asarray(temperatures, dtype=float)
    free_energies = np.asarray(free_energies, dtype=float)
    if not np.all(np.isfinite(free_energies)) or len(temperatures) < 3:
        return np.full(3, np.nan)
    coefficients, *_ = np.linalg.lstsq(design_matrix(temperatures), free_energies, rcond=None)
    return coefficients


def fit_many(temperatures, free_energies):
    """Fit one set of coefficients per row of ``free_energies`` (n_materials, n_temperatures) -> (n, 3)."""
    free_energies = np.atleast_2d(free_energies)
    return np.array([fit_helmholtz(temperatures, row) for row in free_energies])


def transition_temperatures(d_e0, d_coefficients, t_min=-np.inf, t_max=np.inf, shift=0.0):
    """All temperatures at which two polymorphs have the same free energy.

    Solves  d_e0 + d_alpha + shift + d_beta*1e-5*T^2 + d_gamma*1e-10*T^4 = 0 for T in (t_min, t_max).
    The equation is a quadratic in u = T^2; the linear case (d_gamma ~ 0) is handled.

    Args:
        d_e0           (float): E0(2) - E0(1), meV / atom.
        d_coefficients (array): coefficients(2) - coefficients(1), scaled units.
        shift          (float): constant added to the free-energy difference (used for uncertainty bounds).

    Returns:
        numpy.ndarray: sorted transition temperatures (K), possibly empty.
    """
    d_alpha, d_beta, d_gamma = (float(x) for x in d_coefficients)
    c0 = d_e0 + d_alpha + shift
    c1 = d_beta * BETA_SCALE
    c2 = d_gamma * GAMMA_SCALE

    if not np.all(np.isfinite([c0, c1, c2])):
        return np.array([])

    if abs(c2) * 1e6 <= 1e-9 * abs(c1):  # quartic term negligible for T <~ 1000 K -> linear in u
        roots = np.array([-c0 / c1]) if abs(c1) > 0 else np.array([])
    else:
        disc = c1**2 - 4 * c2 * c0
        if disc < 0:
            return np.array([])
        sqrt_disc = np.sqrt(disc)
        roots = np.array([(-c1 + sqrt_disc) / (2 * c2), (-c1 - sqrt_disc) / (2 * c2)])

    roots = roots[np.isfinite(roots) & (roots > 0)]
    temperatures = np.sqrt(roots)
    temperatures = temperatures[(temperatures > t_min) & (temperatures < t_max)]
    return np.sort(temperatures)


def transition_temperature_uncertainty(d_e0, d_coefficients, uncertainty_f, t_c, mode='max'):
    """Uncertainty of a transition temperature given an uncertainty on the free-energy difference (meV / atom).

    The free-energy difference is shifted by -/+ ``uncertainty_f``; the root of each shifted equation that is closest to
    ``t_c`` is the corresponding bound. ``mode='max'`` (default) returns the larger deviation (conservative),
    ``'min'`` the smaller one (what the previous implementation did). NaN if a bound has no root.
    """
    deviations = []
    for shift in (-uncertainty_f, uncertainty_f):
        roots = transition_temperatures(d_e0, d_coefficients, shift=shift)
        if len(roots) == 0:
            deviations.append(np.nan)
        else:
            deviations.append(np.min(np.abs(roots - t_c)))
    deviations = np.array(deviations)
    if np.all(np.isnan(deviations)):
        return np.nan
    if mode == 'min':
        return float(np.nanmin(deviations))
    return float(np.nan) if np.any(np.isnan(deviations)) else float(np.max(deviations))


def entropy_change_si(slope_mev_per_atom_k, mass_per_atom):
    """Convert |d(DeltaF)/dT| (meV / atom / K) to J / (kg K) using the mean atomic mass (g / mol)."""
    if not mass_per_atom or not np.isfinite(mass_per_atom):
        return np.nan
    return abs(slope_mev_per_atom_k) * ENTROPY_CONVERSION / mass_per_atom
