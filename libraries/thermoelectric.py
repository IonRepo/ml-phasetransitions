"""Thermoelectric pre-screening from the electronic DOS, the band gap and a lattice thermal conductivity model.

This is a PROXY for ranking semiconductors, not a transport calculation. What it assumes (all of it is listed here so
that nothing is hidden):

    * rigid bands: doping only moves the chemical potential, the DOS does not change;
    * each band edge (valence / conduction) is described by ONE isotropic parabolic band, with a DOS effective mass
      m_d fitted to the DOS within ``fit_window`` of the edge, and the same mass for the velocity (conductivity mass =
      DOS mass; multi-valley or anisotropic bands are therefore not captured);
    * the transport function is  Sigma(E) = g(E) v^2(E),  g(E) the real DOS and  v^2 = 2 |E - E_edge| / (3 m_d);
    * constant relaxation time ``tau`` (default 1e-14 s, the usual value of high-throughput BoltzTraP screening);
    * electrons and holes are both included, so bipolar compensation reduces S automatically;
    * the electronic thermal conductivity follows Wiedemann-Franz with the Lorenz number of Kim et al. (2015);
    * kappa_L comes from an external (in-house) model. Without it the figure of merit zT is not defined.

For one parabolic band these formulas reduce to the textbook ones (checked in the tests against the Fermi-integral
expression of S and against sigma = n e^2 tau / m). Metals (band gap below ``gap_threshold``) are not screened.

Units: S in uV/K, sigma in S/m, power factor in uW / (cm K^2), kappa in W / (m K), doping in cm^-3.

The in-house kappa_L model is a Python file defining

    def predict_kappa(graphs, temperatures):
        '''graphs: list of torch_geometric Data (with the attributes created by the prediction notebook);
        returns an array (len(graphs), len(temperatures)) of lattice thermal conductivity in W/(m K).'''

and is loaded with ``load_kappa_model(path)``. NOTE: this imports and runs the file, so only point it to code you trust.
"""

import importlib.util
import os
import warnings

import numpy as np
from scipy.special import expit

import libraries.electronic as cle

K_B = cle.K_B               # eV / K
E_CHARGE = 1.602176634e-19  # C
EV = 1.602176634e-19        # J / eV
M_E = 9.1093837015e-31      # kg
HBAR = 1.054571817e-34      # J s

DOPINGS = np.logspace(18, 21, 7)  # cm^-3, scanned for both carrier types


# ----------------------------------------------------------------------------------------------------------------------
# In-house kappa_L model
# ----------------------------------------------------------------------------------------------------------------------

def load_kappa_model(path):
    """Return the ``predict_kappa`` function of the in-house model file, or None if there is no model.

    None is the signal for the notebook to skip the thermoelectric screening.
    """
    if not path:
        print('No kappa_L model given: thermoelectric screening skipped.')
        return None
    if not os.path.isfile(path):
        print(f'kappa_L model {path} not found: thermoelectric screening skipped.')
        return None
    spec = importlib.util.spec_from_file_location('kappa_model', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, 'predict_kappa'):
        raise AttributeError(f'{path} must define predict_kappa(graphs, temperatures)')
    return module.predict_kappa


# ----------------------------------------------------------------------------------------------------------------------
# Bands from the DOS
# ----------------------------------------------------------------------------------------------------------------------

def band_edges(energies, densities, efermi, bandgap, min_gap=0.05):
    """(VBM, CBM) in eV: the empty stretch of the DOS around E_F whose width is closest to the band gap, or None."""
    tol = 1e-6 * max(1.0, densities.max())
    on = energies[densities > tol]
    if len(on) < 2:
        return None
    width = np.diff(on)
    middle = 0.5 * (on[1:] + on[:-1])
    candidates = np.where((width > min_gap) & (np.abs(middle - efermi) <= bandgap + 1.0))[0]
    if not len(candidates):
        return None
    i = candidates[np.argmin(np.abs(width[candidates] - bandgap))]
    return on[i], on[i + 1]


def _effective_mass(energy, density, edge, volume_m3, sign, fit_window):
    """DOS effective mass (kg) from g(E) = c sqrt(|E - edge|) within ``fit_window`` eV of the edge (least squares)."""
    distance = sign * (energy - edge)
    near = (distance > 0) & (distance <= fit_window)
    if near.sum() < 3:
        return None
    s = np.sqrt(distance[near])
    c = np.sum(density[near] * s) / np.sum(s ** 2)             # states / eV^(3/2) / atom
    if c <= 0:
        return None
    c_vol = c / (volume_m3 * EV ** 1.5)                         # 1 / (J^(3/2) m^3)
    m = 0.5 * HBAR ** 2 * (2 * np.pi ** 2 * c_vol) ** (2.0 / 3.0)
    return float(np.clip(m, 0.05 * M_E, 50 * M_E))


def prepare_bands(dos, volume_per_atom, bandgap, natoms=None, window=0.8, step=4e-3, fit_window=0.15):
    """Band-edge data needed by ``thermoelectric_properties``; None if the DOS has no usable gap.

    Args:
        dos             (dict):  as returned by electronic.load_dos.
        volume_per_atom (float): Angstrom^3 / atom.
        bandgap         (float): band gap (eV) used to select the right gap in the DOS.
        window          (float): energy range (eV) kept on each side of the gap.
        step            (float): resampling step (eV) of the DOS in that range (the MP grid is comparable to k_B T).
    """
    n = dos['natoms'] or natoms
    if not n:
        raise ValueError('number of atoms of the DOS cell unknown')
    energies = np.asarray(dos['energies'], dtype=float)
    density = np.clip(np.asarray(dos['densities'], dtype=float), 0, None) / n  # states / eV / atom
    edges = band_edges(energies, density, dos['efermi'], bandgap)
    if edges is None:
        return None
    vbm, cbm = edges

    volume_m3 = volume_per_atom * 1e-30
    valence = np.arange(vbm - window, vbm + 1e-9, step)
    conduction = np.arange(cbm, cbm + window + 1e-9, step)
    g_v, g_c = np.interp(valence, energies, density), np.interp(conduction, energies, density)
    m_v = _effective_mass(valence, g_v, vbm, volume_m3, -1, fit_window)
    m_c = _effective_mass(conduction, g_c, cbm, volume_m3, +1, fit_window)
    if m_v is None or m_c is None:
        return None
    return {'vbm': vbm, 'cbm': cbm, 'volume_m3': volume_m3, 'volume_cm3': volume_per_atom * 1e-24,
            'energy': np.concatenate([valence, conduction]), 'density': np.concatenate([g_v, g_c]),
            'n_valence': len(valence), 'step': step,
            'mass': np.concatenate([np.full(len(valence), m_v), np.full(len(conduction), m_c)]),
            'edge': np.concatenate([np.full(len(valence), vbm), np.full(len(conduction), cbm)]),
            'm_v': m_v / M_E, 'm_c': m_c / M_E}


# ----------------------------------------------------------------------------------------------------------------------
# Transport
# ----------------------------------------------------------------------------------------------------------------------

def _integrate(y, bands):
    """Trapezoid rule on both sides of the gap (never across it); ``y`` has shape (n_targets, n_energies)."""
    nv, dx = bands['n_valence'], bands['step']
    left, right = y[:, :nv], y[:, nv:]
    return dx * (left.sum(axis=1) - 0.5 * (left[:, 0] + left[:, -1]) +
                 right.sum(axis=1) - 0.5 * (right[:, 0] + right[:, -1]))


def thermoelectric_properties(bands, temperature, dopings=DOPINGS, tau=1e-14):
    """S, sigma and kappa_e for holes (row 0) and electrons (row 1) at every doping level.

    Returns:
        dict with arrays of shape (2, len(dopings)): ``S`` (uV/K), ``sigma`` (S/m), ``kappa_e`` (W/mK), ``PF``
        (uW/cm K^2), and ``doping`` (cm^-3).
    """
    energy, g = bands['energy'], bands['density']
    kt = K_B * temperature
    n_dop = len(dopings)
    delta = np.asarray(dopings) * bands['volume_cm3']                 # carriers per atom
    targets = np.concatenate([-delta, delta])                         # net electrons: holes first, then electrons
    occ0 = (np.arange(len(energy)) < bands['n_valence']).astype(float)

    lo = np.full(2 * n_dop, energy[0])
    hi = np.full(2 * n_dop, energy[-1])
    for _ in range(50):                                               # bisection on the chemical potential
        mu = 0.5 * (lo + hi)
        f = expit(-(energy[None, :] - mu[:, None]) / kt)
        net = _integrate((f - occ0[None, :]) * g[None, :], bands)
        below = net < targets
        lo, hi = np.where(below, mu, lo), np.where(below, hi, mu)
    mu = 0.5 * (lo + hi)

    f = expit(-(energy[None, :] - mu[:, None]) / kt)
    minus_df = f * (1 - f) / kt                                       # -df/dE, 1/eV
    distance = np.abs(energy - bands['edge'])
    weight = g * distance / bands['mass'] * minus_df                  # Sigma(E) (-df/dE) up to constants
    l0 = _integrate(weight, bands)
    l1 = _integrate(weight * (energy[None, :] - mu[:, None]), bands)

    sigma = 2 * E_CHARGE ** 2 * tau / (3 * bands['volume_m3']) * l0  # S/m
    seebeck = -1e6 * l1 / np.where(l0 > 0, l0, np.nan) / temperature  # uV/K (E in eV, so E/e is in V)
    lorenz = (1.5 + np.exp(-np.abs(seebeck) / 116.0)) * 1e-8          # W Ohm / K^2 (Kim et al., APL Mater. 3, 041506)
    kappa_e = lorenz * sigma * temperature
    power_factor = seebeck ** 2 * 1e-12 * sigma * 1e4                 # uW / (cm K^2)

    shape = (2, n_dop)
    return {'S': seebeck.reshape(shape), 'sigma': sigma.reshape(shape), 'kappa_e': kappa_e.reshape(shape),
            'PF': power_factor.reshape(shape), 'doping': np.asarray(dopings)}


# ----------------------------------------------------------------------------------------------------------------------
# Screening of graphs
# ----------------------------------------------------------------------------------------------------------------------

COLUMNS = ['label', 'temperature', 'status', 'bandgap', 'bandgap-DOS', 'gap-over-kT', 'carrier', 'doping',
           'S', 'sigma', 'PF', 'kappa-e', 'kappa-L', 'zT', 'm-valence', 'm-conduction']


def _empty_row(label, temperature, status, bandgap):
    row = dict.fromkeys(COLUMNS, np.nan)
    row.update({'label': label, 'temperature': temperature, 'status': status, 'bandgap': bandgap, 'carrier': ''})
    return row


def screen(data, temperatures, kappa_L, gap_threshold=0.1, dopings=DOPINGS, tau=1e-14):
    """One row per temperature for a single graph: the carrier type and doping that maximise zT.

    Args:
        data         (Data):    graph with ``label``, ``bandgap``, ``dos_file``, ``volume_per_atom``.
        temperatures (array):   temperatures (K).
        kappa_L      (array):   lattice thermal conductivity at each temperature (W/mK); None gives PF-based results
                                (zT stays NaN and the doping with the largest power factor is reported).
        gap_threshold (float):  below this band gap (eV) the material is a metal and is not screened.

    ``status`` is 'ok', 'metal', 'no-dos', 'no-volume' or 'no-gap-in-dos'.
    """
    bandgap = float(getattr(data, 'bandgap', np.nan))
    label = getattr(data, 'label', '')
    rows = []
    if np.isnan(bandgap) or bandgap < gap_threshold:
        return [_empty_row(label, T, 'metal' if not np.isnan(bandgap) else 'no-bandgap', bandgap) for T in temperatures]
    dos_file = getattr(data, 'dos_file', None)
    if not dos_file:
        return [_empty_row(label, T, 'no-dos', bandgap) for T in temperatures]
    volume = getattr(data, 'volume_per_atom', None)
    if volume is None:
        return [_empty_row(label, T, 'no-volume', bandgap) for T in temperatures]

    bands = prepare_bands(cle.load_dos(dos_file), volume, bandgap, natoms=data.x.size(0))
    if bands is None:
        return [_empty_row(label, T, 'no-gap-in-dos', bandgap) for T in temperatures]

    for i, T in enumerate(temperatures):
        props = thermoelectric_properties(bands, T, dopings, tau)
        kl = np.nan if kappa_L is None else float(kappa_L[i])
        zt = props['S'] ** 2 * 1e-12 * props['sigma'] * T / (kl + props['kappa_e'])
        score = props['PF'] if np.isnan(kl) else zt
        if np.all(np.isnan(score)):
            rows.append(_empty_row(label, T, 'no-solution', bandgap))
            continue
        c, d = np.unravel_index(np.nanargmax(score), score.shape)
        rows.append({'label': label, 'temperature': T, 'status': 'ok', 'bandgap': bandgap,
                     'bandgap-DOS': bands['cbm'] - bands['vbm'], 'gap-over-kT': (bands['cbm'] - bands['vbm']) / (K_B * T),
                     'carrier': 'p' if c == 0 else 'n', 'doping': props['doping'][d],
                     'S': props['S'][c, d], 'sigma': props['sigma'][c, d], 'PF': props['PF'][c, d],
                     'kappa-e': props['kappa_e'][c, d], 'kappa-L': kl, 'zT': zt[c, d],
                     'm-valence': bands['m_v'], 'm-conduction': bands['m_c']})
    return rows


def screen_dataset(dataset, temperatures, kappa_L, **kwargs):
    """``screen`` for every graph of ``dataset``; ``kappa_L`` has shape (len(dataset), len(temperatures)) or is None.

    Failures of single materials (corrupt DOS file, ...) are reported and recorded with status 'error' instead of
    stopping the whole run.
    """
    rows = []
    for i, data in enumerate(dataset):
        try:
            rows += screen(data, temperatures, None if kappa_L is None else kappa_L[i], **kwargs)
        except Exception as error:  # noqa: BLE001 - one bad file must not stop a screening of thousands
            warnings.warn(f'thermoelectric screening of {getattr(data, "label", i)} failed: {error}')
            rows += [_empty_row(getattr(data, 'label', str(i)), T, 'error', np.nan) for T in temperatures]
    return rows
