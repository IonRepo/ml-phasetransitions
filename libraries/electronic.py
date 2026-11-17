"""Electronic contribution to the free energy from the density of states (DOS).

For a DOS n(e) (states / eV / cell) and Fermi-Dirac occupation f(e; mu, T) at fixed electron number N0:

    U_el(T) = int n(e) e f(e; mu(T), T) de
    S_el(T) = -k_B int n(e) [f ln f + (1 - f) ln(1 - f)] de
    F_el(T) = U_el(T) - U_el(0) - T S_el(T)

so that F_el(0) = 0: the T = 0 electronic energy is already contained in the DFT ground-state energy E0.
The chemical potential mu(T) is fixed by conserving the electron number, N(mu, T) = N0 = N(E_F, 0).

Limits that are checked in the tests:
    * insulator (gap >> k_B T):  F_el -> 0 exponentially
    * metal, flat DOS:           F_el = -(pi^2 / 6) k_B^2 T^2 n(E_F)   (Sommerfeld)

All free energies are returned in meV / atom.

Accepted DOS file formats (file called ``dos``, ``dos.json``, ``dos.dat`` or ``dos.txt`` in the polymorph folder):
    * JSON written by ``data_query.ipynb`` (keys: energies, densities, efermi, natoms)
    * JSON from pymatgen ``Dos.as_dict()`` / ``CompleteDos.as_dict()`` (what the Materials Project API returns)
    * text, columns ``E (eV), DOS`` (optionally a third column with the second spin channel) with an optional
      ``# efermi = <value>`` header line; energies are then taken as absolute (efermi defaults to 0)
"""

import json
import os
import warnings

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, xlogy

K_B = 8.617333262e-5  # Boltzmann constant, eV / K

DOS_FILE_NAMES = ('dos', 'dos.json', 'dos.dat', 'dos.txt')


# ----------------------------------------------------------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------------------------------------------------------

def _find_file(folder, names):
    for name in names:
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            return path
    return None


def find_dos_file(folder):
    """Path of the DOS file in ``folder`` or None."""
    return _find_file(folder, DOS_FILE_NAMES)


def _sum_spin_channels(densities):
    """Total DOS from a pymatgen-style ``{'1': [...], '-1': [...]}`` dict, or from an array/list."""
    if isinstance(densities, dict):
        return np.sum([np.asarray(v, dtype=float) for v in densities.values()], axis=0)
    return np.asarray(densities, dtype=float)


def load_dos(path):
    """Read a DOS file.

    Returns:
        dict: ``energies`` (eV, absolute), ``densities`` (states / eV / cell, spin summed), ``efermi`` (eV) and
            ``natoms`` (int or None, number of atoms in the cell the DOS refers to).
    """
    with open(path, 'r') as f:
        head = f.read(1)
    if head in ('{', '['):
        with open(path, 'r') as f:
            raw = json.load(f)
        # Some tools wrap the DOS in {'dos': {...}} or {'total': {...}}
        for key in ('dos', 'total'):
            if isinstance(raw, dict) and key in raw and 'energies' not in raw:
                raw = raw[key]
        energies = np.asarray(raw['energies'], dtype=float)
        densities = _sum_spin_channels(raw['densities'])
        efermi = float(raw.get('efermi', 0.0))
        natoms = raw.get('natoms')
        if natoms is None and isinstance(raw.get('structure'), dict):
            natoms = len(raw['structure'].get('sites', [])) or None
    else:
        efermi = 0.0
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('#') and 'efermi' in line.lower():
                    efermi = float(line.split('=')[-1].split()[0])
        table = np.loadtxt(path, comments='#', ndmin=2)
        energies = table[:, 0]
        densities = table[:, 1:].sum(axis=1)
        natoms = None
    order = np.argsort(energies)
    return {'energies': energies[order], 'densities': densities[order], 'efermi': efermi,
            'natoms': int(natoms) if natoms else None}


# ----------------------------------------------------------------------------------------------------------------------
# Physics
# ----------------------------------------------------------------------------------------------------------------------

def _trapz(y, x):
    return np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x))


def _occupation(energies, mu, temperature):
    return expit(-(energies - mu) / (K_B * temperature))


def _zero_temperature_occupation(energies, efermi):
    """T = 0 occupation on a discrete grid; states exactly at E_F count half (consistent with f(E_F) = 1/2)."""
    occ = (energies < efermi).astype(float)
    occ[energies == efermi] = 0.5
    return occ


def electronic_free_energy(energies, densities, efermi, temperatures, natoms, bandgap=None):
    """Electronic free energy F_el(T) in meV / atom, with F_el(0) = 0.

    Args:
        energies     (array): DOS energy grid (eV), ascending.
        densities    (array): total DOS (states / eV / cell, spin summed).
        efermi       (float): Fermi level (eV) in the same reference as ``energies``.
        temperatures (array): temperatures (K).
        natoms       (int):   atoms in the cell the DOS refers to.
        bandgap      (float): optional band gap (eV). If given and > 0 the Fermi level is moved to the middle of the
                              gap around ``efermi`` so that the T = 0 occupation is unambiguous for insulators.

    Returns:
        numpy.ndarray: F_el for every temperature, meV / atom.
    """
    energies = np.asarray(energies, dtype=float)
    densities = np.clip(np.asarray(densities, dtype=float), 0, None)
    temperatures = np.atleast_1d(np.asarray(temperatures, dtype=float))

    if bandgap is not None and np.isfinite(bandgap) and bandgap > 0:
        efermi = _centre_of_gap(energies, densities, efermi, bandgap)

    energies, densities = _refine_grid(energies, densities)

    occ0 = _zero_temperature_occupation(energies, efermi)
    n_ref = _trapz(densities * occ0, energies)  # electron count (relative to the bottom of the window)
    u_ref = _trapz(densities * occ0 * energies, energies)

    out = np.zeros_like(temperatures)
    for i, temperature in enumerate(temperatures):
        if temperature <= 0:
            continue
        kt = K_B * temperature

        def excess_electrons(mu):
            return _trapz(densities * _occupation(energies, mu, temperature), energies) - n_ref

        lo, hi = energies[0] - 10 * kt, energies[-1] + 10 * kt
        if excess_electrons(lo) * excess_electrons(hi) > 0:
            warnings.warn('electronic_free_energy: could not bracket the chemical potential (DOS window too narrow?)')
            out[i] = np.nan
            continue
        mu = brentq(excess_electrons, lo, hi, xtol=1e-10)

        f = _occupation(energies, mu, temperature)
        u = _trapz(densities * f * energies, energies)
        entropy = -K_B * _trapz(densities * (xlogy(f, f) + xlogy(1 - f, 1 - f)), energies)  # eV / K per cell
        out[i] = (u - u_ref - temperature * entropy) / natoms * 1e3  # meV / atom
    return out


def _refine_grid(energies, densities, max_step=2e-3, max_points=400_000):
    """Linearly resample the DOS on a finer uniform grid so that the T = 0 step function is resolved well below
    k_B T (the Materials Project grid is ~10 meV, comparable to k_B T at 100 K)."""
    span = energies[-1] - energies[0]
    n_fine = int(np.ceil(span / max_step)) + 1
    if n_fine <= len(energies) or n_fine > max_points:
        return energies, densities
    fine = np.linspace(energies[0], energies[-1], n_fine)
    return fine, np.interp(fine, energies, densities)


def _centre_of_gap(energies, densities, efermi, bandgap):
    """Place E_F in the middle of the gap, using the DOS edges found around the given Fermi level."""
    tol = 1e-6 * max(1.0, densities.max())
    below = energies[(energies <= efermi) & (densities > tol)]
    above = energies[(energies >= efermi) & (densities > tol)]
    if len(below) and len(above):
        vbm, cbm = below.max(), above.min()
        if cbm - vbm > 0:
            return 0.5 * (vbm + cbm)
    return efermi


def electronic_free_energy_from_file(path, temperatures, natoms=None, bandgap=None):
    """Convenience wrapper: read a DOS file and return F_el(T) (meV / atom).

    ``natoms`` is used only if the file does not state the number of atoms of its own cell.
    """
    dos = load_dos(path)
    n = dos['natoms'] or natoms
    if not n:
        raise ValueError(f'{path}: number of atoms of the DOS cell is unknown; pass natoms')
    if dos['natoms'] and natoms and dos['natoms'] != natoms:
        warnings.warn(f'{path}: the DOS cell has {dos["natoms"]} atoms but the structure has {natoms}; '
                      f'using the DOS value')
    return electronic_free_energy(dos['energies'], dos['densities'], dos['efermi'], temperatures, n, bandgap)
