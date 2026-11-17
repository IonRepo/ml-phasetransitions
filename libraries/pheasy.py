"""Reader for the Pheasy (Marzari group) phonon database -> training graphs.

IMPORTANT - assumed layout. The Zenodo record (https://zenodo.org/records/20196565) could not be inspected when this
module was written, so the reader is written against a *generic* layout and reports what it could not recognise.
Run ``scan(root)`` first; if the report shows many unrecognised folders, only ``read_entry`` below needs adapting.

Assumed: one folder per entry (any depth), containing

    structure   POSCAR | CONTCAR | *.cif | structure.json (pymatgen ``Structure.as_dict()``)
    free energy free_energy*.{dat,txt,csv,json} | thermo*.{dat,txt,csv,json}   -> columns temperature (K), F
        or
    phonon DOS  phonon_dos*.{dat,txt,csv,json}                                  -> columns frequency, DOS
                (then F_v(T) is computed here from the harmonic quantum free energy, see ``harmonic_free_energy``)

The target is the vibrational free energy F_v(T) in meV / atom - the same quantity the original model was trained on.
"""

import json
import os
import warnings

import numpy as np
from pymatgen.core import Structure


H_MEV_PER_THZ = 4.135667696      # h * 1 THz in meV
HC_MEV_PER_CM = 0.12398419843    # h c * 1 cm^-1 in meV
K_B_MEV = 8.617333262e-2         # meV / K

STRUCTURE_FILES = ('POSCAR', 'CONTCAR', 'structure.json', 'structure.cif')
FREE_ENERGY_PREFIXES = ('free_energy', 'thermo')
DOS_PREFIXES = ('phonon_dos', 'phdos', 'pdos')
EXTENSIONS = ('.dat', '.txt', '.csv', '.json')


# ----------------------------------------------------------------------------------------------------------------------
# physics
# ----------------------------------------------------------------------------------------------------------------------

def frequencies_to_mev(frequencies, units='THz'):
    f = np.asarray(frequencies, dtype=float)
    if units == 'THz':
        return f * H_MEV_PER_THZ
    if units in ('cm-1', 'cm^-1'):
        return f * HC_MEV_PER_CM
    if units == 'meV':
        return f
    raise ValueError(f'unknown frequency units {units!r}')


def harmonic_free_energy(frequencies, dos, temperatures, n_atoms, units='THz'):
    """Harmonic vibrational free energy in meV / atom from a phonon DOS.

        F(T) = int g(w) [ hw/2 + k_B T ln(1 - exp(-hw / k_B T)) ] dw,   int g dw = 3 N_atoms

    Modes with w <= 0 (imaginary) are dropped and the remaining DOS is renormalised to 3 N_atoms.
    ``imaginary_fraction`` gives the dropped weight so that callers can discard unstable entries.
    """
    w = frequencies_to_mev(frequencies, units)
    g = np.clip(np.asarray(dos, dtype=float), 0, None)
    order = np.argsort(w)
    w, g = w[order], g[order]
    positive = w > 1e-6
    total_weight = np.trapezoid(g, w)
    pos_weight = np.trapezoid(g[positive], w[positive])
    if pos_weight <= 0:
        return np.full(len(np.atleast_1d(temperatures)), np.nan)
    g_norm = g[positive] * (3 * n_atoms / total_weight) if total_weight > 0 else g[positive]
    w = w[positive]

    out = []
    for t in np.atleast_1d(np.asarray(temperatures, dtype=float)):
        zpe = 0.5 * w
        if t <= 0:
            thermal = 0.0 * w
        else:
            thermal = K_B_MEV * t * np.log(-np.expm1(-w / (K_B_MEV * t)))
        out.append(np.trapezoid(g_norm * (zpe + thermal), w) / n_atoms)
    return np.array(out)


def imaginary_fraction(frequencies, dos):
    """Fraction of the DOS weight at (numerically) negative frequencies."""
    f = np.asarray(frequencies, dtype=float)
    g = np.clip(np.asarray(dos, dtype=float), 0, None)
    order = np.argsort(f)
    f, g = f[order], g[order]
    total = np.trapezoid(g, f)
    return float(np.trapezoid(g[f < -1e-3], f[f < -1e-3]) / total) if total > 0 and (f < -1e-3).sum() > 1 else 0.0


# ----------------------------------------------------------------------------------------------------------------------
# file parsing
# ----------------------------------------------------------------------------------------------------------------------

def _find(folder, names=None, prefixes=None, extensions=EXTENSIONS):
    files = sorted(os.listdir(folder))
    if names:
        for n in names:
            if n in files:
                return os.path.join(folder, n)
    if prefixes:
        for f in files:
            stem, ext = os.path.splitext(f)
            if any(stem.lower().startswith(p) for p in prefixes) and ext.lower() in extensions:
                return os.path.join(folder, f)
    return None


def _read_columns(path, json_keys):
    """Two arrays from a text/csv table (first two numeric columns) or from a JSON dict with one of the key pairs."""
    if path.endswith('.json'):
        with open(path) as f:
            raw = json.load(f)
        for kx, ky in json_keys:
            if kx in raw and ky in raw:
                return np.asarray(raw[kx], dtype=float), np.asarray(raw[ky], dtype=float)
        raise ValueError(f'{path}: none of the keys {json_keys} found (has {list(raw)[:8]})')
    rows = []
    with open(path) as f:
        for line in f:
            tokens = line.replace(',', ' ').replace(';', ' ').split()
            if not tokens or tokens[0].startswith('#'):
                continue
            try:
                rows.append([float(t) for t in tokens[:2]])
            except ValueError:
                continue  # header line
    table = np.array(rows)
    if table.ndim != 2 or table.shape[1] < 2:
        raise ValueError(f'{path}: could not read two numeric columns')
    return table[:, 0], table[:, 1]


def _read_structure(folder):
    path = _find(folder, names=STRUCTURE_FILES) or _find(folder, extensions=('.cif',), prefixes=('',))
    if path is None:
        return None
    if path.endswith('.json'):
        with open(path) as f:
            return Structure.from_dict(json.load(f))
    return Structure.from_file(path)


def read_entry(folder, temperatures, free_energy_units='meV/atom', frequency_units='THz', max_imaginary=0.02):
    """Structure and F_v(T) of one folder. Returns (record, None) or (None, reason).

    record: dict(structure, temperatures, fv (meV/atom on ``temperatures``), folder)
    """
    structure = _read_structure(folder)
    if structure is None:
        return None, 'no structure file'
    n = len(structure)

    fe_path = _find(folder, prefixes=FREE_ENERGY_PREFIXES)
    dos_path = _find(folder, prefixes=DOS_PREFIXES)
    if fe_path is not None:
        t_file, f_file = _read_columns(fe_path, [('temperatures', 'free_energy'), ('T', 'F'), ('temperature', 'F_vib')])
        f_file = f_file * {'meV/atom': 1.0, 'eV/atom': 1e3, 'eV/cell': 1e3 / n}[free_energy_units]
        if temperatures is None:
            temperatures, fv = t_file, f_file
        elif temperatures[0] < t_file.min() - 1e-6 or temperatures[-1] > t_file.max() + 1e-6:
            return None, 'free-energy table does not cover the requested temperatures'
        else:
            fv = np.interp(temperatures, t_file, f_file)
    elif dos_path is not None:
        freq, g = _read_columns(dos_path, [('frequencies', 'dos'), ('frequency', 'dos'), ('omega', 'dos')])
        if imaginary_fraction(freq, g) > max_imaginary:
            return None, 'dynamically unstable (imaginary modes)'
        if temperatures is None:
            return None, 'temperatures required to build F_v from a phonon DOS'
        fv = harmonic_free_energy(freq, g, temperatures, n, frequency_units)
    else:
        return None, 'no free-energy table or phonon DOS'

    return {'structure': structure, 'temperatures': np.asarray(temperatures, float), 'fv': np.asarray(fv, float),
            'folder': folder}, None


# ----------------------------------------------------------------------------------------------------------------------
# database level
# ----------------------------------------------------------------------------------------------------------------------

def entry_folders(root):
    for current, dirs, files in os.walk(root):
        if any(f in files for f in STRUCTURE_FILES) or any(f.endswith('.cif') for f in files):
            yield current


def scan(root, n_examples=5):
    """Count folders that look like entries and show what is inside a few of them. Run this first."""
    folders = list(entry_folders(root))
    print(f'{len(folders)} folders with a structure file under {root}')
    for f in folders[:n_examples]:
        print(' ', os.path.relpath(f, root), '->', sorted(os.listdir(f))[:12])
    if not folders:
        print('Top level:', sorted(os.listdir(root))[:20])
    return folders
