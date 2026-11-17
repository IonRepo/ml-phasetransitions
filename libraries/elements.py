"""Per-element node features: atomic mass, charge, electronegativity and first ionization energy.

If ``input/atomic_masses.dat`` exists it is used (same format as before: ``symbol mass charge electronegativity
ionization_energy``, with ``None`` for missing values). Otherwise the table is built from pymatgen, with ``charge``
taken as the most common oxidation state.
"""

import os
import warnings
from functools import lru_cache

import numpy as np

FEATURE_NAMES = ('atomic_mass', 'charge', 'electronegativity', 'ionization_energy')
DEFAULT_TABLE_PATH = 'input/atomic_masses.dat'


def _to_float(token):
    return np.nan if token in ('None', 'nan', 'NaN') else float(token)


def read_element_file(path):
    """Read the whitespace-separated element table. Returns {symbol: np.array(4)}."""
    table = {}
    with open(path, 'r') as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            key, *values = line.split()
            table[key] = np.array([_to_float(v) for v in values[:4]], dtype=float)
    return table


def build_pymatgen_table():
    from pymatgen.core import Element

    table = {}
    for z in range(1, 104):
        element = Element.from_Z(z)
        oxidation = element.common_oxidation_states
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                ionization = element.ionization_energy
            except Exception:
                ionization = None
            try:
                electronegativity = element.X
            except Exception:
                electronegativity = None
        table[element.symbol] = np.array([
            float(element.atomic_mass),
            float(oxidation[0]) if oxidation else 0.0,
            float(electronegativity) if electronegativity is not None and np.isfinite(electronegativity) else np.nan,
            float(ionization) if ionization is not None else np.nan,
        ])
    return table


@lru_cache(maxsize=4)
def load_element_table(path=DEFAULT_TABLE_PATH):
    """Element table, cached. Falls back to pymatgen when ``path`` does not exist."""
    if path and os.path.isfile(path):
        return read_element_file(path)
    return build_pymatgen_table()


def atomic_mass(symbol, path=DEFAULT_TABLE_PATH):
    return float(load_element_table(path)[symbol][0])
