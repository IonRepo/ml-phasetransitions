import json
from types import SimpleNamespace

import numpy as np
from pymatgen.core import Lattice, Structure

from libraries import dataset as ds
from libraries import electronic as el
from libraries import mp_query


class FakeDos:
    def __init__(self, structure):
        self.energies = np.linspace(-10, 10, 401)
        self.densities = {'up': np.ones(401), 'down': np.ones(401)}
        self.efermi = 0.0
        self.structure = structure


def test_write_entry_roundtrip_with_dos(tmp_path):
    s = Structure(Lattice.cubic(4.0), ['Na', 'Cl'], [[0, 0, 0], [0.5, 0.5, 0.5]])
    doc = SimpleNamespace(formula_pretty='NaCl', material_id='mp-22862', structure=s, energy_per_atom=-3.1,
                          band_gap=5.0, is_metal=False, symmetry=SimpleNamespace(symbol='Fm-3m'))
    doc2 = SimpleNamespace(**{**doc.__dict__, 'material_id': 'mp-1', 'symmetry': SimpleNamespace(symbol='Fm-3m')})
    f1 = mp_query.write_entry(str(tmp_path), doc, FakeDos(s))
    f2 = mp_query.write_entry(str(tmp_path), doc2)           # same space group, different id: no overwrite
    assert f1 != f2
    assert mp_query.write_entry(str(tmp_path), doc) is None   # resume: already present
    assert el.find_dos_file(f1) and el.read_bandgap(f1) == 5.0

    graphs, report = ds.read_database(str(tmp_path), targets=False)
    assert report['read'] == 2
    g = [g for g in graphs if g.polymorph.endswith('mp-22862')][0]
    assert g.gs_energy == -3100.0 and g.bandgap == 5.0 and g.dos_path
    dos = el.load_dos(g.dos_path)
    assert dos['natoms'] == 2 and np.allclose(dos['densities'], 2.0)   # spin channels summed
