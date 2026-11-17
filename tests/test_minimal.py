"""Small checks of the new pieces. Run from the project root:  python -m pytest tests -q   (or: python tests/test_minimal.py)"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import libraries.electronic as cle
import libraries.model as clm
import libraries.pheasy as clp
from tests.synthetic import K_B_MEV, einstein_fv, make_database


def test_insulator_has_no_electronic_free_energy():
    e = np.linspace(-10, 10, 4001)
    dos = np.where((e < 0) | (e > 2.0), 1.0, 0.0)  # gap between 0 and 2 eV
    f = cle.electronic_free_energy(e, dos, 0.0, [100, 300, 600], natoms=1, bandgap=2.0)
    assert np.all(np.abs(f) < 1e-3), f  # meV/atom


def test_metal_follows_sommerfeld():
    e = np.linspace(-10, 10, 20001)
    n_ef = 1.5                                  # states / eV / atom, constant
    dos = np.full_like(e, n_ef)
    T = np.array([100., 200., 300.])
    f = cle.electronic_free_energy(e, dos, 0.0, T, natoms=1, bandgap=0.0)
    sommerfeld = -(np.pi ** 2 / 6) * (K_B_MEV * 1e-3) ** 2 * T ** 2 * n_ef * 1e3  # meV
    assert np.allclose(f, sommerfeld, rtol=2e-2, atol=2e-3), (f, sommerfeld)


def test_signed_helmholtz_fit_recovers_both_signs():
    T = np.arange(100, 701, 50.)
    for beta, gamma in [(-3.0, -1.0), (-3.0, 0.5)]:
        F = 10 + beta * 1e-5 * T ** 2 + gamma * 1e-10 * T ** 4
        coef = clm.compute_coefficients(T, F)
        assert np.allclose(np.ravel(coef), [10, beta, gamma], rtol=1e-3, atol=1e-3), coef


def test_harmonic_free_energy_matches_einstein():
    hw = 25.0                                    # meV
    w = np.linspace(0.5, 10, 200001)             # THz grid
    g = np.zeros_like(w)
    k = np.argmin(np.abs(clp.frequencies_to_mev(w) - hw))
    g[k] = 3.0
    T = np.array([100., 300., 600.])
    f = clp.harmonic_free_energy(w, g, T, n_atoms=1)
    hw_k = clp.frequencies_to_mev(w[k])
    ref = 3 * (0.5 * hw_k + K_B_MEV * T * np.log1p(-np.exp(-hw_k / (K_B_MEV * T))))
    assert np.allclose(f, ref, rtol=1e-3), (f, ref)


def test_graph_edges_are_not_duplicated():
    import libraries.graph as clg
    with tempfile.TemporaryDirectory() as tmp:
        make_database(tmp, n_materials=1, n_polymorphs=(1, 1))
        folder = os.path.join(tmp, os.listdir(tmp)[0], 'P0')
        data = clg.generate_graph(folder, label='x P0')
        from pymatgen.core import Structure
        s = Structure.from_file(os.path.join(folder, 'POSCAR'))
        n_expected = sum(len([n for n in nl if n.nn_distance > 0]) for nl in s.get_all_neighbors(6))
        assert data.edge_index.shape[1] == n_expected
        pairs = set(map(tuple, data.edge_index.t().tolist()))
        assert len(pairs) <= data.edge_index.shape[1]


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
