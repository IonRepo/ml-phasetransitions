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


# ----------------------------------------------------------------------------------------------------------------------
# Thermoelectric screening
# ----------------------------------------------------------------------------------------------------------------------

def _parabolic_dos(gap=1.5, mass=1.0, volume=20.0, step=1e-3, span=1.6):
    """Spin-summed DOS (states/eV/cell, 1 atom) of one parabolic conduction and one valence band."""
    import libraries.thermoelectric as clt
    e = np.arange(-span, gap + span, step)
    coefficient = volume * 1e-30 / (2 * np.pi ** 2) * (2 * mass * clt.M_E / clt.HBAR ** 2) ** 1.5 * clt.EV ** 1.5
    dos = np.where(e > gap, coefficient * np.sqrt(np.clip(e - gap, 0, None)), 0) + \
          np.where(e < 0, coefficient * np.sqrt(np.clip(-e, 0, None)), 0)
    return {'energies': e, 'densities': dos, 'efermi': 0.5 * gap, 'natoms': 1}, volume


def _fermi_integral(j, eta):
    from scipy.integrate import quad
    return quad(lambda x: x ** j / (1 + np.exp(x - eta)), 0, 80)[0]


def test_thermoelectric_matches_single_parabolic_band():
    import libraries.thermoelectric as clt
    from scipy.optimize import brentq
    dos, volume = _parabolic_dos(mass=1.0)
    bands = clt.prepare_bands(dos, volume, bandgap=1.5)
    assert abs(bands['m_c'] - 1.0) < 0.05 and abs(bands['m_v'] - 1.0) < 0.05, (bands['m_c'], bands['m_v'])
    T, tau = 300.0, 1e-14
    props = clt.thermoelectric_properties(bands, T, dopings=np.array([1e19, 1e20]), tau=tau)
    kt = clt.K_B * T
    n_c = 2 * (2 * np.pi * clt.M_E * kt * clt.EV / (2 * np.pi * clt.HBAR) ** 2) ** 1.5 * 1e-6  # cm^-3
    for j, n in enumerate([1e19, 1e20]):
        eta = brentq(lambda x: n_c * 2 / np.sqrt(np.pi) * _fermi_integral(0.5, x) - n, -20, 20)
        s_exact = (2.5 * _fermi_integral(1.5, eta) / (1.5 * _fermi_integral(0.5, eta)) - eta) * clt.K_B * 1e6  # uV/K
        sigma_exact = n * 1e6 * clt.E_CHARGE ** 2 * tau / clt.M_E
        for carrier, sign in ((1, -1), (0, +1)):  # electrons: S < 0, holes: S > 0
            assert np.isclose(props['S'][carrier, j], sign * s_exact, rtol=0.05), (n, carrier, props['S'][carrier, j], s_exact)
            assert np.isclose(props['sigma'][carrier, j], sigma_exact, rtol=0.05), (n, carrier, props['sigma'][carrier, j], sigma_exact)


def test_thermoelectric_bipolar_reduces_seebeck_and_metals_are_skipped():
    import types
    import libraries.thermoelectric as clt
    wide, volume = _parabolic_dos(gap=1.5)
    narrow, _ = _parabolic_dos(gap=0.15)
    s_wide = clt.thermoelectric_properties(clt.prepare_bands(wide, volume, 1.5), 400.0, np.array([1e19]))['S'][1, 0]
    s_narrow = clt.thermoelectric_properties(clt.prepare_bands(narrow, volume, 0.15), 400.0, np.array([1e19]))['S'][1, 0]
    assert abs(s_narrow) < 0.8 * abs(s_wide), (s_narrow, s_wide)

    metal = types.SimpleNamespace(label='m x', bandgap=0.0)
    assert clt.screen(metal, [300.0], np.array([2.0]))[0]['status'] == 'metal'


def test_kappa_model_is_optional():
    import libraries.thermoelectric as clt
    assert clt.load_kappa_model(None) is None
    assert clt.load_kappa_model('does/not/exist.py') is None
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'kappa.py')
        with open(path, 'w') as f:
            f.write('import numpy as np\ndef predict_kappa(graphs, temperatures):\n    return np.full((len(graphs), len(temperatures)), 3.0)\n')
        assert clt.load_kappa_model(path)([1, 2], [300, 400]).shape == (2, 2)


# ----------------------------------------------------------------------------------------------------------------------
# Ultrastable polymorphs, transition roots, safe I/O
# ----------------------------------------------------------------------------------------------------------------------

def _fake_polymorph(label, alpha, beta, gamma, uncert=1.0):
    import types
    return types.SimpleNamespace(label=label, space_group=label, gs_energy=0.0, uncertainties=np.array([uncert]),
                                 coefficients=np.array([[alpha], [beta], [gamma]]), bandgap=1.0, conductivity=np.nan)


def test_ultrastable_detection():
    import libraries.stability as cls
    T = np.arange(300, 601, 1.0)
    groups = {
        'stable': [_fake_polymorph('A', 0.0, -1.0, 0.0), _fake_polymorph('B', 50.0, -1.0, 0.0)],        # 50 meV apart
        'crossing': [_fake_polymorph('A', 0.0, -1.0, 0.0), _fake_polymorph('B', 4.0, -3.0, 0.0)],        # cross at 2e-5 T^2 = 4 -> 447 K
        'close': [_fake_polymorph('A', 0.0, -1.0, 0.0), _fake_polymorph('B', 1.0, -1.0, 0.0)],          # 1 meV < 2 sigma
        'alone': [_fake_polymorph('A', 0.0, -1.0, 0.0)],
    }
    rows, single = cls.ultrastable_table(groups, T, n_sigma=2.0)
    by = {r['compound']: r for r in rows}
    assert by['stable']['ultrastable'] and by['stable']['label'] == 'A'
    assert not by['crossing']['ultrastable'] and by['crossing']['label-low-T'] != by['crossing']['label-high-T']
    assert not by['close']['ultrastable']
    assert [s['compound'] for s in single] == ['alone']


def test_transition_roots_and_uncertainty_branch():
    import libraries.model as clm_
    # E + b T^2 = 0 with a tiny quartic term: root near sqrt(-c/b)
    roots = clm_.transition_temperatures(10.0, -1e-4, 1e-12)
    assert len(roots) >= 1 and np.isclose(roots[0], np.sqrt(10 / 1e-4), rtol=1e-2)
    assert len(clm_.transition_temperatures(5.0, 1e-4, 1e-12)) == 0        # no positive root
    # uncertainty: symmetric shift of the free-energy difference moves the root by a similar amount on both sides
    u = clm_.compute_transition_temperature_uncertainty(0.0, 10.0, -1e-4, 0.0, 1.0, np.sqrt(10 / 1e-4))
    assert 15 < u < 17  # roots at sqrt(11/1e-4) = 331.7 and sqrt(9/1e-4) = 300 around 316.2: the larger deviation


def test_excel_export_does_not_create_formulas():
    import pandas as pd
    from openpyxl import load_workbook
    import libraries.dataset as cld_
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'x.xlsx')
        cld_.to_excel_safe(pd.DataFrame({'compound': ['=HYPERLINK("http://x","y")', '@SUM(1)', 'Si'], 'v': [1, 2, 3]}), path)
        sheet = load_workbook(path).active
        assert all(sheet.cell(row=r, column=2).data_type != 'f' for r in range(2, 5))
        assert str(sheet.cell(row=4, column=2).value) == 'Si'


def test_graph_files_load_with_weights_only():
    import torch
    from torch_geometric.data import Data
    import libraries.dataset as cld_
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'g.pt')
        torch.save([Data(x=torch.ones(2, 3), label='a b', bandgap=1.0, dos_file='dos')], path)
        assert cld_.load_graphs(path)[0].label == 'a b'


def test_magnetic_files_are_read_by_graph():
    import libraries.graph as clg
    with tempfile.TemporaryDirectory() as tmp:
        make_database(tmp, n_materials=1, n_polymorphs=(1, 1))
        folder = os.path.join(tmp, os.listdir(tmp)[0], 'P0')
        assert clg.generate_graph(folder, label='x P0').ordering == 'Unknown'
        with open(os.path.join(folder, 'ordering'), 'w') as f:
            f.write('AFM\n')
        np.savetxt(os.path.join(folder, 'total_magnetization'), [2.5])
        data = clg.generate_graph(folder, label='x P0')
        assert data.ordering == 'AFM' and data.total_magnetization == 2.5


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
