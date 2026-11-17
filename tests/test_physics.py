import numpy as np
import pytest

from libraries import electronic as el
from libraries import free_energy as fe
from libraries.transitions import Polymorph, find_transitions


def test_sommerfeld_metal():
    e = np.linspace(-10, 10, 2001)
    n0, natoms = 3.0, 2
    t = np.array([0, 100, 300, 600.])
    f = el.electronic_free_energy(e, np.full_like(e, n0), 0.0, t, natoms)
    sommerfeld = -(np.pi**2 / 6) * el.K_B**2 * t**2 * n0 / natoms * 1e3
    assert f[0] == 0
    np.testing.assert_allclose(f[1:], sommerfeld[1:], rtol=1e-2, atol=1e-3)  # 1 micro-eV/atom grid noise


def test_insulator_has_no_electronic_term():
    e = np.linspace(-10, 10, 2001)
    d = np.where(np.abs(e) < 1.0, 0, 3.0)
    f = el.electronic_free_energy(e, d, 0.0, np.array([300, 700.]), 2, bandgap=2.0)
    assert np.all(np.abs(f) < 1e-3)


def test_efermi_inside_gap_does_not_matter():
    e = np.linspace(-10, 10, 4001)
    d = np.where(np.abs(e) < 1.0, 0, 3.0)
    t = np.array([300, 700.])
    a = el.electronic_free_energy(e, d, -0.9, t, 2, bandgap=2.0)
    b = el.electronic_free_energy(e, d, 0.9, t, 2, bandgap=2.0)
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_dos_file_formats(tmp_path):
    import json
    e = np.linspace(-5, 5, 101)
    d = np.ones_like(e)
    # pymatgen style with spin channels
    p = tmp_path / 'dos'
    p.write_text(json.dumps({'energies': e.tolist(), 'densities': {'1': (d / 2).tolist(), '-1': (d / 2).tolist()},
                             'efermi': 0.5}))
    out = el.load_dos(str(p))
    np.testing.assert_allclose(out['densities'], d)
    assert out['efermi'] == 0.5
    # text
    q = tmp_path / 'dos.dat'
    q.write_text('# efermi = 1.25\n' + '\n'.join(f'{a} {b}' for a, b in zip(e, d)))
    assert el.load_dos(str(q))['efermi'] == 1.25


def test_quartic_fit_recovers_coefficients():
    t = np.arange(300, 601, 50.0)
    f = fe.helmholtz(t, -20.0, -3.0, +0.7)   # note the positive gamma: allowed now
    np.testing.assert_allclose(fe.fit_helmholtz(t, f), [-20.0, -3.0, 0.7], atol=1e-6)


def test_transition_roots_quadratic_and_linear():
    # F2 - F1 = 5 - 0.05*1e-5*... choose roots by construction: d(T) = 1e-10*g*(T^2 - 300^2)(T^2 - 500^2)
    g = 2.0
    c2 = g * 1e-10
    c1 = -c2 * (300**2 + 500**2)
    c0 = c2 * 300**2 * 500**2
    roots = fe.transition_temperatures(0.0, [c0, c1 / 1e-5, c2 / 1e-10], 0, 1000)
    np.testing.assert_allclose(roots, [300, 500], rtol=1e-6)
    # linear: dE + db*1e-5*T^2 = 0 with gamma = 0
    roots = fe.transition_temperatures(10.0, [0.0, -10.0 / (400**2 * 1e-5), 0.0], 0, 1000)
    np.testing.assert_allclose(roots, [400], rtol=1e-6)


def test_uncertainty_symmetric_case():
    d = [0.0, -2e-5 * 1e5 / 1e5, 0.0]  # F2-F1 = -2e-5*... linear in T^2
    d_e0, tc = 10.0, 400.0
    d = [0.0, -d_e0 / (tc**2 * 1e-5), 0.0]
    u = fe.transition_temperature_uncertainty(d_e0, d, 1.0, tc)
    # T = sqrt((dE + shift)/ (|b|)) -> dT ~ tc * shift / (2 dE)
    assert u == pytest.approx(tc * 1.0 / (2 * d_e0), rel=0.05)


def _poly(label, e0, coef_vib, coef_el, phase, gap, group='X', sg='P1'):
    t = np.arange(300, 601, 50.0)
    coef = np.array(coef_vib) + np.array(coef_el)
    return Polymorph(label=label, material=group, space_group=sg, e0=e0, coef=coef, coef_vib=np.array(coef_vib),
                     coef_el=np.array(coef_el), uncertainty=np.full(len(t), 2.0), temperatures=t, bandgap=gap,
                     phase=phase, has_dos=True, mass_per_atom=30.0)


def test_metal_insulator_transition_is_found_and_labelled():
    # insulator: lower E0, no electronic term. metal: higher E0, F_el = -gamma_el T^2 lowers its free energy
    ins = _poly('X I', 0.0, [0, 0, 0], [0, 0, 0], 'insulator', 2.0, sg='I')
    # F_el(T) = -4e-5 T^2 -> at 400 K = -6.4 meV. E0 offset 6.4 meV -> transition at 400 K
    met = _poly('X M', 6.4, [0, 0, 0], [0, -4.0, 0], 'metal', 0.0, sg='M')
    stable, meta = find_transitions([ins, met], 300, 600)
    df = stable if len(stable) else meta
    assert len(df) == 1
    row = df.iloc[0]
    assert row['transition-temperature'] == pytest.approx(400.0, rel=1e-6)
    assert bool(row['MIT']) and row['transition-class'] == 'insulator-metal'
    assert row['polymorph-1'] == 'I' and row['polymorph-2'] == 'M'
    assert row['entropy-change-electronic'] > 0
    # without the electronic term the same pair has no transition in range
    met_no_el = _poly('X M', 6.4, [0, 0, 0], [0, 0, 0], 'metal', 0.0, sg='M')
    s2, m2 = find_transitions([ins, met_no_el], 300, 600)
    assert len(s2) == 0 and len(m2) == 0
