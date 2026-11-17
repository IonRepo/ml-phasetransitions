"""Phase transitions between polymorphs, including metal-insulator transitions.

Total free energy of a polymorph (meV / atom):

    F(T) = E0 + F_v(T) + F_el(T)

F_v is predicted by the GNN, F_el comes from the DOS (``libraries.electronic``), E0 is the DFT ground-state energy.
F_v + F_el is smoothed with the quartic of ``libraries.free_energy`` and a transition between two polymorphs of the same
material is a root of  E0_2 - E0_1 + F_2(T) - F_1(T) = 0.

Every transition is labelled by the electronic character of the two phases (from the band gap): 'insulator-insulator',
'metal-metal' and, new, 'insulator-metal' / 'metal-insulator' (``MIT = True``).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from libraries import electronic as el
from libraries import free_energy as fe
from libraries.predict import predict_free_energies


@dataclass
class Polymorph:
    label: str
    material: str
    space_group: str
    e0: float                       # meV / atom
    coef: np.ndarray                # scaled coefficients of F_v + F_el
    coef_vib: np.ndarray
    coef_el: np.ndarray
    uncertainty: np.ndarray         # ML uncertainty of F_v at each temperature (meV / atom)
    temperatures: np.ndarray
    bandgap: float = np.nan
    phase: str = 'unknown'          # 'metal' | 'insulator' | 'unknown'
    has_dos: bool = False
    centrosymmetry: int = 0
    conductivity: float = np.nan
    mass_per_atom: float = np.nan
    distance: np.ndarray = field(default_factory=lambda: np.array([]))

    def free_energy(self, t):
        return self.e0 + fe.helmholtz(t, *self.coef)


def build_polymorphs(graphs, temperatures, model, standardizer, uncertainty=None, gap_threshold=0.1,
                     include_electronic=True, batch_size=64, device=None):
    """Predict F_v for every graph, add F_el from the DOS, and fit the smooth free energies.

    Returns (polymorphs, notes) where ``notes`` counts metals without a DOS and graphs without usable E0.
    """
    temperatures = np.asarray(temperatures, dtype=float)
    pred = predict_free_energies(graphs, temperatures, model, standardizer, uncertainty, batch_size, device)
    notes = {'metal_without_dos': 0, 'dos_failed': 0, 'no_gs_energy': 0}
    polymorphs = []
    for i, g in enumerate(graphs):
        if not np.isfinite(g.gs_energy):
            notes['no_gs_energy'] += 1
            continue
        fv = pred['fv'][i]
        fel = np.zeros_like(fv)
        has_dos = False
        phase = el.classify_electronic(g.bandgap, gap_threshold)
        if include_electronic and g.dos_path:
            try:
                fel = el.electronic_free_energy_from_file(g.dos_path, temperatures, natoms=int(g.n_atoms),
                                                          bandgap=g.bandgap)
                has_dos = True
            except Exception:
                notes['dos_failed'] += 1
                fel = np.zeros_like(fv)
        if include_electronic and phase == 'metal' and not has_dos:
            notes['metal_without_dos'] += 1
        polymorphs.append(Polymorph(
            label=g.label, material=g.material, space_group=g.space_group, e0=float(g.gs_energy),
            coef=fe.fit_helmholtz(temperatures, fv + fel), coef_vib=fe.fit_helmholtz(temperatures, fv),
            coef_el=fe.fit_helmholtz(temperatures, fel), uncertainty=np.abs(pred['uncertainty'][i]),
            temperatures=temperatures, bandgap=float(g.bandgap), phase=phase, has_dos=has_dos,
            centrosymmetry=int(g.centrosymmetry), conductivity=float(g.conductivity),
            mass_per_atom=float(g.mass_per_atom), distance=pred['distance'][i]))
    return polymorphs, notes


def group_by_material(polymorphs):
    groups = {}
    for p in polymorphs:
        groups.setdefault(p.material, []).append(p)
    return groups


def _max_distance(a, b):
    values = np.concatenate([a.distance, b.distance]) if len(a.distance) + len(b.distance) else np.array([])
    values = values[np.isfinite(values)]
    return float(values.max()) if len(values) else np.nan


def _transition_between(a, b, group, t_min, t_max, dft_uncertainty, uncertainty_mode):
    d_e0 = b.e0 - a.e0
    d_coef = b.coef - a.coef
    roots = fe.transition_temperatures(d_e0, d_coef, t_min, t_max)
    if len(roots) == 0:
        return None
    t_c = float(roots[0])

    f_a, f_b = a.free_energy(t_c), b.free_energy(t_c)
    slope = float(fe.helmholtz_derivative(t_c, *d_coef))
    slope_el = float(fe.helmholtz_derivative(t_c, *(b.coef_el - a.coef_el)))

    # uncertainty from the ML free-energy error at the transition temperature
    idx = int(np.argmin(np.abs(a.temperatures - t_c)))
    u_ml = float(np.sqrt(a.uncertainty[idx] ** 2 + b.uncertainty[idx] ** 2)) if len(a.uncertainty) else np.nan
    unc_ml = fe.transition_temperature_uncertainty(d_e0, d_coef, u_ml, t_c, uncertainty_mode) if np.isfinite(u_ml) else np.nan
    unc_dft = fe.transition_temperature_uncertainty(d_e0, d_coef, dft_uncertainty, t_c, uncertainty_mode)

    # which phase is stable just below the transition
    probe = max(t_c - 5.0, 1.0)
    low, high = (a, b) if a.free_energy(probe) < b.free_energy(probe) else (b, a)

    phases = (low.phase, high.phase)
    mit = set(phases) == {'metal', 'insulator'}
    centro = a.centrosymmetry + b.centrosymmetry
    ltc_change = None
    if centro == 1 and np.isfinite(low.conductivity) and np.isfinite(high.conductivity) and low.conductivity != 0:
        ltc_change = (high.conductivity - low.conductivity) / low.conductivity

    f_min = min(p.free_energy(t_c) for p in group)
    row = {
        'compound': a.material, 'polymorph-1': low.space_group, 'polymorph-2': high.space_group,
        'transition-temperature': t_c, 'all-transition-temperatures': [float(r) for r in roots],
        'entropy-change': fe.entropy_change_si(slope, a.mass_per_atom),
        'entropy-change-electronic': fe.entropy_change_si(slope_el, a.mass_per_atom),
        'entropy-change-meV-per-atom-K': abs(slope),
        'centrosymmetry': centro, 'conductivity-change': ltc_change, 'LTC-1': low.conductivity,
        'LTC-2': high.conductivity, 'uncert-ML': unc_ml, 'uncert-DFT': unc_dft,
        'label-1': low.label, 'label-2': high.label,
        'transition-class': f'{low.phase}-{high.phase}', 'MIT': bool(mit),
        'bandgap-1': low.bandgap, 'bandgap-2': high.bandgap, 'delta-bandgap': high.bandgap - low.bandgap,
        'electronic-term-included': bool(a.has_dos and b.has_dos),
        'extrapolation-distance': _max_distance(a, b),
    }
    stable = bool(np.isclose(a.free_energy(t_c), f_min, rtol=1e-6, atol=1e-3))
    return row, stable


def find_transitions(polymorphs, t_min, t_max, dft_uncertainty=4 * np.sqrt(2), uncertainty_mode='max',
                     verbose=False):
    """All transitions between polymorphs of the same material inside (t_min, t_max).

    ``stable`` transitions happen between phases that are the lowest in free energy at T_c; the others are
    ``metastable``. Both DataFrames include the ``transition-class`` / ``MIT`` columns.
    """
    stable, metastable = [], []
    for material, group in group_by_material(polymorphs).items():
        for k in range(len(group)):
            for m in range(k + 1, len(group)):
                found = _transition_between(group[k], group[m], group, t_min, t_max, dft_uncertainty, uncertainty_mode)
                if found is None:
                    continue
                row, is_stable = found
                if verbose:
                    print(f"{material}: {row['polymorph-1']} -> {row['polymorph-2']} ({row['transition-temperature']:.1f} K, "
                          f"{row['transition-class']})")
                (stable if is_stable else metastable).append(row)
    return pd.DataFrame(stable), pd.DataFrame(metastable)


def metal_insulator(df):
    """Rows of a transitions DataFrame that are metal-insulator transitions."""
    return df[df['MIT']] if len(df) and 'MIT' in df else df
