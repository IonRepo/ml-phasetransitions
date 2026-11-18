"""Ultrastable polymorphs: the thermodynamic ground state over the whole temperature range, by a margin.

For every compound with several polymorphs the smoothed free energies  E0 + F_v + F_el  (the ``coefficients`` fitted
in the prediction notebook) are evaluated on a fine temperature grid. A polymorph is *ultrastable* if

    1. it has the lowest free energy at every temperature of the range (no stable transition inside the range), and
    2. the smallest gap to the second-lowest polymorph over the range is larger than ``n_sigma`` times the combined ML
       uncertainty of the two free energies (and larger than ``min_margin`` meV/atom).

It is a statement about competition between the polymorphs present in the database, not about decomposition into
other compounds (use the energy above the hull for that), and not about kinetics. Compounds with a single polymorph
cannot be judged and are returned separately.
"""

import numpy as np

import libraries.model as clm

ULTRA_COLUMNS = ['compound', 'n-polymorphs', 'ultrastable', 'label', 'polymorph', 'label-low-T', 'label-high-T', 'margin',
                 'T-margin', 'uncert', 'significance', 'bandgap', 'LTC']
SINGLE_COLUMNS = ['compound', 'n-polymorphs', 'n-valid', 'label', 'polymorph', 'bandgap']


def free_energy_curves(polymorphs, temperatures):
    """(n_polymorphs, n_temperatures) array of E0 + smooth F(T) in meV/atom; NaN rows for polymorphs whose fit failed."""
    curves = []
    for data in polymorphs:
        coefficients = np.asarray(data.coefficients, dtype=float)[:, 0]
        if data.gs_energy is None or not np.all(np.isfinite(coefficients)):
            curves.append(np.full(len(temperatures), np.nan))
        else:
            curves.append(data.gs_energy + clm.Helmholtz_free_energy_function(temperatures, *coefficients))
    return np.array(curves)


def ultrastable_table(compound_groups, temperatures, n_sigma=2.0, min_margin=0.0):
    """Stability of every compound's polymorphs over ``temperatures`` (e.g. np.arange(Ti, Tf + 1, 1.0)).

    Args:
        compound_groups (dict):  compound -> list of graphs with ``coefficients``, ``gs_energy``, ``uncertainties``.
        temperatures    (array): fine temperature grid (K) covering the range of interest.
        n_sigma         (float): required margin in units of the combined ML uncertainty.
        min_margin      (float): required margin (meV/atom) regardless of the uncertainty.

    Returns:
        (rows, single): ``rows``, one dict per compound with at least two valid polymorphs; ``single``, one dict per
        compound without competitor (or whose competitors all failed to fit).
        Keys of ``rows``: compound, n-polymorphs, ultrastable, label, polymorph (space group), label-low-T,
        label-high-T, margin (meV/atom), T-margin (K), uncert (meV/atom), significance (margin / uncert).
    """
    temperatures = np.asarray(temperatures, dtype=float)
    rows, single = [], []
    for compound, polymorphs in compound_groups.items():
        curves = free_energy_curves(polymorphs, temperatures)
        valid = [k for k in range(len(polymorphs)) if np.all(np.isfinite(curves[k]))]
        if len(valid) < 2:
            single.append({'compound': compound, 'n-polymorphs': len(polymorphs), 'n-valid': len(valid),
                           'label': polymorphs[valid[0]].label if valid else '',
                           'polymorph': polymorphs[valid[0]].space_group if valid else '',
                           'bandgap': float(getattr(polymorphs[valid[0]], 'bandgap', np.nan)) if valid else np.nan})
            continue

        valid_curves = curves[valid]
        order = np.argsort(valid_curves, axis=0)                       # polymorph ranking at every temperature
        ground = order[0]
        runner_up = order[1]
        margin_T = np.take_along_axis(valid_curves, runner_up[None], 0)[0] - np.take_along_axis(valid_curves, ground[None], 0)[0]
        t_min = int(np.argmin(margin_T))

        constant = bool(np.all(ground == ground[0]))
        g, r = valid[ground[t_min]], valid[runner_up[t_min]]
        uncert = float(np.sqrt(np.max(np.abs(polymorphs[g].uncertainties)) ** 2 +
                               np.max(np.abs(polymorphs[r].uncertainties)) ** 2))
        margin = float(margin_T[t_min])
        significance = margin / uncert if uncert > 0 else np.inf
        best = polymorphs[valid[ground[0]]]
        rows.append({'compound': compound, 'n-polymorphs': len(polymorphs),
                     'ultrastable': constant and margin > max(n_sigma * uncert, min_margin),
                     'label': best.label, 'polymorph': best.space_group,
                     'label-low-T': polymorphs[valid[ground[0]]].label, 'label-high-T': polymorphs[valid[ground[-1]]].label,
                     'margin': margin, 'T-margin': float(temperatures[t_min]), 'uncert': uncert,
                     'significance': significance,
                     'bandgap': float(getattr(best, 'bandgap', np.nan)),
                     'LTC': float(getattr(best, 'conductivity', np.nan))})
    return rows, single
