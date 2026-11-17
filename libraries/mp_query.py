"""Materials Project download -> native database layout.

    root/<formula>/<spacegroup>_<material_id>/{POSCAR, EPA, bandgap, is_metal, dos}

Differences from the previous ``data_query.ipynb``
* metals are kept (``min_gap=None``); the old filter ``band_gap > 0.5`` made metal-insulator transitions impossible
* the DOS is downloaded and stored in a compact JSON (energies, spin-summed densities, E_Fermi, number of atoms)
* the polymorph folder includes the material id, so two polymorphs with the same space group no longer overwrite
  each other
* nothing is swallowed by a bare ``except: pass``: failures are counted by reason and returned
* ``POSCAR`` is written with pymatgen (the old code called an undefined ``composition_concentration``, so every
  write failed silently)
"""

import json
import os
import re
from collections import Counter

import numpy as np
from pymatgen.io.vasp import Poscar

SUMMARY_FIELDS = ['material_id', 'formula_pretty', 'structure', 'symmetry', 'energy_per_atom', 'band_gap', 'is_metal']


def sanitize(name):
    return re.sub(r'[^A-Za-z0-9_.+-]', '-', str(name))


def compact_dos(dos, natoms):
    """JSON-able dict from a pymatgen Dos/CompleteDos (spin channels summed)."""
    densities = np.sum([np.asarray(v, dtype=float) for v in dos.densities.values()], axis=0)
    structure = getattr(dos, 'structure', None)
    return {'energies': np.asarray(dos.energies, dtype=float).tolist(), 'densities': densities.tolist(),
            'efermi': float(dos.efermi), 'natoms': int(len(structure)) if structure is not None else int(natoms)}


def write_entry(root, doc, dos=None, overwrite=False):
    """Write one Materials Project summary document (and optional DOS) to ``root``. Returns the folder or None if it
    already existed."""
    folder = os.path.join(root, sanitize(doc.formula_pretty), f'{sanitize(doc.symmetry.symbol)}_{doc.material_id}')
    if os.path.exists(os.path.join(folder, 'POSCAR')) and not overwrite:
        return None
    os.makedirs(folder, exist_ok=True)
    Poscar(doc.structure, comment=f'{doc.formula_pretty} ({doc.material_id})').write_file(os.path.join(folder, 'POSCAR'))
    np.savetxt(os.path.join(folder, 'EPA'), [doc.energy_per_atom])
    np.savetxt(os.path.join(folder, 'bandgap'), [doc.band_gap])
    np.savetxt(os.path.join(folder, 'is_metal'), [int(bool(getattr(doc, 'is_metal', doc.band_gap < 0.1)))], fmt='%d')
    if dos is not None:
        with open(os.path.join(folder, 'dos'), 'w') as f:
            json.dump(compact_dos(dos, len(doc.structure)), f)
    return folder


def download(api_key, elements, root, exclude_elements=(), min_gap=None, with_dos=True, max_docs_per_element=None,
             verbose=True):
    """Download every material containing each element in ``elements``.

    Args:
        min_gap: keep only materials with ``band_gap > min_gap`` (eV). None keeps everything, metals included.
        with_dos: also fetch the electronic DOS (one extra request per material; materials without one are
            written without a ``dos`` file and counted in the report).

    Returns:
        Counter with the number of written entries and failures by reason.
    """
    from mp_api.client import MPRester  # imported here so the rest of the package works without mp-api

    report = Counter()
    seen = set()
    os.makedirs(root, exist_ok=True)
    with MPRester(api_key) as mpr:
        for element in elements:
            docs = mpr.materials.summary.search(elements=[element], exclude_elements=list(exclude_elements),
                                                fields=SUMMARY_FIELDS)
            if max_docs_per_element:
                docs = docs[:max_docs_per_element]
            if verbose:
                print(f'{element}: {len(docs)} materials')
            for doc in docs:
                key = str(doc.material_id)
                if key in seen:
                    continue  # compounds are returned once per constituent element
                seen.add(key)
                try:
                    if doc.band_gap is None:
                        report['skipped: no band gap'] += 1
                        continue
                    if min_gap is not None and doc.band_gap <= min_gap:
                        report['skipped: gap below min_gap'] += 1
                        continue
                    dos = None
                    if with_dos:
                        try:
                            dos = mpr.get_dos_by_material_id(doc.material_id)
                        except Exception as e:
                            report[f'no DOS ({type(e).__name__})'] += 1
                    folder = write_entry(root, doc, dos)
                    report['written' if folder else 'already present'] += 1
                    if dos is not None and folder:
                        report['written with DOS'] += 1
                except Exception as e:
                    report[f'failed: {type(e).__name__}: {e}'] += 1
    return report
