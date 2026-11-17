"""Synthetic database in the native folder layout, for smoke tests and for trying the notebooks without real data.

    root/<formula>/<polymorph>/{POSCAR, EPA, bandgap, dos, free_energy.dat}

Fv(T) is an Einstein-model vibrational free energy whose frequency depends on the mean atomic mass and the volume per
atom, so a graph network can learn it. Some polymorphs are metals (flat-band DOS, bandgap 0) and some insulators.
"""

import json
import os

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Poscar

K_B_MEV = 8.617333262e-2  # meV / K

ELEMENTS = ['Li', 'Na', 'K', 'Mg', 'Ca', 'Al', 'Si', 'Ge', 'Sn', 'Ga', 'In', 'O', 'S', 'Se', 'Cl', 'Br', 'I', 'Zn', 'Cd']


def einstein_fv(temperatures, mean_mass, volume_per_atom):
    hw = 40.0 * np.sqrt(30.0 / mean_mass) * np.sqrt(20.0 / volume_per_atom)  # meV
    t = np.asarray(temperatures, dtype=float)
    return 3 * (0.5 * hw + K_B_MEV * t * np.log1p(-np.exp(-hw / (K_B_MEV * t))))


def random_structure(rng, species, n_per_species=2, a0=5.0):
    n = len(species) * n_per_species
    for _ in range(200):
        lattice = Lattice.from_parameters(*(a0 * rng.uniform(0.85, 1.2, 3)), *rng.uniform(85, 95, 3))
        coords = rng.uniform(0, 1, (n, 3))
        sp = [s for s in species for _ in range(n_per_species)]
        s = Structure(lattice, sp, coords)
        dm = s.distance_matrix + np.eye(n) * 10
        if dm.min() > 1.6:
            return s
    raise RuntimeError('could not place atoms')


def dos_json(rng, metal, natoms, gap):
    e = np.linspace(-15, 15, 3001)
    centre = rng.uniform(-6, -3)
    valence = natoms * 2.0 * np.exp(-((e - centre) / 3.0) ** 2) / (3.0 * np.sqrt(np.pi))
    if metal:
        # one band straddling E_F with a smooth slope -> finite n(E_F)
        dos = valence + natoms * 0.8 * np.exp(-((e - rng.uniform(-0.3, 0.3)) / 2.5) ** 2)
        efermi = 0.0
    else:
        cond = natoms * 1.2 * np.exp(-((e - (gap + 3)) / 2.5) ** 2)
        dos = np.where(e < 0, valence, 0) + np.where(e > gap, cond, 0)
        efermi = 0.0
    return {'energies': e.tolist(), 'densities': dos.tolist(), 'efermi': efermi, 'natoms': natoms}


def make_database(root, n_materials=40, n_polymorphs=(2, 3), temperatures=np.arange(100, 701, 100), seed=0):
    rng = np.random.RandomState(seed)
    os.makedirs(root, exist_ok=True)
    for m in range(n_materials):
        species = list(rng.choice(ELEMENTS, size=rng.randint(1, 3), replace=False))
        formula = ''.join(species) + str(m)
        base_e0 = rng.uniform(-6, -3)
        for p in range(rng.randint(*n_polymorphs) if n_polymorphs[0] != n_polymorphs[1] else n_polymorphs[0]):
            s = random_structure(rng, species)
            folder = os.path.join(root, formula, f'P{p}')
            os.makedirs(folder, exist_ok=True)
            Poscar(s).write_file(os.path.join(folder, 'POSCAR'))
            mean_mass = s.composition.weight / len(s)
            fv = einstein_fv(temperatures, mean_mass, s.volume / len(s))
            np.savetxt(os.path.join(folder, 'free_energy.dat'), np.column_stack([temperatures, fv]))
            np.savetxt(os.path.join(folder, 'EPA'), [base_e0 + rng.normal(0, 0.003) ])
            metal = rng.rand() < 0.35
            gap = 0.0 if metal else float(rng.uniform(0.8, 4.0))
            np.savetxt(os.path.join(folder, 'bandgap'), [gap])
            with open(os.path.join(folder, 'dos'), 'w') as f:
                json.dump(dos_json(rng, metal, len(s), gap), f)
    return root
