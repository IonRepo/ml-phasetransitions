# ml-phasetransitions

GNN prediction of vibrational free energies F_v(T) + DFT electronic (DOS) free energy -> phase transitions
between polymorphs, including metal-insulator transitions. See `CHANGES.md` for what changed and the bugs fixed.

    libraries/electronic.py    F_el(T) from the DOS; band-gap classification
    libraries/free_energy.py   quartic smoothing, transition roots and their uncertainty
    libraries/transitions.py   polymorph records, pair search, stable/metastable, MIT labelling
    libraries/model.py         gated + attention GNN, checkpoints
    libraries/training.py      hold-out / k-fold / full / fine-tuning (also a CLI: python -m libraries.training)
    libraries/dataset.py       graphs database, standardizer, material-grouped splits
    libraries/pheasy.py        Pheasy reader (check `scan()` against the real layout)
    libraries/mp_query.py      Materials Project download (metals + DOS)
    data_query.ipynb / train-GNN.ipynb / predict-phase-transitions.ipynb
    tests/                     `pytest tests` (21 tests, ~15 s, CPU)

Native database layout: `root/<material>/<polymorph>/{POSCAR, EPA (eV/atom), bandgap (eV), dos, Fv (T, F in meV/atom), LTC}`.
`input/atomic_masses.dat` is optional (same format as before); without it element features come from pymatgen.

Quick start: `python -c "from tests.synthetic import make_database; make_database('toy-db')"`, then set
`DATA_SOURCE='native'`, `DATABASE='toy-db'` in `train-GNN.ipynb`.
