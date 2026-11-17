import os

import numpy as np
import pytest
import torch
from pymatgen.core import Lattice, Structure
from torch_geometric.data import Batch

from libraries import dataset as ds
from libraries import graph as lg
from libraries import pheasy
from libraries import training as tr
from libraries.elements import load_element_table
from libraries.model import GNN, ModelConfig, load_checkpoint, save_checkpoint
from libraries.predict import predict_free_energies
from libraries.uncertainty import EmbeddingUncertainty
from tests.synthetic import make_database


@pytest.fixture(scope='module')
def db(tmp_path_factory):
    root = str(tmp_path_factory.mktemp('db'))
    make_database(root, n_materials=24, seed=1)
    graphs, report = ds.read_database(root)
    assert report['read'] == len(graphs) > 40
    return root, graphs


def test_graph_edges_are_not_duplicated():
    s = Structure(Lattice.cubic(4.0), ['Na', 'Cl'], [[0, 0, 0], [0.5, 0.5, 0.5]])
    x, ei, ea = lg.structure_to_graph_tensors(s, load_element_table(None), cutoff=5.0, max_neighbors=None)
    pairs = set()
    for a, b, d in zip(ei[0].tolist(), ei[1].tolist(), ea[:, 0].tolist()):
        pairs.add((a, b, round(d, 6)))
    n_unique_images = len(ei[0])
    # 1 NaCl cell: each atom has 8 neighbours at 3.46 A and 6 at 4.0 A -> 14 per atom within 5 A
    assert n_unique_images == 2 * (8 + 6)
    # max_neighbors keeps the closest ones
    _, ei2, ea2 = lg.structure_to_graph_tensors(s, load_element_table(None), cutoff=5.0, max_neighbors=8)
    assert ei2.shape[1] == 16 and float(ea2.max()) < 3.5


def test_element_table_is_finite_for_common_elements():
    table = load_element_table(None)
    for el in ['Li', 'O', 'Si', 'Zn', 'I']:
        assert np.all(np.isfinite(table[el]))


def test_standardizer_statistics(db):
    _, graphs = db
    std = ds.Standardizer.fit(graphs)
    nodes = torch.cat([g.x for g in std.standardize_graphs(graphs)])
    assert torch.allclose(nodes.mean(0), torch.zeros(4), atol=1e-4)
    assert torch.allclose(nodes.std(0, unbiased=False), torch.ones(4), atol=1e-3)
    y = np.concatenate([std.standardize_target(g.fv.numpy()) for g in graphs])
    assert abs(y.mean()) < 1e-6 and abs(y.std() - 1) < 1e-6
    # originals untouched
    assert not torch.allclose(graphs[0].x, std.standardize_graphs(graphs)[0].x)
    # round trip
    std2 = ds.Standardizer.from_dict(std.to_dict())
    np.testing.assert_allclose(std2.destandardize_target(std.standardize_target([1.0, 2.0])), [1.0, 2.0], rtol=1e-5)


def test_splits_do_not_leak(db):
    _, graphs = db
    labels = ds.group_labels(graphs)
    folds = ds.kfold_split(labels, 4, seed=3)
    seen = np.concatenate([f['val'] for f in folds])
    assert sorted(seen) == sorted(np.unique(labels))
    for f in folds:
        assert not set(f['train']) & set(f['val'])
    h = ds.holdout_split(labels, 0.6, 0.2, 0.2, seed=1)
    assert len(set(h['train']) | set(h['val']) | set(h['test'])) == len(np.unique(labels))
    assert not (set(h['train']) & set(h['test'])) and not (set(h['val']) & set(h['test']))
    # the old split_dataset dropped the validation set when test_ratio == 0
    h0 = ds.holdout_split(labels, 0.8, 0.2, 0.0)
    assert len(h0['val']) > 0 and len(h0['test']) == 0


def test_temperature_is_last_node_feature(db):
    _, graphs = db
    std = ds.Standardizer.fit(graphs)
    d = ds.TemperatureGraphDataset(std.standardize_graphs(graphs), std)
    item = d[0]
    assert item.x.shape[1] == 5
    t = std.standardize_temperature(float(graphs[0].temperatures[0]))
    assert float(item.x[0, -1]) == pytest.approx(t, abs=1e-5)


def test_model_shapes_and_checkpoint(db, tmp_path):
    _, graphs = db
    std = ds.Standardizer.fit(graphs)
    d = ds.TemperatureGraphDataset(std.standardize_graphs(graphs[:6]), std)
    batch = Batch.from_data_list([d[i] for i in range(0, 30, 5)])
    model = GNN(ModelConfig(in_features=5, hidden=32, n_layers=2, heads=4)).eval()
    out, emb = model(batch, return_embedding=True)
    assert out.shape == (batch.num_graphs,) and emb.shape == (batch.num_graphs, 32)
    save_checkpoint(model, tmp_path / 'm.pt', std)
    model2, std_dict, _ = load_checkpoint(str(tmp_path / 'm.pt'))
    assert torch.allclose(model2(batch), out, atol=1e-6)
    assert std_dict['t_mean'] == pytest.approx(std.t_mean)


def test_model_is_invariant_to_atom_order(db):
    _, graphs = db
    std = ds.Standardizer.fit(graphs)
    g = std.standardize_graphs([graphs[0]])
    d = ds.TemperatureGraphDataset(g, std)
    a = d[0]
    perm = torch.randperm(a.x.size(0))
    inv = torch.argsort(perm)
    b = a.clone()
    b.x = a.x[perm]
    b.edge_index = inv[a.edge_index]
    model = GNN(ModelConfig(in_features=5, hidden=32, n_layers=2)).eval()
    with torch.no_grad():
        assert torch.allclose(model(Batch.from_data_list([a])), model(Batch.from_data_list([b])), atol=1e-5)


def test_uncertainty_estimator_roundtrip(tmp_path):
    rng = np.random.RandomState(0)
    emb = rng.normal(size=(200, 4))
    err = np.abs(emb[:, 0]) + 0.1
    u = EmbeddingUncertainty(emb, err, k=8)
    # duplicate embeddings must not break it (the old RBFInterpolator(smoothing=0) did)
    u2 = EmbeddingUncertainty(np.vstack([emb, emb]), np.concatenate([err, err]), k=8)
    value, dist = u2(emb[:5])
    assert np.all(value > 0) and np.all(dist < 1e-6)
    u.save(tmp_path / 'u.npz')
    v2, _ = EmbeddingUncertainty.load(tmp_path / 'u.npz')(emb[:5])
    np.testing.assert_allclose(u(emb[:5])[0], v2)
    far, dfar = u(np.full((1, 4), 50.0))
    assert dfar[0] > 40


def test_end_to_end_training_and_prediction(db, tmp_path):
    _, graphs = db
    out = str(tmp_path / 'run')
    cfg = tr.TrainConfig(out_dir=out, graphs_path=os.path.join(out, 'd.pt'), mode='kfold', n_splits=3, epochs=15,
                         batch_size=64, hidden=32, n_layers=2, patience=8, warmup_epochs=1)
    tr.run(cfg, graphs)
    summary, _, oof = tr.aggregate(cfg)
    assert summary['n_oof_samples'] == sum(len(g.temperatures) for g in graphs)  # every sample validated once
    ref = tr.finalize(cfg, graphs, verbose=False)
    model, std_dict, _ = load_checkpoint(os.path.join(out, 'deployed_model.pt'))
    std = ds.Standardizer.from_dict(std_dict)
    t = np.array([200., 400.])
    pred = predict_free_energies(graphs[:5], t, model, std, ref)
    assert pred['fv'].shape == (5, 2) and np.all(pred['uncertainty'] > 0)
    # inputs not mutated by prediction
    assert graphs[0].x.mean().abs() > 0.5


def test_split_signature_guards_against_stale_files(db, tmp_path):
    _, graphs = db
    cfg = tr.TrainConfig(out_dir=str(tmp_path), mode='kfold', n_splits=3)
    tr.load_or_make_splits(cfg, graphs)
    cfg2 = tr.TrainConfig(out_dir=str(tmp_path), mode='kfold', n_splits=4)
    with pytest.raises(RuntimeError):
        tr.load_or_make_splits(cfg2, graphs)


def test_finetuning_uses_pretrained_standardizer(db, tmp_path):
    _, graphs = db
    out = str(tmp_path / 'pre')
    cfg = tr.TrainConfig(out_dir=out, mode='holdout', epochs=3, batch_size=64, hidden=32, n_layers=2)
    tr.run(cfg, graphs)
    ft = tr.TrainConfig(out_dir=str(tmp_path / 'ft'), mode='holdout', epochs=3, batch_size=64,
                        pretrained=os.path.join(out, 'holdout_model.pt'), freeze_epochs=1)
    tr.run(ft, graphs)
    _, s1, _ = load_checkpoint(os.path.join(out, 'holdout_model.pt'))
    _, s2, _ = load_checkpoint(os.path.join(str(tmp_path / 'ft'), 'holdout_model.pt'))
    assert s1 == s2


def test_pheasy_reader_from_phonon_dos_and_table(tmp_path):
    from pymatgen.io.vasp import Poscar
    root = tmp_path / 'ph'
    s = Structure(Lattice.cubic(4.2), ['Na', 'Cl'], [[0, 0, 0], [0.5, 0.5, 0.5]])
    for name, mode in [('a', 'table'), ('b', 'dos'), ('c', 'unstable')]:
        d = root / 'NaCl' / name
        d.mkdir(parents=True)
        Poscar(s).write_file(str(d / 'POSCAR'))
        T = np.arange(100, 701, 100.)
        if mode == 'table':
            np.savetxt(d / 'free_energy.dat', np.column_stack([T, -0.05 * T]), header='T F')
        else:
            nu = np.linspace(-1, 8, 900)
            g = np.exp(-((nu - 4) / 1.0) ** 2) + (0.5 * np.exp(-((nu + 0.5) / 0.2) ** 2) if mode == 'unstable' else 0)
            np.savetxt(d / 'phonon_dos.dat', np.column_stack([nu, g]))
    pheasy.scan(str(root))
    graphs, report = pheasy.read_pheasy(str(root), temperatures=np.arange(100, 701, 100.))
    assert len(graphs) == 2 and report['dynamically unstable (imaginary modes)'] == 1
    assert all(len(g.fv) == 7 for g in graphs)
