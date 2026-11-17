"""Unified training: hold-out split, grouped k-fold cross-validation, and full-data refit share one code path.

Every run is a *split* with three disjoint sets of material groups:
    train - weights are fitted here
    val   - early stopping only
    test  - reported metrics (the held-out fold in k-fold mode)

    mode='holdout'  one split: train / val / test by ratio
    mode='kfold'    n_splits splits; test = fold k, val = a small slice carved out of the other folds, so the fold
                    that is reported never influences early stopping (the old k-fold notebook used it for both)
    mode='full'     one split with no test set: the deployable model fitted on everything (val is a small slice)

Fine-tuning on a custom database is the same call with ``pretrained=<checkpoint>``: architecture and standardizer
come from the checkpoint, the learning rate defaults to lr / 10, and the encoder can be frozen for a few epochs.

Typical use (see ``train-GNN.ipynb``):
    cfg = TrainConfig(mode='kfold', n_splits=5)
    run(cfg, graphs)              # every fold, optionally each in a fresh subprocess (isolate=True)
    summary = aggregate(cfg)      # CV metrics + out-of-fold predictions
    finalize(cfg, graphs)         # deployable model + uncertainty reference
"""

import argparse
import copy
import json
import math
import os
import random
import subprocess
import sys
import warnings
from dataclasses import dataclass, asdict, field

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from libraries import dataset as ds
from libraries.model import ModelConfig, build_model, save_checkpoint, load_checkpoint
from libraries.predict import collect, get_device
from libraries.uncertainty import EmbeddingUncertainty


@dataclass
class TrainConfig:
    out_dir: str = 'model'
    graphs_path: str = 'model/dataset.pt'
    # splitting
    mode: str = 'kfold'                 # 'holdout' | 'kfold' | 'full'
    n_splits: int = 10
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    inner_val_ratio: float = 0.1        # kfold / full: fraction of training groups used for early stopping
    seed: int = 42
    # architecture
    hidden: int = 128
    n_layers: int = 3
    heads: int = 4
    n_rbf: int = 32
    cutoff: float = 6.0
    dropout: float = 0.1
    embedding_dim: int = 32
    # optimisation
    epochs: int = 200
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    loss: str = 'huber'                 # 'huber' | 'mse'
    huber_delta: float = 1.0            # in standardized units
    patience: int = 30
    min_delta: float = 0.0              # meV / atom
    num_workers: int = 0
    # fine-tuning
    pretrained: str = ''                # path to a checkpoint ('' = train from scratch)
    finetune_lr: float = 0.0            # 0 -> lr / 10
    freeze_epochs: int = 0              # epochs with only the readout trainable
    # uncertainty
    uncertainty_k: int = 16
    uncertainty_statistic: str = 'rmse'

    def to_json(self, path):
        ds.save_json(asdict(self), path)

    @classmethod
    def from_json(cls, path):
        return cls(**ds.load_json(path))


# ----------------------------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _path(cfg, name, suffix):
    return os.path.join(cfg.out_dir, f'{name}_{suffix}')


def make_splits(cfg, labels):
    """List of dicts {name, train, val, test} with arrays of group labels."""
    unique = np.unique(labels)
    if cfg.mode == 'holdout':
        s = ds.holdout_split(labels, cfg.train_ratio, cfg.val_ratio, cfg.test_ratio, cfg.seed)
        return [{'name': 'holdout', **s}]
    if cfg.mode == 'kfold':
        splits = []
        for k, fold in enumerate(ds.kfold_split(labels, cfg.n_splits, cfg.seed)):
            if cfg.inner_val_ratio > 0:
                train, val = ds.carve_validation(fold['train'], cfg.inner_val_ratio, cfg.seed + k)
            else:
                warnings.warn('inner_val_ratio=0: the held-out fold is also used for early stopping')
                train, val = fold['train'], fold['val']
            splits.append({'name': f'fold{k}', 'train': train, 'val': val, 'test': fold['val']})
        return splits
    if cfg.mode == 'full':
        train, val = ds.carve_validation(unique, max(cfg.inner_val_ratio, 0.05), cfg.seed)
        return [{'name': 'full', 'train': train, 'val': val, 'test': np.array([], dtype=unique.dtype)}]
    raise ValueError(f"mode must be 'holdout', 'kfold' or 'full', got {cfg.mode!r}")


def load_or_make_splits(cfg, graphs):
    """Splits are computed once and stored, so every later session (or subprocess) reuses the same assignment."""
    labels = ds.group_labels(graphs)
    signature = {'mode': cfg.mode, 'n_splits': cfg.n_splits, 'seed': cfg.seed, 'ratios': [
        cfg.train_ratio, cfg.val_ratio, cfg.test_ratio], 'inner_val_ratio': cfg.inner_val_ratio,
        'n_groups': int(len(np.unique(labels)))}
    path = os.path.join(cfg.out_dir, f'splits_{cfg.mode}.json')
    if os.path.exists(path):
        saved = ds.load_json(path)
        if saved['signature'] != signature:
            raise RuntimeError(f'{path} was made with a different configuration/dataset:\n  saved   {saved["signature"]}'
                               f'\n  current {signature}\nDelete it (and the old fold results) or use another out_dir.')
        return [{k: (np.array(v) if k != 'name' else v) for k, v in s.items()} for s in saved['splits']]
    splits = make_splits(cfg, labels)
    os.makedirs(cfg.out_dir, exist_ok=True)
    ds.save_json({'signature': signature, 'splits': splits}, path)
    return splits


class EarlyStopping:
    """Keeps the best weights in memory. Improvement means val < best - min_delta; the best score is never replaced
    by a worse value (the previous version moved it to any value within ``delta``)."""

    def __init__(self, patience, min_delta=0.0):
        self.patience, self.min_delta = patience, min_delta
        self.best = np.inf
        self.best_epoch = -1
        self.counter = 0
        self.best_state = None

    def step(self, value, model, epoch):
        if value < self.best - self.min_delta:
            self.best, self.best_epoch, self.counter = value, epoch, 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def _lr_factor(epoch, cfg):
    if epoch < cfg.warmup_epochs:
        return (epoch + 1) / cfg.warmup_epochs
    progress = (epoch - cfg.warmup_epochs) / max(1, cfg.epochs - cfg.warmup_epochs)
    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))


def _make_loader(dataset, cfg, shuffle):
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle, num_workers=cfg.num_workers,
                      pin_memory=torch.cuda.is_available(), drop_last=False)


def train_epoch(model, loader, optimizer, criterion, device, grad_clip):
    model.train()
    total, count = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(batch), batch.y)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item() * batch.num_graphs
        count += batch.num_graphs
    return total / max(count, 1)


@torch.no_grad()
def evaluate(model, dataset, std, cfg, device):
    """MAE / RMSE in meV / atom plus predictions. (The old code reported RMSE under the name MAE.)"""
    if len(dataset) == 0:
        return {'mae': np.nan, 'rmse': np.nan, 'n': 0}, None
    pred, _, gidx, temps = collect(model, dataset, cfg.batch_size, device, cfg.num_workers)
    y_pred = std.destandardize_target(pred)
    y_true = np.array([dataset.targets[i][j] for i, j in dataset.index])
    err = y_pred - y_true
    uid = [f'{dataset.graphs[i].label}@{dataset.temps[i][j]:g}' for i, j in dataset.index]
    metrics = {'mae': float(np.mean(np.abs(err))), 'rmse': float(np.sqrt(np.mean(err**2))), 'n': int(len(err))}
    return metrics, {'uid': np.array(uid), 'y_true': y_true, 'y_pred': y_pred, 'temperature': temps.ravel()}


# ----------------------------------------------------------------------------------------------------------------------
# one split
# ----------------------------------------------------------------------------------------------------------------------

def train_split(cfg, graphs, split, split_index=0, verbose=True):
    """Train on one split; saves ``<name>_model.pt``, ``<name>_result.json`` and ``<name>_pred.npz``."""
    device = get_device()
    name = split['name']
    set_seed(cfg.seed + split_index)
    os.makedirs(cfg.out_dir, exist_ok=True)

    sets = {k: [g for g in ds.select_groups(graphs, split[k]) if hasattr(g, 'fv')] for k in ('train', 'val', 'test')}
    if not sets['train']:
        raise RuntimeError(f'{name}: empty training set')

    if cfg.pretrained:
        model, std_dict, _ = load_checkpoint(cfg.pretrained, device, mode='train')
        std = ds.Standardizer.from_dict(std_dict)
        lr = cfg.finetune_lr or cfg.lr / 10
    else:
        std = ds.Standardizer.fit(sets['train'])
        model = build_model(ModelConfig(in_features=sets['train'][0].x.size(1) + 1, hidden=cfg.hidden,
                                        n_layers=cfg.n_layers, heads=cfg.heads, n_rbf=cfg.n_rbf, cutoff=cfg.cutoff,
                                        dropout=cfg.dropout, embedding_dim=cfg.embedding_dim), device)
        lr = cfg.lr

    data = {k: ds.TemperatureGraphDataset(std.standardize_graphs(v), std) for k, v in sets.items()}
    train_loader = _make_loader(data['train'], cfg, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: _lr_factor(e, cfg))
    criterion = (torch.nn.HuberLoss(delta=cfg.huber_delta) if cfg.loss == 'huber' else torch.nn.MSELoss())
    stopper = EarlyStopping(cfg.patience, cfg.min_delta)

    readout = ('to_embedding', 'out')
    history = {'train_loss': [], 'val_mae': []}
    for epoch in range(cfg.epochs):
        frozen = epoch < cfg.freeze_epochs
        for pname, p in model.named_parameters():
            p.requires_grad = (not frozen) or pname.startswith(readout)
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, cfg.grad_clip)
        scheduler.step()
        monitor_set = data['val'] if len(data['val']) else data['train']
        val_metrics, _ = evaluate(model, monitor_set, std, cfg, device)
        history['train_loss'].append(train_loss)
        history['val_mae'].append(val_metrics['mae'])
        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            print(f'[{name}] epoch {epoch + 1:4d}  train loss {train_loss:.4f}  val MAE {val_metrics["mae"]:.3f} meV/atom')
        if stopper.step(val_metrics['mae'], model, epoch):
            if verbose:
                print(f'[{name}] early stopping at epoch {epoch + 1}, best epoch {stopper.best_epoch + 1}')
            break
    stopper.restore(model)

    metrics, preds = {}, {}
    for key in ('train', 'val', 'test'):
        metrics[key], preds[key] = evaluate(model, data[key], std, cfg, device)
    if verbose:
        print(f'[{name}] MAE (meV/atom): ' + ', '.join(f'{k} {metrics[k]["mae"]:.3f}' for k in metrics))

    save_checkpoint(model, _path(cfg, name, 'model.pt'), std, {'split': name, 'best_epoch': stopper.best_epoch})
    result = {'name': name, 'metrics': metrics, 'history': history, 'best_epoch': stopper.best_epoch,
              'n_train_groups': len(split['train']), 'n_val_groups': len(split['val']),
              'n_test_groups': len(split['test']), 'config': asdict(cfg)}
    ds.save_json(result, _path(cfg, name, 'result.json'))

    rows = {}
    for key in ('val', 'test'):
        if preds[key] is not None:
            rows[f'{key}_uid'] = preds[key]['uid']
            rows[f'{key}_true'] = preds[key]['y_true']
            rows[f'{key}_pred'] = preds[key]['y_pred']
            rows[f'{key}_T'] = preds[key]['temperature']
    if preds['train'] is not None:  # subsample for parity plots
        rng = np.random.RandomState(0)
        sel = rng.choice(len(preds['train']['y_true']), min(5000, len(preds['train']['y_true'])), replace=False)
        rows['train_true'], rows['train_pred'] = preds['train']['y_true'][sel], preds['train']['y_pred'][sel]
    np.savez_compressed(_path(cfg, name, 'pred.npz'), **rows)

    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


# ----------------------------------------------------------------------------------------------------------------------
# all splits
# ----------------------------------------------------------------------------------------------------------------------

def run(cfg, graphs=None, fold=None, isolate=False, config_path=None):
    """Train every split of ``cfg.mode`` (or only split number ``fold``).

    isolate=True runs each split in a fresh Python process (nothing - CUDA memory, RNG state, leftover objects - can
    carry over). That subprocess re-reads ``cfg.graphs_path`` so the graphs must be saved there.
    """
    os.makedirs(cfg.out_dir, exist_ok=True)
    if isolate:
        config_path = config_path or os.path.join(cfg.out_dir, 'train_config.json')
        cfg.to_json(config_path)
        graphs = ds.load_graphs(cfg.graphs_path)
        splits = load_or_make_splits(cfg, graphs)
        del graphs
        indices = range(len(splits)) if fold is None else [fold]
        for k in indices:
            subprocess.run([sys.executable, '-m', 'libraries.training', '--config', config_path, '--fold', str(k)],
                           check=True)
        return

    graphs = graphs if graphs is not None else ds.load_graphs(cfg.graphs_path)
    splits = load_or_make_splits(cfg, graphs)
    indices = range(len(splits)) if fold is None else [fold]
    return [train_split(cfg, graphs, splits[k], split_index=k) for k in indices]


def aggregate(cfg):
    """Read every split's saved result: mean/std of the test metric and out-of-fold predictions."""
    n = cfg.n_splits if cfg.mode == 'kfold' else 1
    names = [f'fold{k}' for k in range(n)] if cfg.mode == 'kfold' else [cfg.mode]
    missing = [nm for nm in names if not os.path.exists(_path(cfg, nm, 'result.json'))]
    if missing:
        raise RuntimeError(f'Missing results for {missing}; train them first')
    results = [ds.load_json(_path(cfg, nm, 'result.json')) for nm in names]
    mae = np.array([r['metrics']['test']['mae'] for r in results])
    rmse = np.array([r['metrics']['test']['rmse'] for r in results])
    oof = {k: [] for k in ('uid', 'true', 'pred', 'T')}
    for nm in names:
        with np.load(_path(cfg, nm, 'pred.npz')) as f:
            for k in oof:
                oof[k].append(f[f'test_{k}'])
    oof = {k: np.concatenate(v) for k, v in oof.items()}
    summary = {'mode': cfg.mode, 'n_splits': n, 'test_mae': mae.tolist(), 'test_rmse': rmse.tolist(),
               'mean_mae': float(mae.mean()), 'std_mae': float(mae.std()), 'mean_rmse': float(rmse.mean()),
               'n_oof_samples': int(len(oof['true']))}
    ds.save_json(summary, os.path.join(cfg.out_dir, 'cv_summary.json'))
    np.savez_compressed(os.path.join(cfg.out_dir, 'oof_predictions.npz'), **oof)
    return summary, results, oof


def finalize(cfg, graphs=None, verbose=True):
    """Make the deployable model and the uncertainty reference.

    kfold:   refit on all materials (mode='full'); the uncertainty reference uses the *out-of-fold* error of every
             (polymorph, T) sample (so errors are honest) with embeddings from the refitted model.
    holdout: the holdout model is deployed; the reference is made of its val + test samples.
    Writes ``deployed_model.pt`` (architecture + weights + standardizer) and ``uncertainty_reference.npz``.
    """
    graphs = graphs if graphs is not None else ds.load_graphs(cfg.graphs_path)
    device = get_device()

    if cfg.mode == 'kfold':
        final_cfg = TrainConfig(**{**asdict(cfg), 'mode': 'full'})
        splits = load_or_make_splits(final_cfg, graphs)
        train_split(final_cfg, graphs, splits[0], split_index=10_000, verbose=verbose)
        model_path, name = _path(cfg, 'full', 'model.pt'), 'full'
        _, _, oof = aggregate(cfg)
        errors = dict(zip(oof['uid'].tolist(), np.abs(oof['pred'] - oof['true'])))
    elif cfg.mode == 'holdout':
        model_path = _path(cfg, 'holdout', 'model.pt')
        with np.load(_path(cfg, 'holdout', 'pred.npz')) as f:
            errors = {}
            for key in ('val', 'test'):
                errors.update(dict(zip(f[f'{key}_uid'].tolist(), np.abs(f[f'{key}_pred'] - f[f'{key}_true']))))
    else:
        raise ValueError("finalize() needs mode 'kfold' or 'holdout'")

    model, std_dict, extra = load_checkpoint(model_path, device)
    std = ds.Standardizer.from_dict(std_dict)
    with_targets = [g for g in graphs if hasattr(g, 'fv')]
    dataset = ds.TemperatureGraphDataset(std.standardize_graphs(with_targets), std)
    _, emb, _, _ = collect(model, dataset, cfg.batch_size, device, cfg.num_workers)
    uids = [f'{dataset.graphs[i].label}@{dataset.temps[i][j]:g}' for i, j in dataset.index]
    keep = np.array([u in errors for u in uids])
    if not keep.any():
        raise RuntimeError('no samples with out-of-sample errors for the uncertainty reference')
    reference = EmbeddingUncertainty(emb[keep], np.array([errors[u] for u, k in zip(uids, keep) if k]),
                                     cfg.uncertainty_k, cfg.uncertainty_statistic)
    reference.save(os.path.join(cfg.out_dir, 'uncertainty_reference.npz'))
    save_checkpoint(model, os.path.join(cfg.out_dir, 'deployed_model.pt'), std, extra)
    ds.save_json(std.to_dict(), os.path.join(cfg.out_dir, 'standardizer.json'))
    if verbose:
        print(f'Deployed model: {model_path} -> {cfg.out_dir}/deployed_model.pt; uncertainty reference with '
              f'{int(keep.sum())} out-of-sample (polymorph, T) samples')
    return reference


def main():
    parser = argparse.ArgumentParser(description='Train one split in this process (used by run(isolate=True)).')
    parser.add_argument('--config', required=True)
    parser.add_argument('--fold', type=int, default=None)
    parser.add_argument('--finalize', action='store_true')
    args = parser.parse_args()
    cfg = TrainConfig.from_json(args.config)
    if args.finalize:
        finalize(cfg)
    else:
        run(cfg, fold=args.fold)


if __name__ == '__main__':
    main()
