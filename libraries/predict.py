"""Inference helpers: free energies (and uncertainties) of base graphs on a temperature grid."""

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from libraries.dataset import TemperatureGraphDataset


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@torch.no_grad()
def collect(model, dataset, batch_size=64, device=None, num_workers=0):
    """Run the model over a ``TemperatureGraphDataset``.

    One forward pass gives both the prediction and the graph embedding (the previous code ran the network twice).
    Returns (prediction_standardized, embedding, graph_index, temperature) as numpy arrays in dataset order.
    """
    device = device or get_device()
    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    preds, embs, gidx, temps = [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        p, e = model(batch, return_embedding=True)
        preds.append(p.cpu().numpy())
        embs.append(e.cpu().numpy())
        gidx.append(batch.graph_index.cpu().numpy())
        temps.append(batch.temperature.cpu().numpy())
    return (np.concatenate(preds), np.concatenate(embs), np.concatenate(gidx), np.concatenate(temps))


def predict_free_energies(graphs, temperatures, model, standardizer, uncertainty=None, batch_size=64, device=None):
    """Vibrational free energy of every base graph at ``temperatures``.

    The input graphs are not modified.

    Returns:
        dict of arrays shaped (n_graphs, n_temperatures): ``fv`` (meV / atom), ``uncertainty`` (meV / atom, NaN if no
        estimator is given) and ``distance`` (embedding-space distance to the nearest reference, NaN likewise).
    """
    temperatures = np.asarray(temperatures, dtype=np.float64)
    dataset = TemperatureGraphDataset(standardizer.standardize_graphs(graphs), standardizer, temperatures)
    pred, emb, gidx, _ = collect(model, dataset, batch_size, device)
    shape = (len(graphs), len(temperatures))
    out = {'fv': standardizer.destandardize_target(pred).reshape(shape)}
    if uncertainty is not None:
        unc, dist = uncertainty(emb)
        out['uncertainty'], out['distance'] = unc.reshape(shape), dist.reshape(shape)
    else:
        out['uncertainty'] = out['distance'] = np.full(shape, np.nan)
    return out
