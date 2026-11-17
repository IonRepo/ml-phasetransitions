"""Uncertainty of the predicted free energy from the model's graph embedding.

A reference set of (embedding, error) pairs is stored; the uncertainty of a new prediction is the distance-weighted
RMS (or mean absolute) error of its k nearest reference embeddings. This keeps the idea of the previous
implementation (error of known materials that the network represents similarly) but

* needs no reference *graphs* at prediction time (only a small ``.npz``),
* never builds a dense N x N system (the old ``RBFInterpolator(..., smoothing=0)`` on all points needs O(N^2)
  memory and fails on duplicated embeddings),
* is always positive, and also reports the distance to the nearest reference as an extrapolation indicator.
"""

import numpy as np
from scipy.spatial import cKDTree


class EmbeddingUncertainty:
    def __init__(self, embeddings, errors, k=16, statistic='rmse'):
        embeddings = np.asarray(embeddings, dtype=np.float64)
        errors = np.abs(np.asarray(errors, dtype=np.float64))
        if len(embeddings) != len(errors):
            raise ValueError('embeddings and errors must have the same length')
        if statistic not in ('rmse', 'mae'):
            raise ValueError("statistic must be 'rmse' or 'mae'")
        self.embeddings, self.errors, self.k, self.statistic = embeddings, errors, int(k), statistic
        self.tree = cKDTree(embeddings)
        # typical nearest-neighbour spacing inside the reference set, to judge extrapolation
        nn_dist = self.tree.query(embeddings, k=2)[0][:, 1] if len(embeddings) > 1 else np.zeros(1)
        self.reference_spacing = float(np.median(nn_dist))

    def __call__(self, embeddings):
        """Returns (uncertainty, distance_to_nearest_reference), both with one value per row."""
        embeddings = np.atleast_2d(np.asarray(embeddings, dtype=np.float64))
        k = min(self.k, len(self.errors))
        dist, idx = self.tree.query(embeddings, k=k)
        dist, idx = dist.reshape(len(embeddings), k), idx.reshape(len(embeddings), k)
        weights = 1.0 / (dist + 1e-6)
        weights /= weights.sum(axis=1, keepdims=True)
        err = self.errors[idx]
        value = (weights * err**2).sum(1) ** 0.5 if self.statistic == 'rmse' else (weights * err).sum(1)
        return value, dist[:, 0]

    def save(self, path):
        np.savez_compressed(path, embeddings=self.embeddings, errors=self.errors, k=self.k,
                            statistic=self.statistic)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as f:
            return cls(f['embeddings'], f['errors'], int(f['k']), str(f['statistic']))
