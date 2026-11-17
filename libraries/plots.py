"""Plot helpers shared by the notebooks."""

import matplotlib.pyplot as plt
import numpy as np


def parity_plot(sets, figsize=(4, 4), save_to=None, unit='meV/atom'):
    """``sets``: {name: (computed, predicted)}. Sets that are empty or None are skipped."""
    fig, ax = plt.subplots(figsize=figsize)
    lo, hi = np.inf, -np.inf
    for name, pair in sets.items():
        if pair is None or len(pair[0]) == 0:
            continue
        x, y = np.asarray(pair[0]), np.asarray(pair[1])
        ax.plot(x, y, '.', ms=3, alpha=0.6, label=f'{name} (MAE {np.mean(np.abs(x - y)):.2f})')
        lo, hi = min(lo, x.min(), y.min()), max(hi, x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], '-r', lw=1)
    ax.set_xlabel(f'Computed ({unit})')
    ax.set_ylabel(f'Predicted ({unit})')
    ax.legend(loc='best', fontsize=8)
    if save_to:
        fig.savefig(save_to, dpi=150, bbox_inches='tight')
    return fig


def loss_curves(results, key='val_mae', figsize=(6, 4), save_to=None, log=True):
    """Validation curves of every split in ``results`` (list of ``<name>_result.json`` dicts)."""
    fig, ax = plt.subplots(figsize=figsize)
    for r in results:
        ax.plot(r['history'][key], label=r['name'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation MAE (meV/atom)')
    if log:
        ax.set_yscale('log')
    ax.legend(loc='best', fontsize=7, ncol=2)
    if save_to:
        fig.savefig(save_to, dpi=150, bbox_inches='tight')
    return fig


def hist2d_annotated(x, y, x_edges, y_edges, xlabel, ylabel, title='', vmax=None, y_ticks=None, save_to=None,
                     figsize=(6, 4)):
    """2D histogram with the count written in each cell (replaces the six copy-pasted blocks of the old notebook)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    counts = np.histogram2d(x[ok], y[ok], bins=(list(x_edges), list(y_edges)))[0].T
    fig, ax = plt.subplots(figsize=figsize)
    mesh = ax.pcolormesh(x_edges, y_edges, counts, shading='auto', cmap='viridis', vmin=0, vmax=vmax)
    xc, yc = 0.5 * (np.asarray(x_edges)[1:] + np.asarray(x_edges)[:-1]), 0.5 * (np.asarray(y_edges)[1:] + np.asarray(y_edges)[:-1])
    for i, yy in enumerate(yc):
        for j, xx in enumerate(xc):
            ax.text(xx, yy, f'{int(counts[i, j])}', ha='center', va='center', color='white', size=10)
    if y_ticks:
        ax.set_yticks(*y_ticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax).set_label(r'$N_{PT}$')
    if save_to:
        fig.savefig(save_to, dpi=150, bbox_inches='tight')
    return fig
