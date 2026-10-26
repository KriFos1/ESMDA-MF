"""Rank a 50%-tempered cascade ensemble and select members for the final 2D stage.

Members are scored by their final cascade-stage (1D) reduced misfit against
the original measurement covariance. RML starts are drawn at random from the
best ``quantile`` fraction; EnRML takes the ``count`` best members.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def member_misfits(predictions, observations, covariance):
    predictions = np.asarray(predictions, dtype=float)
    if predictions.shape[0] != len(observations) or not np.isfinite(predictions).all():
        raise ValueError('Predictions must be finite with one row per observation')
    return np.mean((predictions - np.asarray(observations)[:, None]) ** 2
                   / np.asarray(covariance)[:, None], axis=0)


def ranked(misfits):
    return np.argsort(misfits, kind='stable')


def random_from_top(misfits, count, quantile, seed):
    """``count`` distinct members drawn uniformly from the best ``quantile`` fraction."""
    pool = ranked(misfits)[:int(np.ceil(quantile * len(misfits)))]
    if not 0 < quantile <= 1 or count > len(pool):
        raise ValueError('Cannot draw that many members from the requested quantile')
    return np.sort(np.random.default_rng(seed).choice(pool, size=count, replace=False))


def top(misfits, count):
    if count > len(misfits):
        raise ValueError('Requested more members than the ensemble holds')
    return np.sort(ranked(misfits)[:count])


def load_cascade(directory):
    """Final cascade ensemble, its final-stage predictions and per-member misfits."""
    directory = Path(directory)
    study = json.loads((directory / 'study.json').read_text())
    state_file = directory / f"state-{study['mda_steps_executed']:02d}-1d.npz"
    with np.load(state_file) as state, np.load(directory / 'mda-final-predictions.npz') as saved:
        fields = state['fields'].copy()
        predictions, observations, covariance = (saved[key].copy() for key in
                                                 ('predictions', 'observations', 'covariance'))
        mask = saved['mask'].copy()
    if fields.shape[1] != predictions.shape[1]:
        raise ValueError('Cascade state and predictions disagree on ensemble size')
    misfits = member_misfits(predictions, observations, covariance)
    provenance = {'cascade': str(directory), 'state_file': state_file.name,
                  'state_sha256': digest(state_file),
                  'predictions_sha256': digest(directory / 'mda-final-predictions.npz'),
                  'mda_likelihood_weight': study['mda_likelihood_weight'],
                  'covariance_inflation': study['covariance_inflation'],
                  'logging_index': study['logging_index']}
    return {'fields': fields, 'predictions': predictions, 'observations': observations,
            'covariance': covariance, 'mask': mask, 'misfits': misfits, 'provenance': provenance}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cascade', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rml-count', type=int, default=10)
    parser.add_argument('--rml-quantile', type=float, default=.5)
    parser.add_argument('--rml-seed', type=int, default=4510)
    parser.add_argument('--enrml-count', type=int, default=100)
    args = parser.parse_args(argv)
    cascade = load_cascade(args.cascade)
    misfits = cascade['misfits']
    selection = {**cascade['provenance'],
                 'score': 'final cascade-stage (1D) reduced misfit, original Cd',
                 'misfit_quantiles': dict(zip(('min', 'p25', 'median', 'p75', 'max'),
                                              np.quantile(misfits, [0, .25, .5, .75, 1]).tolist())),
                 'rml': {'count': args.rml_count, 'quantile': args.rml_quantile, 'seed': args.rml_seed,
                         'members': random_from_top(misfits, args.rml_count, args.rml_quantile,
                                                    args.rml_seed).tolist()},
                 'enrml': {'count': args.enrml_count,
                           'members': top(misfits, args.enrml_count).tolist()},
                 'member_misfits': misfits.tolist()}
    args.output.write_text(json.dumps(selection, indent=2) + '\n')
    print(json.dumps({key: selection[key] for key in ('misfit_quantiles', 'rml')}, indent=1))


if __name__ == '__main__':
    main()
