"""Read user-prepared arrays with the paper's exact sample and participant scopes."""
import csv
import json
from pathlib import Path

import numpy as np

from .config import ROOT, ScopeError
from .data import Answers, Scope


def load_inputs(dataset, embeddings, lexical=None):
    metadata = Path(__file__).resolve().parent/'protocol'/f'{dataset.lower()}_samples.csv'
    with metadata.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    matrix = np.load(embeddings, allow_pickle=False)
    indices = np.array([int(row['embedding_row']) for row in rows])
    if not isinstance(matrix, np.ndarray) or matrix.shape != (len(rows), 768) or matrix.dtype != np.float32:
        raise ScopeError('embeddings must be an N by 768 float32 NPY array in the documented sample order')
    if set(indices) != set(range(len(rows))):
        raise ScopeError('invalid protocol embedding-row mapping')
    n = len(rows)
    values = np.full((n, 36), np.nan, dtype=np.float64)
    mask = np.zeros((n, 36), dtype=bool)
    reason = np.ones((n, 36), dtype=np.int64)
    if lexical is not None:
        with np.load(lexical, allow_pickle=False) as measurements:
            values = measurements['psy']
            mask = measurements['mask']
            reason = measurements['reason'] if 'reason' in measurements else (~mask).astype(np.int64)
    data = Answers(
        dataset, tuple(row['sample_id'] for row in rows),
        tuple(row['participant_id'] for row in rows),
        np.array([int(row['label']) for row in rows], dtype=np.int8), matrix[indices],
        values, mask, reason, np.array([int(row['nlex']) for row in rows]),
        tuple(row['source_id'] for row in rows), 'REAL_AUDIT_ONLY',
    )
    scopes = [Scope(**row) for row in json.loads((Path(__file__).resolve().parent/'protocol/scopes.json').read_text())
              if row['dataset'] == dataset]
    validate_scopes(data, scopes)
    return data, scopes


def validate_scopes(data, scopes):
    if len(scopes) != 25 or len({s.key for s in scopes}) != 25:
        raise ScopeError('five outer folds and four inner folds per outer fold are required')
    population = set(data.participants)
    tested = []
    for outer in range(5):
        refit = next(s for s in scopes if s.outer_fold == outer and s.inner_fold is None)
        refit.check(data)
        if set(refit.fit_ids) | set(refit.test_ids) != population:
            raise ScopeError('incomplete outer participant coverage')
        tested.extend(refit.test_ids)
        validated = []
        for inner in range(4):
            scope = next(s for s in scopes if s.outer_fold == outer and s.inner_fold == inner)
            scope.check(data)
            if (set(scope.fit_ids) | set(scope.validation_ids) != set(refit.fit_ids)
                    or scope.test_ids != refit.test_ids):
                raise ScopeError('inner fold does not partition its outer training participants')
            validated.extend(scope.validation_ids)
        if len(validated) != len(set(validated)) or set(validated) != set(refit.fit_ids):
            raise ScopeError('inner validation participants must appear exactly once')
    if len(tested) != len(population) or set(tested) != population:
        raise ScopeError('outer test participants must appear exactly once')
