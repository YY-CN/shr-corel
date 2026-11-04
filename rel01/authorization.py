"""Bind a public reproduction run to the features explicitly supplied by its user."""
from .config import ScopeError


class Permit:
    def __init__(self, data):
        self.dataset = data.dataset
        self.fingerprint = data.fingerprint

    def validate(self):
        return {'dataset': self.dataset, 'data_fingerprint': self.fingerprint}

    def check_data(self, data):
        if data.dataset != self.dataset or data.fingerprint != self.fingerprint:
            raise ScopeError('input data changed within a reproduction run')


def require_execution(data, action, permit=None):
    if data.kind == 'SYNTHETIC':
        from .data import require_synthetic
        require_synthetic(data, action)
        return
    if not isinstance(permit, Permit):
        raise ScopeError(action + ': load and validate the supplied features first')
    permit.check_data(data)
