"""Regression checks for the portable prepared-feature reader and run binding."""
from dataclasses import replace
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from rel01.authorization import Permit
from rel01.config import ScopeError
from rel01.inputs import load_inputs


class PublicInputTests(TestCase):
    def matrix(self):
        matrix = np.zeros((320, 768), dtype=np.float32)
        matrix[:, 0] = 1
        return matrix

    def test_preserves_frozen_integer_dtype_and_scope_coverage(self):
        with patch('rel01.inputs.np.load', return_value=self.matrix()):
            data, scopes = load_inputs('MU3D', 'prepared.npy')
        self.assertEqual(data.labels.dtype, np.dtype('int8'))
        self.assertEqual(len(scopes), 25)
        self.assertEqual(len(set(data.participants)), 80)
        self.assertFalse(data.mask.any())
        self.assertTrue(np.isnan(data.psy).all())

    def test_rejects_wrong_shape_and_precision(self):
        for matrix in [self.matrix().astype(np.float64), self.matrix()[:319]]:
            with self.subTest(shape=matrix.shape, dtype=matrix.dtype):
                with patch('rel01.inputs.np.load', return_value=matrix):
                    with self.assertRaises(ScopeError):
                        load_inputs('MU3D', 'prepared.npy')

    def test_run_input_binding_rejects_different_measurements(self):
        with patch('rel01.inputs.np.load', return_value=self.matrix()):
            data, _ = load_inputs('MU3D', 'prepared.npy')
        permit = Permit(data)
        permit.check_data(data)
        changed = replace(data, nlex=data.nlex+1)
        with self.assertRaises(ScopeError):
            permit.check_data(changed)
