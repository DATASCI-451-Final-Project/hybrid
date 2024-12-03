
"""
NHANES Cancer Type Prediction using Bayesian Model Averaging
---------------------------------------------------------
Author: Yuhan Ye

Date: December 2, 2024
Version: 2.0

Description: Test for nhanes_bma.py
"""

import unittest
import numpy as np
from nhanes_bma import NHANESBayesianModelAveraging

class TestBMABasics(unittest.TestCase):
    def setUp(self):
        self.bma = NHANESBayesianModelAveraging()
        self.n_samples = 5
        self.n_cancer_types = 29
        
    def test_initialization(self):
        """Test if BMA initializes correctly"""
        self.assertEqual(self.bma.model_weights['BLR'], 0.5)
        self.assertEqual(self.bma.model_weights['GP'], 0.5)
        
    def test_weight_update(self):
        """Test if weights update correctly"""
        blr_likelihood = -100.0
        gp_likelihood = -120.0
        self.bma.compute_model_weights(blr_likelihood, gp_likelihood)
        self.assertAlmostEqual(sum(self.bma.model_weights.values()), 1.0)
        
    def test_prediction_probabilities(self):
        """Test if predictions are valid probabilities"""
        blr_pred = np.random.random((self.n_samples, self.n_cancer_types))
        blr_pred = blr_pred / blr_pred.sum(axis=1)[:, np.newaxis]
        
        gp_pred = np.random.random((self.n_samples, self.n_cancer_types))
        gp_pred = gp_pred / gp_pred.sum(axis=1)[:, np.newaxis]
        
        combined_pred = self.bma.combine_predictions(blr_pred, gp_pred)
        row_sums = np.sum(combined_pred, axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(self.n_samples))
        
    def test_top_k_predictions(self):
        """Test if top-k predictions are valid"""
        blr_pred = np.random.random((self.n_samples, self.n_cancer_types))
        blr_pred = blr_pred / blr_pred.sum(axis=1)[:, np.newaxis]
        
        gp_pred = np.random.random((self.n_samples, self.n_cancer_types))
        gp_pred = gp_pred / gp_pred.sum(axis=1)[:, np.newaxis]
        
        cancer_types = [f"Cancer_{i}" for i in range(self.n_cancer_types)]
        predictions = self.bma.predict_top_k(blr_pred, gp_pred, cancer_types, k=3)
        
        self.assertEqual(len(predictions), self.n_samples)
        self.assertEqual(len(predictions[0]), 3)

if __name__ == '__main__':
    unittest.main()