
"""
NHANES Cancer Type Prediction using Bayesian Model Averaging
---------------------------------------------------------
Author: Yuhan Ye
         
Date: December 2, 2024
Version: 2.0

Description:
    Implementation of Bayesian Model Averaging (BMA) for combining predictions
    from Bayesian Logistic Regression (BLR) and Gaussian Process (GP) models
    to predict cancer types using NHANES 2021-2023 dataset.
"""

import numpy as np
from scipy.special import softmax
from typing import List, Tuple, Dict

class NHANESBayesianModelAveraging:
    def __init__(self):
        """Initialize BMA with equal weights for both models"""
        self.model_weights = {
            'BLR': 0.5,
            'GP': 0.5
        }
        
    def compute_model_weights(self,
                            blr_likelihood: float,
                            gp_likelihood: float):
        """
        Update model weights based on provided likelihoods from BLR and GP models.
        
        Args:
            blr_likelihood: Log likelihood from BLR model
            gp_likelihood: Log likelihood from GP model
        """
        # Convert log likelihoods to weights using softmax
        likelihoods = np.array([blr_likelihood, gp_likelihood])
        weights = softmax(likelihoods)
        
        # Update weights
        self.model_weights['BLR'] = weights[0]
        self.model_weights['GP'] = weights[1]
        
    def combine_predictions(self,
                          blr_pred: np.ndarray,
                          gp_pred: np.ndarray) -> np.ndarray:
        """
        Combine predictions from BLR and GP models using current weights.
        
        Args:
            blr_pred: Predictions from BLR model [n_samples, n_cancer_types]
            gp_pred: Predictions from GP model [n_samples, n_cancer_types]
            
        Returns:
            Combined predictions [n_samples, n_cancer_types]
        """
        return (self.model_weights['BLR'] * blr_pred + 
                self.model_weights['GP'] * gp_pred)
    
    def predict_top_k(self,
                     blr_pred: np.ndarray,
                     gp_pred: np.ndarray,
                     cancer_types: List[str],
                     k: int = 3) -> List[List[Tuple[str, float]]]:
        """
        Get top k cancer predictions with probabilities.
        
        Args:
            blr_pred: BLR model predictions
            gp_pred: GP model predictions
            cancer_types: List of cancer type names
            k: Number of top predictions to return
            
        Returns:
            List of (cancer_type, probability) tuples for each sample
        """
        combined_pred = self.combine_predictions(blr_pred, gp_pred)
        results = []
        
        for pred in combined_pred:
            # Get indices of top k predictions
            top_k_indices = np.argsort(pred)[-k:][::-1]
            
            # Create list of (cancer_type, probability) tuples
            sample_results = [
                (cancer_types[idx], float(pred[idx]))
                for idx in top_k_indices
            ]
            results.append(sample_results)
            
        return results