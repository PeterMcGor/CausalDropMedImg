"""
Main Shapley value computation logic from original paper
with extensibility for different estimation methods
"""

import numpy as np
from itertools import chain, combinations
from typing import List, Dict

class CausalShapley:
    """Main attribution calculator preserving original paper's logic"""

    def __init__(self, cgm, density_estimator, exclude_dists=None):
        """
        Args match original paper's CGExplainerDR parameters:
        - cgm: Causal graph model
        - density_estimator: Configured density ratio estimator
        - exclude_dists: List of distributions to exclude
        """
        self.cgm = cgm
        self.density_estimator = density_estimator
        self.exclude_dists = exclude_dists or []

    def explain(self, model, metric, source_df, target_df):
        """Preserve original paper's interface with DataFrame inputs"""
        shifts = self._get_valid_shifts()
        source_metric = metric(model, source_df)

        # Compute Shapley values using original paper's logic
        values = {}
        for shift in shifts:
            contribution = 0
            for coalition in self._powerset([s for s in shifts if s != shift]):
                val_with = self._coalition_value(coalition + [shift], model, metric, source_df, source_metric)
                val_without = self._coalition_value(coalition, model, metric, source_df, source_metric)
                contribution += (val_with - val_without) * self._shapley_weight(len(coalition), len(shifts))
            values[shift] = contribution

        return values

    def _coalition_value(self, coalition, model, metric, source_df, source_metric):
        """Adapted from original _delta method"""
        weights = np.ones(len(source_df))

        for shift in coalition:
            # Original paper's weight computation logic
            if shift in self.cgm.root_nodes:
                feat_names = self._get_feature_names(shift)
                weights *= self.density_estimator.compute_ratios(
                    source_df[feat_names].values
                )
            else:
                # Handle conditional shifts
                parent_feats = self._get_feature_names(shift[0])
                child_feats = self._get_feature_names(shift[1])
                weights *= (
                    self.density_estimator.compute_ratios(source_df[parent_feats + child_feats].values) /
                    self.density_estimator.compute_ratios(source_df[parent_feats].values)
                )

        return metric(model, source_df, weights) - source_metric