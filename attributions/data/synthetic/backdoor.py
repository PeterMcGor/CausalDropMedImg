import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Tuple, List
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass
class BackdoorParams:
    """Parameters for BackdoorSpurious dataset generation"""
    data_seed: int
    n: int
    test_pct: float
    spu_q: float = 0.9           # Target-predictor correlation
    spu_y_noise: float = 0.25    # Label noise
    spu_mu_add: float = 3.0      # Spurious feature offset
    spu_x1_weight: float = 1.0   # Feature importance weight

class BackdoorSpurious:
    """Synthetic dataset with spurious correlations and backdoor paths.

    Implements causal structure:
    G -> Y, G -> X2, G -> X3
    Y -> X1, Y -> X2, Y -> X3

    Where:
    - G is a binary confounder
    - Y is binary target
    - X1,X2,X3 are continuous features affected by both Y and G
    """

    # Dataset constants
    TARGET_NAME = 'Y'
    FEATURE_NAMES = ['X1', 'X2', 'X3']
    NODE_NAMES = ['G', 'Y', 'X1', 'X2', 'X3']

    def __init__(self, params: BackdoorParams):
        """
        Args:
            params: Parameters for data generation
        """
        self.params = params

        # Create causal graph using networkx
        self.graph = self._create_graph()

    @classmethod
    def _create_graph(cls) -> nx.DiGraph:
        """Create causal graph using networkx"""
        # Create directed graph
        g = nx.DiGraph()

        # Add nodes
        g.add_nodes_from(cls.NODE_NAMES)

        # Add edges
        edges = [
            ('G', 'Y'),
            ('G', 'X2'),
            ('G', 'X3'),
            ('Y', 'X1'),
            ('Y', 'X2'),
            ('Y', 'X3')
        ]
        g.add_edges_from(edges)

        return g

    def generate_data(
        self,
        n_samples: int,
        rng: np.random.RandomState,
        params: BackdoorParams
    ) -> pd.DataFrame:
        """Generate synthetic data according to causal structure"""
        # Generate confounder G
        G = rng.random(size=(n_samples, 1)) >= 0.5

        # Generate target Y with dependence on G
        Y = np.logical_xor(G, rng.random(size=(n_samples, 1)) >= params.spu_q)

        # Add noise to Y for features
        Y_noised_1 = np.logical_xor(Y, rng.random(size=(n_samples, 1)) <= params.spu_y_noise)
        Y_noised_2 = np.logical_xor(Y, rng.random(size=(n_samples, 1)) <= params.spu_y_noise)
        Y_noised_3 = np.logical_xor(Y, rng.random(size=(n_samples, 1)) <= params.spu_y_noise)

        # Generate features with dependencies on Y and G
        X1 = rng.normal(loc=params.spu_x1_weight * Y_noised_1, size=(n_samples, 1))
        X2 = rng.normal(loc=Y_noised_2 + G, size=(n_samples, 1))
        X3 = rng.normal(loc=Y_noised_3 + params.spu_mu_add * G, size=(n_samples, 1))

        # Combine into dataframe
        return pd.DataFrame({
            'G': G.squeeze().astype(int),
            'Y': Y.squeeze().astype(int),
            'X1': X1.squeeze(),
            'X2': X2.squeeze(),
            'X3': X3.squeeze()
        })

    def get_source_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get source distribution train/test data"""
        rng = np.random.RandomState(self.params.data_seed)
        data = self.generate_data(self.params.n, rng, self.params)
        return train_test_split(
            data,
            test_size=self.params.test_pct,
            random_state=self.params.data_seed
        )

    def get_target_data(
        self,
        target_params: BackdoorParams
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get target distribution train/test data"""
        if target_params.data_seed == self.params.data_seed:
            raise ValueError("Target must have different random seed than source")

        rng = np.random.RandomState(target_params.data_seed)
        data = self.generate_data(target_params.n, rng, target_params)
        return train_test_split(
            data,
            test_size=target_params.test_pct,
            random_state=target_params.data_seed
        )