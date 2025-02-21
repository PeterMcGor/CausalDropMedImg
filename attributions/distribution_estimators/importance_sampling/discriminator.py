from typing import Any, Dict, List, Optional, Union, Tuple
import torch
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss

from ...core.distribution_base import DensityRatioEstimator, MechanismSpec

class DiscriminatorRatioEstimator(DensityRatioEstimator):
    """Estimate density ratios using binary classifier approach.

    Based on the idea that density ratio r(x) = P(target|x)/P(source|x)
    can be estimated using a classifier trained to distinguish source vs target data.
    """

    def __init__(
        self,
        discriminator_model: Any,
        calibrate: bool = False,
        clip_probabilities: Optional[float] = None,
        clip_ratios: Optional[float] = None
    ):
        """Initialize discriminator-based estimator.

        Args:
            discriminator_model: Binary classifier model (sklearn compatible)
            calibrate: Whether to calibrate classifier probabilities
            clip_probabilities: Clip probabilities to [1-clip, clip]
            clip_ratios: Clip final ratios to [1/clip, clip]
        """
        super().__init__()
        self.base_model = discriminator_model
        self.calibrate = calibrate
        self.clip_probabilities = clip_probabilities
        self.clip_ratios = clip_ratios

        if clip_probabilities is not None:
            assert 0.5 <= clip_probabilities <= 1.0
        if clip_ratios is not None:
            assert clip_ratios >= 1.0

        self.fitted_models: Dict[str, Any] = {}
        self.model_metrics: Dict[str, Dict[str, float]] = {}

    def _prepare_discriminator_data(
        self,
        source_data: Union[pd.DataFrame, np.ndarray],
        target_data: Union[pd.DataFrame, np.ndarray],
        mechanism: MechanismSpec
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for discriminator training according to mechanism structure"""
        # Convert to numpy if needed
        if isinstance(source_data, pd.DataFrame):
            source_arr = source_data.values
            target_arr = target_data.values
        else:
            source_arr = source_data
            target_arr = target_data

        # Get relevant features for mechanism
        features = []
        if mechanism.parents:
            features.extend(mechanism.parents)
        features.extend(mechanism.variables)

        # Return combined training data and labels
        X = np.concatenate([source_arr, target_arr])
        y = np.concatenate([
            np.zeros(len(source_arr)),
            np.ones(len(target_arr))
        ])

        return X, y

    def _fit_mechanism_model(
        self,
        source_data: Union[pd.DataFrame, np.ndarray],
        target_data: Union[pd.DataFrame, np.ndarray],
        mechanism: MechanismSpec
    ) -> None:
        """Fit discriminator for a specific mechanism"""
        X, y = self._prepare_discriminator_data(source_data, target_data, mechanism)

        # Clone and fit base model
        model = clone(self.base_model).fit(X, y)

        # Optionally calibrate
        if self.calibrate:
            model = CalibratedClassifierCV(
                base_estimator=model,
                method='isotonic',
                cv='prefit'
            ).fit(X, y)

        # Store model
        key = self._get_mechanism_key(mechanism)
        self.fitted_models[key] = model

        # Compute and store metrics
        probs = model.predict_proba(X)[:, 1]
        self.model_metrics[key] = {
            'roc_auc': roc_auc_score(y, probs),
            'brier': brier_score_loss(y, probs)
        }

    def fit(
        self,
        source_data: Union[pd.DataFrame, np.ndarray],
        target_data: Union[pd.DataFrame, np.ndarray],
        mechanisms: List[MechanismSpec]
    ) -> None:
        """Fit discriminators for all mechanisms"""
        for mechanism in mechanisms:
            self._fit_mechanism_model(source_data, target_data, mechanism)
        self.fitted = True

    def estimate_ratio(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        mechanisms: List[MechanismSpec]
    ) -> np.ndarray:
        """Estimate density ratio for data points.

        The total ratio is a product of mechanism-specific ratios following
        the causal factorization.
        """
        self.check_is_fitted()

        # Start with all ones
        ratios = np.ones(len(data))

        # Multiply ratios for each mechanism
        for mechanism in mechanisms:
            mechanism_ratio = self._estimate_mechanism_ratio(data, mechanism)
            ratios *= mechanism_ratio

        return ratios

    def _estimate_mechanism_ratio(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        mechanism: MechanismSpec
    ) -> np.ndarray:
        """Estimate ratio for a specific mechanism"""
        if isinstance(data, pd.DataFrame):
            data = data.values

        # Get model for this mechanism
        key = self._get_mechanism_key(mechanism)
        model = self.fitted_models[key]

        # Get probabilities
        probs = model.predict_proba(data)

        # Clip probabilities if needed
        if self.clip_probabilities is not None:
            probs = np.clip(
                probs,
                1 - self.clip_probabilities,
                self.clip_probabilities
            )

        # Compute ratios
        ratios = probs[:, 1] / probs[:, 0]

        # Clip ratios if needed
        if self.clip_ratios is not None:
            ratios = np.clip(
                ratios,
                1 / self.clip_ratios,
                self.clip_ratios
            )

        return ratios

    @staticmethod
    def _get_mechanism_key(mechanism: MechanismSpec) -> str:
        """Create unique key for mechanism"""
        if mechanism.parents:
            return f"P({mechanism.variables[0]}|{','.join(mechanism.parents)})"
        return f"P({mechanism.variables[0]})"

    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics for all mechanism discriminators"""
        self.check_is_fitted()
        return self.model_metrics