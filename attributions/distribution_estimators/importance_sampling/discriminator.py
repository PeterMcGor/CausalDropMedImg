from typing import Any, Dict, List, Optional, Union, Tuple, Callable
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

    #TODO this has to be moved to the base class and make it work for no dataframes data
    def _prepare_discriminator_data(
        self,
        features: List[str],
        source_data: Union[pd.DataFrame, np.ndarray],
        target_data: Union[pd.DataFrame, np.ndarray] = None # In inference
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for discriminator training according to mechanism structure"""


        # Extract features from source/target data
        #print("Source data", type(source_data))
        source_subset = source_data[features]
        if target_data is not None:
            target_subset = target_data[features]

            # Combine into training data
            X = np.concatenate([source_subset, target_subset])
            y = np.concatenate([np.zeros(len(source_subset)), np.ones(len(target_subset))])
            return X, y
        else:
            return np.concatenate([source_subset])#X


    # TODO this just work with dataframes and sklearn
    def _fit_mechanism_models(
        self,
        source_data: Union[pd.DataFrame, np.ndarray],
        target_data: Union[pd.DataFrame, np.ndarray],
        input_features: List[str],
        register_key:str
    ) -> None:
        """Fit discriminator for a specific mechanism"""
        X, y = self._prepare_discriminator_data(input_features, source_data, target_data)

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
        #key = self._get_mechanism_key(mechanism)
        #key = MechanismSpec.sort_string_list(input_features)
        self.fitted_models[register_key] = model

        # Compute and store metrics
        probs = model.predict_proba(X)[:, 1]
        self.model_metrics[register_key] = {
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
            print("Fitting", mechanism)
            self._fit_mechanism_models(source_data, target_data, mechanism.variables, mechanism.variables_key)
            if not mechanism.is_root:
                if mechanism.parents_key not in self.fitted_models.keys():
                    print("Fitting parents", mechanism.parents)
                    self._fit_mechanism_models(source_data, target_data, mechanism.parents, mechanism.parents_key)
        self.fitted = True
        print("Fitted models", self.fitted_models, len(self.fitted_models))

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
        for i,mechanism in enumerate(mechanisms):
            mechanism_ratio = self._estimate_mechanism_ratio(data, mechanism)
            ratios *= mechanism_ratio

        return ratios

    def _estimate_mechanism_ratio(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        mechanism: MechanismSpec,
        root_node_prior_coef:int = 1,
    ) -> np.ndarray:
        """Estimate ratio for a specific mechanism"""
        def get_dicrimination_ratios(variables, variables_key):
            # Get model for this mechanism
            data_filtered = self._prepare_discriminator_data(features=variables, source_data=data, target_data=None)
            model = self.fitted_models[variables_key]
            probs = model.predict_proba(data_filtered)
            # Clip probabilities if needed
            if self.clip_probabilities is not None:
                probs = np.clip(
                    probs,
                    1 - self.clip_probabilities,
                    self.clip_probabilities
                )
            ratios = probs[:, 1] / probs[:, 0]
            # Clip ratios if needed
            if self.clip_ratios is not None:
                ratios = np.clip(
                    ratios,
                    1 / self.clip_ratios,
                    self.clip_ratios
                )
            return ratios

        ratios = get_dicrimination_ratios(mechanism.variables, mechanism.variables_key)

        if mechanism.is_root:
            print("Root Mechanism:", mechanism, ratios.mean())
            return ratios*root_node_prior_coef
        else:
            print("Not Root:", mechanism, ratios.mean())
            return ratios/get_dicrimination_ratios(mechanism.parents, mechanism.parents_key)


    def estimate_performance_shift(
        self,
        source_data: pd.DataFrame,
        mechanisms: List[MechanismSpec],
        model: torch.nn.Module,
        metric_fn: Callable,
        target_data: pd.DataFrame = None,
        **metric_kwargs
    ) -> float:
        """Compute performance change when only the `mechanisms` shift."""
        # Step 1: Compute importance weights for the shifted mechanisms
        print("Estimating rtaio for ", mechanisms)
        weights = self.estimate_ratio(
            data=source_data,
            mechanisms=mechanisms
        )

        # Add weight diagnostics
        print(f" Weight stats - Mean: {weights.mean():.2f}, Std: {weights.std():.2f}")
        print(f" Min: {weights.min():.2f}, Max: {weights.max():.2f}")

        # Step 2: Evaluate model performance under shifted distribution
        shifted_performance = metric_fn(
            model,
            source_data,
            weights=weights,
            **metric_kwargs
        )
        return shifted_performance

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