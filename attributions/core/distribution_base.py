from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import torch
from torch.utils.data import Dataset

class DistributionType(Enum):
    """Types of distributions that can be estimated"""
    MARGINAL = "marginal"      # P(X)
    CONDITIONAL = "conditional" # P(Y|X)
    JOINT = "joint"           # P(X,Y)


@dataclass
class MechanismSpec:
    """Specification of a causal mechanism to analyze."""

    name: str
    variables: List[str]
    parents: Optional[List[str]] = None
    separator: str = '_'

    def __post_init__(self):
        """Ensures instance variables are properly initialized."""
        self.variables_key = self.get_variables_string()
        self.parents_key = self.get_parents_string()
        self.is_root = True if self.parents is None or len(self.parents) == 0 else False

    @staticmethod
    def sort_string_list(string_list: List[str]) -> List[str]:
        """Sorts a list of strings alphabetically."""
        return sorted(string_list)

    def get_variables_string(self) -> str:
        """Returns the variables list as a sorted, joined string."""
        return self.separator.join(self.sort_string_list(self.variables))

    def get_parents_string(self) -> str:
        """Returns the parents list as a sorted, joined string."""
        return self.separator.join(self.sort_string_list(self.parents or []))

class DistributionEstimator(ABC):
    """Base class for estimating distributions under different approaches"""

    def __init__(self):
        self.fitted = False

    @abstractmethod
    def estimate_performance_shift(
        self,
        train_env_data: Union[Dataset, torch.Tensor, Any],
        inference_env_data: Union[Dataset, torch.Tensor, Any],
        mechanisms: List[MechanismSpec],
        model: torch.nn.Module,
        metric_fn: Callable,
        **metric_kwargs
    ) -> float:
        """
        Estimate performance shift between train_env and inference_env distributions
        for given mechanisms using specified metric

        Args:
            train_env_data: Data from train_env distribution
            inference_env_data: Data from inference_env distribution
            mechanisms: List of mechanisms being analyzed
            model: Model to evaluate
            metric_fn: Metric function to evaluate shift
            metric_kwargs: Additional arguments for metric function

        Returns:
            Estimated performance shift between distributions
        """
        pass

class ImportanceSamplingEstimator(DistributionEstimator):
    """Distribution estimation via importance sampling"""

    def __init__(self, density_ratio_estimator: "DensityRatioEstimator"):
        super().__init__()
        self.density_ratio_estimator = density_ratio_estimator
class DoCausalEstimator(DistributionEstimator):
    """Distribution estimation using do-calculus"""

    def __init__(self, scm_model: Any):
        super().__init__()
        self.scm = scm_model

    def estimate_performance_shift(
        self,
        train_env_data: Union[Dataset, torch.Tensor, Any],
        inference_env_data: Union[Dataset, torch.Tensor, Any],
        mechanisms: List[MechanismSpec],
        model: torch.nn.Module,
        metric_fn: Callable,
        **metric_kwargs
    ) -> float:
        """Estimate performance shift using do-operations"""
        raise NotImplementedError

class CounterfactualEstimator(DistributionEstimator):
    """Distribution estimation using counterfactual analysis"""

    def __init__(self, scm_model: Any):
        super().__init__()
        self.scm = scm_model

    def estimate_performance_shift(
        self,
        train_env_data: Union[Dataset, torch.Tensor, Any],
        inference_env_data: Union[Dataset, torch.Tensor, Any],
        mechanisms: List[MechanismSpec],
        model: torch.nn.Module,
        metric_fn: Callable,
        **metric_kwargs
    ) -> float:
        """Estimate performance shift using counterfactuals"""
        raise NotImplementedError

class GenerativeModelEstimator(DistributionEstimator):
    """Distribution estimation using generative models"""

    def __init__(self, generator_model: Any):
        super().__init__()
        self.generator = generator_model

    def estimate_performance_shift(
        self,
        train_env_data: Union[Dataset, torch.Tensor, Any],
        inference_env_data: Union[Dataset, torch.Tensor, Any],
        mechanisms: List[MechanismSpec],
        model: torch.nn.Module,
        metric_fn: Callable,
        **metric_kwargs
    ) -> float:
        """Estimate performance shift using generated samples"""
        raise NotImplementedError

class DensityRatioEstimator(ABC):
    """Base class for density ratio estimation methods"""

    def __init__(self):
        self.fitted = False

    @abstractmethod
    def fit(
        self,
        train_env_data: Union[Dataset, torch.Tensor, Any],
        inference_env_data: Union[Dataset, torch.Tensor, Any],
        mechanisms: List[MechanismSpec]
    ) -> None:
        """Fit the density ratio estimator"""
        pass

    @abstractmethod
    def estimate_ratio(
        self,
        train_env_data: Union[Dataset, torch.Tensor, Any],
        inference_env_data: Union[Dataset, torch.Tensor, Any],
        mechanisms: List[MechanismSpec]
    ) -> torch.Tensor:
        """Compute density ratio for given data"""
        pass

    def check_is_fitted(self):
        """Check if estimator has been fitted"""
        if not self.fitted:
            raise RuntimeError("Estimator must be fitted before computing ratios")