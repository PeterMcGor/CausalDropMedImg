from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional, Callable
from dataclasses import dataclass
import numpy as np
from sklearn.base import BaseEstimator
import torch
import networkx as nx
from dowhy import gcm
from torch.utils.data import Dataset

from attributions.core.metrics.metrics import compute_weighted_metrics_merged_dataset
from attributions.models.base_models import MetricConfig, MetricGoal

from attributions.core.distribution_base import DistributionEstimator, MechanismSpec
from dowhy.gcm import shapley


class CausalMechanismShift:
    """Analyzes performance changes due to shifts in causal mechanisms using Shapley values.
    # TODO this class is bases in shaprley values, whatever are got. Not valid for other apporaches
    Each mechanism represents a component of the causal factorization P(V) = ∏ P(Xi|Pa(Xi)).
    """

    def __init__(
        self,
        distribution_estimator: DistributionEstimator,
        causal_graph: nx.DiGraph,
        shapley_config: Optional[shapley.ShapleyConfig] = None,
        metric_goal: MetricGoal = MetricGoal.MAXIMIZE,
    ):
        """
        Args:
            distribution_estimator: Method to estimate mechanism distributions
            causal_graph: NetworkX DiGraph defining causal relationships
            shapley_config: Configuration for Shapley value computation
        """
        self.distribution_estimator = distribution_estimator
        self.graph = causal_graph
        self.shapley_config = shapley_config or shapley.ShapleyConfig(n_jobs=1)

        # Extract mechanisms from graph
        self.mechanisms = self._extract_mechanisms_from_graph()
        self.baseline_performance = None
        self.metric_goal = metric_goal

    # TODO Mechanism doesntlook a good name her esince normally the mechanism is directly the function conecting the nodes
    def _extract_mechanisms_from_graph(self) -> List[MechanismSpec]:
        mechanisms = []
        for node in self.graph.nodes():
            parents = list(self.graph.predecessors(node))
            # Define mechanism as P(node | parents)
            mechanisms.append(
                MechanismSpec(
                    name=f"P({node}|{','.join(parents)})" if parents else f"P({node})",
                    variables=[node]+parents,
                    parents=parents
                )
            )
        return mechanisms

    def _compute_permutation_shift_value(
        self,
        shifted_mechanisms: List[MechanismSpec],
        train_env_data: Union[Dataset, torch.Tensor, Any],
        inference_env_data: Union[Dataset, torch.Tensor, Any],
        model: torch.nn.Module,
        metric_fn: Callable,
        **metric_kwargs
    ) -> float:
        """Compute performance change when only the specified mechanisms shift.

        This implements v(S) in the Shapley formula, representing the value when
        mechanisms S shift from train_env to inference_env distribution while others remain
        at train_env distribution.
        """
        # Ensure mechanisms follow causal ordering
        #ordered_mechanisms = self._order_mechanisms_topologically(shifted_mechanisms)

        shifted_perf = self.distribution_estimator.estimate_performance_shift(
            train_env_data=train_env_data,
            inference_env_data=inference_env_data,
            mechanisms=shifted_mechanisms,#ordered_mechanisms,
            model=model,
            metric_fn=metric_fn,
            **metric_kwargs
        )

        performance_shift = shifted_perf - self.baseline_performance if self.metric_goal == MetricGoal.MINIMIZE else self.baseline_performance - shifted_perf

        print(f"Performance change: {performance_shift:.4f}")

        return performance_shift

    def _order_mechanisms_topologically(
        self,
        mechanisms: List[MechanismSpec]
    ) -> List[MechanismSpec]:
        """Order mechanisms according to causal graph's topological ordering"""
        # Get topological ordering of full graph
        topo_order = list(nx.topological_sort(self.graph))

        # Filter and order the given mechanisms
        mechanism_dict = {m.name: m for m in mechanisms}
        ordered = [
            mechanism_dict[node]
            for node in topo_order
            if node in mechanism_dict
        ]

        return ordered

    def analyze_shift(
        self,
        train_env_data: Union[Dataset, torch.Tensor, Any],
        inference_env_data: Union[Dataset, torch.Tensor, Any],
        model: Union[torch.nn.Module, BaseEstimator],
        metric_fn: Callable,
        **metric_kwargs
    ) -> Dict[str, float]:
        """Analyze performance shift and attribute to mechanisms using Shapley values.

        Args:
            train_env_data: Data from train_env distribution
            inference_env_data: Data from inference_env distribution
            model: Model to evaluate. The one employed to train the `train_env_data`
            metric_fn: Metric to measure performance
            metric_kwargs: Additional metric arguments

        Returns:
            Dictionary mapping mechanism names to their Shapley values
        """
        # Define set function for Shapley calculation
        def mechanism_value_function(mechanism_mask: np.ndarray) -> float:
            """Value function v(S) for Shapley calculation"""
            # Get mechanisms corresponding to 1s in mask
            shifted = [m for i, m in enumerate(self.mechanisms) if mechanism_mask[i]]

            # Compute value when these mechanisms shift
            return self._compute_permutation_shift_value(
                shifted, train_env_data, inference_env_data,
                model, metric_fn, **metric_kwargs
            )

        # I need the change in performance repsect to the baseline as shapely value
        # Performing this here is not need to do it each timeI want to measure a change
        self.baseline_performance = metric_fn(model, train_env_data, **metric_kwargs)

        # Compute Shapley values using dowhy implementation
        shapley_values = shapley.estimate_shapley_values(
            set_func=mechanism_value_function,
            num_players=len(self.mechanisms),
            shapley_config=self.shapley_config
        )

        # Map values back to mechanism names
        return {
            mech.name: value
            for mech, value in zip(self.mechanisms, shapley_values)
        }

    def get_mechanism_ordering(
        self,
        train_env_data: Union[Dataset, torch.Tensor, Any],
        inference_env_data: Union[Dataset, torch.Tensor, Any],
        model: torch.nn.Module,
        metric_fn: Callable,
        **metric_kwargs
    ) -> List[str]:
        """Get mechanisms ordered by magnitude of contribution to shift."""
        attributions = self.analyze_shift(
            train_env_data, inference_env_data, model, metric_fn, **metric_kwargs
        )
        sorted_items = sorted(
            attributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return [item[0] for item in sorted_items]

    def get_markov_blanket(self, node: str) -> List[str]:
        """Get Markov blanket of a node (parents, children, children's parents)"""
        parents = list(self.graph.predecessors(node))
        children = list(self.graph.successors(node))
        spouses = []
        for child in children:
            spouses.extend(self.graph.predecessors(child))

        # Remove duplicates and the node itself
        blanket = list(set(parents + children + spouses))
        if node in blanket:
            blanket.remove(node)

        return blanket

class CausalMechanismShiftMed(CausalMechanismShift):
    def analyze_shift(
        self,
        train_env_data: Union[Dataset, torch.Tensor, Any],
        inference_env_data: Union[Dataset, torch.Tensor, Any],
        csv_data:str,
        measure=MetricConfig('F1_score', MetricGoal.MAXIMIZE),
        estimator=None
    ) -> Dict[str, float]:
        """Analyze performance shift and attribute to mechanisms using Shapley values.

        Args:
            train_env_data: Data from train_env distribution
            inference_env_data: Data from inference_env distribution
            metric_fn: Metric to measure performance
            metric_kwargs: Additional metric arguments

        Returns:
            Dictionary mapping mechanism names to their Shapley values
        """
        self.metric_goal = measure.goal
        # Define set function for Shapley calculation
        def mechanism_value_function(mechanism_mask: np.ndarray) -> float:
            """Value function v(S) for Shapley calculation"""
            # Get mechanisms corresponding to 1s in mask
            shifted = [m for i, m in enumerate(self.mechanisms) if mechanism_mask[i]]

            # Compute value when these mechanisms shift
            return self._compute_permutation_shift_value(
                shifted, train_env_data, inference_env_data,
                csv_data, [measure.name], estimator=estimator
            )

        # I need the change in performance repsect to the baseline as shapely value
        # Performing this here is not need to do it each timeI want to measure a change
        self.baseline_performance = compute_weighted_metrics_merged_dataset(csv_data, train_env_data, measures=[measure.name])

        # Compute Shapley values using dowhy implementation
        shapley_values = shapley.estimate_shapley_values(
            set_func=mechanism_value_function,
            num_players=len(self.mechanisms),
            shapley_config=self.shapley_config
        )

        # Map values back to mechanism names
        return {
            mech.name: value
            for mech, value in zip(self.mechanisms, shapley_values)
        }


    def _compute_permutation_shift_value(
        self,
        shifted_mechanisms: List[MechanismSpec],
        train_env_data: Union[Dataset, torch.Tensor, Any],
        inference_env_data: Union[Dataset, torch.Tensor, Any],
        csv_data:str,
        measure=['F1_score'],
        estimator = None,
    ) -> float:
        """Compute performance change when only the specified mechanisms shift.

        This implements v(S) in the Shapley formula, representing the value when
        mechanisms S shift from train_env to inference_env distribution while others remain
        at train_env distribution.
        """
        # Ensure mechanisms follow causal ordering
        #ordered_mechanisms = self._order_mechanisms_topologically(shifted_mechanisms)
        weights = estimator.estimate_ratio(
            train_env_dat=train_env_data,
            inference_env_data=inference_env_data,
            mechanisms=shifted_mechanisms,
        )
        # TODO check this weights
        shifted_perf = compute_weighted_metrics_merged_dataset(csv_data, train_env_data, measures=measure, weights=weights)

        # Debug prints
        #print(f"\nMechanisms: {[m.name for m in shifted_mechanisms]}")
        #print(f"train_env perf: {train_env_perf:.4f}, Shifted perf: {shifted_perf:.4f}")
        performance_shift = shifted_perf - self.baseline_performance if self.metric_goal == MetricGoal.MINIMIZE else self.baseline_performance - shifted_perf
        print(f"Performance change: {performance_shift:.4f}")
        return performance_shift
