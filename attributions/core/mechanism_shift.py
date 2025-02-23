from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional, Callable
from dataclasses import dataclass
import numpy as np
import torch
import networkx as nx
from dowhy import gcm
from torch.utils.data import Dataset

from .distribution_base import DistributionEstimator, MechanismSpec
from dowhy.gcm import shapley


class CausalMechanismShift:
    """Analyzes performance changes due to shifts in causal mechanisms using Shapley values.
    # TODO
    This class implements the approach from the paper "Why did the Model Fail?", using
    Shapley values to attribute performance changes to shifts in causal mechanisms.
    Each mechanism represents a component of the causal factorization P(V) = ∏ P(Xi|Pa(Xi)).
    """

    def __init__(
        self,
        distribution_estimator: DistributionEstimator,
        causal_graph: nx.DiGraph,
        shapley_config: Optional[shapley.ShapleyConfig] = None,
    ):
        """
        Args:
            distribution_estimator: Method to estimate mechanism distributions
            causal_graph: NetworkX DiGraph defining causal relationships
            shapley_config: Configuration for Shapley value computation
        """
        self.distribution_estimator = distribution_estimator
        self.graph = causal_graph
        self.shapley_config = shapley_config or shapley.ShapleyConfig()

        # Extract mechanisms from graph
        self.mechanisms = self._extract_mechanisms_from_graph()

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

    def _compute_mechanism_shift_value(
        self,
        shifted_mechanisms: List[MechanismSpec],
        source_data: Union[Dataset, torch.Tensor, Any],
        target_data: Union[Dataset, torch.Tensor, Any],
        model: torch.nn.Module,
        metric_fn: Callable,
        **metric_kwargs
    ) -> float:
        """Compute performance change when only the specified mechanisms shift.

        This implements v(S) in the Shapley formula, representing the value when
        mechanisms S shift from source to target distribution while others remain
        at source distribution.
        """
        # Ensure mechanisms follow causal ordering
        #ordered_mechanisms = self._order_mechanisms_topologically(shifted_mechanisms)
        source_perf = metric_fn(model, source_data, **metric_kwargs)#TODO why is done all times?

        shifted_perf = self.distribution_estimator.estimate_performance_shift(
            source_data=source_data,
            target_data=target_data,
            mechanisms=shifted_mechanisms,#ordered_mechanisms,
            model=model,
            metric_fn=metric_fn,
            **metric_kwargs
        )
        # Debug prints
        #print(f"\nMechanisms: {[m.name for m in shifted_mechanisms]}")
        #print(f"Source perf: {source_perf:.4f}, Shifted perf: {shifted_perf:.4f}")
        print(f"Delta: {shifted_perf - source_perf:.4f}")

        return shifted_perf - source_perf

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
        source_data: Union[Dataset, torch.Tensor, Any],
        target_data: Union[Dataset, torch.Tensor, Any],
        model: torch.nn.Module,
        metric_fn: Callable,
        **metric_kwargs
    ) -> Dict[str, float]:
        """Analyze performance shift and attribute to mechanisms using Shapley values.

        Args:
            source_data: Data from source distribution
            target_data: Data from target distribution
            model: Model to evaluate
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
            return self._compute_mechanism_shift_value(
                shifted, source_data, target_data,
                model, metric_fn, **metric_kwargs
            )

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
        source_data: Union[Dataset, torch.Tensor, Any],
        target_data: Union[Dataset, torch.Tensor, Any],
        model: torch.nn.Module,
        metric_fn: Callable,
        **metric_kwargs
    ) -> List[str]:
        """Get mechanisms ordered by magnitude of contribution to shift."""
        attributions = self.analyze_shift(
            source_data, target_data, model, metric_fn, **metric_kwargs
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