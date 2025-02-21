import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

# Import our implementations
from attributions.core.mechanism_shift import CausalMechanismShift
from attributions.distribution_estimators.importance_sampling.discriminator import DiscriminatorRatioEstimator
from attributions.data.synthetic.backdoor import BackdoorParams, BackdoorSpurious
from dowhy.gcm.shapley import ShapleyConfig

if __name__ == "__main__":
    # 1. Generate data
    source_params = BackdoorParams(
        data_seed=42,
        n=1000,
        test_pct=0.2,
        spu_q=0.9,          # Strong correlation in source
        spu_y_noise=0.25,
        spu_mu_add=3.0,
        spu_x1_weight=1.0
    )

    target_params = BackdoorParams(
        data_seed=43,
        n=1000,
        test_pct=0.2,
        spu_q=0.5,          # Weaker correlation in target
        spu_y_noise=0.25,
        spu_mu_add=3.0,
        spu_x1_weight=1.0
    )

    # Create dataset
    dataset = BackdoorSpurious(source_params)
    source_train, source_test = dataset.get_source_data()
    target_train, target_test = dataset.get_target_data(target_params)

    # 2. Create simple model
    class SimpleClassifier(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.linear = nn.Linear(input_dim, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            return self.sigmoid(self.linear(x))

    model = SimpleClassifier(input_dim=3)  # X1, X2, X3

    # 3. Define metric function
    def compute_accuracy(model, data, weights=None):
        x = torch.tensor(data[['X1', 'X2', 'X3']].values, dtype=torch.float32)
        y = torch.tensor(data['Y'].values, dtype=torch.float32)

        with torch.no_grad():
            outputs = model(x).squeeze()
            preds = (outputs >= 0.5).float()

            if weights is not None:
                # Weighted accuracy for importance sampling
                weights = torch.tensor(weights)
                correct = (preds == y).float() * weights
                return correct.sum() / weights.sum()
            return (preds == y).float().mean().item()

    # 4. Setup mechanism shift analysis
    shapley_config = ShapleyConfig(num_subset_samples=100)  # For faster example

    # Create estimator with logistic regression discriminator
    estimator = DiscriminatorRatioEstimator(
        discriminator_model=LogisticRegression(),
        calibrate=True,
        clip_ratios=10.0  # Clip extreme ratios
    )

    # Create analyzer
    analyzer = CausalMechanismShift(
        distribution_estimator=estimator,
        causal_graph=dataset.graph,
        shapley_config=shapley_config
    )

    # 5. Analyze shifts
    attributions = analyzer.analyze_shift(
        source_data=source_test,
        target_data=target_test,
        model=model,
        metric_fn=compute_accuracy
    )

    print("\nShapley attributions for performance change:")
    for mechanism, value in attributions.items():
        print(f"{mechanism}: {value:.4f}")

    # 6. Get ordered mechanisms
    ordered = analyzer.get_mechanism_ordering(
        source_data=source_test,
        target_data=target_test,
        model=model,
        metric_fn=compute_accuracy
    )

    print("\nMechanisms ordered by impact:")
    for i, mech in enumerate(ordered, 1):
        print(f"{i}. {mech}")

    # 7. Print discriminator performance
    print("\nDiscriminator metrics:")
    for mech, metrics in estimator.get_metrics().items():
        print(f"{mech}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")