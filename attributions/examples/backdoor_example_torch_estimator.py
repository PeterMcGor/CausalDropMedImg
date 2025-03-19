import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from attributions.core.metrics.metrics import compute_brier

# Import our implementations
from attributions.core.mechanism_shift import CausalMechanismShift
from attributions.distribution_estimators.importance_sampling.discriminator import TorchDiscriminatorRatioEstimator
from attributions.data.synthetic.backdoor import BackdoorParams, BackdoorSpurious
from dowhy.gcm.shapley import ShapleyConfig
import networkx as nx

from attributions.models.base_models import CriterionConfig, MetricConfig, MetricGoal, OptimizerConfig, TrainingConfig


class DiscriminatorNet(nn.Module):
    """Simple PyTorch discriminator network"""
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Single output for binary classification
        )

    def forward(self, x):
        return self.net(x).squeeze()

    def get_init_params(self):
        """Return initialization parameters for model cloning"""
        return (self.net[0].in_features,)

if __name__ == "__main__":

    # Keep your original data generation and model setup
    train_env_params = BackdoorParams(
        data_seed=42,
        n=20000,
        test_pct=0.25,
        spu_q=0.9,          # Strong correlation in train_env
        spu_y_noise=0.25,
        spu_mu_add=3.0,
        spu_x1_weight=1.0
    )

    inference_env_params = BackdoorParams(
        data_seed=1,
        n=20000,
        test_pct=0.25,
        spu_q=0.1,          # Weaker correlation in inference_env
        spu_y_noise=0.25,
        spu_mu_add=3.0,
        spu_x1_weight=1.0,
    )

    # Create dataset
    dataset = BackdoorSpurious(train_env_params, is_scm=True)
    train_env_train, train_env_test = dataset.get_train_env_data()
    inference_env_train, inference_env_test = dataset.get_inference_env_data(inference_env_params)
    print(f"Train env data going to train the discriminator: {train_env_train.shape}")
    print(f"Train env data for transportation: {train_env_test.shape}")
    print(f"Inference env data going to train the discriminator: {inference_env_train.shape}")
    print(f"Inference env data for model drop estimation: {inference_env_test.shape}")


    # Keep your original XGBoost model
    model = GridSearchCV(
        estimator=XGBClassifier(random_state=42, n_jobs=-1),
        param_grid={"max_depth": [1, 2, 3, 4, 5]},
        scoring="roc_auc_ovr",
        cv=3,
        refit=True
    ).fit(
        train_env_train[["X1", "X2", "X3"]],
        train_env_train["Y"]
    )
    best_model = model.best_estimator_

    # Create estimator with PyTorch discriminator
    estimator = TorchDiscriminatorRatioEstimator(
        discriminator_model=DiscriminatorNet(input_dim=3),  # 3 features
        optimizer_config=OptimizerConfig(
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs={'lr': 0.001}
        ),
        criterion_config=CriterionConfig(
            criterion_class=nn.BCEWithLogitsLoss
        ),
        training_config=TrainingConfig(
            num_epochs=50,
            val_interval=5,
            num_train_iterations_per_epoch=100,
            num_val_iterations_per_epoch=10,
            metric=MetricConfig(
                name='roc_auc',
                goal=MetricGoal.MAXIMIZE
            )
        ),
        clip_ratios=1000.0,
        clip_probabilities=0.9999
    )

    # Create analyzer (same as before)
    analyzer = CausalMechanismShift(
        distribution_estimator=estimator,
        causal_graph=dataset.graph if isinstance(dataset, nx.DiGraph) else dataset.graph.graph,
        shapley_config=ShapleyConfig(num_subset_samples=500),
        metric_goal=MetricGoal.MINIMIZE
    )

    # Rest of your analysis code remains the same
    print(f"Mechanisms in causal graph:{analyzer.mechanisms}")
    estimator.fit(
        train_env_data=train_env_train,
        inference_env_data=inference_env_train,
        mechanisms=analyzer.mechanisms
    )

    # Compute metrics using original XGBoost model
    train_env_perf = compute_brier(best_model, train_env_test)
    inference_env_perf = compute_brier(best_model, inference_env_test)
    print(f"train_env Brier: {train_env_perf:.4f}")
    print(f"inference_env Brier: {inference_env_perf:.4f}")
    print(f"Performance Drop: {inference_env_perf - train_env_perf:.4f}")

    # Analyze shifts
    attributions = analyzer.analyze_shift(
        train_env_data=train_env_test,
        inference_env_data=inference_env_test,
        model=best_model,
        metric_fn=compute_brier
    )

    print("\nShapley attributions for performance change:")
    for mechanism, value in attributions.items():
        print(f"{mechanism}: {value:.4f}")