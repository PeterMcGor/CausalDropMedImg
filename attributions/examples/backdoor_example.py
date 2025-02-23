import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import brier_score_loss

# Import our implementations
from attributions.core.mechanism_shift import CausalMechanismShift
from attributions.distribution_estimators.importance_sampling.discriminator import DiscriminatorRatioEstimator
from attributions.data.synthetic.backdoor import BackdoorParams, BackdoorSpurious
from dowhy.gcm.shapley import ShapleyConfig
from dowhy import gcm
import networkx as nx

if __name__ == "__main__":

    def compute_brier(model, data, weights=None, subset_cols=None, return_average=True):
        """Replicates the paper's metric function."""
        if subset_cols is None:
            subset_cols = ["X1", "X2", "X3"]

        X = data[subset_cols]
        y = data["Y"]

        # Get predicted probabilities for class 1
        probs = model.predict_proba(X)[:, 1]

        if not return_average:
             return (probs - y)**2

        # Compute weighted Brier score
        if weights is not None:
            return brier_score_loss(y, probs, sample_weight=weights)
        return brier_score_loss(y, probs)

        # 3. Define metric function
    def compute_accuracy(model, data, weights=None):
        x = torch.tensor(data[['X1', 'X2', 'X3']].values, dtype=torch.float32)
        y = torch.tensor(data['Y'].values, dtype=torch.float32)
        outputs = model(x).squeeze()
        preds = (outputs >= 0.5).float()

        if weights is not None:
            weights = torch.tensor(weights, dtype=torch.float32)
            correct = (preds == y).float() * weights
            return (correct.sum() / weights.sum()).item()
        else:
            return (preds == y).float().mean().item()

    class SimpleClassifier(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.linear = nn.Linear(input_dim, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            return self.sigmoid(self.linear(x))



    # 1. Generate data
    source_params = BackdoorParams(
        data_seed=42,
        n=20000,
        test_pct=0.25,
        spu_q=0.9,          # Strong correlation in source
        spu_y_noise=0.25,
        spu_mu_add=3.0,
        spu_x1_weight=1.0
    )

    target_params = BackdoorParams(
        data_seed=1,
        n=20000,
        test_pct=0.25,
        spu_q=0.1,          # Weaker correlation in target
        spu_y_noise=0.25,
        spu_mu_add=3.0,
        spu_x1_weight=1.0,
    )

    # Create dataset
    dataset = BackdoorSpurious(source_params, is_scm=True)
    print(f"Dataset: {dataset}")

    #print(f"Dataset causal mechanisms: {dataset.graph.causal_mechanisms}")

    source_train, source_test = dataset.get_source_data()
    target_train, target_test = dataset.get_target_data(target_params)

    # In backdoor_example.py
    print("Source Y=1 when G=1:", source_train[source_train['G'] == 1]['Y'].mean())
    print("Target Y=1 when G=1:", target_train[target_train['G'] == 1]['Y'].mean())

    model = GridSearchCV(
        estimator=XGBClassifier(random_state=42, n_jobs=-1),
        param_grid={"max_depth": [1, 2, 3, 4, 5]},
        scoring="roc_auc_ovr",
        cv=3,
        refit=True
    ).fit(
        source_train[["X1", "X2", "X3"]],
        source_train["Y"]
    )
    best_model = model.best_estimator_
    """
    Try dowhy automatic on distributions

    source_train['Y_prime'] = model.predict_proba(source_train[['X1', 'X2', 'X3']])[:,1]
    source_train['brier'] = (source_train['Y'] - source_train['Y_prime'])**2
    target_train['Y_prime'] = model.predict_proba(target_train[['X1', 'X2', 'X3']])[:,1]
    target_train['brier'] = (target_train['Y'] - target_train['Y_prime'])**2
    #dataset.graph.graph.add_edges_from([('G','Y_prime'), ('Y','brier'), ('Y_prime','brier')])
    #dataset.graph.graph.add_edges_from([('X1','Y_prime'),('X2','Y_prime'), (('X3','Y_prime')), ('Y','brier'), ('Y_prime','brier')])
    #dataset.graph.graph.add_edges_from([('Y_prime','X1'),('Y_prime', 'X2'), (('Y_prime', 'X3')), ('Y','brier'), ('Y_prime','brier')])
    dataset.graph.graph.add_edge('X1', 'Y_prime', bidirected=True)
    dataset.graph.graph.add_edge('X2', 'Y_prime', bidirected=True)
    dataset.graph.graph.add_edge('X3', 'Y_prime', bidirected=True)
    gcm.auto.assign_causal_mechanisms(dataset.graph, source_train)
    attributions = gcm.distribution_change(dataset.graph, source_train, target_train, 'brier')
    print('Attributions',attributions)
    """

    # 4. Setup mechanism shift analysis
    shapley_config = ShapleyConfig(num_subset_samples=500)  # For faster example

    # Create estimator with logistic regression discriminator
    estimator = DiscriminatorRatioEstimator(
        discriminator_model= GridSearchCV(
            estimator = XGBClassifier(random_state = 42, n_jobs = -1),
            param_grid = {'max_depth': [1, 2, 3, 4, 5]},
            scoring = 'roc_auc_ovr',
            cv = 3,
            refit = True
        ),#LogisticRegression(), #CalibratedClassifierCV(base_estimator = clf, method = 'isotonic', cv = 'prefit').fit(X, Y)
        calibrate=True,
        clip_ratios=1000.0,  # Clip extreme ratios
        clip_probabilities=0.9999
    )

    # Create analyzer
    analyzer = CausalMechanismShift(
        distribution_estimator=estimator,
        causal_graph=dataset.graph if isinstance(dataset, nx.DiGraph) else dataset.graph.graph,
        shapley_config=shapley_config
    )

    print(f"Mechanisms in causal graph:{analyzer.mechanisms}")
    estimator.fit(
        source_data=source_train,
        target_data=target_train,
        mechanisms=analyzer.mechanisms  # Use the mechanisms extracted by the analyzer
        )

    source_perf = compute_brier(model, source_test)
    target_perf = compute_brier(model, target_test)
    print(f"Source Brier: {source_perf:.4f}")
    print(f"Target Brier: {target_perf:.4f}")
    print(f"Performance Drop: {target_perf - source_perf:.4f}")


    # 5. Analyze shifts
    attributions = analyzer.analyze_shift(
        source_data=source_test,
        target_data=target_test,
        model=best_model,
        metric_fn=compute_brier
    )

    print("\nShapley attributions for performance change:")
    for mechanism, value in attributions.items():
        print(f"{mechanism}: {value:.4f}")

    # 6. Get ordered mechanisms
    #ordered = analyzer.get_mechanism_ordering(
    #    source_data=source_test,
    #    target_data=target_test,
    #    model=model,
    #    metric_fn=compute_accuracy
    #)

    #print("\nMechanisms ordered by impact:")
    #for i, mech in enumerate(ordered, 1):
    #    print(f"{i}. {mech}")

    # 7. Print discriminator performance
    #print("\nDiscriminator metrics:")
    #for mech, metrics in estimator.get_metrics().items():
    #    print(f"{mech}:")
    #    for metric, value in metrics.items():
    #        print(f"  {metric}: {value:.4f}")