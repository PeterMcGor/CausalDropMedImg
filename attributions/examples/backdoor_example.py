import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from attributions.core.metrics.metrics import compute_brier

# Import our implementations
from attributions.core.mechanism_shift import CausalMechanismShift
from attributions.distribution_estimators.importance_sampling.discriminator import DiscriminatorRatioEstimator, SklearnDiscriminatorRatioEstimator
from attributions.data.synthetic.backdoor import BackdoorParams, BackdoorSpurious
from dowhy.gcm.shapley import ShapleyConfig
from dowhy import gcm
import networkx as nx

if __name__ == "__main__":
    class SimpleClassifier(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.linear = nn.Linear(input_dim, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            return self.sigmoid(self.linear(x))



    # 1. Generate data
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
    print(f"Dataset: {dataset}")

    #print(f"Dataset causal mechanisms: {dataset.graph.causal_mechanisms}")

    train_env_train, train_env_test = dataset.get_train_env_data()
    inference_env_train, inference_env_test = dataset.get_inference_env_data(inference_env_params)

    # In backdoor_example.py
    print("train_env Y=1 when G=1:", train_env_train[train_env_train['G'] == 1]['Y'].mean())
    print("inference_env Y=1 when G=1:", inference_env_train[inference_env_train['G'] == 1]['Y'].mean())

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
    """
    Try dowhy automatic on distributions

    train_env_train['Y_prime'] = model.predict_proba(train_env_train[['X1', 'X2', 'X3']])[:,1]
    train_env_train['brier'] = (train_env_train['Y'] - train_env_train['Y_prime'])**2
    inference_env_train['Y_prime'] = model.predict_proba(inference_env_train[['X1', 'X2', 'X3']])[:,1]
    inference_env_train['brier'] = (inference_env_train['Y'] - inference_env_train['Y_prime'])**2
    #dataset.graph.graph.add_edges_from([('G','Y_prime'), ('Y','brier'), ('Y_prime','brier')])
    #dataset.graph.graph.add_edges_from([('X1','Y_prime'),('X2','Y_prime'), (('X3','Y_prime')), ('Y','brier'), ('Y_prime','brier')])
    #dataset.graph.graph.add_edges_from([('Y_prime','X1'),('Y_prime', 'X2'), (('Y_prime', 'X3')), ('Y','brier'), ('Y_prime','brier')])
    dataset.graph.graph.add_edge('X1', 'Y_prime', bidirected=True)
    dataset.graph.graph.add_edge('X2', 'Y_prime', bidirected=True)
    dataset.graph.graph.add_edge('X3', 'Y_prime', bidirected=True)
    gcm.auto.assign_causal_mechanisms(dataset.graph, train_env_train)
    attributions = gcm.distribution_change(dataset.graph, train_env_train, inference_env_train, 'brier')
    print('Attributions',attributions)
    """

    # 4. Setup mechanism shift analysis
    shapley_config = ShapleyConfig(num_subset_samples=1000)  # For faster example

    # Create estimator with logistic regression discriminator
    estimator = SklearnDiscriminatorRatioEstimator(
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
        train_env_data=train_env_train,
        inference_env_data=inference_env_train,
        mechanisms=analyzer.mechanisms  # Use the mechanisms extracted by the analyzer
        )

    train_env_perf = compute_brier(model, train_env_test)
    inference_env_perf = compute_brier(model, inference_env_test)
    print(f"train_env Brier: {train_env_perf:.4f}")
    print(f"inference_env Brier: {inference_env_perf:.4f}")
    print(f"Performance Drop: {inference_env_perf - train_env_perf:.4f}")


    # 5. Analyze shifts
    attributions = analyzer.analyze_shift(
        train_env_data=train_env_test,
        inference_env_data=inference_env_test,
        model=best_model,
        metric_fn=compute_brier
    )

    print("\nShapley attributions for performance change:")
    for mechanism, value in attributions.items():
        print(f"{mechanism}: {value:.4f}")

    # 6. Get ordered mechanisms
    #ordered = analyzer.get_mechanism_ordering(
    #    train_env_data=train_env_test,
    #    inference_env_data=inference_env_test,
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