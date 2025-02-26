
from sklearn.metrics import brier_score_loss
import torch


def compute_brier(model, data, weights=None, subset_cols=None, return_average=True):
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
