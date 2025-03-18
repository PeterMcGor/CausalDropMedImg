from sklearn.metrics import brier_score_loss
import torch
import pandas as pd
import numpy as np
import re


def compute_brier(model, data, weights=None, subset_cols=None, return_average=True):
    if subset_cols is None:
        subset_cols = ["X1", "X2", "X3"]

        X = data[subset_cols]
        y = data["Y"]

        # Get predicted probabilities for class 1
        probs = model.predict_proba(X)[:, 1]

        if not return_average:
            return (probs - y) ** 2

        # Compute weighted Brier score
        if weights is not None:
            return brier_score_loss(y, probs, sample_weight=weights)
        return brier_score_loss(y, probs)

        # 3. Define metric function


def compute_accuracy(model, data, weights=None):
    x = torch.tensor(data[["X1", "X2", "X3"]].values, dtype=torch.float32)
    y = torch.tensor(data["Y"].values, dtype=torch.float32)
    outputs = model(x).squeeze()
    preds = (outputs >= 0.5).float()

    if weights is not None:
        weights = torch.tensor(weights, dtype=torch.float32)
        correct = (preds == y).float() * weights
        return (correct.sum() / weights.sum()).item()
    else:
        return (preds == y).float().mean().item()

def compute_weighted_metrics(
    data,
    case_col="id",
    weights=None,
    measures=None,
    subset_subjects=None,
    return_average=True,
):
    """
    Compute weighted averages of metrics across subjects.

    Parameters:
    -----------
    data : pandas.DataFrame or str
        DataFrame or path to CSV file containing the metrics
    case_col : str
        Column containing the case identifier (exact value will be used as subject ID)
    weights : dict or None
        Dictionary mapping subject IDs to weights. If None, equal weights are used.
        Keys must exactly match the values in the case_col column.
    measures : list or None
        List of measure columns to compute weighted averages for. If None, uses all numeric columns
        except id, dataset, and ref.
    subset_subjects : list or None
        List of subject IDs to include in the calculation. If None, all subjects are included.
    return_average : bool
        If True, returns the weighted averages as a dictionary. If False, returns the raw values.

    Returns:
    --------
    dict or numpy.ndarray
        If return_average is True, returns a dictionary mapping measure names to their weighted averages.
        If return_average is False, returns the raw measure values without averaging.
    """
    # Load data if it's a file path
    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        df = data.copy()

    # Make sure the case column exists
    if case_col not in df.columns:
        raise ValueError(f"Case column '{case_col}' not found in the data")

    # Filter to subset_subjects if provided
    if subset_subjects is not None:
        df = df[df[case_col].isin(subset_subjects)]

    # Get unique case values in the data
    unique_cases = set(df[case_col].dropna().unique())

    # If no weights provided, use equal weights for all subjects
    if weights is None:
        weights = {id_val: 1.0 for id_val in unique_cases}

    # Check if all weights keys have corresponding values in the case_col
    if weights is not None:
        missing_keys = [key for key in weights.keys() if key not in unique_cases]
        if missing_keys:
            raise ValueError(
                f"The following keys in the weights dictionary do not have corresponding values "
                f"in the {case_col} column: {missing_keys}"
            )

        # Check for duplicate keys in the data
        value_counts = df[case_col].value_counts()
        duplicate_keys = [key for key in weights.keys() if key in value_counts.index and value_counts[key] > 1]
        if duplicate_keys:
            raise ValueError(
                f"The following keys in the weights dictionary appear multiple times "
                f"in the {case_col} column: {duplicate_keys}. Each key should be unique."
            )

    # If no measures specified, use all numeric columns except known non-measure columns
    if measures is None:
        exclude_cols = ["id", case_col, "ref", "dataset", "labelsTs"]
        measures = [
            col
            for col in df.columns
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
        ]

    # If not returning average, return the raw measure values
    if not return_average:
        return df[measures].values

    # Initialize results dictionary
    results = {}

    # Calculate weighted average for each measure
    for measure in measures:
        # Skip if the measure is not in the DataFrame
        if measure not in df.columns:
            continue

        # Filter out rows where the measure is NaN
        valid_data = df[~df[measure].isna()].copy()

        if len(valid_data) == 0:
            results[measure] = np.nan
            continue

        # Add weight column based on case_col value
        valid_data["weight"] = valid_data[case_col].map(lambda x: weights.get(x, 0))

        # Calculate weighted sum and sum of weights for valid data
        weighted_sum = (valid_data[measure] * valid_data["weight"]).sum()
        sum_of_weights = valid_data["weight"].sum()

        # Avoid division by zero
        if sum_of_weights > 0:
            results[measure] = weighted_sum / sum_of_weights
        else:
            results[measure] = np.nan

    if len(measures) == 1:
        return results[measures[0]]
    return results


def compute_weighted_metrics_merged_dataset(
    data,
    merged_dataset=None,
    case_col="id",
    weights=None,
    measures=None,
    subset_subjects=None,
    return_average=True,
    key_transform=lambda k: k + ".nii",
):
    """
    Compute weighted metrics using either a provided weights dictionary or creating one from merged_dataset.

    Parameters:
    -----------
    data : dataframe
        The data to compute metrics on
    merged_dataset : dict, optional
        Dictionary to extract keys from when weights is not a dict
    case_col : str, default='id'
        Column name for case identifiers
    weights : dict or list, optional
        Either a dictionary of weights or a list of weights corresponding to merged_dataset keys
    measures : list, optional
        List of measures to compute
    subset_subjects : list, optional
        List of subjects to include
    return_average : bool, default=True
        Whether to return average metrics
    key_transform : callable, optional
        Function to transform keys. Default adds '.nii' extension

    Returns:
    --------
    Result of compute_weighted_metrics
    """
    # Determine how to get the weights dictionary
    if isinstance(weights, dict):
        # If weights is already a dictionary, transform its keys
        weights_dict = {key_transform(k): v for k, v in weights.items()}
    else:
        # If weights is not a dictionary, we need merged_dataset
        if merged_dataset is None:
            raise ValueError(
                "merged_dataset is required when weights is not a dictionary"
            )

        dict_keys = list(merged_dataset.keys())

        # If weights is not provided, create a list of 1.0s
        if weights is None:
            weights = [1.0] * len(dict_keys)
        # Otherwise, check that weights has the right length
        elif len(weights) != len(dict_keys):
            raise ValueError(
                f"Length of weights ({len(weights)}) must match length of merged_dataset keys ({len(dict_keys)})"
            )

        # Apply the key_transform to each key from merged_dataset
        weights_dict = {key_transform(k): weights[i] for i, k in enumerate(dict_keys)}

    return compute_weighted_metrics(
        data,
        case_col=case_col,
        weights=weights_dict,
        measures=measures,
        subset_subjects=subset_subjects,
        return_average=return_average,
    )