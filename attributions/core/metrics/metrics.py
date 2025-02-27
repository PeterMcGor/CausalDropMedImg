
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

def compute_weighted_metrics(data, case_col='id', weights=None, measures=None, subset_subjects=None, return_average=True):
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

    # If no measures specified, use all numeric columns except known non-measure columns
    if measures is None:
        exclude_cols = ['id', case_col, 'ref', 'dataset', 'labelsTs']
        measures = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]

    # If not returning average, return the raw measure values
    if not return_average:
        return df[measures].values

    # If no weights provided, use equal weights for all subjects
    if weights is None:
        unique_ids = df[case_col].dropna().unique()
        weights = {id_val: 1.0 for id_val in unique_ids}

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
        valid_data['weight'] = valid_data[case_col].map(lambda x: weights.get(x, 0))

        # Calculate weighted sum and sum of weights for valid data
        weighted_sum = (valid_data[measure] * valid_data['weight']).sum()
        sum_of_weights = valid_data['weight'].sum()

        # Avoid division by zero
        if sum_of_weights > 0:
            results[measure] = weighted_sum / sum_of_weights
        else:
            results[measure] = np.nan

    if len(measures) == 1:
        return results[measures[0]]
    return results

def compute_weighted_metrics_merged_dataset(data, merged_dataset, case_col='id', weights=None, measures=None, subset_subjects=None, return_average=True):
    dict_keys = merged_dataset.keys()
    if weights is None:
        weights = [1.0] * len(dict_keys)
    weights_dict =  {k+'.nii':weights[i] for i,k in enumerate(dict_keys)}
    return compute_weighted_metrics(data, case_col='id', weights=weights_dict , measures=measures, subset_subjects=subset_subjects, return_average=return_average)


if __name__ == "__main__":
    # Create a mock DataFrame with the same structure as the provided CSV
    # Including some empty and NaN values to demonstrate handling
    mock_data = {
        'dataset': ['test'] * 15,
        'id': [
            'P3_T3.nii', 'P5_T2.nii', 'P49_T2.nii', 'P3_T1.nii', 'P35_T1.nii',
            'P49_T1.nii', 'P15_T1.nii', 'P31_T2.nii', 'P5_T1.nii', 'P31_T1.nii',
            'P3_T4.nii', 'P3_T2.nii', 'P7_T1.nii', 'P8_T1.nii', 'P9_T1.nii'
        ],
        'ref': [599] * 15,
        'labelsTs': ['labelsTs'] * 15,
        'Jaccard': [0.534574, 0.559676, 0.389657, 0.566943, 0.633868,
                   0.731755, 0.604536, 0.581162, 0.218918, 0.589146,
                   0.444399, 0.550077, np.nan, 0.612345, 0.723456],
        'Dice': [0.696707, 0.717682, 0.560796, 0.723629, 0.775911,
                0.845102, 0.753534, 0.735107, 0.359200, 0.741462,
                0.615341, 0.709742, 0.680123, np.nan, 0.810234],
        'Sensitivity': [0.590308, 0.721627, 0.402808, 0.642497, 0.702926,
                       0.756000, 0.844957, 0.596710, 0.414024, 0.665036,
                       0.569393, 0.680432, 0.712345, 0.653421, np.nan],
        'Specificity': [0.999970, 0.999854, 0.999964, 0.999945, 0.999821,
                       0.999840, 0.999904, 0.999976, 0.999502, 0.999923,
                       0.999926, 0.999948, np.nan, 0.999876, 0.999912],
        'PPV': [0.849894, 0.713781, 0.922691, 0.828213, 0.865808,
              0.958014, 0.679963, 0.957090, 0.317198, 0.837735,
              0.669355, 0.741690, 0.801234, 0.732156, np.nan],
        'F1_score': [0.761194, 0.736196, 0.645598, 0.799154, 0.762431,
                   0.686747, 1.000000, 0.799695, 0.534979, 0.737705,
                   0.740741, 0.767932, np.nan, 0.720123, 0.812345]
    }

    # Create DataFrame
    df = pd.DataFrame(mock_data)

    print("Mock DataFrame (first 5 rows):")
    print(df.head())
    print("\nDataFrame contains NaN values:", df.isna().any().any())

    # Example 1: Compute weighted metrics with custom weights
    print("\nExample 1: Compute weighted metrics with custom weights")

    # Now weights use the exact ID values as keys
    weights = {
        "P3_T3.nii": 1.5,
        "P3_T1.nii": 1.5,
        "P3_T4.nii": 1.5,
        "P3_T2.nii": 1.5,
        "P5_T2.nii": 0.8,
        "P5_T1.nii": 0.8,
        "P49_T2.nii": 1.2,
        "P49_T1.nii": 1.2,
        "P31_T2.nii": 1.0,
        "P31_T1.nii": 1.0,
        "P15_T1.nii": 0.7,
        "P35_T1.nii": 0.9,
        "P7_T1.nii": 1.1,
        "P8_T1.nii": 0.6,
        "P9_T1.nii": 0.5
    }

    print("\nUnique IDs in the case column:")
    print(df['id'].unique())

    results = compute_weighted_metrics(df, case_col='id', weights=weights)
    print("\nWeighted average metrics for all subjects:")
    for metric, value in results.items():
        print(f"{metric}: {value:.6f}")

    # Example 2: Compute metrics for specific measures only
    print("\nExample 2: Compute weighted metrics for specific measures only")
    selected_measures = ['Dice', 'Jaccard', 'F1_score']
    results = compute_weighted_metrics(df, case_col='id', weights=weights, measures=selected_measures)
    print("\nWeighted average for selected metrics:")
    for metric, value in results.items():
        print(f"{metric}: {value:.6f}")

    # Example 3: Compute metrics for subset of subjects
    print("\nExample 3: Compute weighted metrics for subset of subjects")
    # Use complete filename values for subsetting
    subset = ["P3_T3.nii", "P3_T1.nii", "P3_T4.nii", "P3_T2.nii", "P5_T2.nii", "P5_T1.nii"]
    results = compute_weighted_metrics(df, case_col='id', weights=weights,
                                     subset_subjects=subset, measures=selected_measures)
    print("\nWeighted average for P3 and P5 subjects only:")
    for metric, value in results.items():
        print(f"{metric}: {value:.6f}")

    # Example 4: Handling completely missing data for one measure
    print("\nExample 4: Handling missing data")
    # Create a column with all NaN values
    df['CompletelyMissing'] = np.nan
    # Add some missing values to existing column
    df.loc[0:3, 'Jaccard'] = np.nan

    results = compute_weighted_metrics(df, case_col='id', weights=weights,
                                     measures=['Jaccard', 'CompletelyMissing', 'Dice'])
    print("\nResults with missing data:")
    for metric, value in results.items():
        if pd.isna(value):
            print(f"{metric}: NaN (completely missing)")
        else:
            print(f"{metric}: {value:.6f}")

    # Example 5: Return raw values instead of averages
    print("\nExample 5: Return raw values instead of averages")
    raw_values = compute_weighted_metrics(df, case_col='id', measures=['Dice', 'Jaccard'],
                                        return_average=False)
    print("\nRaw values (not averaged):")
    print(raw_values)