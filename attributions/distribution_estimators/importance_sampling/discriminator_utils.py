import numpy as np


def extract_probabilities(data):
    """Extract probabilities array from data, regardless of format."""
    if isinstance(data, list) and isinstance(data[0], dict):
        # Data is a list of dictionaries with metadata
        all_probs = []
        for item in data:
            all_probs.append(item["probabilities"])
        return np.vstack(all_probs)
    else:
        # Data is already a numpy array
        return data

def clip_values(data, min_val, max_val, field="probabilities"):
    """
    Clip values in data, handling both array and metadata formats.

    Args:
        data: Either a numpy array or a list of dictionaries
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        field: Field name to clip if data is a list of dictionaries

    Returns:
        Clipped data in the same format as input
    """
    if isinstance(data, dict):
        # Data is a list of dictionaries with metadata
        for k,item in data.items():
            item[field] = np.clip(item[field], min_val, max_val)
        return data
    else:
        # Data is a numpy array
        return np.clip(data, min_val, max_val)

def compute_ratios(probabilities_data, clip=None):
    """
    Compute ratios from probabilities, handling both array and metadata formats.

    Args:
        probabilities_data: Either a numpy array or a list of dictionaries

    Returns:
        If input is numpy array: numpy array of ratios
        If input is list of dictionaries: Same list with added 'ratio' field
    """
    if isinstance(probabilities_data, dict):
        # Data is a list of dictionaries with metadata
        for k,item in probabilities_data.items():
            probs = item["probabilities"]
            item["ratio"] = probs[1] / probs[0]
            if clip is not None:
                item["ratio"] = np.clip(item["ratio"],  1 / clip, clip)
        return probabilities_data
    else:
        # Data is a numpy array
        return probabilities_data[:, 1] / probabilities_data[:, 0] if clip is None else np.clip(
            probabilities_data[:, 1] / probabilities_data[:, 0], 1 / clip, clip
        )
