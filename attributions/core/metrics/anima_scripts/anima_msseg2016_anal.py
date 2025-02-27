#!/usr/bin/env python3
"""
Script to compare predictions in predictionsTs folder with all labelsTs_* folders
for multiple datasets, extracting the annotator number for each dataset.
"""

import os
import glob
import re
import argparse
import pandas as pd
from tqdm import tqdm
import sys

# Import the get_anima_mesures function from anima_utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # Ensure the script directory is in the path
from anima_utils import get_anima_mesures

def extract_annotator_number(dataset_name):
    """
    Extract the annotator number from the dataset name.

    Args:
        dataset_name (str): Name of the dataset, e.g., 'Dataset001_MSSEG_FLAIR_Annotator1'

    Returns:
        str: The annotator number or 'Consensus' for consensus datasets
    """
    if "Consensus" in dataset_name:
        return "Consensus"

    # Extract the annotator number using regex
    match = re.search(r'Annotator(\d+)', dataset_name)
    if match:
        return match.group(1)
    return "Unknown"

def process_dataset(dataset_path):
    """
    Process a single dataset by comparing predictions with all available labelsTs_* folders.

    Args:
        dataset_path (str): Path to the dataset directory

    Returns:
        pandas.DataFrame: Results of all comparisons
    """
    dataset_name = os.path.basename(dataset_path)
    print(f"Processing dataset: {dataset_name}")

    # Extract annotator number
    annotator = extract_annotator_number(dataset_name)

    # Find all reference label folders (labelsTs_*)
    if "Consensus" in dataset_name:
        reference_folders = [os.path.join(dataset_path, "labelsTs")]
        if not os.path.exists(reference_folders[0]):
            print(f"Warning: No labelsTs folder found in {dataset_path}")
            return None
    else:
        reference_folders = sorted(glob.glob(os.path.join(dataset_path, "labelsTs_*")))
        if not reference_folders:
            print(f"Warning: No labelsTs_* folders found in {dataset_path}")
            return None

    # Find predictions folder
    predictions_folder = os.path.join(dataset_path, "predictionsTs")
    if not os.path.exists(predictions_folder):
        # Check for alternative capitalization
        alt_predictions_folder = os.path.join(dataset_path, "PredictionsTs")
        if os.path.exists(alt_predictions_folder):
            predictions_folder = alt_predictions_folder
        else:
            print(f"Warning: No predictionsTs folder found in {dataset_path}")
            return None

    # Get all prediction files
    prediction_files = sorted(glob.glob(os.path.join(predictions_folder, "*.nii.gz")))
    if not prediction_files:
        print(f"Warning: No prediction files found in {predictions_folder}")
        return None

    all_results = []

    # Process each reference folder
    for ref_folder in reference_folders:
        ref_folder_name = os.path.basename(ref_folder)
        print(f"  Comparing predictions with reference: {ref_folder_name}")

        # Get all reference files
        reference_files = sorted(glob.glob(os.path.join(ref_folder, "*.nii.gz")))

        # Process each reference file
        for ref_file in tqdm(reference_files, desc=f"Processing {ref_folder_name}"):
            subject_name = os.path.basename(ref_file)
            pred_file = os.path.join(predictions_folder, subject_name)

            # Skip if prediction doesn't exist
            if not os.path.exists(pred_file):
                print(f"    Warning: No prediction found for {subject_name}")
                continue

            try:
                # Get measurements
                measures = get_anima_mesures(pred_file, ref_file, remove_files=True, match_images=False)

                # Add to results
                all_results.append({
                    'id': os.path.splitext(subject_name)[0],
                    'dataset': dataset_name,
                    'annotator': annotator,
                    'ref': ref_folder_name,
                    **measures
                })
            except Exception as e:
                print(f"    Error processing {subject_name}: {str(e)}")

    # Convert results to DataFrame
    if all_results:
        results_df = pd.DataFrame(all_results)

        # Save to CSV in the dataset folder
        csv_path = os.path.join(dataset_path, f"{dataset_name}_anima_measures.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")

        return results_df
    else:
        print("No results to save")
        return None

def process_all_datasets(base_path):
    """
    Process all datasets in the base path.

    Args:
        base_path (str): Base path containing all dataset folders

    Returns:
        dict: Dictionary of dataset names to result DataFrames
    """
    # Find all dataset folders
    dataset_pattern = os.path.join(base_path, "Dataset*_MSSEG_FLAIR*")
    dataset_paths = sorted(glob.glob(dataset_pattern))

    if not dataset_paths:
        print(f"No dataset folders found in {base_path} matching {dataset_pattern}")
        return {}

    results = {}

    # Process each dataset
    for dataset_path in dataset_paths:
        dataset_name = os.path.basename(dataset_path)
        results[dataset_name] = process_dataset(dataset_path)

    return results

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Compare predictions with multiple reference labels for multiple datasets')
    parser.add_argument('-b', '--base_path', type=str, required=True, help='Base path containing dataset folders')
    parser.add_argument('-d', '--dataset', type=str, help='Specific dataset to process (optional)')
    args = parser.parse_args()

    # Process single dataset or all datasets
    if args.dataset:
        dataset_path = os.path.join(args.base_path, args.dataset)
        if os.path.exists(dataset_path):
            process_dataset(dataset_path)
        else:
            print(f"Dataset not found: {args.dataset}")
    else:
        process_all_datasets(args.base_path)