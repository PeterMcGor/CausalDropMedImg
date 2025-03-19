#!/usr/bin/env python3
"""
Script to compare:
1. predictionsTs against modified labels (labelsTs_N_dilate2, etc.) as reference
2. predictionsTs against labelsTs_N (matching annotator number)
3. predictionsTr against labelsTr
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

def compare_folders(predictions_folder, reference_folder, dataset_name, annotator, comparison_type):
    """
    Compare prediction files against reference files.

    Args:
        predictions_folder (str): Path to the predictions folder
        reference_folder (str): Path to the reference folder
        dataset_name (str): Name of the dataset
        annotator (str): Annotator number
        comparison_type (str): Type of comparison being made

    Returns:
        list: List of dictionaries with comparison results
    """
    results = []

    # Get all reference files
    reference_files = sorted(glob.glob(os.path.join(reference_folder, "*.nii.gz")))
    if not reference_files:
        print(f"Warning: No .nii.gz files found in {reference_folder}")
        return results

    for ref_file in tqdm(reference_files, desc=f"Processing {os.path.basename(predictions_folder)} vs {os.path.basename(reference_folder)}"):
        subject_name = os.path.basename(ref_file)
        pred_file = os.path.join(predictions_folder, subject_name)

        # Skip if prediction file doesn't exist
        if not os.path.exists(pred_file):
            print(f"  Warning: No corresponding prediction found for {subject_name} in {predictions_folder}")
            continue

        try:
            # Get measurements (pred_file is the segmentation to evaluate, ref_file is the ground truth)
            measures = get_anima_mesures(pred_file, ref_file, remove_files=True, match_images=False)

            # Add to results
            results.append({
                'id': os.path.splitext(subject_name)[0],
                'dataset': dataset_name,
                'annotator': annotator,
                'reference': os.path.basename(reference_folder),
                'predictions': os.path.basename(predictions_folder),
                'comparison_type': comparison_type,
                **measures
            })
        except Exception as e:
            print(f"  Error processing {subject_name}: {str(e)}")

    return results

def process_dataset(dataset_path):
    """
    Process a single dataset by comparing:
    1. predictionsTs against modified labels (labelsTs_N_dilate2, etc.) as reference
    2. predictionsTs against labelsTs_N (matching annotator number)
    3. predictionsTr against labelsTr

    Args:
        dataset_path (str): Path to the dataset directory

    Returns:
        pandas.DataFrame: Results of all comparisons
    """
    dataset_name = os.path.basename(dataset_path)
    print(f"Processing dataset: {dataset_name}")

    # Extract annotator number
    annotator = extract_annotator_number(dataset_name)

    all_results = []

    # Find predictionsTs folder (with alternative capitalization check)
    predictions_folder = os.path.join(dataset_path, "predictionsTs")
    if not os.path.exists(predictions_folder):
        alt_predictions_folder = os.path.join(dataset_path, "PredictionsTs")
        if os.path.exists(alt_predictions_folder):
            predictions_folder = alt_predictions_folder
        else:
            print(f"Warning: No predictionsTs folder found in {dataset_path}")
            predictions_folder = None

    if predictions_folder:
        # 1. Compare predictionsTs against labelsTs_N (matching annotator number)
        if "Consensus" in dataset_name:
            reference_folder = os.path.join(dataset_path, "labelsTs")
        else:
            reference_folder = os.path.join(dataset_path, f"labelsTs_{annotator}")

        if os.path.exists(reference_folder):
            print(f"Comparing predictionsTs against {os.path.basename(reference_folder)}")
            results = compare_folders(
                predictions_folder,
                reference_folder,
                dataset_name,
                annotator,
                "original_labels"
            )
            all_results.extend(results)
        else:
            print(f"Warning: {os.path.basename(reference_folder)} folder not found in {dataset_path}")

        # 2. Compare predictionsTs against modified labels
        if "Consensus" in dataset_name:
            modified_pattern = os.path.join(dataset_path, "labelsTs_*")
        else:
            modified_pattern = os.path.join(dataset_path, f"labelsTs_{annotator}_*")

        modified_folders = sorted(glob.glob(modified_pattern))
        for mod_folder in modified_folders:
            mod_folder_name = os.path.basename(mod_folder)

            # Check if it's one of the target modifications (dilate2, erode8, reduce5)
            if any(mod in mod_folder_name for mod in ["dilate2", "erode8", "reduce5"]):
                print(f"Comparing predictionsTs against {mod_folder_name}")
                results = compare_folders(
                    predictions_folder,
                    mod_folder,
                    dataset_name,
                    annotator,
                    "modified_labels"
                )
                all_results.extend(results)

    # 3. Compare predictionsTr against labelsTr
    labelsTr_folder = os.path.join(dataset_path, "labelsTr")
    predictionsTr_folder = os.path.join(dataset_path, "predictionsTr")

    if os.path.exists(labelsTr_folder) and os.path.exists(predictionsTr_folder):
        print("Comparing predictionsTr against labelsTr")
        results = compare_folders(
            predictionsTr_folder,
            labelsTr_folder,
            dataset_name,
            annotator,
            "training_data"
        )
        all_results.extend(results)
    else:
        if not os.path.exists(labelsTr_folder):
            print(f"Warning: labelsTr folder not found in {dataset_path}")
        if not os.path.exists(predictionsTr_folder):
            print(f"Warning: predictionsTr folder not found in {dataset_path}")

    # Convert results to DataFrame
    if all_results:
        results_df = pd.DataFrame(all_results)

        # Save to CSV in the dataset folder
        csv_path = os.path.join(dataset_path, f"{dataset_name}_label_comparisons.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")

        return results_df
    else:
        print("No results to save")
        return None

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Compare predictions with references for a specific dataset')
    parser.add_argument('-d', '--dataset_path', type=str, required=True,
                        help='Path to the dataset directory')
    args = parser.parse_args()

    if os.path.exists(args.dataset_path):
        process_dataset(args.dataset_path)
    else:
        print(f"Dataset not found: {args.dataset_path}")