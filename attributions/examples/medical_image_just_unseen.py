import numpy as np
import pandas as pd
import torch.nn as nn
from pathlib import Path
import os
import networkx as nx
import argparse
from attributions.core.mechanism_shift import (
    CausalMechanismShift,
    CausalMechanismShiftMed,
)
from attributions.core.metrics.metrics import compute_weighted_metrics_merged_dataset
from attributions.distribution_estimators.importance_sampling.nnunet_discriminators import FlexibleNNunetBinaryDiscriminatorRatioEstimator
from attributions.models.base_models import TrainingConfig, MetricConfig, MetricGoal
from attributions.core.distribution_base import MechanismSpec
from dowhy.gcm.shapley import ShapleyConfig
from dowhy import gcm


# ImagesTr&LabelsTr vs imagesTs&LabelsTs: Test in portion of imagesTs&LabelsTs
# all same centers. Real case
def main(args):
    # Common configuration
    training_config = TrainingConfig(
        num_epochs=args.num_epochs,
        val_interval=args.val_interval,
        num_train_iterations_per_epoch=args.num_train_iterations_per_epoch,
        num_val_iterations_per_epoch=args.num_val_iterations_per_epoch,
        metric=MetricConfig("balanced_acc", MetricGoal.MAXIMIZE), #f1
        device=args.device,
        verbosity=1,
        log_path=None,
        save_path=None,
        exp_name=f"Exp2_Annotator{args.train_annotator}_vs_Annotator{args.test_annotator}"
    )

    def center_number_groupby(key):
        parts = key.split("_")
        for i, part in enumerate(parts):
            if part.lower() == "center" and i + 1 < len(parts):
                return parts[i + 1]
        return "unknown"

    # Initialize with minimal settings - skip default preprocessing
    nnunet_folders_path = args.nnunet_folders_path

    # Construct dataset name with specified annotator
    nnunet_dataset = (
        f"Dataset00{args.train_annotator}_MSSEG_FLAIR_Annotator{args.train_annotator}"
    )

    # Create results folder explicitly
    result_folder = os.path.join(nnunet_folders_path, nnunet_dataset, "results")
    os.makedirs(result_folder, exist_ok=True)

    discriminator = FlexibleNNunetBinaryDiscriminatorRatioEstimator(
        nnunet_dataset=nnunet_dataset,
        nnunet_folders_path=nnunet_folders_path,
        training_config=training_config,
        stratify_by=center_number_groupby,
        batch_size=args.batch_size_estimator_training,
        clip_probabilities=0.9999,
        clip_ratios=1000,
        result_folder=result_folder,  # Explicitly provide result folder
        skip_default_preprocessing=True,  # Skip automatic preprocessing
    )

    # Training data for the discriminator from nnunet training data with specified annotator
    train_nnunet_path = discriminator.preprocess_custom_folder(
        input_images_folder=f'{nnunet_folders_path}/nnUNet_raw/{nnunet_dataset}/imagesTr',
        input_segs_folder=f'{nnunet_folders_path}/nnUNet_raw/{nnunet_dataset}/labelsTr',
        folder_name='train_data'
     )
    #train_nnunet_path = "/tmp/tmp4owrcfo4/train_data"
    training_nnunet_dataset = discriminator.create_dataset_from_folder(
        train_nnunet_path, is_test_data=False
    )

    # Same domain since it's label from the same annotator
    test_nnunet = discriminator.preprocess_custom_folder(
        input_images_folder=f'{nnunet_folders_path}/nnUNet_raw/{nnunet_dataset}/imagesTs',
        input_segs_folder=f'{nnunet_folders_path}/nnUNet_raw/{nnunet_dataset}/labelsTs_{args.train_annotator}',
        folder_name='train_transport'
     )
    #test_nnunet = "/tmp/tmpd084h9y5/train_transport"
    # here test:data mean different domain. Since here we supousse training domain Center=[1,7,8] and annotator=1. Same domain
    test_nnunet_dataset = discriminator.create_dataset_from_folder(
        test_nnunet, is_test_data=False
    )

    def exclude_center_pattern(case_id: str) -> bool:
        return "Center_03" not in case_id

    # center 03 not in the source, want just label effect here. Workaround to just keep subject in domain
    just_in_train_centers_dataset_unseen_for_transport = (
        test_nnunet_dataset.subset_by_pattern(exclude_center_pattern)
    )

    # Now the out-of-domain dataset with a different annotator
    test_path = discriminator.preprocess_custom_folder(
        input_images_folder=f'{nnunet_folders_path}/nnUNet_raw/{nnunet_dataset}/imagesTs',#
        input_segs_folder=f'{nnunet_folders_path}/nnUNet_raw/{nnunet_dataset}/labelsTs_{args.test_annotator}',
        folder_name='test'
    )
    #test_path = "/tmp/tmpjkj6fye6/test"

    out_of_dataset = discriminator.create_dataset_from_folder(
        test_path, is_test_data=True,
    )
    out_of_domain_with_same_images = out_of_dataset.subset_by_pattern(
        exclude_center_pattern
    )
    test_data_for_disc, test_data_for_infer = (
        out_of_domain_with_same_images.stratified_split(0.7)
    )

    just_in_train_for_infer = just_in_train_centers_dataset_unseen_for_transport.subset(test_data_for_infer.keys())

    train_data, val_data = training_nnunet_dataset.merge_and_split(
        test_data_for_disc, split_ratio=0.8, split_type=None,check_conflicting_cases = True #True always for original experiment
    )
    discriminator.set_datasets(
        train_data=train_data,
        val_data=val_data,
        test_dataset_train_domain=test_data_for_infer,
        test_dataset_inference_domain=test_data_for_infer,
    )

    # Create analyzer
    g = nx.DiGraph()
    g.add_nodes_from(["images", "labels"])
    g.add_edges_from([("images", "labels")])
    dataset_graph = gcm.StructuralCausalModel(g)
    analyzer = CausalMechanismShiftMed(
        distribution_estimator=discriminator,
        causal_graph=(
            dataset_graph.graph
            if isinstance(dataset_graph, nx.DiGraph)
            else dataset_graph.graph
        ),
        shapley_config=ShapleyConfig(
            num_subset_samples=args.shapley_samples, n_jobs=args.n_jobs
        ),
    )

    # Rest of analysis code
    print(f"Mechanisms in causal graph: {analyzer.mechanisms}")
    # print("TYPES", type(just_in_train_centers_dataset_unseen_for_transport), type(test_data_for_infer))
    discriminator.fit(
        train_env_data=None, inference_env_data=None, mechanisms=analyzer.mechanisms
    )

    # Compute metrics using original model
    data_csv = f"{nnunet_folders_path}/nnUNet_raw/{nnunet_dataset}/{nnunet_dataset}_anima_measures.csv"
    data_csv = pd.read_csv(data_csv)
    data_csv_train_labels = data_csv[
        (data_csv["annotator"] == args.train_annotator)
        & (data_csv["ref"] == "labelsTs_" + str(args.train_annotator))
    ]
    data_csv_test_labels = data_csv[
        (data_csv["annotator"] == args.train_annotator)
        & (data_csv["ref"] == "labelsTs_" + str(args.test_annotator))
    ]
    train_env_perf_f1 = compute_weighted_metrics_merged_dataset(
        data_csv_train_labels,
        test_data_for_infer,
        measures=["F1_score"],
    )
    inference_env_perf_f1 = compute_weighted_metrics_merged_dataset(
        data_csv_test_labels, test_data_for_infer, measures=["F1_score"]
    )

    train_env_perf_dice = compute_weighted_metrics_merged_dataset(
        data_csv_train_labels, test_data_for_infer, measures=["Dice"]
    )
    inference_env_perf_dice = compute_weighted_metrics_merged_dataset(
        data_csv_test_labels, test_data_for_infer, measures=["Dice"]
    )

    # Calculate performance differences
    f1_diff = inference_env_perf_f1 - train_env_perf_f1
    dice_diff = inference_env_perf_dice - train_env_perf_dice

    # Analyze shifts for F1 score
    attributions_f1 = analyzer.analyze_shift(
        just_in_train_for_infer,
        test_data_for_infer,
        data_csv_train_labels,
        measure=MetricConfig('F1_score', MetricGoal.MAXIMIZE),
        estimator=discriminator,
    )

    print("\nShapley attributions for performance change:")
    for mechanism, value in attributions_f1.items():
        print(f"{mechanism}: {value:.4f}")

    # Analyze shifts for Dice
    attributions_dice = analyzer.analyze_shift(
        just_in_train_for_infer,
        test_data_for_infer,
        data_csv_train_labels,
        measure=MetricConfig('Dice', MetricGoal.MAXIMIZE),
        estimator=discriminator,
    )

    print("\nShapley attributions for performance Dice change:")
    for mechanism, value in attributions_dice.items():
        print(f"{mechanism}: {value:.4f}")

    # Create DataFrames from each attribution dictionary
    df_f1 = pd.DataFrame(attributions_f1.items(), columns=["mechanism", "value"])
    df_f1["metric"] = "F1"
    df_f1["initial_train_perf"] = train_env_perf_f1
    df_f1["initial_inference_perf"] = inference_env_perf_f1
    df_f1["performance_difference"] = f1_diff
    df_f1["train_annotator"] = args.train_annotator
    df_f1["test_annotator"] = args.test_annotator

    df_dice = pd.DataFrame(attributions_dice.items(), columns=["mechanism", "value"])
    df_dice["metric"] = "Dice"
    df_dice["initial_train_perf"] = train_env_perf_dice
    df_dice["initial_inference_perf"] = inference_env_perf_dice
    df_dice["performance_difference"] = dice_diff
    df_dice["train_annotator"] = args.train_annotator
    df_dice["test_annotator"] = args.test_annotator

    # Combine the DataFrames
    combined_df = pd.concat([df_f1, df_dice], ignore_index=True)

    # Create a pivot table
    pivot_columns = [
        "initial_train_perf",
        "initial_inference_perf",
        "performance_difference",
        "value",
    ]
    pivot_df = combined_df.pivot(
        index="mechanism", columns="metric", values=pivot_columns
    )

    # Flatten the multi-level column names for easier access
    pivot_df.columns = ["_".join(col).strip() for col in pivot_df.columns.values]

    # Add annotator information to pivot
    pivot_df["train_annotator"] = args.train_annotator
    pivot_df["test_annotator"] = args.test_annotator

    # Generate output filenames with annotator info
    output_base = f"Exp_2_annotator{args.train_annotator}_vs_annotator{args.test_annotator}"
    combined_output = os.path.join(
        result_folder, f"{output_base}_combined_attributions.csv"
    )
    pivot_output = os.path.join(
        result_folder, f"{output_base}_pivoted_attributions.csv"
    )

    # Print the results
    print("\nCombined DataFrame:")
    print(combined_df)

    print("\nPivot DataFrame:")
    print(pivot_df)

    # Save the DataFrames to CSV
    combined_df.to_csv(combined_output, index=False)
    pivot_df.to_csv(pivot_output)

    print(f"\nResults saved to {combined_output} and {pivot_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run causal mechanism shift analysis with different annotators"
    )

    parser.add_argument(
        "--train_annotator",
        type=int,
        required=True,
        choices=range(1, 8),
        help="Annotator number for training (1-7)",
    )

    parser.add_argument(
        "--test_annotator",
        type=int,
        required=True,
        choices=range(1, 8),
        help="Annotator number for testing (1-7, different from train)",
    )

    parser.add_argument(
        "--nnunet_folders_path",
        type=str,
        default="/home/jovyan/nnunet_data/",
        help="Path to nnUNet folders",
    )

    parser.add_argument(
        "--num_epochs", type=int, default=50, help="Number of epochs for training"
    )

    parser.add_argument("--val_interval", type=int, default=5, help="validation freq.")

    parser.add_argument(
        "--num_train_iterations_per_epoch",
        type=int,
        default=100,
        help="validation freq.",
    )

    parser.add_argument(
        "--num_val_iterations_per_epoch", type=int, default=100, help="validation freq."
    )

    parser.add_argument(
        "--batch_size_estimator_training", type=int, default=4, help="Batch size for training"
    )

    parser.add_argument(
        "--device", type=str, default="cuda", help="Device for training (cuda or cpu)"
    )

    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for Shapley computation",
    )

    parser.add_argument(
        "--shapley_samples",
        type=int,
        default=500,
        help="Number of subset samples for Shapley computation",
    )

    args = parser.parse_args()

    # Validate that train and test annotators are different
    if args.train_annotator == args.test_annotator:
        parser.error("Train and test annotators must be different")

    main(args)
