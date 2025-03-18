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
        exp_name=f"Discriminator_Annotator{args.train_annotator}_vs_Annotator{args.test_annotator}",
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
    in_domain = discriminator.preprocess_custom_folder(
        input_images_folder=f'{nnunet_folders_path}/nnUNet_raw/{nnunet_dataset}/imagesTs',
        input_segs_folder=f'{nnunet_folders_path}/nnUNet_raw/{nnunet_dataset}/labelsTs_{args.train_annotator}',
        folder_name='train_data'
     )
    #in_domain = "/tmp/tmpxu0ra2iq/train_data"

    # Same domain since it's label from the same annotator
    out_of_domain = discriminator.preprocess_custom_folder(
        input_images_folder=f'{nnunet_folders_path}/nnUNet_raw/{nnunet_dataset}/imagesTs',
        input_segs_folder=f'{nnunet_folders_path}/nnUNet_raw/{nnunet_dataset}/labelsTs_{args.test_annotator}',
        folder_name='train_transport'
    )
    #out_of_domain = "/tmp/tmprkr5d_4n/train_transport"
    #test_nnunet = "/tmp/tmpcc35y15n/train_transport"
    # here test:data mean different domain. Since here we supousse training domain Center=[1,7,8] and annotator=1. Same domain
    in_domain_dataset = discriminator.create_dataset_from_folder(
        in_domain, is_test_data=False
    )

    in_domain_dataset.transform_keys(lambda k: 'workaround_'+k)
    # DONT KEEP THIS in this experiment <<<<<<<<<<
    out_of_domain_dataset = discriminator.create_dataset_from_folder(
        out_of_domain , is_test_data=True,
    )

    #test_data_for_disc, test_data_for_infer = (
    #    out_of_domain_dataset.random_split(0.95)
    #)

    train_data, val_data = in_domain_dataset.merge_and_split(
        out_of_domain_dataset, split_ratio=0.7, check_conflicting_cases = True #True always for original experiment
    )
    discriminator.set_datasets(
        train_data=train_data,
        val_data=val_data,
        test_dataset_train_domain=train_data,# Not used in here. Obviusly in a real experiment cannot be here
        test_dataset_inference_domain=train_data, # Not used here
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
    discriminator.fit(
        train_env_data=None, inference_env_data=None, mechanisms=analyzer.mechanisms
    )


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
        help="validation iters.",
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
