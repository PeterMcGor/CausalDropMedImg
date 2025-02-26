from pathlib import Path
import torch
import os
from datetime import datetime

from attributions.models.monai_binary import (
    MonaiBinaryClassifier,
    BinaryClassifierConfig,
    MonaiModelType,
    BinaryMergedNNUNetTrainer,
    BinaryClassificationMetricComputer,
    MonaiBinaryClassifierInference
)

from attributions.models.base_models import (
    TrainingConfig,
    MetricConfig,
    MetricGoal,
    OptimizerConfig,
    MergedNNUNetDataLoaderSpecs,
    InferenceConfig
)

from nnunet_utils.dataset_utils import MergerNNUNetDataset
#from monai_binary_inference import MonaiBinaryClassifierInference, InferenceConfig

# Example usage function
def train_and_infer():
    # ----- Parameters -----
    dataset = 'Dataset824_FLAWS-HCO'  # Update with your dataset name
    dataset_folder = '/home/jovyan/nnunet_data/nnUNet_preprocessed/' + dataset
    folder1 = os.path.join(dataset_folder, 'nnUNetPlans_3d_fullres_train_images')
    folder2 = os.path.join(dataset_folder, 'nnUNetPlans_3d_fullres_test_images')

    # ----- Helper function to create datasets -----
    def create_split_datasets(train_folder, test_folder, split_ratio=(0.2, 0.8), seed=42):
        def process_case_id(case_id, nnunet_dataset=None):
            return 'deployment' in case_id

        dataset1 = MergerNNUNetDataset(
            train_folder,
            additional_data={'source': train_folder, 'deployment_data': process_case_id, 'test_data': 0}
        )
        dataset2 = MergerNNUNetDataset(
            test_folder,
            additional_data={'source': test_folder, 'deployment_data': process_case_id, 'test_data': 1}
        )
        return dataset1.merge_and_split(dataset2, split_ratio, seed=seed)

    # ----- 1. Training phase -----
    print("===== Starting Training =====")

    # Create model configuration
    classifier_config = BinaryClassifierConfig(
        model_type=MonaiModelType.DENSENET121,
        num_input_channels=2,  # data + target channels
        dropout_prob=0.2
    )

    # Create training configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = Path(f'./models_save/monai_binary_classifier/test_monai_binary_{dataset}_{timestamp}')

    training_config = TrainingConfig(
        num_epochs=1,  # Reduced for example
        val_interval=1,
        num_train_iterations_per_epoch=50,  # Reduced for example
        num_val_iterations_per_epoch=25,    # Reduced for example
        metric=MetricConfig('f1', MetricGoal.MAXIMIZE),
        log_path=Path(f'./logs/monai_binary_classifier/{dataset}_{timestamp}'),
        save_path=model_save_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbosity=2
    )

    # Create optimizer configuration
    optimizer_config = OptimizerConfig(
        optimizer_class=torch.optim.AdamW,
        optimizer_kwargs={'lr': 1e-4}
    )

    # Create datasets
    train_ds, val_ds = create_split_datasets(folder1, folder2, split_ratio=(0.2, 0.8))
    print(f"Training dataset size: {len(train_ds)}")
    print(f"Validation dataset size: {len(val_ds)}")

    # Create dataloader specifications
    dataloader_specs = MergedNNUNetDataLoaderSpecs(
        dataset_json_path=os.path.join(dataset_folder, 'dataset.json'),
        dataset_plans_path=os.path.join(dataset_folder, 'nnUNetPlans.json'),
        dataset_train=train_ds,
        dataset_val=val_ds,
        batch_size=4,  # Reduced for example
        num_processes=4,
        unpack_data=True,
        cleanup_unpacked=True
    )

    # Create trainer
    trainer = BinaryMergedNNUNetTrainer.create(
        classifier_config=classifier_config,
        training_config=training_config,
        optimizer_config=optimizer_config,
        dataloader_specs=dataloader_specs
    )

    # Train model
    results = trainer.train()
    model_path = results['model_path']
    print(f"Training completed. Best model saved at: {model_path}")

    # ----- 2. Inference phase -----
    print("\n===== Starting Inference =====")

    # Create test dataset for inference
    test_ds = MergerNNUNetDataset(
        folder2,  # Use test folder
        additional_data={'source': folder2, 'test_data': 1}
    )
    print(f"Test dataset size: {len(test_ds)}")

    # Create dataloader specifications for inference
    inference_dataloader_specs = MergedNNUNetDataLoaderSpecs(
        dataset_json_path=os.path.join(dataset_folder, 'dataset.json'),
        dataset_plans_path=os.path.join(dataset_folder, 'nnUNetPlans.json'),
        dataset_train=test_ds,  # Use test_ds for both train and val to satisfy constructor, dataset_val will be uesed
        dataset_val=test_ds,
        batch_size=4,
        num_processes=4,
        unpack_data=True,
        cleanup_unpacked=True
    )

    # Create inference configuration
    inference_config = InferenceConfig(
        model_path=Path(model_path),
        device="cuda" if torch.cuda.is_available() else "cpu",
        output_path=model_save_path / "inference_results",
        verbosity=2,
        save_outputs=True,
        save_format="npy"
    )

    # Method 1: Create inference from existing model (reusing the trained model)
    inference_tool = MonaiBinaryClassifierInference(
        model=trainer.model,  # Reuse the already trained model
        config=inference_config,
        dataloader_specs=inference_dataloader_specs,
        output_transform=torch.nn.Softmax(dim=1),
        post_process=lambda x: x.cpu().numpy()
    )

    # Run inference
    print("Running inference with the trained model...")
    results1 = inference_tool.run_inference()
    print(f"Inference completed with {len(results1.case_identifiers)} cases")
    print(f"Class distribution: {sum(results1.predictions)} positive, {len(results1.predictions) - sum(results1.predictions)} negative")

    # Method 2: Create inference from checkpoint (loading model from saved file)
    print("\nRunning inference by loading model from checkpoint...")
    inference_tool2 = MonaiBinaryClassifierInference.from_checkpoint(
        model_class=MonaiBinaryClassifier,
        model_args={'config': classifier_config},
        config=inference_config,
        dataloader_specs=inference_dataloader_specs,
        output_transform=torch.nn.Softmax(dim=1),
        post_process=lambda x: x.cpu().numpy()
    )

    # Run inference
    results2 = inference_tool2.run_inference()
    print(f"Inference completed with {len(results2.case_identifiers)} cases")
    print(f"Class distribution: {sum(results2.predictions)} positive, {len(results2.predictions) - sum(results2.predictions)} negative")

    # Verify that results are the same using both methods
    predictions_match = all(p1 == p2 for p1, p2 in zip(results1.predictions, results2.predictions))
    print(f"Predictions match between both methods: {predictions_match}")

    # Calculate metrics if ground truth is available
    # In this example, we use test_data field as our "ground truth" for demonstration
    # (In real scenarios, you would use actual ground truth labels)
    if hasattr(test_ds, 'get_additional_properties'):
        true_labels = {}
        for i, case_id in enumerate(results1.case_identifiers):
            properties = test_ds.get_additional_properties(case_id)
            if properties and 'test_data' in properties:
                true_labels[case_id] = properties['test_data']

        if true_labels:
            print("\nPerformance metrics:")
            metrics = results1.get_performance_metrics(true_labels)
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric_name}: {value:.4f}")
                else:
                    print(f"  {metric_name}: {value}")

    print("\nDone!")
    return results1

if __name__ == "__main__":
    train_and_infer()