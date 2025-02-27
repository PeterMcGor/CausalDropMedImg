from enum import Enum
import os
from pathlib import Path
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple, Type, Optional, Union
import monai.networks.nets as monai_nets
import torch

from attributions.models.base_models import CriterionConfig, MetricConfig, MetricGoal, OptimizerConfig, TrainingConfig, InferenceConfig
from nnunet_utils.dataloader_utils import get_nnunet_dataloaders
from nnunet_utils.dataset_utils import MergerNNUNetDataset
from datetime import datetime

from attributions.models.merge_nnunet_trainers_inferers import MergedNNUNetDataLoaderSpecs, MergedNNUNetDataLoaderSpecs, MergedNNUNetDatasetTrainer, MergedNNUNetDatasetTrainerImages, MergedNNUNetInference
from nnunet_utils.dataset_utils import MergerNNUNetDataset

class MonaiModelType(Enum):
    """Some of the MONAI models, check monai.networks.nets for more:
    https://docs.monai.io/en/stable/networks.html
    """
    DENSENET121 = "DenseNet121"
    RESNET50 = "ResNet50"
    EFFICIENTNETB0 = "EfficientNetB0"
    EFFICIENTNETB7 = "EfficientNetB7"
    VGG16 = "VGG16"
    SENET154 = "SENet154"

@dataclass
class BinaryClassifierConfig:
    """Configuration for binary classifier"""
    model_type: MonaiModelType = MonaiModelType.DENSENET121
    spatial_dims: int = 3  # Fixed for 3D medical images
    num_input_channels: int = 2  # Default for data + target
    dropout_prob: float = 0.2
    pretrained: bool = False
    out_channels=2  # Binary classification

class BinaryClassificationMetricComputer:
    """Compute metrics for binary classification"""
    def compute(self, outputs: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        predictions = outputs.argmax(dim=1)
        pred_np = predictions.cpu().numpy()
        labels_np = labels.cpu().numpy()

        tp = np.sum((pred_np == 1) & (labels_np == 1))
        tn = np.sum((pred_np == 0) & (labels_np == 0))
        fp = np.sum((pred_np == 1) & (labels_np == 0))
        fn = np.sum((pred_np == 0) & (labels_np == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        balanced_acc = (recall + specificity) / 2

        return {
            'f1': f1,
            'balanced_acc': balanced_acc,
            'precision': precision,
            'recall': recall,
            'specificity': specificity
        }

class MonaiBinaryClassifier(torch.nn.Module):
    """Wrapper for MONAI classification models configured for binary classification"""
    def __init__(self, config: BinaryClassifierConfig):
        super().__init__()
        self.config = config

        # Get model class from MONAI
        model_class = getattr(monai_nets, config.model_type.value)

        # Initialize model with appropriate configuration
        model_params = self._get_model_params()
        self.model = model_class(**model_params)

        # Ensure output layer is appropriate for binary classification
        self._adjust_output_layer()

    def _get_model_params(self) -> dict:
        """Get model-specific parameters"""
        common_params = {
            'spatial_dims': self.config.spatial_dims,
            'in_channels': self.config.num_input_channels,
            'out_channels': self.config.out_channels  # Changed from num_classes to out_channels
        }

        # Add model-specific parameters
        if self.config.model_type in [MonaiModelType.DENSENET121]:
            common_params.update({
                'dropout_prob': self.config.dropout_prob,
                'pretrained': self.config.pretrained
            })
        elif self.config.model_type in [MonaiModelType.RESNET50]:
            common_params.update({
                'n_groups': 8,
                'pretrained': self.config.pretrained
            })
        # Add other model-specific configurations as needed

        return common_params

    def _adjust_output_layer(self):
        """Ensure output layer is configured for binary classification"""
        # This method can be expanded if specific models need output layer adjustments
        pass

    def forward(self, x):
        return self.model(x)

class BinaryMergedNNUNetTrainer(MergedNNUNetDatasetTrainer):
    """Specialized trainer for binary classification with MONAI networks"""
    @classmethod
    def create(
        cls,
        classifier_config: BinaryClassifierConfig,
        training_config: TrainingConfig,
        dataloader_specs: MergedNNUNetDataLoaderSpecs,
        optimizer_config: Optional[OptimizerConfig] = None,
        criterion_config: Optional[CriterionConfig] = None
    ) -> 'BinaryMergedNNUNetTrainer':
        """
        Factory method to create binary classification trainer

        Args:
            classifier_config: Configuration for the binary classifier
            training_config: General training configuration
            optimizer_config: Optional optimizer configuration
            criterion_config: Optional criterion configuration
        """
        # Create model
        model = MonaiBinaryClassifier(classifier_config)

        # Use binary classification specific metric computer
        metric_computer = BinaryClassificationMetricComputer()

        # Set default criterion for binary classification if not provided
        if criterion_config is None:
            criterion_config = CriterionConfig(
                criterion_class=torch.nn.CrossEntropyLoss,
                criterion_kwargs={}
            )

        # Create trainer using parent's create method
        return super().create(
            model=model,
            config=training_config,
            metric_computer=metric_computer,
            optimizer_config=optimizer_config,
            criterion_config=criterion_config,
            dataloader_specs=dataloader_specs,
            output_transform=torch.nn.Softmax(dim=1)
        )

    def _validate_inputs(self, batch: Dict[str, Any]) -> None:
        """Additional validation specific to binary classification"""
        if not isinstance(self.model, MonaiBinaryClassifier):
            raise ValueError("Model must be an instance of MonaiBinaryClassifier")

        # Validate input channels match configuration
        expected_channels = self.model.config.num_input_channels
        actual_channels = batch['data'].shape[1] + batch['target'].shape[1]
        if actual_channels != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} input channels but got {actual_channels}"
            )
class BinaryMergedNNUNetTrainerImages(BinaryMergedNNUNetTrainer,MergedNNUNetDatasetTrainerImages):

    def _validate_inputs(self, batch: Dict[str, Any]) -> None:
        """Additional validation specific to binary classification"""
        if not isinstance(self.model, MonaiBinaryClassifier):
            raise ValueError("Model must be an instance of MonaiBinaryClassifier")

        # Validate input channels match configuration
        expected_channels = self.model.config.num_input_channels
        actual_channels = batch['data'].shape[1] #+ batch['target'].shape[1]
        if actual_channels != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} input channels but got {actual_channels}"
            )



class BinaryClassificationInferenceResults:
    """Helper class to process and analyze binary classification results"""
    def __init__(self, outputs: List[np.ndarray], case_identifiers: List[str]):
        self.outputs = outputs
        self.case_identifiers = case_identifiers
        self.predictions = [np.argmax(output, axis=0) if output.ndim > 0 else int(output > 0.5) for output in outputs]
        self.probabilities = [output[1] if output.ndim > 0 and output.shape[0] > 1 else output for output in outputs]

    def get_predictions_dict(self) -> Dict[str, int]:
        """Return case_id to prediction mapping"""
        return {case_id: pred for case_id, pred in zip(self.case_identifiers, self.predictions)}

    def get_probabilities_dict(self) -> Dict[str, float]:
        """Return case_id to probability mapping"""
        return {case_id: float(prob) for case_id, prob in zip(self.case_identifiers, self.probabilities)}

    def get_results_by_threshold(self, threshold: float = 0.5) -> Dict[str, int]:
        """Get binary predictions using custom threshold"""
        return {
            case_id: int(prob >= threshold)
            for case_id, prob in self.get_probabilities_dict().items()
        }

    def get_performance_metrics(self, true_labels: Dict[str, int]) -> Dict[str, float]:
        """
        Calculate performance metrics given ground truth labels

        Args:
            true_labels: Dictionary mapping case_id to true label (0 or 1)

        Returns:
            Dictionary with performance metrics (accuracy, precision, recall, f1, etc.)
        """
        predictions = self.get_predictions_dict()

        # Get case IDs that exist in both predictions and true_labels
        common_cases = set(predictions.keys()).intersection(true_labels.keys())

        if not common_cases:
            return {"error": "No matching case IDs between predictions and true labels"}

        y_pred = [predictions[case_id] for case_id in common_cases]
        y_true = [true_labels[case_id] for case_id in common_cases]

        # Calculate metrics
        tp = sum((p == 1 and t == 1) for p, t in zip(y_pred, y_true))
        tn = sum((p == 0 and t == 0) for p, t in zip(y_pred, y_true))
        fp = sum((p == 1 and t == 0) for p, t in zip(y_pred, y_true))
        fn = sum((p == 0 and t == 1) for p, t in zip(y_pred, y_true))

        accuracy = (tp + tn) / len(y_true) if y_true else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        balanced_acc = (recall + specificity) / 2

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "balanced_acc": balanced_acc,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn
        }


class MonaiBinaryClassifierInferenceAlpha(MergedNNUNetInference):
    """Specialized inference class for MONAI binary classifiers"""

    def __init__(
        self,
        model: torch.nn.Module,
        config: InferenceConfig,
        dataloader_specs: MergedNNUNetDataLoaderSpecs,
        output_transform: Optional[Callable] = None,
        post_process: Optional[Callable] = None,
        probability_threshold: float = 0.5
    ):
        # Set default output transform to softmax if not provided
        if output_transform is None:
            output_transform = torch.nn.Softmax(dim=1)

        super().__init__(
            model=model,
            config=config,
            dataloader_specs=dataloader_specs,
            output_transform=output_transform,
            post_process=post_process
        )
        self.probability_threshold = probability_threshold

    @classmethod
    def from_checkpoint(
        cls,
        model_class: type,
        model_args: Dict[str, Any],
        config: InferenceConfig,
        dataloader_specs: MergedNNUNetDataLoaderSpecs,
        output_transform: Optional[Callable] = None,
        post_process: Optional[Callable] = None,
        probability_threshold: float = 0.5
    ) -> 'MonaiBinaryClassifierInference':
        """Create binary classifier inference from checkpoint"""
        # Initialize the model

        # Load weights from checkpoint

        #import numpy as np
        #from torch.serialization import add_safe_globals
        #add_safe_globals([np.core.multiarray.scalar])
        checkpoint = torch.load(config.model_path, map_location=config.device, weights_only=False)
        model = model_class(**model_args)

        model.load_state_dict(checkpoint['model_state_dict'])

        return cls(
            model=model,
            config=config,
            dataloader_specs=dataloader_specs,
            output_transform=output_transform,
            post_process=post_process,
            probability_threshold=probability_threshold
        )

    def run_inference(self, dataloader=None):
        """Run inference and return structured binary classification results"""
        # Call parent method to get raw results
        raw_results = super().run_inference(dataloader)

        # Process results for binary classification
        return BinaryClassificationInferenceResults(
            outputs=raw_results['outputs'],
            case_identifiers=raw_results['case_identifiers']
        )

    def _process_batch(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Process batch from MergerNNUNetDataset format for binary classification inference"""
        # Combine data and target same as in training
        inputs = torch.cat([
            batch['data'],
            batch['target']
        ], dim=1).to(self.device)

        return inputs

    def _extract_case_identifiers(self, batch: Dict[str, Any]) -> List[str]:
        """Extract case identifiers from nnUNet batch properties"""
        # Try to extract case identifiers from properties
        if 'properties' in batch and all('case_identifier' in p for p in batch['properties']):
            return [p['case_identifier'] for p in batch['properties']]

        # Fallback - try to extract from other property fields
        if 'properties' in batch and all('dataset_identifier' in p for p in batch['properties']):
            return [f"{p.get('dataset_identifier', '')}_{i}" for i, p in enumerate(batch['properties'])]

        # Generic fallback
        return [f"case_{i}" for i in range(len(batch['data']))]

    def _extract_metadata(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Extract additional metadata from batch"""
        metadata = super()._extract_metadata(batch)

        # Extract test_data flag if available
        if 'properties' in batch and all('test_data' in p for p in batch['properties']):
            metadata['test_data'] = [p['test_data'] for p in batch['properties']]

        # Extract any other metadata specific to your implementation
        if 'properties' in batch and all('deployment_data' in p for p in batch['properties']):
            metadata['deployment_data'] = [p['deployment_data'] for p in batch['properties']]

        return metadata


# Example usage function
def run_monai_binary_inference(
    model_path: Union[str, Path],
    dataset_folder: str,
    output_folder: Union[str, Path] = None,
    device: str = "cuda"
) -> BinaryClassificationInferenceResults:
    """
    Run inference with MONAI binary classifier

    Args:
        model_path: Path to saved model checkpoint
        dataset_folder: Path to nnUNet preprocessed dataset folder
        output_folder: Path to save inference results (if None, uses model directory)
        device: Device to run inference on ('cuda' or 'cpu')

    Returns:
        Processed inference results
    """


    # Set output folder if not provided
    if output_folder is None:
        model_path = Path(model_path)
        output_folder = model_path.parent / f"inference_{model_path.stem}"

    # Create output directory
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Load model checkpoint to get model configuration
    checkpoint = torch.load(model_path, map_location='cpu')

    # Create inference configuration
    inference_config = InferenceConfig(
        model_path=Path(model_path),
        device=device,
        output_path=Path(output_folder),
        verbosity=2,
        save_outputs=True,
        save_format="npy"
    )

    # Create dataset for inference (adapt this to your dataset structure)
    test_folder = None  # Set this based on your dataset structure
    if test_folder is None:
        # Try to find test folder based on dataset structure
        for folder in Path(dataset_folder).glob("*test*"):
            if folder.is_dir():
                test_folder = str(folder)
                break

    if test_folder is None:
        # Fallback to using any available folder
        for folder in Path(dataset_folder).glob("nnUNetPlans_3d_fullres*"):
            if folder.is_dir():
                test_folder = str(folder)
                break

    if test_folder is None:
        raise ValueError(f"Could not find a suitable test folder in {dataset_folder}")

    print(f"Using test folder: {test_folder}")

    # Create dataset for inference
    dataset_inference = MergerNNUNetDataset(
        test_folder,
        additional_data={'source': test_folder, 'test_data': 1}  # Set all as test data
    )

    # Create dataloader specs
    dataloader_specs = MergedNNUNetDataLoaderSpecs(
        dataset_json_path=str(Path(dataset_folder) / 'dataset.json'),
        dataset_plans_path=str(Path(dataset_folder) / 'nnUNetPlans.json'),
        dataset_train=None,  # Not needed for inference
        dataset_val=dataset_inference,
        batch_size=4,
        num_processes=4,
        unpack_data=True,
        cleanup_unpacked=True
    )

    # Create model configuration (adapt parameters as needed)
    classifier_config = BinaryClassifierConfig(
        model_type=MonaiModelType.DENSENET121,  # Update if using a different model
        num_input_channels=2,  # data + target channels
        dropout_prob=0.2
    )

    # Create inference object
    inference_tool = MonaiBinaryClassifierInference.from_checkpoint(
        model_class=MonaiBinaryClassifier,
        model_args={'config': classifier_config},
        config=inference_config,
        dataloader_specs=dataloader_specs,
        # Default output_transform is softmax
        # Default post_process is None (leaves as torch tensor)
        post_process=lambda x: x.cpu().numpy()  # Convert to numpy array
    )

    # Run inference
    results = inference_tool.run_inference()

    # Save prediction results to CSV
    import pandas as pd
    predictions_df = pd.DataFrame({
        'case_id': results.case_identifiers,
        'prediction': results.predictions,
        'probability': results.probabilities
    })

    csv_path = Path(output_folder) / "predictions.csv"
    predictions_df.to_csv(csv_path, index=False)
    print(f"Saved predictions to {csv_path}")

    return results


class MonaiBinaryClassifierInference(MergedNNUNetInference):
    """Specialized inference class for MONAI binary classifiers"""

    def __init__(
        self,
        model: torch.nn.Module,
        config: InferenceConfig,
        dataloader_specs: MergedNNUNetDataLoaderSpecs,
        output_transform: Optional[Callable] = None,
        post_process: Optional[Callable] = None,
        probability_threshold: float = 0.5,
        num_processes: int = max(1, os.cpu_count() - 2)  # CHANGED: Added this parameter
    ):
        # Set default output transform to softmax if not provided
        if output_transform is None:
            output_transform = torch.nn.Softmax(dim=1)

        super().__init__(
            model=model,
            config=config,
            dataloader_specs = dataloader_specs,
            output_transform=output_transform,
            post_process=post_process,
            num_processes=num_processes
        )
        self.probability_threshold = probability_threshold
    @classmethod
    def from_checkpoint(
        cls,
        model_class: type,
        model_args: Dict[str, Any],
        config: InferenceConfig,
        dataloader_specs: MergedNNUNetDataLoaderSpecs,
        output_transform: Optional[Callable] = None,
        post_process: Optional[Callable] = None,
        probability_threshold: float = 0.5,
        num_processes: int = max(1, os.cpu_count() - 2)  # CHANGED
    ) -> 'MonaiBinaryClassifierInference':
        """Create binary classifier inference from checkpoint"""
        # Initialize the model
        model = model_class(**model_args)

        # Load weights from checkpoint
        checkpoint = torch.load(config.model_path, map_location=config.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        return cls(
            model=model,
            config=config,
            dataloader_specs = dataloader_specs,
            output_transform=output_transform,
            post_process=post_process,
            probability_threshold=probability_threshold,
            num_processes=num_processes               # CHANGED
        )

class MonaiBinaryClassifierInferenceImages(MonaiBinaryClassifierInference):

    def _process_batch(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Process batch using only batch['data'] for inputs (no target concatenation)"""
        # Only use the data part, don't try to concatenate with target
        inputs = batch['data'].to(self.device)
        return inputs



# Example usage
def main():
    dataset = 'Dataset824_FLAWS-HCO'
    dataset_folder = '/home/jovyan/nnunet_data/nnUNet_preprocessed/'+dataset

    def create_split_datasets(
        train_folder: str,
        test_folder: str,
        split_ratio: Union[Tuple[float, float], float] = 0.8,
        seed: int = 42
    ) -> Tuple[MergerNNUNetDataset, MergerNNUNetDataset]:
        """
        Create training and validation datasets by:
        1. Splitting each folder into train/val
        2. Merging the train portions and val portions separately
        """
        def process_case_id(case_id: str, nnunet_dataset=None) -> int:
            return 'deployment' in case_id

        # Create datasets for each folder
        dataset1 = MergerNNUNetDataset(
            train_folder,
            additional_data={'source': train_folder, 'deployment_data': process_case_id, 'test_data': 0}
        )
        dataset2 = MergerNNUNetDataset(
            test_folder,
            additional_data={'source': test_folder, 'deployment_data': process_case_id, 'test_data': 1}
        )
        return dataset1.merge_and_split(dataset2, split_ratio, seed=seed) #train_train, train_val
    # Create configurations
    classifier_config = BinaryClassifierConfig(
        model_type=MonaiModelType.DENSENET121,
        num_input_channels= 2,#1,#2,  # data + target channels
        dropout_prob=0.5
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_config = TrainingConfig(
        num_epochs=50,
        val_interval=5,
        num_train_iterations_per_epoch=250, #250
        num_val_iterations_per_epoch=20, #150
        metric=MetricConfig('f1', MetricGoal.MAXIMIZE),
        log_path=Path(f'./logs/monay_binary_classifier_test_just_images_50B/{dataset}_{timestamp}'),
        save_path=Path(f'./models_save/monai_binary_classifier_test_just_images_50B/{dataset}_{timestamp}'),
        device="cuda",
        verbosity=2
    )

    # Optional custom optimizer configuration
    optimizer_config = OptimizerConfig(
        optimizer_class=torch.optim.AdamW,
        optimizer_kwargs={'lr': 1e-4}
    )

    dataset_folder = '/home/jovyan/nnunet_data/nnUNet_preprocessed/Dataset824_FLAWS-HCO'
    folder1 = os.path.join(dataset_folder, 'nnUNetPlans_3d_fullres_train_images')
    folder2 = os.path.join(dataset_folder, 'nnUNetPlans_3d_fullres_test_images')

    # Create datasets
    train_ds, val_ds = create_split_datasets(folder1, folder2, split_ratio=(0.2, 0.8))
    print(f"Training dataset size: {len(train_ds)}")
    print(f"Validation dataset size: {len(val_ds)}")

    dataloader_specs = MergedNNUNetDataLoaderSpecs(
        dataset_json_path=os.path.join(dataset_folder, 'dataset.json'),
        dataset_plans_path=os.path.join(dataset_folder, 'nnUNetPlans.json'),
        dataset_train=train_ds,
        dataset_val=val_ds,
        batch_size=4,
        num_processes=12,
        unpack_data=True
        )

    # Create trainer
    #BinaryMergedNNUNetTrainerImages.create( #
    trainer = BinaryMergedNNUNetTrainer.create(
        classifier_config=classifier_config,
        training_config=training_config,
        optimizer_config=optimizer_config,
        dataloader_specs=dataloader_specs
    )

    # Train
    results = trainer.train()

if __name__ == '__main__':
    main()