from enum import Enum
import os
from pathlib import Path
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Type, Optional, Union
import monai.networks.nets as monai_nets
import torch

from attributions.models.base_models import CriterionConfig, MergedNNUNetDatasetTrainer, MetricConfig, MetricGoal, OptimizerConfig, TrainingConfig
from nnunet_utils.dataloader_utils import get_nnunet_dataloaders
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

# Example usage
def main():

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
        """
        # Split each dataset
        if isinstance(split_ratio, float):
            split_ratio = (split_ratio, split_ratio)

        train_train, train_val = dataset1.random_split(split_ratio=split_ratio[0], shuffle=True)
        test_train, test_val = dataset2.random_split(split_ratio=split_ratio[1], shuffle=True)
        print(f"From the training of nnUNet for training domain classifier: {len(train_train)}, for validation: {len(train_val)}")
        print(f"From the test of nnUNet for trainining domain classifier: {len(test_train)}, for validation {len(test_val)}")

        # Merge training sets
        train_train.merge(test_train)
        train_val.merge(test_val)
        print(f"Training domain classifier dataset size: {len(train_train)}")
        print(f"Validation domain classifier dataset size: {len(train_val)}")
        """

        return dataset1.merge_and_split(dataset2, split_ratio, seed=seed) #train_train, train_val
    # Create configurations
    classifier_config = BinaryClassifierConfig(
        model_type=MonaiModelType.DENSENET121,
        num_input_channels=2,  # data + target channels
        dropout_prob=0.2
    )

    training_config = TrainingConfig(
        num_epochs=100,
        val_interval=5,
        num_train_iterations_per_epoch=250,
        num_val_iterations_per_epoch=150,
        metric=MetricConfig('f1', MetricGoal.MAXIMIZE),
        log_path=Path('./logs/monay_binary_classifier_test/'),
        save_path=Path('./models_save/monai_binary_classifier_test/'),
        device="cuda",
        verbosity=2
    )

    # Optional custom optimizer configuration
    optimizer_config = OptimizerConfig(
        optimizer_class=torch.optim.AdamW,
        optimizer_kwargs={'lr': 1e-4}
    )

    # Create trainer
    trainer = BinaryMergedNNUNetTrainer.create(
        classifier_config=classifier_config,
        training_config=training_config,
        optimizer_config=optimizer_config
    )

    # Get your dataloaders

    dataset_folder = '/home/jovyan/shared/pedro-maciasgordaliza/ms-data/nnunet_folders/nnUNet_preprocessed/Dataset001_MSSEG_FLAIR_Annotator1'
    folder1 = os.path.join(dataset_folder, 'nnUNetPlans_3d_fullres_train_images')
    folder2 = os.path.join(dataset_folder, 'nnUNetPlans_3d_fullres_test_images_ann2')


    # Create datasets
    train_ds, val_ds = create_split_datasets(folder1, folder2, split_ratio=(0.2, 0.8))
    print(f"Training dataset size: {len(train_ds)}")
    print(f"Validation dataset size: {len(val_ds)}")


    train_loader, val_loader = get_nnunet_dataloaders(
        dataset_json_path=os.path.join(dataset_folder, 'dataset.json'),
        dataset_plans_path=os.path.join(dataset_folder, 'nnUNetPlans.json'),
        dataset_tr=train_ds,
        dataset_val=val_ds,
        batch_size=8,
        num_processes=4
    )

    # Train
    results = trainer.train(train_loader, val_loader)

if __name__ == '__main__':
    main()