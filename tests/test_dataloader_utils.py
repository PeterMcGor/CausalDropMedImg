import pytest
import numpy as np
import os
from pathlib import Path
import json
import pickle
from unittest.mock import MagicMock, patch
import warnings

# Filter deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunet_utils.dataloader_utils import (
    AugmentationConfig,
    calculate_rotation_range,
    get_augmentation_config,
    get_nnunet_dataloaders
)
from nnunet_utils.dataset_utils import MergerNNUNetDataset

class TestDatasetCreator:
    """Helper class to create mock nnUNet dataset structure for testing"""
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.plans_name = "nnUNetPlans_3d_fullres"
        print(f"Creating test dataset at: {self.base_path}")

    def create_dataset_structure(self, case_identifiers):
        """Creates necessary dataset files including dataset.json and plans"""
        plans_dir = self.base_path / self.plans_name
        plans_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created plans directory at: {plans_dir}")

        for case_id in case_identifiers:
            np.random.seed(42)  # For reproducibility
            data = np.random.rand(1, 16, 16, 16)
            seg = np.random.randint(0, 2, size=(1, 16, 16, 16))

            # Save data files
            np.savez_compressed(plans_dir / f"{case_id}.npz", data=data)
            np.save(plans_dir / f"{case_id}_seg.npy", seg)

            # Create properties pickle matching real structure
            properties = {
                'sitk_stuff': {
                    'spacing': (1.0, 1.0, 1.0),
                    'origin': (0.0, 0.0, 0.0),
                    'direction': (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
                },
                'spacing': [1.0, 1.0, 1.0],
                'shape_before_cropping': (16, 16, 16),
                'bbox_used_for_cropping': [0, 16, 0, 16, 0, 16],
                'shape_after_cropping_and_before_resampling': (16, 16, 16),
                'class_locations': {
                    1: np.random.randint(0, 16, size=(10000, 4))  # Mock class locations
                },
                'test_data': 1,
                'deployment_data': True
            }

            with open(plans_dir / f"{case_id}.pkl", 'wb') as f:
                pickle.dump(properties, f)

        # Create dataset.json
        dataset_json = {
            "channel_names": {"0": "T1w"},
            "labels": {"background": 0, "CL": 1},
            "numTraining": 347,
            "file_ending": ".nii.gz",
            "name": "Dataset301_CL_Multisite"
        }
        with open(self.base_path / 'dataset.json', 'w') as f:
            json.dump(dataset_json, f)
        print(f"Created dataset.json at: {self.base_path / 'dataset.json'}")

        # Create plans file
        plans = {
            "dataset_name": "Dataset301_CL_Multisite",
            "plans_name": "nnUNetPlans",
            "original_median_spacing_after_transp": [1.0, 1.0, 1.0],
            "original_median_shape_after_transp": [138, 153, 174],
            "configurations": {
                "3d_fullres": {
                    "data_identifier": "nnUNetPlans_3d_fullres",
                    "preprocessor_name": "DefaultPreprocessor",
                    "batch_size": 2,
                    "patch_size": [112, 128, 160],
                    "median_image_size_in_voxels": [136.0, 153.0, 171.0],
                    "spacing": [1.0, 1.0, 1.0],
                    "normalization_schemes": ["ZScoreNormalization"],
                    "use_mask_for_norm": [True],
                    "UNet_class_name": "PlainConvUNet",
                    "UNet_base_num_features": 32,
                    "n_conv_per_stage_encoder": [2, 2, 2, 2, 2, 2],
                    "n_conv_per_stage_decoder": [2, 2, 2, 2, 2],
                    "num_pool_per_axis": [4, 5, 5],
                    "pool_op_kernel_sizes": [[1, 1, 1], [2, 2, 2], [2, 2, 2],
                                          [2, 2, 2], [2, 2, 2], [1, 2, 2]],
                    "conv_kernel_sizes": [[3, 3, 3]] * 6,
                    "unet_max_num_features": 320
                }
            }
        }
        with open(self.base_path / 'nnUNetPlans.json', 'w') as f:
            json.dump(plans, f)
        print(f"Created nnUNetPlans.json at: {self.base_path / 'nnUNetPlans.json'}")

        return plans_dir

def test_augmentation_config_creation():
    """Test creation of augmentation configuration"""
    patch_size = [112, 128, 160]  # Using actual patch size from your plans
    config = get_augmentation_config(patch_size)

    assert isinstance(config, AugmentationConfig)
    assert len(config.rotation_range) == 2
    assert isinstance(config.use_dummy_2d, bool)
    assert len(config.initial_patch_size) == 3
    assert len(config.mirror_axes) == 3

def test_rotation_range_calculation():
    """Test rotation range calculation for different patch sizes"""
    patch_size_real = [112, 128, 160]  # Your actual patch size
    range_real = calculate_rotation_range(patch_size_real)
    assert len(range_real) == 2

    patch_size_aniso = [32, 128, 160]  # Anisotropic case
    range_aniso = calculate_rotation_range(patch_size_aniso)
    assert len(range_aniso) == 2

@pytest.mark.parametrize("num_cases", [4, 8])
def test_get_nnunet_dataloaders(tmp_path_factory, num_cases):
    """Test the main dataloader creation function"""
    # Create test directory and dataset
    test_path = tmp_path_factory.mktemp("nnunet_test_data")
    creator = TestDatasetCreator(test_path)

    # Create case IDs and dataset structure
    case_ids = [f"case_{i}" for i in range(num_cases)]
    plans_dir = creator.create_dataset_structure(case_ids)

    # Create MergerNNUNetDataset instead of base nnUNetDataset
    base_dataset = MergerNNUNetDataset(
        folder=str(plans_dir),
        additional_data={'test_data': 1, 'deployment_data': lambda x, y: True}
    )

    # Split into train and validation
    train_cases = case_ids[:-2]  # Leave last 2 cases for validation
    val_cases = case_ids[-2:]

    dataset_tr = base_dataset.subset(train_cases)
    dataset_val = base_dataset.subset(val_cases)

    # Create dataloaders
    train_loader, val_loader = get_nnunet_dataloaders(
        dataset_json_path=str(test_path / 'dataset.json'),
        dataset_plans_path=str(test_path / 'nnUNetPlans.json'),
        dataset_tr=dataset_tr,
        dataset_val=dataset_val,
        batch_size=2,
        num_processes=0  # Use single thread for testing
    )

    # Test basic functionality
    assert train_loader is not None
    assert val_loader is not None

    # Test batch generation (catch any runtime errors)
    try:
        train_batch = next(train_loader)
        assert 'data' in train_batch
        assert 'properties' in train_batch

        val_batch = next(val_loader)
        assert 'data' in val_batch
        assert 'properties' in val_batch

    finally:
        # Cleanup
        if hasattr(train_loader, '_finish'):
            train_loader._finish()
        if hasattr(val_loader, '_finish'):
            val_loader._finish()