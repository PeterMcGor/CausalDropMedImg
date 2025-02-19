import os
from functools import wraps
from dataclasses import dataclass
from torch import device as torch_device

from typing import Any, Dict, Union, List, Tuple,Optional
import numpy as np
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from nnunetv2.configuration import ANISO_THRESHOLD
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from batchgenerators.utilities.file_and_folder_operations import load_json
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter

class CustomnnUNetDataLoader3D(nnUNetDataLoader3D):
    def __init__(self, *args, retain_from_properties=['test_data','deployment_data'], **kwargs):
        super().__init__(*args, **kwargs)
        self._temp_properties = []
        self.retain_from_properties = retain_from_properties

        # Wrap the original load_case method to capture properties
        original_load_case = self._data.load_case
        @wraps(original_load_case)
        def load_case_wrapper(*args, **kwargs):
            data, seg, properties = original_load_case(*args, **kwargs)
            self._temp_properties.append(properties)
            return data, seg, properties

        self._data.load_case = load_case_wrapper

    def generate_train_batch(self):
        # Clear temporary properties before new batch
        self._temp_properties = []

        # Get batch from parent class (which will now populate _temp_properties)
        batch = super().generate_train_batch()

        # Add collected properties to the batch
        batch['properties'] = [
            {k: prop[k] for k in set(self.retain_from_properties) & set(prop)}
            for prop in self._temp_properties
            ]
        return batch

@dataclass
class AugmentationConfig:
    """Configuration for data augmentation parameters."""
    rotation_range: Tuple[float, float]
    use_dummy_2d: bool
    initial_patch_size: Union[List[int], Tuple[int, ...]]
    mirror_axes: Tuple[int, ...]

def calculate_rotation_range(patch_size: Union[List[int], Tuple[int, ...]]) -> Tuple[float, float]:
    """Calculate rotation range based on patch dimensions."""
    if len(patch_size) == 2:
        return (
            (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            if max(patch_size) / min(patch_size) > 1.5
            else (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
        )
    return (
        (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
        if max(patch_size) / patch_size[0] > ANISO_THRESHOLD
        else (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
    )

def get_augmentation_config(patch_size: Union[List[int], Tuple[int, ...]]) -> AugmentationConfig:
    """
    Generate augmentation configuration based on patch size.

    Args:
        patch_size: Input patch dimensions

    Returns:
        AugmentationConfig with all necessary parameters

    Raises:
        ValueError: If patch_size dimensionality is not 2 or 3
    """
    dim = len(patch_size)
    if dim not in (2, 3):
        raise ValueError(f"Unsupported patch size dimensionality: {dim}D")

    use_dummy_2d = dim == 3 and (max(patch_size) / patch_size[0]) > ANISO_THRESHOLD
    rotation_range = calculate_rotation_range(patch_size)
    mirror_axes = (0, 1) if dim == 2 else (0, 1, 2)

    initial_size = get_patch_size(
        patch_size[-dim:],
        *([rotation_range] * 3),
        (0.85, 1.25)
    )

    if use_dummy_2d:
        initial_size[0] = patch_size[0]

    return AugmentationConfig(
        rotation_range=rotation_range,
        use_dummy_2d=use_dummy_2d,
        initial_patch_size=initial_size,
        mirror_axes=mirror_axes
    )

class DataloaderFactory:
    """Factory class for creating appropriate dataloaders."""

    def __init__(
        self,
        batch_size: int,
        label_manager: Any,
        oversample_foreground_percent: float,
        device: Union[str, torch_device]
    ):
        self.batch_size = batch_size
        self.label_manager = label_manager
        self.oversample_percent = oversample_foreground_percent
        self.device = device

    def create_dataloader(
        self,
        dataset: nnUNetDataset,
        patch_size: Union[List[int], Tuple[int, ...]],
        transforms: Any,
        is_training: bool = True
    ) -> Union[nnUNetDataLoader2D, nnUNetDataLoader3D]:
        """Create appropriate dataloader based on dimensionality."""
        initial_size = patch_size if not is_training else get_augmentation_config(patch_size).initial_patch_size

        loader_cls = nnUNetDataLoader2D if len(patch_size) == 2 else CustomnnUNetDataLoader3D #nnUNetDataLoader3D
        return loader_cls(
            data=dataset,
            batch_size=self.batch_size,
            patch_size=initial_size,
            final_patch_size=patch_size,
            label_manager=self.label_manager,
            oversample_foreground_percent=self.oversample_percent,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=transforms
        )

def create_augmenter(
    dataloader: Union[nnUNetDataLoader2D, nnUNetDataLoader3D],
    num_processes: int,
    is_training: bool,
    device: Union[str, torch_device]
) -> Union[SingleThreadedAugmenter, NonDetMultiThreadedAugmenter]:
    """Create appropriate augmenter based on number of processes."""
    if num_processes == 0:
        return SingleThreadedAugmenter(dataloader, None)

    is_cuda = str(device).lower() == 'cuda'
    not_train_processes = max(3, num_processes // 4)
    return NonDetMultiThreadedAugmenter(
        data_loader=dataloader,
        transform=None,
        num_processes=num_processes if is_training else not_train_processes,
        num_cached=max(6, num_processes // 2) if is_training else not_train_processes,
        seeds=None,
        pin_memory=is_cuda,
        wait_time=0.002
    )

def get_nnunet_dataloaders(
    #patch_size: Union[List[int], Tuple[int, ...]],
    dataset_json_path: str,
    dataset_plans_path: str,
    dataset_tr: nnUNetDataset,
    dataset_val: nnUNetDataset,
    configuration: str = '3d_fullres',
    deep_supervision_scales: Optional[List[float]] = None,
    is_cascaded: bool = False,
    batch_size: int = 2,
    oversample_foreground_percent: float = 0.33,
    num_processes: int = max(1, os.cpu_count() - 2),
    device: Union[str, torch_device] = 'cuda'
) -> Tuple[Union[SingleThreadedAugmenter, NonDetMultiThreadedAugmenter],
           Union[SingleThreadedAugmenter, NonDetMultiThreadedAugmenter]]:
    """
    Create training and validation dataloaders with appropriate augmentation.

    Args:
        patch_size: Dimensions of the input patches
        dataset_json_path: Path to dataset JSON configuration
        dataset_plans_path: Path to dataset plans
        dataset_tr: Training dataset
        dataset_val: Validation dataset
        configuration: Configuration name
        deep_supervision_scales: Scales for deep supervision
        is_cascaded: Whether this is a cascaded network
        batch_size: Batch size for training
        oversample_foreground_percent: Percentage of foreground oversampling
        num_processes: Number of processes for data loading
        device: Device to use for training

    Returns:
        Tuple of (training_augmenter, validation_augmenter)
    """
    # Initialize configuration
    dataset_json = load_json(dataset_json_path)
    plans_manager = PlansManager(dataset_plans_path)
    label_manager = plans_manager.get_label_manager(dataset_json)
    config_manager = plans_manager.get_configuration(configuration)

    # Get augmentation parameters
    aug_config = get_augmentation_config(config_manager.patch_size)

    # Create transforms
    tr_transforms = nnUNetTrainer.get_training_transforms(
        patch_size=config_manager.patch_size,
        rotation_for_DA=aug_config.rotation_range,
        deep_supervision_scales=deep_supervision_scales,
        mirror_axes=aug_config.mirror_axes,
        do_dummy_2d_data_aug=aug_config.use_dummy_2d,
        use_mask_for_norm=config_manager.use_mask_for_norm,
        is_cascaded=is_cascaded,
        foreground_labels=label_manager.foreground_labels,
        regions=label_manager.foreground_regions if label_manager.has_regions else None,
        ignore_label=label_manager.ignore_label
    )

    val_transforms = nnUNetTrainer.get_validation_transforms(
        deep_supervision_scales=deep_supervision_scales,
        is_cascaded=is_cascaded,
        foreground_labels=label_manager.foreground_labels,
        regions=label_manager.foreground_regions if label_manager.has_regions else None,
        ignore_label=label_manager.ignore_label
    )

    # Create dataloaders
    factory = DataloaderFactory(batch_size, label_manager, oversample_foreground_percent, device)
    train_loader = factory.create_dataloader(dataset_tr, config_manager.patch_size, tr_transforms, True)
    val_loader = factory.create_dataloader(dataset_val, config_manager.patch_size, val_transforms, False)

    # Create and initialize augmenters
    train_augmenter = create_augmenter(train_loader, num_processes,True, device)
    val_augmenter = create_augmenter(val_loader, num_processes, False, device)

    # Initialize augmenters
    _ = next(train_augmenter)
    _ = next(val_augmenter)

    return train_augmenter, val_augmenter


# Example usage:
if __name__ == "__main__":
    from time import time

    # Example configuration
    dataset_folder = '/opt/nnunet_resources/nnUNet_preprocessed/Dataset301_CL_Multisite/'
    folder = os.path.join(dataset_folder,'nnUNetPlans_3d_fullres')

    def process_case_id(case_id: str, nnunet_dataset=None) -> int:
        return 'deployment' in case_id

    print("Example 1: Basic Dataset Creation and Merging")
    print("-" * 50)
    # Create base dataset
    base_dataset = MergerNNUNetDataset(folder, additional_data={'test_data': 0, 'deployment_data': process_case_id})
    other_dataset = MergerNNUNetDataset(
        os.path.join(dataset_folder,'nnUNetPlans_3d_fullres_test_images'),
        additional_data={'test_data': 1,'deployment_data': process_case_id}
    )

    # Merge datasets
    base_dataset.merge(other_dataset)
    print(f"Total cases after merge: {len(base_dataset)}")
    # get a case and print the properties
    for case_id in base_dataset.dataset.keys():
        print(f'Case {case_id}: {base_dataset.dataset[case_id][base_dataset.DATASET_INFO]}')
        break  # Only print first case

    print("\nExample 2: Dataset Splitting")
    print("-" * 50)
    dataset_tr, dataset_val = base_dataset.random_split(split_ratio=0.8, shuffle=True)
    print(f"Training cases: {len(dataset_tr)}")
    print(f"Validation cases: {len(dataset_val)}")

    print("\nExample 3: Testing Dataloaders")
    print("-" * 50)

    # Create and setup dataloaders
    train_loader, val_loader = get_nnunet_dataloaders(
        dataset_json_path=os.path.join(dataset_folder, 'dataset.json'),
        dataset_plans_path=os.path.join(dataset_folder, 'nnUNetPlans_original.json'),
        dataset_tr=dataset_tr,
        dataset_val=dataset_val,
        batch_size=2,
        num_processes=8
    )

    # Explicitly start the dataloaders
    train_loader._start()
    val_loader._start()

    try:
        print("Starting dataloader test...")
        st = time()

        # Test training dataloader
        print("\nTesting training dataloader:")
        for i in range(5):  # Test 5 batches
            batch = next(train_loader)
            print(f"Batch {i}:")
            print(f"- Shape: {batch['data'].shape}")
            print(f"- Keys: {batch['properties']}")

        # Test validation dataloader
        print("\nTesting validation dataloader:")
        for i in range(5):  # Test 5 batches
            batch = next(val_loader)
            print(f"Batch {i}:")
            print(f"- Shape: {batch['data'].shape}")
            print(f"- Keys: {batch['properties']}")

        end = time()
        print(f"\nTime taken for test: {end - st:.2f} seconds")

    except Exception as e:
        print(f"Error during testing: {str(e)}")
        raise e

    finally:
        # Proper cleanup as in the original library
        print("\nCleaning up...")
        train_loader._finish()
        val_loader._finish()

