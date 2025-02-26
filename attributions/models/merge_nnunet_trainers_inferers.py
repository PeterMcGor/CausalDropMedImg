
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional
import numpy as np
import torch

from nnunetv2.training.dataloading.utils import unpack_dataset

from attributions.models.base_models import BaseTrainerWithSpecs, CriterionConfig, DataLoaderSpecs, OptimizerConfig, TrainingConfig, InferenceConfig
from nnunet_utils.dataset_utils import MergerNNUNetDataset

from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from nnunet_utils.dataloader_utils import get_nnunet_dataloaders
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from batchgenerators.utilities.file_and_folder_operations import load_json
from nnunetv2.inference.data_iterators import preprocessing_iterator_fromfiles


class MergedNNUNetDataLoaderSpecs(DataLoaderSpecs):
    """Specifications for MergerNNUNetDataset and its dataloaders"""
    def __init__(self,
                 dataset_json_path: str,
                 dataset_plans_path: str,
                 dataset_train: MergerNNUNetDataset,
                 dataset_val: MergerNNUNetDataset,
                 unpack_data=True,
                 cleanup_unpacked=True,
                 inference=False,
                   **kwargs
                 ):
        super().__init__(
            loader_type=(SingleThreadedAugmenter, NonDetMultiThreadedAugmenter),
            dataset_type=MergerNNUNetDataset,
            batch_keys=[MergerNNUNetDataset.BATCH_IMAGES_KEY, MergerNNUNetDataset.BATCH_SEG_KEY, MergerNNUNetDataset.BATCH_PROPERTIES_KEY],
            **kwargs
        )
        assert isinstance(dataset_json_path, str), "dataset_json_path must be a string"
        assert isinstance(dataset_plans_path, str), "dataset_plans_path must be a string"
        assert isinstance(dataset_train, MergerNNUNetDataset), "dataset_train must be an instance of MergerNNUNetDataset"
        assert isinstance(dataset_val, MergerNNUNetDataset), "dataset_val must be an instance of MergerNNUNetDataset"

        self.dataset_json_path = dataset_json_path
        self.dataset_plans_path = dataset_plans_path
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.unpack_data = unpack_data
        self.cleanup_unpacked = cleanup_unpacked
        self.inference = inference

        self.train_loader, self.val_loader = get_nnunet_dataloaders(
        dataset_json_path=self.dataset_json_path,
        dataset_plans_path=self.dataset_plans_path,
        dataset_tr=self.dataset_train,
        dataset_val=self.dataset_val,
        batch_size=self.batch_size,
        num_processes=self.num_processes,
        inference=self.inference
         )

    def validate_batch(self, batch: Dict[str, Any]) -> bool:
        """Validate batch structure from nnUNet dataloader"""
        if not super().validate_batch(batch):
            return False

        # Validate properties contain required fields
        #if not all('test_data' in p for p in batch['properties']):
        #    return False

        # Validate data and inference_env shapes match in batch dimension
        if batch[MergerNNUNetDataset.BATCH_IMAGES_KEY].shape[0] != batch[MergerNNUNetDataset.BATCH_SEG_KEY].shape[0]:
            return False

        return True

class MergedNNUNetDatasetTrainer(BaseTrainerWithSpecs):
    """Trainer for models using MergerNNUNetDataset and nnUNet dataloaders"""
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        config: TrainingConfig,
        metric_computer: Any,
        dataloader_specs: MergedNNUNetDataLoaderSpecs,
        output_transform: Optional[Callable] = None
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
            metric_computer=metric_computer,
            dataloader_specs=dataloader_specs
        )
        self.output_transform = output_transform or (lambda x: x)

    @classmethod
    def create(
        cls,
        model: torch.nn.Module,
        config: TrainingConfig,
        metric_computer: Any,
        dataloader_specs: MergedNNUNetDataLoaderSpecs,
        optimizer_config: Optional[OptimizerConfig] = None,
        criterion_config: Optional[CriterionConfig] = None,
        output_transform: Optional[Callable] = None
    ) -> 'MergedNNUNetDatasetTrainer':
        """Factory method to create a trainer instance"""
        optimizer_config = optimizer_config or OptimizerConfig()
        criterion_config = criterion_config or CriterionConfig()

        optimizer = optimizer_config.optimizer_class(
            model.parameters(),
            **optimizer_config.optimizer_kwargs
        )

        criterion = criterion_config.criterion_class(
            **criterion_config.criterion_kwargs
        )

        return cls(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
            dataloader_specs=dataloader_specs,
            metric_computer=metric_computer,
            output_transform=output_transform
        )

    def train(self):
        if self.dataloader_specs.unpack_data: # from nnUNet code
            folders_unpack = {ds.folder for ds in (
                                self.dataloader_specs.dataset_train.get_merged_datasets() +
                                self.dataloader_specs.dataset_val.get_merged_datasets()
                            )}
            super()._log(f'unpacking datasets...{folders_unpack}')
            for folder in folders_unpack:
                unpack_dataset(folder, unpack_segmentation=True, overwrite_existing=False, num_processes=self.dataloader_specs.num_processes, verify_npy=True)
            self._log('unpacking done...')

        # call the real training
        results = super().train(self.dataloader_specs.train_loader, self.dataloader_specs.val_loader)

        # Delete unpacked .npy files after training
        if self.dataloader_specs.cleanup_unpacked:
            for folder in folders_unpack:
                self._log(f"Cleaining unpacked files in {folder}")
                for npy_file in Path(folder).rglob('*.npy'):
                    try:
                        os.remove(npy_file)
                        self._log(f"Deleted file: {npy_file}")
                    except Exception as e:
                        self._log(f"Error deleting file {npy_file}: {e}", level=2)
        return results

    def _process_batch(self, batch: Dict[str, Any]) -> tuple:
        """Process batch from MergerNNUNetDataset format"""
        inputs = torch.cat([
            batch['data'],
            batch['target']
        ], dim=1).to(self.config.device)

        labels = torch.tensor(
            [p['test_data'] for p in batch['properties']],
            dtype=torch.long
        ).to(self.config.device)

        return inputs, labels

    def _compute_batch_metrics(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """Compute metrics for a batch"""
        transformed_outputs = self.output_transform(outputs)
        return self.metric_computer.compute(transformed_outputs, labels)

class MergedNNUNetDatasetTrainerImages(MergedNNUNetDatasetTrainer):
    """Trainer for models using MergerNNUNetDataset, but only uses batch['data'] as input."""

    def _process_batch(self, batch: Dict[str, Any]) -> tuple:
        """Process batch using only batch['data'] for inputs (no target concatenation)"""
        inputs = batch['data'].to(self.config.device)

        # Keep label extraction logic unchanged
        labels = torch.tensor(
            [p['test_data'] for p in batch['properties']],
            dtype=torch.long
        ).to(self.config.device)

        return inputs, labels




class MergedNNUNetInference:
    """Simplified inference class for nnUNet-based models"""
    def __init__(
        self,
        model: torch.nn.Module,
        config: InferenceConfig,
        dataloader_specs: MergedNNUNetDataLoaderSpecs,
        output_transform: Optional[Callable] = None,
        post_process: Optional[Callable] = None,
        num_processes: int = max(1, os.cpu_count() - 2)
    ):
        self.model = model
        self.config = config
        self.output_transform = output_transform or (lambda x: x)
        self.post_process = post_process or (lambda x: x)
        self.device = torch.device(config.device if isinstance(config.device, str) else config.device)
        self.model = self.model.to(self.device)
        self.dataloader = dataloader_specs.val_loader
        self.dataloader.infinite = False

        # Setup nnUNet components
        """
        self.plans_manager = PlansManager(dataset_plans_path)
        self.dataset_json = load_json(dataset_json_path)
        self.config_manager = self.plans_manager.get_configuration(architecture_config)

        # Create the data iterator
        self.dataloader = preprocessing_iterator_fromfiles(
            list_of_lists=config.input_folders,
            list_of_segs_from_prev_stage_files=None,
            output_filenames_truncated=None,
            plans_manager=self.plans_manager,
            dataset_json=self.dataset_json,
            configuration_manager=self.config_manager,
            num_processes=num_processes,
            pin_memory="cuda" in str(config.device),
            verbose=config.verbosity > 0
        )
        """


    def run_inference(self):
        """Run inference using the configured dataloader"""
        self.model.eval()
        results = []

        print("Starting inference...")
        with torch.no_grad():
            for batch_idx, preprocessed in enumerate(self.dataloader):
                # Get data properties and case information
                properties = preprocessed['keys']
                case_id = properties#os.path.basename(properties.get('list_of_data_files', ['unknown'])[0]).split('.')[0]
                print(f"Processing batch {batch_idx}, case: {case_id}")


                # Process batch for binary classification (combine data and target channels)
                inputs = self._process_batch(preprocessed)
                # Move data to device
                inputs = inputs.to(self.device)

                # Run model
                outputs = self.model(inputs)

                # Apply output transform (e.g., softmax)
                transformed_outputs = self.output_transform(outputs)

                # Get predictions
                predictions = torch.argmax(transformed_outputs, dim=1)

                # Move results to CPU
                predictions = predictions.cpu().numpy()
                probabilities = transformed_outputs.cpu().numpy()

                # Store results
                case_result = {
                    'case_id': case_id,
                    'prediction': predictions,
                    'probabilities': probabilities,
                    'properties': properties
                }
                results.append(case_result)

                print(f"Finished case {case_id}. Prediction: {predictions}")

                # Save results if output path is specified
                if self.config.output_path:
                    output_file = self.config.output_path / f"{case_id}_result.npz"
                    np.savez(
                        output_file,
                        prediction=predictions,
                        probabilities=probabilities
                    )
                    print(f"Results saved to {output_file}")

        print(f"Inference completed for {len(results)} cases")
        return results


    def _process_batch(self, batch: Dict[str, Any]) -> tuple:
        """Process batch from MergerNNUNetDataset format"""
        inputs = torch.cat([
            batch['data'],
            batch['target']
        ], dim=1).to(self.config.device)
        return inputs