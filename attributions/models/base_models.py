import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Union, Type, Tuple
import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter


import tqdm


from nnunetv2.inference.data_iterators import preprocessing_iterator_fromfiles
from nnunet_utils.dataset_utils import MergerNNUNetDataset
from batchgenerators.utilities.file_and_folder_operations import load_json


# ===== Configuration Classes =====
class MetricGoal(Enum):
    MAXIMIZE = 'maximize'
    MINIMIZE = 'minimize'

@dataclass
class MetricConfig:
    """Configuration for a metric"""
    name: str
    goal: MetricGoal
    better_than: Callable[[float, float], bool] = None

    def __post_init__(self):
        if self.better_than is None:
            self.better_than = (lambda x, y: x > y) if self.goal == MetricGoal.MAXIMIZE else (lambda x, y: x < y)

@dataclass
class TrainingConfig:
    """Configuration for model training"""
    num_epochs: int
    val_interval: int
    metric: MetricConfig
    num_train_iterations_per_epoch: int
    num_val_iterations_per_epoch: int
    save_path: Path = Path("models")
    log_path: Path = Path("logs")
    device: Union[str, torch.device] = "cuda"
    verbosity: int = 1

@dataclass
class OptimizerConfig:
    """Configuration for optimizer"""
    optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam
    optimizer_kwargs: Dict[str, Any] = field(default_factory=lambda: {'lr': 1e-4})

@dataclass
class CriterionConfig:
    """Configuration for loss function"""
    criterion_class: Type[torch.nn.Module] = torch.nn.CrossEntropyLoss
    criterion_kwargs: Dict[str, Any] = field(default_factory=dict)

# ===== Dataloader Specifications =====
class DataLoaderSpecs:
    """Base class for dataloader specifications"""
    def __init__(
        self,
        loader_type: Union[Type, Tuple[Type, ...]],
        dataset_type: Union[Type, Tuple[Type, ...]],
        batch_keys: list[str],
        batch_size: int = 2,
        num_processes: int = max(1, os.cpu_count() - 2)
    ):
        self.loader_type = loader_type
        self.dataset_type = dataset_type
        self.batch_keys = batch_keys
        self.batch_size = batch_size
        self.num_processes = num_processes

    def validate_loader(self, loader: Any) -> bool:
        """Validate that loader meets specifications"""
        return isinstance(loader, self.loader_type)

    def validate_dataset(self, dataset: Any) -> bool:
        """Validate that dataset meets specifications"""
        return isinstance(dataset, self.dataset_type)

    def validate_batch(self, batch: Dict[str, Any]) -> bool:
        """Validate that batch contains required keys"""
        return all(key in batch for key in self.batch_keys)

# ===== Base Trainer =====
class BaseTrainerWithSpecs(ABC):
    """Base trainer class that works with dataloader specifications"""
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        config: TrainingConfig,
        metric_computer: Any,
        dataloader_specs: DataLoaderSpecs
    ):
        # Convert device string to torch.device if needed
        self.device = (torch.device(config.device)
                      if isinstance(config.device, str)
                      else config.device)
        # Move model to correct device first
        self.model = model.to(config.device)
        # Create optimizer after model is on correct device
        self.optimizer = optimizer
        # Move criterion to same device
        self.criterion = criterion.to(config.device)
        self.config = config
        self.metric_computer = metric_computer
        self.dataloader_specs = dataloader_specs

        # Setup paths and logging
        self._setup_paths_and_logging()

    def _ensure_device_consistency(self, x: torch.Tensor) -> None:
        """Ensure model and tensors are on the same device"""
        model_device = next(self.model.parameters()).device
        if x.device != model_device:
            self.model = self.model.to(x.device)
            self.criterion = self.criterion.to(x.device)

    def _setup_paths_and_logging(self):
        """Setup paths and initialize logging"""
        self.config.save_path.mkdir(parents=True, exist_ok=True)
        self.config.log_path.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(
            log_dir=str(self.config.log_path),
            flush_secs=30
        )

        # Initialize metric tracking based on goal
        self.best_metric_value = float('inf') if self.config.metric.goal == MetricGoal.MINIMIZE else float('-inf')
        self.best_epoch = -1

    def _validate_dataloader(self, dataloader: Any) -> bool:
        """Validate dataloader meets specifications"""
        if not hasattr(dataloader, '__iter__'):
            return False
        return True#self.dataloader_specs.validate_loader(dataloader)

    def _log(self, message: str, level: int = 1):
        """Log message if verbosity level is sufficient"""
        if self.config.verbosity >= level:
            print(message)

    @abstractmethod
    def _process_batch(self, batch: Dict[str, Any]) -> tuple:
        """Process a batch of data"""
        pass

    @abstractmethod
    def _compute_batch_metrics(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """Compute metrics for a batch"""
        pass

    def train_epoch(self, train_loader: Any, epoch: int) -> float:
        """Run single training epoch"""
        if not isinstance(train_loader, Iterator):
            train_loader = iter(train_loader)

        if not self._validate_dataloader(train_loader):
            raise ValueError("Invalid train_loader type")

        self.model.train()
        epoch_loss = 0

        # Use configured number of iterations
        if self.config.verbosity >= 2:
            iterator = tqdm.tqdm(range(self.config.num_train_iterations_per_epoch))
        else:
            iterator = range(self.config.num_train_iterations_per_epoch)

        for i in iterator:
            batch = next(train_loader)
            if not self.dataloader_specs.validate_batch(batch):
                raise ValueError("Invalid batch structure")

            inputs, labels = self._process_batch(batch)
            # Ensure device consistency
            #self._ensure_device_consistency(inputs)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

            # Update progress bar if using detailed logging
            if self.config.verbosity >= 2:
                iterator.set_postfix({'loss': loss.item()})

        avg_loss = epoch_loss / self.config.num_train_iterations_per_epoch
        self.writer.add_scalar('Loss/train', avg_loss, epoch)

        # Normal verbosity logging
        if self.config.verbosity >= 1:
            self._log(f"Epoch {epoch + 1} - Average training loss: {avg_loss:.4f}")

        return avg_loss

    def validate(self, val_loader: Any, epoch: int) -> Dict[str, float]:
        if not isinstance(val_loader, Iterator):
            val_loader = iter(val_loader)
        """Run validation"""
        if not self._validate_dataloader(val_loader):
            raise ValueError("Invalid val_loader type")

        self.model.eval()
        metrics_sum = {}

        with torch.no_grad():
            for i in range(self.config.num_val_iterations_per_epoch):
                # Get next batch from the infinite loader
                batch = next(val_loader)
                if not self.dataloader_specs.validate_batch(batch):
                    raise ValueError("Invalid batch structure")

                inputs, labels = self._process_batch(batch)
                # Ensure device consistency
                # self._ensure_device_consistency(inputs)
                outputs = self.model(inputs)
                batch_metrics = self._compute_batch_metrics(outputs, labels)

                for key, value in batch_metrics.items():
                    metrics_sum[key] = metrics_sum.get(key, 0) + value

        # Average metrics
        metrics_avg = {
            key: value / self.config.num_val_iterations_per_epoch
            for key, value in metrics_sum.items()
        }

        self.writer.add_scalar(
            f'Metric/{self.config.metric.name}',
            metrics_avg[self.config.metric.name],
            epoch
        )

        # Detailed validation logging
        if self.config.verbosity >= 2:
            self._log("\nValidation Metrics:", 2)
            for key, value in metrics_avg.items():
                self._log(f"  {key}: {value:.4f}", 2)

        return metrics_avg

    def save_checkpoint(self, metrics: Dict[str, float], epoch: int):
        """Save model if metric improves"""
        current_metric = metrics[self.config.metric.name]

        if self.config.metric.better_than(current_metric, self.best_metric_value):
            self.best_metric_value = current_metric
            self.best_epoch = epoch

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'best_metric': {self.config.metric.name: current_metric}
            }

            torch.save(
                checkpoint,
                self.config.save_path / 'best_model.pth'
            )

    def train(self, train_loader: Any, val_loader: Any) -> Dict[str, Any]:
        """Run complete training process"""
        self._log("\nStarting training...")
        self._log(f"Model will be saved to: {self.config.save_path}")
        self._log(f"Tensorboard logs will be saved to: {self.config.log_path}")

        for epoch in range(self.config.num_epochs):
            train_loss = self.train_epoch(train_loader, epoch)

            if (epoch + 1) % self.config.val_interval == 0:
                metrics = self.validate(val_loader, epoch)
                self.save_checkpoint(metrics, epoch)

                if self.config.verbosity >= 1:
                    self._log(
                        f"Epoch {epoch + 1}: "
                        f"loss = {train_loss:.4f}, "
                        f"{self.config.metric.name} = {metrics[self.config.metric.name]:.4f}"
                    )
                    if metrics[self.config.metric.name] == self.best_metric_value:
                        self._log("New best model saved!")


        final_report = {
            'best_epoch': self.best_epoch,
            'best_metric': {
                self.config.metric.name: self.best_metric_value
            },
            'model_path': str(self.config.save_path / 'best_model.pth')
        }


        self._log("\nTraining completed!")
        self._log(f"Best {self.config.metric.name}: {self.best_metric_value:.4f} at epoch {self.best_epoch}")
        self._log(f"Model saved at: {final_report['model_path']}")
        self._log(f"To view training curves, run: tensorboard --logdir {self.config.log_path}")

        self.writer.close()
        return final_report

@dataclass
class InferenceConfig:
    """Simple configuration for inference"""
    def __init__(
        self,
        model_path: Path,
        device: Union[str, torch.device] = "cuda",
        output_path: Optional[Path] = None,
        verbosity: int = 1,
    ):
        self.model_path = model_path
        self.device = device
        self.output_path = output_path
        self.verbosity = verbosity

        # Setup output path if specified
        if self.output_path is not None:
            self.output_path.mkdir(parents=True, exist_ok=True)

# ===== Base Inferer ======= #
class BaseInferenceWithSpecs:
    """Base inference class that works with dataloader specifications"""
    def __init__(
        self,
        model: torch.nn.Module,
        config: InferenceConfig,
        dataloader_specs: 'DataLoaderSpecs',
        output_transform: Optional[Callable] = None,
        post_process: Optional[Callable] = None
    ):
        # Convert device string to torch.device if needed
        self.device = (torch.device(config.device)
                      if isinstance(config.device, str)
                      else config.device)

        self.model = model.to(self.device)
        self.config = config
        self.dataloader_specs = dataloader_specs
        self.output_transform = output_transform or (lambda x: x)
        self.post_process = post_process or (lambda x: x)

        # Setup output path if specified
        if self.config.output_path is not None:
            self.config.output_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_checkpoint(
        cls,
        model_class: Type[torch.nn.Module],
        model_args: Dict[str, Any],
        config: InferenceConfig,
        dataloader_specs: 'DataLoaderSpecs',
        output_transform: Optional[Callable] = None,
        post_process: Optional[Callable] = None
    ) -> 'BaseInferenceWithSpecs':
        """Create inference object by loading model from checkpoint"""
        # Initialize model
        model = model_class(**model_args)

        # Load weights from checkpoint
        checkpoint = torch.load(config.model_path, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])

        return cls(
            model=model,
            config=config,
            dataloader_specs=dataloader_specs,
            output_transform=output_transform,
            post_process=post_process
        )

    def _log(self, message: str, level: int = 1):
        """Log message if verbosity level is sufficient"""
        if self.config.verbosity >= level:
            print(message)

    def _validate_dataloader(self, dataloader: Any) -> bool:
        """Validate dataloader meets specifications"""
        if not hasattr(dataloader, '__iter__'):
            return False
        return True  # Further validation in specific implementations

    def _process_batch(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Process a batch of data for inference - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement batch processing")

    def _save_outputs(self, outputs: List[Any], case_identifiers: List[str]):
        """Save inference outputs"""
        if not self.config.save_outputs or self.config.output_path is None:
            return

        for output, identifier in zip(outputs, case_identifiers):
            output_file = self.config.output_path / f"{identifier}.{self.config.save_format}"

            if self.config.save_format == "npy":
                np.save(output_file, output)
            elif self.config.save_format == "pt":
                torch.save(output, output_file)
            # Add other formats as needed

            self._log(f"Saved output to {output_file}", level=2)

    def run_inference(self, dataloader: Any) -> Dict[str, List[Any]]:
        """Run inference on dataloader"""
        if not isinstance(dataloader, Iterator):
            dataloader = iter(dataloader)

        if not self._validate_dataloader(dataloader):
            raise ValueError("Invalid dataloader type")

        self.model.eval()
        results = {
            'outputs': [],
            'case_identifiers': [],
            'metadata': []
        }

        self._log("Starting inference...")

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if not self.dataloader_specs.validate_batch(batch):
                    raise ValueError(f"Invalid batch structure at index {batch_idx}")

                # Extract case identifiers for saving results
                case_identifiers = self._extract_case_identifiers(batch)
                results['case_identifiers'].extend(case_identifiers)

                # Process batch and get inputs for model
                inputs = self._process_batch(batch)

                # Run model
                raw_outputs = self.model(inputs)

                # Transform outputs if needed
                transformed_outputs = self.output_transform(raw_outputs)

                # Post-process outputs if needed
                processed_outputs = [
                    self.post_process(output) for output in transformed_outputs
                ]

                # Store results
                results['outputs'].extend(processed_outputs)

                # Extract any additional metadata if needed
                metadata = self._extract_metadata(batch)
                results['metadata'].append(metadata)

                # Save outputs if configured
                self._save_outputs(processed_outputs, case_identifiers)

                self._log(f"Processed batch {batch_idx+1}", level=2)

        self._log("Inference completed!")
        return results

    def _extract_case_identifiers(self, batch: Dict[str, Any]) -> List[str]:
        """Extract case identifiers from batch - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement case identifier extraction")

    def _extract_metadata(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from batch - can be overridden by subclasses"""
        return {}  # Default implementation returns empty metadata



def check_preprocessing_iterator():
    """Simple example to check what the preprocessing_iterator_fromfiles returns"""

    # Update these paths to your actual data paths
    dataset_folder = '/home/jovyan/nnunet_data/nnUNet_raw/Dataset824_FLAWS-HCO/imagesTs/'
    test_folder = '/home/jovyan/nnunet_data/nnUNet_raw/Dataset824_FLAWS-HCO/imagesTs/'

    # Find test files
    test_files = [f for f in os.listdir(test_folder) if f.endswith('.nii.gz')][:2]  # Just use first 2 files

    # Create input folders list - each item is a list of files that make up one case
    input_folders = [[os.path.join(test_folder, f)] for f in test_files]

    print(f"Found {len(input_folders)} test files")
    for i, files in enumerate(input_folders):
        print(f"Case {i}: {files}")

    # Load plans and dataset json
    try:
        plans_file = os.path.join(dataset_folder, 'nnUNetPlans.json')
        dataset_json_file = os.path.join(dataset_folder, 'dataset.json')

        plans_manager = PlansManager(plans_file)
        dataset_json = load_json(dataset_json_file)

        # Get the configuration manager for 3d_fullres
        configuration_manager = plans_manager.get_configuration('3d_fullres')

        # Make sure num_processes is at least 1
        num_processes = 1

        print("\nInitializing preprocessing iterator...")
        data_iterator = preprocessing_iterator_fromfiles(
            list_of_lists=input_folders,
            list_of_segs_from_prev_stage_files=None,
            output_filenames_truncated=None,
            plans_manager=plans_manager,
            dataset_json=dataset_json,
            configuration_manager=configuration_manager,
            num_processes=num_processes,
            pin_memory=False,  # Set to True if using CUDA
            verbose=True
        )

        print("\nExamining first batch from iterator:")

        # Get the first batch
        first_batch = next(data_iterator)

        print("Keys in batch:", list(first_batch.keys()))

        # Check data
        if isinstance(first_batch['data'], str):
            print("Data is a file path:", first_batch['data'])
            # Load the data to examine it
            data = np.load(first_batch['data'])
            print("Data shape:", data.shape)
        else:
            print("Data shape:", first_batch['data'].shape)

        # Check properties
        print("\nData properties keys:", list(first_batch['data_properties'].keys()))
        if 'list_of_data_files' in first_batch['data_properties']:
            print("Source files:", first_batch['data_properties']['list_of_data_files'])

        # Get case identifier
        case_id = os.path.basename(first_batch['data_properties'].get('list_of_data_files', ['unknown'])[0]).split('.')[0]
        print("Case ID:", case_id)

        # Try to get a second batch
        try:
            second_batch = next(data_iterator)
            print("\nSecond batch successfully retrieved")

            # Get case identifier for second batch
            case_id2 = os.path.basename(second_batch['data_properties'].get('list_of_data_files', ['unknown'])[0]).split('.')[0]
            print("Second case ID:", case_id2)
        except StopIteration:
            print("\nNo second batch available, iterator exhausted")
        except Exception as e:
            print(f"\nError getting second batch: {e}")

    except Exception as e:
        print(f"Error in preprocessing iterator setup: {e}")

if __name__ == "__main__":
    check_preprocessing_iterator()
    print("-----run_example-----------------------")
    run_example()