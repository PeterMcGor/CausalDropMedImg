import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union, Type, Tuple
import torch
from torch.utils.tensorboard import SummaryWriter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
import tqdm
from nnunet_utils.dataloader_utils import get_nnunet_dataloaders
from nnunetv2.training.dataloading.utils import unpack_dataset
from nnunet_utils.dataset_utils import MergerNNUNetDataset

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
    num_train_iterations_per_epoch: int  # Add this
    num_val_iterations_per_epoch: int    # Add this
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

class MergedNNUNetDataLoaderSpecs(DataLoaderSpecs):
    """Specifications for MergerNNUNetDataset and its dataloaders"""
    def __init__(self,
                 dataset_json_path: str,
                 dataset_plans_path: str,
                 dataset_train: MergerNNUNetDataset,
                 dataset_val: MergerNNUNetDataset,
                 unpack_data=True,
                 cleanup_unpacked=True,
                **kwargs):
        super().__init__(
            loader_type=(SingleThreadedAugmenter, NonDetMultiThreadedAugmenter),
            dataset_type=MergerNNUNetDataset,
            batch_keys=['data', 'target', 'properties']
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

        self.train_loader, self.val_loader = get_nnunet_dataloaders(
        dataset_json_path=self.dataset_json_path,
        dataset_plans_path=self.dataset_plans_path,
        dataset_tr=self.dataset_train,
        dataset_val=self.dataset_val,
        batch_size=self.batch_size,
        num_processes=4
         )

    def validate_batch(self, batch: Dict[str, Any]) -> bool:
        """Validate batch structure from nnUNet dataloader"""
        if not super().validate_batch(batch):
            return False

        # Validate properties contain required fields
        #if not all('test_data' in p for p in batch['properties']):
        #    return False

        # Validate data and target shapes match in batch dimension
        if batch['data'].shape[0] != batch['target'].shape[0]:
            return False

        return True

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
        return self.dataloader_specs.validate_loader(dataloader)

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

# ===== Merged nnUNet Dataset Trainer =====
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
        optimizer_config: Optional[OptimizerConfig] = None,
        criterion_config: Optional[CriterionConfig] = None,
        dataloader_specs: Optional[MergedNNUNetDataLoaderSpecs] = None,
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
        super().train(self.dataloader_specs.train_loader, self.dataloader_specs.val_loader)

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
