import copy
from datetime import datetime
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss

from attributions.models.monai_binary import BinaryClassifierConfig
from nnunet_utils.dataset_utils import MergerNNUNetDataset
from nnunet_utils.preprocess import AnyFolderPreprocessor

from attributions.core.distribution_base import DensityRatioEstimator, MechanismSpec

import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tempfile
from attributions.models.base_models import (
    DataLoaderSpecs,
    MetricConfig,
    MetricGoal,
    TrainingConfig,
    OptimizerConfig,
    CriterionConfig,
    BaseTrainerWithSpecs,
)
from attributions.models.tensor_datasets_trainers_inferers import (
    TorchTensorDataLoaderSpecs,
    TorchTensorDiscriminatorTrainer,
)


class DiscriminatorRatioEstimator(DensityRatioEstimator):
    """Estimate density ratios using binary classifier approach.

    Based on the idea that density ratio r(x) = P(inference_env|x)/P(train_env|x)
    can be estimated using a classifier trained to distinguish train_env vs inference_env data.
    """

    def __init__(
        self,
        discriminator_model: Any,
        clip_probabilities: Optional[float] = None,
        clip_ratios: Optional[float] = None,
    ):
        """Initialize discriminator-based estimator.

        Args:
            discriminator_model: Binary classifier model
            clip_probabilities: Clip probabilities to [1-clip, clip]
            clip_ratios: Clip final ratios to [1/clip, clip]
        """
        super().__init__()
        self.base_model = discriminator_model
        self.clip_probabilities = clip_probabilities
        self.clip_ratios = clip_ratios

        if clip_probabilities is not None:
            assert 0.5 <= clip_probabilities <= 1.0
        if clip_ratios is not None:
            assert clip_ratios >= 1.0

        self.fitted_models: Dict[str, Any] = {}
        self.model_metrics: Dict[str, Dict[str, float]] = {}
        self.ratios_at_variables: Dict[str, Any] = {}

    def _prepare_discriminator_data(
        self,
        features: List[str],
        train_env_data: Union[pd.DataFrame, np.ndarray],
        inference_env_data: Union[pd.DataFrame, np.ndarray] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Prepare data for discriminator training according to mechanism structure.
        To be implemented by child classes based on model type.
        """
        raise NotImplementedError

    def _fit_mechanism_models(
        self,
        train_env_data: Union[pd.DataFrame, np.ndarray],
        inference_env_data: Union[pd.DataFrame, np.ndarray],
        input_features: List[str],
        register_key: str,
    ) -> None:
        """Fit discriminator for a specific mechanism.
        To be implemented by child classes based on model type.
        """
        raise NotImplementedError

    def fit(
        self,
        train_env_data: Union[pd.DataFrame, np.ndarray],
        inference_env_data: Union[pd.DataFrame, np.ndarray],
        mechanisms: List[MechanismSpec],
    ) -> None:
        """Fit discriminators for all mechanisms"""
        for mechanism in mechanisms:
            self._fit_mechanism_models(
                train_env_data,
                inference_env_data,
                mechanism.variables,
                mechanism.variables_key,
            )
            if not mechanism.is_root:
                if mechanism.parents_key not in self.fitted_models.keys():
                    self._fit_mechanism_models(
                        train_env_data,
                        inference_env_data,
                        mechanism.parents,
                        mechanism.parents_key,
                    )
        self.fitted = True

    def estimate_ratio(
        self,
        train_env_dat: Union[
            pd.DataFrame, np.ndarray
        ],  # This data should bedifferent to the ones used for fitting
        inference_env_data: Union[pd.DataFrame, np.ndarray],
        mechanisms: List[MechanismSpec],
    ) -> np.ndarray:
        """Estimate density ratio for data points."""
        train_env_data_shift_samples = len(train_env_dat)
        ratios = np.ones(train_env_data_shift_samples)
        for mechanism in mechanisms:
            mechanism_ratio = self._estimate_mechanism_ratio(
                train_env_dat,
                mechanism,
                root_node_prior_coef=train_env_data_shift_samples
                / len(inference_env_data),
            )
            ratios *= mechanism_ratio
        return ratios

    def _get_probabilities(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        variables: List[str],
        variables_key: str,
    ) -> np.ndarray:
        """Get predicted probabilities from the model.
        To be implemented by child classes based on model type.
        """
        raise NotImplementedError

    def _estimate_mechanism_ratio(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        mechanism: MechanismSpec,
        root_node_prior_coef: float = 1.0,
    ) -> np.ndarray:
        """Estimate ratio for a specific mechanism using model probabilities."""

        def get_discrimination_ratios(variables, variables_key):
            if variables_key in self.ratios_at_variables.keys():
                return self.ratios_at_variables[variables_key]

            probs = self._get_probabilities(data, variables, variables_key)

            if self.clip_probabilities is not None:
                probs = np.clip(
                    probs, 1 - self.clip_probabilities, self.clip_probabilities
                )

            ratios = probs[:, 1] / probs[:, 0]

            if self.clip_ratios is not None:
                ratios = np.clip(ratios, 1 / self.clip_ratios, self.clip_ratios)

            self.ratios_at_variables[variables_key] = ratios
            return ratios

        ratios = get_discrimination_ratios(mechanism.variables, mechanism.variables_key)

        if mechanism.is_root:
            return ratios * root_node_prior_coef
        else:
            return ratios / get_discrimination_ratios(
                mechanism.parents, mechanism.parents_key
            )

    def estimate_performance_shift(
        self,
        train_env_data: pd.DataFrame,
        mechanisms: List[MechanismSpec],
        model: torch.nn.Module,
        metric_fn: Callable,
        inference_env_data: pd.DataFrame = None,
        **metric_kwargs,
    ) -> float:
        """Compute performance change when only the `mechanisms` shift."""
        self.check_is_fitted()
        weights = self.estimate_ratio(
            train_env_dat=train_env_data,
            inference_env_data=inference_env_data,
            mechanisms=mechanisms,
        )

        print(f" Weight stats - Mean: {weights.mean():.2f}, Std: {weights.std():.2f}")
        print(f" Min: {weights.min():.2f}, Max: {weights.max():.2f}")

        shifted_performance = metric_fn(
            model, train_env_data, weights=weights, **metric_kwargs
        )
        return shifted_performance

    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics for all mechanism discriminators"""
        self.check_is_fitted()
        return self.model_metrics


class SklearnDiscriminatorRatioEstimator(DiscriminatorRatioEstimator):
    """Sklearn-specific implementation of discriminator-based density ratio estimation."""

    def __init__(
        self,
        discriminator_model: BaseEstimator,
        calibrate: bool = False,
        clip_probabilities: Optional[float] = None,
        clip_ratios: Optional[float] = None,
    ):
        super().__init__(
            discriminator_model=discriminator_model,
            clip_probabilities=clip_probabilities,
            clip_ratios=clip_ratios,
        )
        self.calibrate = calibrate

    def _prepare_discriminator_data(
        self,
        features: List[str],
        train_env_data: Union[pd.DataFrame, np.ndarray],
        inference_env_data: Union[pd.DataFrame, np.ndarray] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Prepare data specifically for sklearn models"""
        train_env_subset = train_env_data[features]

        if inference_env_data is not None:
            inference_env_subset = inference_env_data[features]
            X = np.concatenate([train_env_subset, inference_env_subset])
            y = np.concatenate(
                [np.zeros(len(train_env_subset)), np.ones(len(inference_env_subset))]
            )
            return X, y

        return np.array(train_env_subset)

    def _fit_mechanism_models(
        self,
        train_env_data: Union[pd.DataFrame, np.ndarray],
        inference_env_data: Union[pd.DataFrame, np.ndarray],
        input_features: List[str],
        register_key: str,
    ) -> None:
        """Fit sklearn discriminator for a specific mechanism"""
        X, y = self._prepare_discriminator_data(
            input_features, train_env_data, inference_env_data
        )

        model = clone(self.base_model).fit(X, y)

        if self.calibrate:
            model = CalibratedClassifierCV(
                base_estimator=model, method="isotonic", cv="prefit"
            ).fit(X, y)

        self.fitted_models[register_key] = model

        probs = model.predict_proba(X)[:, 1]
        self.model_metrics[register_key] = {
            "roc_auc": roc_auc_score(y, probs),
            "brier": brier_score_loss(y, probs),
        }

    def _get_probabilities(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        variables: List[str],
        variables_key: str,
    ) -> np.ndarray:
        """Get predicted probabilities from sklearn model"""
        data_filtered = self._prepare_discriminator_data(
            features=variables, train_env_data=data
        )
        model = self.fitted_models[variables_key]
        return model.predict_proba(data_filtered)


class TorchDiscriminatorRatioEstimator(DiscriminatorRatioEstimator):
    """PyTorch-specific implementation of discriminator-based density ratio estimation."""

    def __init__(
        self,
        discriminator_model: nn.Module,
        optimizer_config: Optional[OptimizerConfig] = None,
        criterion_config: Optional[CriterionConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        clip_probabilities: Optional[float] = None,
        clip_ratios: Optional[float] = None,
    ):
        """Initialize PyTorch discriminator estimator.

        Args:
            discriminator_model: PyTorch model that outputs probabilities
            optimizer_config: Configuration for optimizer
            criterion_config: Configuration for loss function
            training_config: Configuration for training
            clip_probabilities: Clip probabilities to [1-clip, clip]
            clip_ratios: Clip final ratios to [1/clip, clip]
        """
        super().__init__(
            discriminator_model=discriminator_model,
            clip_probabilities=clip_probabilities,
            clip_ratios=clip_ratios,
        )
        self.optimizer_config = optimizer_config or OptimizerConfig()
        self.criterion_config = criterion_config or CriterionConfig(
            criterion_class=nn.BCEWithLogitsLoss
        )
        self.training_config = training_config or TrainingConfig(
            num_epochs=100,
            val_interval=5,
            metric=None,  # Will be set during training
            num_train_iterations_per_epoch=100,
            num_val_iterations_per_epoch=10,
        )

    def _prepare_discriminator_data(
        self,
        features: List[str],
        train_env_data: Union[pd.DataFrame, np.ndarray],
        inference_env_data: Union[pd.DataFrame, np.ndarray] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Prepare data specifically for PyTorch models"""
        # Create loader specs for these features
        loader_specs = TorchTensorDataLoaderSpecs(
            feature_columns=features, batch_size=32  # Could make this configurable
        )

        if inference_env_data is not None:
            # For training, need both environments and labels
            train_features = loader_specs.prepare_data(train_env_data)
            inference_features = loader_specs.prepare_data(inference_env_data)

            X = torch.cat([train_features, inference_features])
            y = torch.cat(
                [torch.zeros(len(train_features)), torch.ones(len(inference_features))]
            )
            return X, y

        # For inference, just need features
        return loader_specs.prepare_data(train_env_data)

    def _fit_mechanism_models(
        self,
        train_env_data: Union[pd.DataFrame, np.ndarray],
        inference_env_data: Union[pd.DataFrame, np.ndarray],
        input_features: List[str],
        register_key: str,
    ) -> None:
        """Fit PyTorch discriminator for a specific mechanism"""
        X, y = self._prepare_discriminator_data(
            input_features, train_env_data, inference_env_data
        )

        # Create new model with adjusted input dimension
        n_features = len(input_features)
        original_params = self.base_model.get_init_params()
        new_input_dim = n_features
        new_params = (new_input_dim,) + original_params[1:]  # Adjust input_dim
        model = self.base_model.__class__(*new_params)

        # Split data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y.numpy()
        )

        # Create datasets and dataloaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        loader_specs = TorchTensorDataLoaderSpecs(
            feature_columns=input_features, batch_size=32
        )
        train_loader = DataLoader(
            train_dataset, batch_size=loader_specs.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=loader_specs.batch_size, shuffle=False
        )

        # Setup trainer
        trainer = TorchTensorDiscriminatorTrainer(
            model=model,
            optimizer=self.optimizer_config.optimizer_class(
                model.parameters(), **self.optimizer_config.optimizer_kwargs
            ),
            criterion=self.criterion_config.criterion_class(
                **self.criterion_config.criterion_kwargs
            ),
            config=self.training_config,
            dataloader_specs=loader_specs,
        )

        # Train model
        trainer.train(
            train_loader, val_loader
        )  # Using same loader for train/val for simplicity
        self.fitted_models[register_key] = model

    def _get_probabilities(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        variables: List[str],
        variables_key: str,
    ) -> np.ndarray:
        """Get predicted probabilities from PyTorch model"""
        data_tensor = self._prepare_discriminator_data(
            features=variables, train_env_data=data
        )
        data_tensor = data_tensor.to(
            self.training_config.device
        )  # workaround here. Must be a dataloader like before
        model = self.fitted_models[variables_key]

        model.eval()
        with torch.no_grad():
            logits = model(data_tensor)
            probs = logits.sigmoid().cpu().numpy()
            return np.column_stack([1 - probs, probs])  # Return [P(y=0), P(y=1)]


class NNunetBinaryDiscriminatorRatioEstimator(DiscriminatorRatioEstimator):
    def __init__(
        self,
        nnunet_folders_path: str,  # = '/home/jovyan/nnunet_data/nnUNet_preprocessed/',
        nnunet_dataset: str,  # dataset name
        imagesTs="imagesTs",
        labelsTs="labelsTs",
        imagesTr="imagesTr",
        labelsTr="labelsTr",
        num_processes: int = max(1, os.cpu_count() - 2),
        discriminator_model: BinaryClassifierConfig = None,
        optimizer_config: Optional[OptimizerConfig] = None,
        criterion_config: Optional[CriterionConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        clip_probabilities: Optional[float] = None,
        clip_ratios: Optional[float] = None,
    ):
        self.nnunet_folders_path = nnunet_folders_path
        self.nnunet_dataset = nnunet_dataset
        self.imagesTs = imagesTs
        self.labelsTs = labelsTs
        self.imagesTr = imagesTr
        self.labelsTr = labelsTr
        self.num_processes = num_processes
        self.raw_dataset_folder = os.path.join(
            self.nnunet_folders_path, "nnUNet_raw", self.nnunet_dataset
        )
        self.preprocessed_dataset_folder = os.path.join(
            self.nnunet_folders_path, "nnUNet_preprocessed", self.nnunet_dataset
        )
        self.result_folder = os.path.join(
            self.nnunet_folders_path, "nnUNet_results", self.nnunet_dataset
        )
        self.plans_file = os.path.join(
            self.preprocessed_dataset_folder, "nnUNetPlans.json"
        )
        self.dataset_json_file = os.path.join(
            self.preprocessed_dataset_folder, "dataset.json"
        )
        # Use default config if none provided
        if discriminator_model is None:
            discriminator_model = BinaryClassifierConfig()
        self.discriminator_model = discriminator_model
        super().__init__(
            discriminator_model=discriminator_model,
            clip_probabilities=clip_probabilities,
            clip_ratios=clip_ratios,
        )

        if optimizer_config is None:
            optimizer_config = OptimizerConfig(
                optimizer_class=torch.optim.AdamW, optimizer_kwargs={"lr": 1e-4}
            )
        self.optimizer_config = optimizer_config
        # Set default criterion for binary classification if not provided
        if criterion_config is None:
            criterion_config = CriterionConfig(
                criterion_class=torch.nn.CrossEntropyLoss, criterion_kwargs={}
            )
        self.criterion_config = criterion_config

        if training_config is None:
            training_config = TrainingConfig(
                num_epochs=250,
                val_interval=5,
                num_train_iterations_per_epoch=250,  # 250
                num_val_iterations_per_epoch=150,  # 150
                metric=MetricConfig("f1", MetricGoal.MAXIMIZE),
                log_path=None,
                save_path=None,
                device="cuda",
                verbosity=1,
            )

        # prepare data for later trrainings anf inferences
        self.train_env_data_path, self.inference_env_data_path = (
            self._preprocess_discriminator_data()
        )

        # Create datasets for each folder
        self.train_env_dataset = MergerNNUNetDataset(
            self.train_env_data_path, additional_data={"test_data": 0}
        )
        self.inference_env_dataset = MergerNNUNetDataset(
            self.inference_env_data_path, additional_data={"test_data": 1}
        )



    def _fit_mechanism_models(
        self,
        train_env_data: Union[pd.DataFrame, np.ndarray],
        inference_env_data: Union[pd.DataFrame, np.ndarray],
        input_features: List[str],
        register_key: str,
    ) -> None:
        """Fit nnunet model discriminator for a specific mechanism"""

        # Create new model with adjusted input dimension
        n_features = len(input_features)
        original_params = self.base_model.get_init_params()
        new_input_dim = n_features
        new_params = (new_input_dim,) + original_params[1:]  # Adjust input_dim
        model = self.base_model.__class__(*new_params)

        # Split data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y.numpy()
        )

        # Create datasets and dataloaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        loader_specs = TorchTensorDataLoaderSpecs(
            feature_columns=input_features, batch_size=32
        )
        train_loader = DataLoader(
            train_dataset, batch_size=loader_specs.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=loader_specs.batch_size, shuffle=False
        )

        # Setup trainer
        trainer = TorchTensorDiscriminatorTrainer(
            model=model,
            optimizer=self.optimizer_config.optimizer_class(
                model.parameters(), **self.optimizer_config.optimizer_kwargs
            ),
            criterion=self.criterion_config.criterion_class(
                **self.criterion_config.criterion_kwargs
            ),
            config=self.training_config,  # TODO change paths if none timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataloader_specs=loader_specs,
        )

        # Train model
        trainer.train(
            train_loader, val_loader
        )  # Using same loader for train/val for simplicity
        self.fitted_models[register_key] = model

    def _preprocess_discriminator_data(self, output_folder: str = None):
        output_folder = tempfile.mkdtemp() if output_folder is None else output_folder
        inference_env_data_path = os.path.join(output_folder, "inference_env_data")
        preprocessor_test_data = AnyFolderPreprocessor(
            input_images_folder=os.path.join(self.raw_dataset_folder, self.imagesTs),
            input_segs_folder=os.path.join(self.raw_dataset_folder, self.labelsTs),
            output_folder=inference_env_data_path,
            plans_file=self.plans_file,
            dataset_json_file=self.dataset_json_file,
        )
        train_env_data_path = os.path.join(output_folder, "train_env_data")
        preprocessor_train_data = AnyFolderPreprocessor(
            input_images_folder=os.path.join(self.raw_dataset_folder, self.imagesTr),
            input_segs_folder=os.path.join(self.raw_dataset_folder, self.labelsTr),
            output_folder=train_env_data_path,
            plans_file=self.plans_file,
            dataset_json_file=self.dataset_json_file,
        )
        preprocessor_test_data.run("3d_fullres", self.num_processes)
        preprocessor_train_data.run("3d_fullres", self.num_processes)
        return train_env_data_path, inference_env_data_path

    def _create_enviroments_datasets(self, split_ratio: float = 0.65):
        n_train_env_data = len(self.train_env_dataset)
        n_inference_env_data = len(self.inference_env_dataset)
        train_data_env_train, test_data_env_train = self.train_env_data_path.random_split(split_ratio=split_ratio)
        inference_ratio = len(train_data_env_train) / n_inference_env_data
        inference_ratio = min(inference_ratio, 1/inference_ratio)
        train_data_env_test, test_data_env_test = self.inference_env_data_path.random_split(split_ratio=inference_ratio)



    def _prepare_discriminator_data(self):
        pass

if __name__ == "__main__":
    nnunet_folders_path = '/home/jovyan/nnunet_data/'
    nnunet_dataset = 'Dataset001_MSSEG_FLAIR_Annotator1'
    NNunetBinaryDiscriminatorRatioEstimator(nnunet_dataset=nnunet_dataset, nnunet_folders_path=nnunet_folders_path,labelsTs='labelsTs_1')