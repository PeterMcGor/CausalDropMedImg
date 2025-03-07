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

from attributions.utils import get_available_cpus

from attributions.models.merge_nnunet_trainers_inferers import (
    MergedNNUNetDataLoaderSpecs,
)
from attributions.models.monai_binary import (
    BinaryClassifierConfig,
    BinaryMergedNNUNetTrainer,
    BinaryMergedNNUNetTrainerImages,
    MonaiBinaryClassifier,
    MonaiBinaryClassifierInference,
    MonaiBinaryClassifierInferenceImages,
)
from nnunet_utils.dataset_utils import MergerNNUNetDataset
from nnunet_utils.preprocess import AnyFolderPreprocessor

from attributions.core.distribution_base import DensityRatioEstimator, MechanismSpec

import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tempfile
from attributions.models.base_models import (
    DataLoaderSpecs,
    InferenceConfig,
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
        num_processes: int = None,
        unpack_data: bool = True,
        batch_size: int = 4,
        stratify_by: Union[Callable, None] = None,
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
        self.num_processes = num_processes if num_processes is not None else get_available_cpus()
        self.batch_size = batch_size
        self.stratify_by = stratify_by
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
                num_epochs=100,
                val_interval=5,
                num_train_iterations_per_epoch=250,  # 250
                num_val_iterations_per_epoch=150,  # 150
                metric=MetricConfig("f1", MetricGoal.MAXIMIZE),
                log_path=None,
                save_path=None,
                device="cuda",
                verbosity=1,
            )
        self.training_config = training_config

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
        self._create_enviroments_datasets()

    def _fit_mechanism_models(
        self,
        train_env_data: None,
        inference_env_data: None,
        input_features: List[str],
        register_key: str,
    ) -> None:
        """Fit nnunet model discriminator for a specific mechanism"""
        binary_config = copy.deepcopy(self.discriminator_model)
        binary_config.num_input_channels = len(input_features)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_config = copy.deepcopy(self.training_config)

        optimizer_config = copy.deepcopy(self.optimizer_config)
        # mergennunet_trainer_dataloader_specs = copy.deepcopy(self.mergennunet_trainer_dataloader_specs)

        if training_config.log_path is None:
            training_config.log_path = Path(
                os.path.join(
                    self.result_folder,
                    "logs",
                    f"{register_key}_{self.nnunet_dataset}_{timestamp}",
                )
            )
        if training_config.save_path is None:
            training_config.save_path = Path(
                os.path.join(
                    self.result_folder,
                    "models_save",
                    f"{register_key}_{self.nnunet_dataset}_{timestamp}",
                )
            )

        mergennunet_trainer_dataloader_specs = MergedNNUNetDataLoaderSpecs(
            dataset_json_path=self.dataset_json_file,
            dataset_plans_path=self.plans_file,
            dataset_train=self.train_data,
            dataset_val=self.val_data,
            batch_size=self.batch_size,
            num_processes=self.num_processes,
            unpack_data=True,
            cleanup_unpacked=False,
            inference=False,
        )

        if binary_config.num_input_channels == 2:
            print("Mix trainer")
            trainer = BinaryMergedNNUNetTrainer.create(
                classifier_config=binary_config,
                training_config=training_config,
                optimizer_config=optimizer_config,
                dataloader_specs=mergennunet_trainer_dataloader_specs,
            )
        else:  # just the images # TODO extend with mor evariables. Fututre work
            print("Images trainer")
            trainer = BinaryMergedNNUNetTrainerImages.create(
                classifier_config=binary_config,
                training_config=training_config,
                optimizer_config=optimizer_config,
                dataloader_specs=mergennunet_trainer_dataloader_specs,
            )

        # Train model
        results = trainer.train()  # Using same loader for train/val for simplicity
        self.fitted_models[register_key] = results[
            "model_path"
        ]  # so the path to a file not the model for now

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
        if self.stratify_by is None:
            train_data_env_train, test_data_env_train = (
                self.train_env_dataset.random_split(split_ratio=split_ratio)
            )
        else:
            train_data_env_train, test_data_env_train = (
                self.train_env_dataset.stratified_split(
                    split_ratio=split_ratio,
                    min_samples_per_group=3,
                    groupby_func=self.stratify_by,
                )
            )
        inference_ratio = len(train_data_env_train) / n_inference_env_data
        inference_ratio = min(inference_ratio, 1 / inference_ratio)
        if self.stratify_by is None:
            train_data_env_test, test_data_env_test = (
                self.inference_env_dataset.random_split(split_ratio=inference_ratio)
            )
        else:
            train_data_env_test, test_data_env_test = (
                self.inference_env_dataset.stratified_split(
                    split_ratio=inference_ratio,
                    min_samples_per_group=3,
                    groupby_func=self.stratify_by,
                )
            )

        self.train_data, self.val_data = train_data_env_train.merge_and_split(
            train_data_env_test, split_ratio=split_ratio
        )  # these goes to the discrimiantors
        self.test_data_env_train = (
            test_data_env_train  # this is for transportation through weights
        )
        self.test_data_env_test = test_data_env_test
        print(
            f"For training classifiers: From the training domain {len(train_data_env_train)}, from the development domain {len(train_data_env_test)}."
        )
        print(
            f"From those for the trainin fo the domain classifier {len(self.train_data)}, for validation {len(self.val_data)}"
        )
        print(
            f"For shifts transportation: From the training domain {len(test_data_env_train)}, from the development domain {len(test_data_env_test)}"
        )

    def _get_probabilities(
        self,
        data: None,
        variables: List[str],
        variables_key: str,
    ) -> np.ndarray:
        """Get predicted probabilities from nnunet model"""

        # Create inference config
        config = InferenceConfig(  # Load themodel each time, efficient for memory not for speed
            model_path=Path(self.fitted_models[variables_key]),
            device="cuda" if torch.cuda.is_available() else "cpu",
            output_path=Path(
                os.path.join(
                    os.path.dirname(self.fitted_models[variables_key]),
                    "inference_results",
                )
            ),
            verbosity=1,
        )
        binary_config = copy.deepcopy(self.discriminator_model)
        binary_config.num_input_channels = len(variables)

        dataloader_specs = MergedNNUNetDataLoaderSpecs(
            dataset_json_path=self.dataset_json_file,
            dataset_plans_path=self.plans_file,
            dataset_train=self.test_data_env_train,
            dataset_val=self.test_data_env_train,
            batch_size=self.batch_size,
            num_processes=self.num_processes,
            unpack_data=True,
            inference=True,
        )
        if binary_config.num_input_channels == 2:
            inference_tool = MonaiBinaryClassifierInference.from_checkpoint(
                model_class=MonaiBinaryClassifier,
                model_args={"config": binary_config},
                config=config,
                dataloader_specs=dataloader_specs,
                output_transform=torch.nn.Softmax(dim=1),
                post_process=lambda x: x.cpu().numpy(),
            )
        else:
            inference_tool = MonaiBinaryClassifierInferenceImages.from_checkpoint(
                model_class=MonaiBinaryClassifier,
                model_args={"config": binary_config},
                config=config,
                dataloader_specs=dataloader_specs,
                output_transform=torch.nn.Softmax(dim=1),
                post_process=lambda x: x.cpu().numpy(),
            )
        results = inference_tool.run_inference()
        all_probabilities = []
        # Loop through each dictionary in the data list
        for item in results:
            # Extract the probabilities array from each item and add it to our collection
            all_probabilities.append(item["probabilities"])

        # Combine all probability arrays into a single 2D array
        # This will stack all the arrays vertically (along axis 0)
        return np.vstack(all_probabilities)
        # return np.column_stack([1 - probs, probs])  # Return [P(y=0), P(y=1)]

    def _prepare_discriminator_data(self):
        pass


class FlexibleNNunetBinaryDiscriminatorRatioEstimator(DiscriminatorRatioEstimator):
    def __init__(
        self,
        # Basic configuration
        discriminator_model: BinaryClassifierConfig = None,
        optimizer_config: Optional[OptimizerConfig] = None,
        criterion_config: Optional[CriterionConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        clip_probabilities: Optional[float] = None,
        clip_ratios: Optional[float] = None,
        batch_size: int = 4,
        num_processes: int = None,
        stratify_by: Union[Callable, None] = None,
        # Path-based approach (like original)
        nnunet_folders_path: Optional[str] = None,
        nnunet_dataset: Optional[str] = None,
        skip_default_preprocessing: bool = True,  # Default to skip auto preprocessing
        imagesTs: str = "imagesTs",
        labelsTs: str = "labelsTs",
        imagesTr: str = "imagesTr",
        labelsTr: str = "labelsTr",
        # Dataset-based approach (direct datasets)
        train_dataset: Optional[MergerNNUNetDataset] = None,
        test_dataset_train_domain: Optional[MergerNNUNetDataset] = None,
        test_dataset_inference_domain: Optional[MergerNNUNetDataset] = None,
        # Pre-split train/val datasets
        train_data: Optional[MergerNNUNetDataset] = None,
        val_data: Optional[MergerNNUNetDataset] = None,
        # Manual file paths
        plans_file: Optional[str] = None,
        dataset_json_file: Optional[str] = None,
        result_folder: Optional[str] = None,
        # Splitting parameters
        train_val_split_ratio: float = 0.65,
    ):
        # Store basic configuration
        self.batch_size = batch_size
        self.num_processes = num_processes if num_processes is not None else get_available_cpus()
        self.stratify_by = stratify_by
        self.train_val_split_ratio = train_val_split_ratio

        # Set up model configuration
        if discriminator_model is None:
            discriminator_model = BinaryClassifierConfig()
        self.discriminator_model = discriminator_model

        super().__init__(
            discriminator_model=discriminator_model,
            clip_probabilities=clip_probabilities,
            clip_ratios=clip_ratios,
        )

        # Set up optimizer configuration
        if optimizer_config is None:
            optimizer_config = OptimizerConfig(
                optimizer_class=torch.optim.AdamW, optimizer_kwargs={"lr": 1e-4}
            )
        self.optimizer_config = optimizer_config

        # Set up criterion configuration
        if criterion_config is None:
            criterion_config = CriterionConfig(
                criterion_class=torch.nn.CrossEntropyLoss, criterion_kwargs={}
            )
        self.criterion_config = criterion_config

        # Set up training configuration
        if training_config is None:
            training_config = TrainingConfig(
                num_epochs=100,
                val_interval=5,
                num_train_iterations_per_epoch=250,
                num_val_iterations_per_epoch=150,
                metric=MetricConfig("f1", MetricGoal.MAXIMIZE),
                log_path=None,
                save_path=None,
                device="cuda",
                verbosity=1,
            )
        self.training_config = training_config

        # Dictionary to store model paths - key will be input_features signature
        self.fitted_models = {}

        # Initialize datasets as None
        self.train_env_dataset = None
        self.inference_env_dataset = None
        self.train_data = None
        self.val_data = None
        self.test_data_env_train = None
        self.test_data_env_test = None

        # Set up paths based on dataset
        if nnunet_dataset is not None and nnunet_folders_path is not None:
            self.nnunet_folders_path = nnunet_folders_path
            self.nnunet_dataset = nnunet_dataset
            self.imagesTs = imagesTs
            self.labelsTs = labelsTs
            self.imagesTr = imagesTr
            self.labelsTr = labelsTr

            # Set up paths
            self.raw_dataset_folder = os.path.join(
                self.nnunet_folders_path, "nnUNet_raw", self.nnunet_dataset
            )
            self.preprocessed_dataset_folder = os.path.join(
                self.nnunet_folders_path, "nnUNet_preprocessed", self.nnunet_dataset
            )

            # Use default paths if not provided
            self.plans_file = plans_file or os.path.join(
                self.preprocessed_dataset_folder, "nnUNetPlans.json"
            )
            self.dataset_json_file = dataset_json_file or os.path.join(
                self.preprocessed_dataset_folder, "dataset.json"
            )

            # Set result folder with more robust path construction
            if result_folder:
                self.result_folder = result_folder
            else:
                # If no result folder provided, create one in a sensible location
                self.result_folder = os.path.join(
                    self.nnunet_folders_path, self.nnunet_dataset, "results"
                )
                # Also create the directory if it doesn't exist
                os.makedirs(self.result_folder, exist_ok=True)
                # Create subdirectories for logs and models
                os.makedirs(os.path.join(self.result_folder, "logs"), exist_ok=True)
                os.makedirs(os.path.join(self.result_folder, "models"), exist_ok=True)

            # Only run default preprocessing if not skipped
            if not skip_default_preprocessing:
                try:
                    print(
                        "Running default preprocessing for standard folder structure..."
                    )
                    self.train_env_data_path, self.inference_env_data_path = (
                        self._preprocess_discriminator_data()
                    )

                    self.train_env_dataset = MergerNNUNetDataset(
                        self.train_env_data_path, additional_data={"test_data": 0}
                    )
                    self.inference_env_dataset = MergerNNUNetDataset(
                        self.inference_env_data_path, additional_data={"test_data": 1}
                    )
                    self._create_enviroments_datasets()
                except Exception as e:
                    print(f"Warning: Default preprocessing failed: {str(e)}")
                    print(
                        "You'll need to use preprocess_custom_folder() and set datasets manually."
                    )
        else:
            # Require manual configuration when not using dataset approach
            if plans_file is None or dataset_json_file is None or result_folder is None:
                raise ValueError(
                    "When not using nnunet_dataset, must provide plans_file, "
                    "dataset_json_file, and result_folder"
                )
            self.plans_file = plans_file
            self.dataset_json_file = dataset_json_file
            self.result_folder = result_folder

        # Handle direct dataset approach
        if train_dataset is not None:
            self.train_env_dataset = train_dataset

            # Set test datasets if provided
            if test_dataset_train_domain is not None:
                self.test_data_env_train = test_dataset_train_domain

            if test_dataset_inference_domain is not None:
                self.test_data_env_test = test_dataset_inference_domain

            # Split train dataset for training/validation if not already split
            if train_data is None or val_data is None:
                print("Splitting training dataset for train/val...")
                if self.stratify_by is None:
                    self.train_data, self.val_data = (
                        self.train_env_dataset.random_split(
                            split_ratio=self.train_val_split_ratio
                        )
                    )
                else:
                    self.train_data, self.val_data = (
                        self.train_env_dataset.stratified_split(
                            split_ratio=self.train_val_split_ratio,
                            min_samples_per_group=3,
                            groupby_func=self.stratify_by,
                        )
                    )
                print(
                    f"Split result: Train {len(self.train_data)}, Val {len(self.val_data)}"
                )

        # Handle pre-split datasets
        if train_data is not None and val_data is not None:
            self.train_data = train_data
            self.val_data = val_data

        # Report dataset status
        self._report_dataset_status()

    def _report_dataset_status(self):
        """Print summary of available datasets"""
        print("\n=== Dataset Status ===")
        if self.train_data is not None:
            print(f"Train: {len(self.train_data)} samples")
        else:
            print("Train: None")

        if self.val_data is not None:
            print(f"Val: {len(self.val_data)} samples")
        else:
            print("Val: None")

        if self.test_data_env_train is not None:
            print(f"Test (train domain): {len(self.test_data_env_train)} samples")
        else:
            print("Test (train domain): None")

        if self.test_data_env_test is not None:
            print(f"Test (inference domain): {len(self.test_data_env_test)} samples")
        else:
            print("Test (inference domain): None")
        print("====================\n")

    def _preprocess_discriminator_data(self, output_folder: str = None):
        """Preprocess data for discriminator training (path-based approach)"""
        output_folder = tempfile.mkdtemp() if output_folder is None else output_folder

        # Check if the required folders exist
        train_images_folder = os.path.join(self.raw_dataset_folder, self.imagesTr)
        train_labels_folder = os.path.join(self.raw_dataset_folder, self.labelsTr)
        test_images_folder = os.path.join(self.raw_dataset_folder, self.imagesTs)
        test_labels_folder = os.path.join(self.raw_dataset_folder, self.labelsTs)

        if not os.path.exists(train_images_folder):
            raise FileNotFoundError(
                f"Training images folder not found: {train_images_folder}"
            )
        if not os.path.exists(train_labels_folder):
            raise FileNotFoundError(
                f"Training labels folder not found: {train_labels_folder}"
            )
        if not os.path.exists(test_images_folder):
            raise FileNotFoundError(
                f"Test images folder not found: {test_images_folder}"
            )
        if not os.path.exists(test_labels_folder):
            raise FileNotFoundError(
                f"Test labels folder not found: {test_labels_folder}"
            )

        inference_env_data_path = os.path.join(output_folder, "inference_env_data")
        preprocessor_test_data = AnyFolderPreprocessor(
            input_images_folder=test_images_folder,
            input_segs_folder=test_labels_folder,
            output_folder=inference_env_data_path,
            plans_file=self.plans_file,
            dataset_json_file=self.dataset_json_file,
        )

        train_env_data_path = os.path.join(output_folder, "train_env_data")
        preprocessor_train_data = AnyFolderPreprocessor(
            input_images_folder=train_images_folder,
            input_segs_folder=train_labels_folder,
            output_folder=train_env_data_path,
            plans_file=self.plans_file,
            dataset_json_file=self.dataset_json_file,
        )

        print(
            f"Preprocessing test data from {test_images_folder} and {test_labels_folder}..."
        )
        preprocessor_test_data.run("3d_fullres", self.num_processes)

        print(
            f"Preprocessing training data from {train_images_folder} and {train_labels_folder}..."
        )
        preprocessor_train_data.run("3d_fullres", self.num_processes)

        return train_env_data_path, inference_env_data_path

    def _create_enviroments_datasets(self):
        """Original split logic for path-based approach with train and inference datasets"""
        if self.train_env_dataset is None or self.inference_env_dataset is None:
            raise ValueError(
                "Train and inference datasets must be set before creating environments"
            )

        n_train_env_data = len(self.train_env_dataset)
        n_inference_env_data = len(self.inference_env_dataset)

        # Split training environment data
        if self.stratify_by is None:
            train_data_env_train, test_data_env_train = (
                self.train_env_dataset.random_split(
                    split_ratio=self.train_val_split_ratio
                )
            )
        else:
            train_data_env_train, test_data_env_train = (
                self.train_env_dataset.stratified_split(
                    split_ratio=self.train_val_split_ratio,
                    min_samples_per_group=3,
                    groupby_func=self.stratify_by,
                )
            )

        # Calculate inference ratio and split inference environment data
        inference_ratio = len(train_data_env_train) / n_inference_env_data
        inference_ratio = min(inference_ratio, 1 / inference_ratio)

        if self.stratify_by is None:
            train_data_env_test, test_data_env_test = (
                self.inference_env_dataset.random_split(split_ratio=inference_ratio)
            )
        else:
            train_data_env_test, test_data_env_test = (
                self.inference_env_dataset.stratified_split(
                    split_ratio=inference_ratio,
                    min_samples_per_group=3,
                    groupby_func=self.stratify_by,
                )
            )

        # Merge and split for final train/val datasets
        self.train_data, self.val_data = train_data_env_train.merge_and_split(
            train_data_env_test, split_ratio=self.train_val_split_ratio
        )

        # Store test datasets
        self.test_data_env_train = test_data_env_train
        self.test_data_env_test = test_data_env_test

        print(
            f"For training classifiers: From the training domain {len(train_data_env_train)}, "
            f"from the development domain {len(train_data_env_test)}."
        )
        print(
            f"From those for the training of the domain classifier {len(self.train_data)}, "
            f"for validation {len(self.val_data)}"
        )
        print(
            f"For shifts transportation: From the training domain {len(test_data_env_train)}, "
            f"from the development domain {len(test_data_env_test)}"
        )

    def preprocess_custom_folder(
        self,
        input_images_folder: str,
        input_segs_folder: str,
        output_folder: str = None,
        folder_name: str = "processed_data",
    ) -> str:
        """
        Preprocess a custom data folder and return the path to preprocessed data.

        Args:
            input_images_folder: Path to input images folder
            input_segs_folder: Path to input segmentations folder
            output_folder: Path to output folder (will be created if None)
            folder_name: Name for the subfolder to store processed data

        Returns:
            Path to preprocessed data
        """
        # Validate inputs
        if not os.path.exists(input_images_folder):
            raise FileNotFoundError(f"Images folder not found: {input_images_folder}")
        if not os.path.exists(input_segs_folder):
            raise FileNotFoundError(
                f"Segmentations folder not found: {input_segs_folder}"
            )

        # Create output folder
        output_folder = tempfile.mkdtemp() if output_folder is None else output_folder
        os.makedirs(output_folder, exist_ok=True)
        data_path = os.path.join(output_folder, folder_name)

        print(
            f"Preprocessing data from {input_images_folder} and {input_segs_folder}..."
        )
        print(f"Output will be saved to {data_path}")

        # Create and run preprocessor
        preprocessor = AnyFolderPreprocessor(
            input_images_folder=input_images_folder,
            input_segs_folder=input_segs_folder,
            output_folder=data_path,
            plans_file=self.plans_file,
            dataset_json_file=self.dataset_json_file,
        )
        preprocessor.run("3d_fullres", self.num_processes)

        print(f"Preprocessing complete. Data saved to {data_path}")
        return data_path

    def create_dataset_from_folder(
        self, folder_path: str, is_test_data: bool = False
    ) -> MergerNNUNetDataset:
        """
        Create a MergerNNUNetDataset from a preprocessed folder.

        Args:
            folder_path: Path to preprocessed folder
            is_test_data: Whether this is test data (affects additional_data)

        Returns:
            MergerNNUNetDataset
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Preprocessed folder not found: {folder_path}")

        print(f"Creating dataset from {folder_path}")
        dataset = MergerNNUNetDataset(
            folder_path, additional_data={"test_data": 1 if is_test_data else 0}
        )
        print(f"Created dataset with {len(dataset)} samples")
        return dataset

    def set_datasets(
        self,
        train_dataset=None,
        test_dataset_train_domain=None,
        test_dataset_inference_domain=None,
        train_data=None,
        val_data=None,
    ):
        """
        Set datasets for the discriminator.

        Args:
            train_dataset: Training dataset (will be split if train_data is None)
            test_dataset_train_domain: Test dataset from training domain
            test_dataset_inference_domain: Test dataset from inference domain
            train_data: Pre-split training data
            val_data: Pre-split validation data
        """
        if train_dataset is not None:
            self.train_env_dataset = train_dataset

            # Split for train/val if not provided
            if train_data is None or val_data is None:
                print("Splitting training dataset for train/val...")
                if self.stratify_by is None:
                    self.train_data, self.val_data = (
                        self.train_env_dataset.random_split(
                            split_ratio=self.train_val_split_ratio
                        )
                    )
                else:
                    self.train_data, self.val_data = (
                        self.train_env_dataset.stratified_split(
                            split_ratio=self.train_val_split_ratio,
                            min_samples_per_group=3,
                            groupby_func=self.stratify_by,
                        )
                    )
                print(
                    f"Split result: Train {len(self.train_data)}, Val {len(self.val_data)}"
                )

        # Set pre-split datasets if provided
        if train_data is not None and val_data is not None:
            self.train_data = train_data
            self.val_data = val_data

        # Set test datasets
        if test_dataset_train_domain is not None:
            self.test_data_env_train = test_dataset_train_domain

        if test_dataset_inference_domain is not None:
            self.test_data_env_test = test_dataset_inference_domain

        self._report_dataset_status()

    def _fit_mechanism_models(
        self,
        train_data,
        val_data,
        input_features: List[str],
        register_key: str
    ) -> None:
        """
        Fit nnunet model discriminator for a specific mechanism.

        Args:
            input_features: List of input feature names
            register_key: Key to register the fitted model under
            train_data: Optional override for training data
            val_data: Optional override for validation data
        """
        # Check if we have training and validation data
        if self.train_data is None or self.val_data is None:
            if train_data is None or val_data is None:
                raise ValueError(
                    "No training or validation data available. "
                    "Use set_datasets() to set datasets first."
                )

        # Create a unique key based on input features to prevent conflicts
        # feature_key = "_".join(sorted(input_features))
        # model_key = f"{register_key}_{feature_key}"

        binary_config = copy.deepcopy(self.discriminator_model)
        binary_config.num_input_channels = len(input_features)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_config = copy.deepcopy(self.training_config)
        optimizer_config = copy.deepcopy(self.optimizer_config)

        # Use provided datasets or default ones
        train_dataset = train_data if train_data is not None else self.train_data
        val_dataset = val_data if val_data is not None else self.val_data

        # Create specific log and save paths for this model configuration
        if training_config.log_path is None:
            log_path = os.path.join(
                self.result_folder, "logs", f"{register_key}_{timestamp}"
            )
            training_config.log_path = Path(log_path)

        if training_config.save_path is None:
            save_path = os.path.join(
                self.result_folder, "models", f"{register_key}_{timestamp}"
            )
            training_config.save_path = Path(save_path)

        # Ensure output directories exist
        os.makedirs(training_config.log_path, exist_ok=True)
        os.makedirs(training_config.save_path, exist_ok=True)

        mergennunet_trainer_dataloader_specs = MergedNNUNetDataLoaderSpecs(
            dataset_json_path=self.dataset_json_file,
            dataset_plans_path=self.plans_file,
            dataset_train=train_dataset,
            dataset_val=val_dataset,
            batch_size=self.batch_size,
            num_processes=self.num_processes,
            unpack_data=True,
            cleanup_unpacked=False,
            inference=False,
        )

        if binary_config.num_input_channels == 2:
            print("Mix trainer")
            trainer = BinaryMergedNNUNetTrainer.create(
                classifier_config=binary_config,
                training_config=training_config,
                optimizer_config=optimizer_config,
                dataloader_specs=mergennunet_trainer_dataloader_specs,
            )
        else:
            print("Images trainer")
            trainer = BinaryMergedNNUNetTrainerImages.create(
                classifier_config=binary_config,
                training_config=training_config,
                optimizer_config=optimizer_config,
                dataloader_specs=mergennunet_trainer_dataloader_specs,
            )

        # Train model
        print(f"Training model for {register_key}...")
        results = trainer.train()
        model_path = results["model_path"]
        print(f"Model saved to {model_path}")

        self.fitted_models[register_key] = results[
            "model_path"
        ]  # so the path to a file not the model for now

    def _get_probabilities(
        self,
        test_dataset,
        variables: List[str],
        register_key: str,
        domain: str = "train",
    ) -> np.ndarray:
        """
        Get predicted probabilities from nnunet model.

        Args:
            variables: List of input variable names
            variables_key: Key to use for accessing the fitted model
            test_dataset: Optional override for test dataset
            domain: Which domain to use if test_dataset is None ('train' or 'inference')

        Returns:
            Array of predicted probabilities
        """
        # Create a variables signature key for matching with the correct model
        model_path = self.fitted_models[register_key]

        # Use provided test dataset or select based on domain
        if test_dataset is not None:
            print("Test data is not none", test_dataset)
            test_data = test_dataset
        else:
            if domain == "train":
                test_data = self.test_data_env_train
            elif domain == "inference":
                test_data = self.test_data_env_test
            else:
                raise ValueError(f"Domain must be 'train' or 'inference', got {domain}")

        if test_data is None:
            raise ValueError(
                f"No test data available for domain '{domain}'. "
                "Please set test datasets using set_datasets()."
            )

        # Create inference config
        config = InferenceConfig(
            model_path=Path(model_path),
            device="cuda" if torch.cuda.is_available() else "cpu",
            output_path=Path(
                os.path.join(os.path.dirname(model_path), "inference_results")
            ),
            verbosity=1,
        )

        binary_config = copy.deepcopy(self.discriminator_model)
        binary_config.num_input_channels = len(variables)

        dataloader_specs = MergedNNUNetDataLoaderSpecs(
            dataset_json_path=self.dataset_json_file,
            dataset_plans_path=self.plans_file,
            dataset_train=test_data,
            dataset_val=test_data,
            batch_size=1,#self.batch_size,
            num_processes=1,#self.num_processes,
            unpack_data=True,
            inference=True,
        )

        print(f"Running inference for {register_key} using {domain} domain data...")

        # Use the appropriate inference tool based on input channels
        try:
            if binary_config.num_input_channels == 2:
                inference_tool = MonaiBinaryClassifierInference.from_checkpoint(
                    model_class=MonaiBinaryClassifier,
                    model_args={"config": binary_config},
                    config=config,
                    dataloader_specs=dataloader_specs,
                    output_transform=torch.nn.Softmax(dim=1),
                    post_process=lambda x: x.cpu().numpy(),
                )
            else:
                inference_tool = MonaiBinaryClassifierInferenceImages.from_checkpoint(
                    model_class=MonaiBinaryClassifier,
                    model_args={"config": binary_config},
                    config=config,
                    dataloader_specs=dataloader_specs,
                    output_transform=torch.nn.Softmax(dim=1),
                    post_process=lambda x: x.cpu().numpy(),
                )
        except RuntimeError as e:
            # Handle model mismatch error with more clarity
            error_message = str(e)
            if "size mismatch" in error_message:
                print(
                    f"ERROR: Model input dimension mismatch. The model was trained with different input features."
                )
                print(
                    f"Make sure you're using the correct model for the variables: {variables}"
                )
                print(f"Original error: {error_message}")
                raise ValueError(
                    f"Model mismatch error. Make sure to use the same feature set for training and inference."
                )
            else:
                raise

        results = inference_tool.run_inference()
        all_probabilities = []

        # Loop through each dictionary in the data list
        for item in results:
            # Extract the probabilities array from each item and add it to our collection
            all_probabilities.append(item["probabilities"])

        # Combine all probability arrays into a single 2D array
        probabilities = np.vstack(all_probabilities)
        print(f"Generated probabilities with shape {probabilities.shape}")
        return probabilities


if __name__ == "__main__":
    nnunet_folders_path = "/home/jovyan/nnunet_data/"
    nnunet_dataset = "Dataset001_MSSEG_FLAIR_Annotator1"
    training_config = TrainingConfig(
        num_epochs=2,
        val_interval=1,
        num_train_iterations_per_epoch=20,  # 250
        num_val_iterations_per_epoch=10,  # 150
        metric=MetricConfig("f1", MetricGoal.MAXIMIZE),
        log_path=None,
        save_path=None,
        device="cuda",
        verbosity=1,
    )

    def center_number_groupby(key):
        parts = key.split("_")
        for i, part in enumerate(parts):
            if part.lower() == "center" and i + 1 < len(parts):
                # Return just the center number
                return parts[i + 1]  # This will give "01", "07", etc.
        return "unknown"  # Fallback if no center is found

    discriminator = NNunetBinaryDiscriminatorRatioEstimator(
        nnunet_dataset=nnunet_dataset,
        nnunet_folders_path=nnunet_folders_path,
        labelsTs="labelsTs_1",
        training_config=training_config,
        stratify_by=center_number_groupby,
    )
    discriminator._fit_mechanism_models(
        None, None, ["images", "labels"], "images_labels"
    )
    discriminator._fit_mechanism_models(None, None, ["images"], "images")
    probs_mix = discriminator._get_probabilities(
        None, ["images", "labels"], "images_labels"
    )
    print("Ratio mix", probs_mix[:, 1] / probs_mix[:, 0])
    probs_imgs = discriminator._get_probabilities(None, ["images"], "images")
    print("Ratio imgs", probs_imgs[:, 1] / probs_imgs[:, 0])
