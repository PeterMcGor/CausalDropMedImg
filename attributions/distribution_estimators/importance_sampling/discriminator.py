import copy
from datetime import datetime
import gc
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

from attributions.distribution_estimators.importance_sampling.discriminator_utils import clip_values, compute_ratios
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
    RATIOS_INFO = "RATIOS_INFO"
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
        """Fit discriminators for all mechanisms with improved memory management"""
        self.fitted_models = {}  # Reset to ensure clean state

        def train_single_model(input_features, key_name):
            """Train a single model in isolated scope"""
            print(f"\n==== Training model for {key_name} ====")
            try:
                self._fit_mechanism_models(
                    train_env_data,
                    inference_env_data,
                    input_features,
                    key_name,
                )
                # Force garbage collection after each model
                gc.collect()
                torch.cuda.empty_cache()
                print(f"==== Completed training for {key_name} ====\n")
            except Exception as e:
                print(f"Error training model for {key_name}: {e}")
                raise

        # Process mechanisms one by one with cleanup between each
        for mechanism in mechanisms:
            # Train mechanism model
            if mechanism.variables_key not in self.fitted_models:
                train_single_model(mechanism.variables, mechanism.variables_key)

            # Train parent model if needed
            if not mechanism.is_root:
                if mechanism.parents_key not in self.fitted_models:
                    train_single_model(mechanism.parents, mechanism.parents_key)

        # Final cleanup
        gc.collect()
        torch.cuda.empty_cache()
        self.fitted = True

    def estimate_ratio(
        self,
        train_env_dat: Union[
            pd.DataFrame, np.ndarray
        ],  # This data should bedifferent to the ones used for fitting
        inference_env_data: Union[pd.DataFrame, np.ndarray],
        mechanisms: List[MechanismSpec],
    ) -> Union [np.ndarray, Dict]:
        """Estimate density ratio for data points."""
        train_env_data_shift_samples = len(train_env_dat)

        # Set keep_metadata flag based on input type
        #print("Types", "train_env_dat", type(train_env_dat), "inference_env_data", type(inference_env_data))
        self.keep_metadata = True if isinstance(train_env_dat, MergerNNUNetDataset) else False

        if self.keep_metadata:
            # Make a deep copy to avoid modifying the cached values
            aux_ratio_var = 'permutation_cum_ratio'
            aux_to_transport_data = copy.deepcopy(train_env_dat)
            if isinstance(aux_to_transport_data, MergerNNUNetDataset):
                aux_to_transport_data.extend_dataset({aux_ratio_var: 1.0}, DiscriminatorRatioEstimator.RATIOS_INFO)
            else:
                raise TypeError("aux_to_transport_data must be a MergerNNUNetDataset instance")

            # Process remaining mechanisms
            for mechanism in mechanisms:
                mech_ratios = self._estimate_mechanism_ratio(
                    aux_to_transport_data,
                    mechanism,
                    root_node_prior_coef= train_env_data_shift_samples / len(inference_env_data),
                )
                # Update cumulative ratios at permutation
                for case_id, data in aux_to_transport_data.items():
                    data[DiscriminatorRatioEstimator.RATIOS_INFO][aux_ratio_var] *= mech_ratios[case_id]['ratio']

            return {case_id:ratio[aux_ratio_var] for case_id,ratio in  aux_to_transport_data.extract_per_case_id(DiscriminatorRatioEstimator.RATIOS_INFO).items()}
        else:
            # array-based logic
            ratios = np.ones(train_env_data_shift_samples)
            for mechanism in mechanisms:
                mechanism_ratio = self._estimate_mechanism_ratio(
                    train_env_dat,
                    mechanism,
                    root_node_prior_coef= train_env_data_shift_samples / len(inference_env_data),
                )
                ratios *= mechanism_ratio
            return ratios
        """Estimate density ratio for data points.
        train_env_data_shift_samples = len(train_env_dat)
        ratios = np.ones(train_env_data_shift_samples)
        self.keep_metadata = False if isinstance(train_env_dat, np.ndarray) else True

        for mechanism in mechanisms:
            mechanism_ratio = self._estimate_mechanism_ratio(
                train_env_dat,
                mechanism,
                root_node_prior_coef=train_env_data_shift_samples
                / len(inference_env_data),
            )
            ratios *= mechanism_ratio
        return ratios
        """

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
                return copy.deepcopy(self.ratios_at_variables[variables_key])

            ratios_data = self._get_probabilities(data, variables, variables_key) # here not ratios yet, add the probabilties to a new dict

            # Apply clipping to probabilities if needed
            if self.clip_probabilities is not None:
                ratios_data = clip_values(
                    ratios_data,
                    1 - self.clip_probabilities,
                    self.clip_probabilities,
                    field="probabilities"
                )

            #ratios = probs[:, 1] / probs[:, 0]
            # Compute ratios
            ratios_data = compute_ratios(ratios_data, clip=self.clip_ratios)

            # Store the ratios for future use
            self.ratios_at_variables[variables_key] = copy.deepcopy(ratios_data)
            return ratios_data
            #self.ratios_at_variables[variables_key] = ratios
            #return ratios

        #ratios = get_discrimination_ratios(mechanism.variables, mechanism.variables_key)
        mech_ratios = get_discrimination_ratios(mechanism.variables, mechanism.variables_key)

        # Apply root node adjustment if needed. I cold cave operation in here by saving this value directly # TODO
        if mechanism.is_root:
            if isinstance(mech_ratios, dict): # if self.keep_metadata
                # Data is a list of dictionaries with metadata
                for case,item in mech_ratios.items():
                    item["ratio"] = item["ratio"] * root_node_prior_coef
                    #item["adjusted_ratio"] = item["ratio"] * root_node_prior_coef
                return mech_ratios
            else:
                # Data is a numpy array
                return mech_ratios * root_node_prior_coef
        else:
            # Get parent ratios
            parent_ratios = get_discrimination_ratios(
                mechanism.parents, mechanism.parents_key
            )

            if isinstance(mech_ratios, dict):
                # Data is a list of dictionaries with metadata
                if isinstance(parent_ratios, dict):
                    # Both are lists of dictionaries
                    # Assuming they have matching indices/order
                    for case, item in mech_ratios.items():
                        item["ratio"] = item["ratio"] / parent_ratios[case]["ratio"]
                        #item["adjusted_ratio"] = item["ratio"] / parent_ratios[case]["ratio"]
                else:
                    # Parent ratios is a numpy array
                    parent_array = parent_ratios
                    for case, item in mech_ratios.items():
                        item["ratio"] = item["ratio"] / parent_array[case]
                        #item["adjusted_ratio"] = item["ratio"] / parent_array[case]
                return mech_ratios
            else:
                # Both are numpy arrays
                return mech_ratios / parent_ratios

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
                    f"{training_config.exp_name}_{optimizer_config.name}_lr_{optimizer_config.optimizer_kwargs.get('lr', 0.0)}_{register_key}_{self.nnunet_dataset}_{timestamp}",
                )
            )
        if training_config.save_path is None:
            training_config.save_path = Path(
                os.path.join(
                    self.result_folder,
                    "models_save",
                    f"{training_config.exp_name}_{optimizer_config.name}_lr_{optimizer_config.optimizer_kwargs.get('lr', 0.0)}_{register_key}_{self.nnunet_dataset}_{timestamp}",
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
