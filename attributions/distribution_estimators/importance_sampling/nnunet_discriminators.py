
import copy
from datetime import datetime
import gc
import os
import tempfile
from pathlib import Path
from typing import Callable, List, Optional, Union
import numpy as np
import torch
from attributions.distribution_estimators.importance_sampling.discriminator import DiscriminatorRatioEstimator
from attributions.models.base_models import CriterionConfig, InferenceConfig, MetricConfig, MetricGoal, OptimizerConfig, TrainingConfig
from attributions.models.merge_nnunet_trainers_inferers import MergedNNUNetDataLoaderSpecs
from attributions.models.monai_binary import BinaryClassifierConfig, BinaryMergedNNUNetTrainer, BinaryMergedNNUNetTrainerImages, MonaiBinaryClassifier, MonaiBinaryClassifierInference, MonaiBinaryClassifierInferenceImages
from nnunet_utils.dataset_utils import MergerNNUNetDataset
from nnunet_utils.preprocess import AnyFolderPreprocessor
from attributions.utils import get_available_cpus


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
        #if optimizer_config is None:
        #    optimizer_config = OptimizerConfig(
        #        optimizer_class=torch.optim.AdamW, optimizer_kwargs={"lr": 1e-4}
        #    )
        # trying this to capture subtle differences
        if optimizer_config is None:
            optimizer_config = OptimizerConfig(
            optimizer_class=torch.optim.AdamW,
            optimizer_kwargs={
                "lr": 5e-5,
                "betas": (0.9, 0.999),  # Default values, but you could try (0.95, 0.999)
                "eps": 1e-8,
                "weight_decay": 0.01    # Increase slightly from default
            }
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
        def _get_n_test_data(merge_nnunet_dataset):
            return sum([test_or_not_test['test_data'] == 1 for test_or_not_test in merge_nnunet_dataset.extract_per_case_id('dataset_info').values()])
        print("\n=== Dataset Status ===")
        if self.train_data is not None:
            n_train_data = len(self.train_data)
            n_from_new_enviroment = _get_n_test_data(self.train_data)
            print(f"Train: {n_train_data} samples. {n_train_data - n_from_new_enviroment} from the training environment, {n_from_new_enviroment} from the new environment")
        else:
            print("Train: None")

        if self.val_data is not None:
            n_val_data = len(self.val_data)
            n_from_new_enviroment = _get_n_test_data(self.val_data)
            print(f"Val: {n_val_data} samples. {n_val_data - n_from_new_enviroment} from the training environment, {n_from_new_enviroment} from the new environment")
        else:
            print("Val: None")

        if self.test_data_env_train is not None:
            n_test_data = len(self.test_data_env_train)
            n_from_new_enviroment = _get_n_test_data(self.test_data_env_train)
            print(f"Test (train environment) for transport: {n_test_data} samples. {n_test_data - n_from_new_enviroment} from the training environment, {n_from_new_enviroment} from the new environment. Should be 0")
        else:
            print("Test (train environment): None")

        if self.test_data_env_test is not None:
            n_test_data = len(self.test_data_env_test)
            n_from_new_enviroment = _get_n_test_data(self.test_data_env_test)
            print(f"Test (inference environment): {n_test_data} samples. {n_test_data - n_from_new_enviroment} from the training environment (should be 0), {n_from_new_enviroment} from the new environment")
        else:
            print("Test (inference environment): None")
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
        #if register_key == 'images':
        #    self.fitted_models[register_key] = "/home/jovyan/nnunet_data/Dataset001_MSSEG_FLAIR_Annotator1/results/models/Exp3_Annotator1_vs_Annotator1_AdamW_lr_5e-05_images_Dataset001_MSSEG_FLAIR_Annotator1_20250319_212945/best_model.pth" # f1 trained
        #    return
        #if register_key == 'images_labels':
        #    self.fitted_models[register_key] = "/home/jovyan/nnunet_data/Dataset001_MSSEG_FLAIR_Annotator1/results/models/Exp3_Annotator1_vs_Annotator1_AdamW_lr_5e-05_images_labels_Dataset001_MSSEG_FLAIR_Annotator1_20250319_213256/best_model.pth" #
        #    return

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
                self.result_folder, "logs", f"{training_config.exp_name}_{optimizer_config.name}_lr_{optimizer_config.optimizer_kwargs.get('lr', 0.0)}_{register_key}_{self.nnunet_dataset}_{timestamp}"
            )
            training_config.log_path = Path(log_path)

        if training_config.save_path is None:
            save_path = os.path.join(
                self.result_folder, "models", f"{training_config.exp_name}_{optimizer_config.name}_lr_{optimizer_config.optimizer_kwargs.get('lr', 0.0)}_{register_key}_{self.nnunet_dataset}_{timestamp}"
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
            print("Creating mixed (images+labels) trainer")
            trainer = BinaryMergedNNUNetTrainer.create(
                classifier_config=binary_config,
                training_config=training_config,
                optimizer_config=optimizer_config,
                dataloader_specs=mergennunet_trainer_dataloader_specs,
            )
        else:
            print("Creating images-only trainer")
            trainer = BinaryMergedNNUNetTrainerImages.create(
                classifier_config=binary_config,
                training_config=training_config,
                optimizer_config=optimizer_config,
                dataloader_specs=mergennunet_trainer_dataloader_specs,
            )

        # Monitor memory before training
        print("\n--- Memory before training ---")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")


        # Train model
        print(f"Training model for {register_key}...")
        results = trainer.train()
        model_path = results["model_path"]
        print(f"Model saved to {model_path}")

        self.fitted_models[register_key] = results[
            "model_path"
        ]  # so the path to a file not the model for now

        # Explicit cleanup
        print("\nCleaning up resources...")
        del trainer
        del binary_config
        del training_config
        del optimizer_config
        del mergennunet_trainer_dataloader_specs

        # Monitor memory after cleanup
        gc.collect()
        torch.cuda.empty_cache()
        print("\n--- Memory after cleanup ---")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

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

        # If keeping metadata, return the full results
        if hasattr(self, 'keep_metadata') and self.keep_metadata:
            return results
        else:
            # Otherwise extract just the probability arrays
            all_probabilities = []
            for item in results:
                all_probabilities.append(item["probabilities"])
            probabilities = np.vstack(all_probabilities)
            print(f"Generated probabilities with shape {probabilities.shape}")
            return probabilities

