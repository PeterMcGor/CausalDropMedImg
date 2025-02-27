
# Example usage with multiple initialization approaches
import os
from attributions.distribution_estimators.importance_sampling.discriminator import FlexibleNNunetBinaryDiscriminatorRatioEstimator
from attributions.models.base_models import MetricConfig, MetricGoal, TrainingConfig


# Example usage with flexible approach
if __name__ == "__main__":
    # Common configuration
    training_config = TrainingConfig(
        num_epochs=10,
        val_interval=2,
        num_train_iterations_per_epoch=40,
        num_val_iterations_per_epoch=10,
        metric=MetricConfig("f1", MetricGoal.MAXIMIZE),
        device="cuda",
        verbosity=1,
        log_path=None,
        save_path=None,
    )

    def center_number_groupby(key):
        parts = key.split('_')
        for i, part in enumerate(parts):
            if part.lower() == "center" and i + 1 < len(parts):
                return parts[i+1]
        return "unknown"

    # Initialize with minimal settings - skip default preprocessing
    nnunet_folders_path = '/home/jovyan/nnunet_data/'
    nnunet_dataset = 'Dataset001_MSSEG_FLAIR_Annotator1'

    # Create results folder explicitly
    result_folder = os.path.join(nnunet_folders_path, nnunet_dataset, 'results')
    os.makedirs(result_folder, exist_ok=True)

    discriminator = FlexibleNNunetBinaryDiscriminatorRatioEstimator(
        nnunet_dataset=nnunet_dataset,
        nnunet_folders_path=nnunet_folders_path,
        training_config=training_config,
        stratify_by=center_number_groupby,
        result_folder=result_folder,  # Explicitly provide result folder
        skip_default_preprocessing=True  # Skip automatic preprocessing
    )

    #### Training data for the discriminator from nnunet training data training with annotator 1####
    train_nnunet_path = discriminator.preprocess_custom_folder(
        input_images_folder=f'{nnunet_folders_path}/nnUNet_raw/{nnunet_dataset}/imagesTr',
        input_segs_folder=f'{nnunet_folders_path}/nnUNet_raw/{nnunet_dataset}/labelsTr',
        folder_name='train_data'
    )
    training_nnunet_dataset = discriminator.create_dataset_from_folder(train_nnunet_path, is_test_data=False)

    ######## Same domain since is label from annotator 1 #########
    test_nnunet = discriminator.preprocess_custom_folder(
        input_images_folder=f'{nnunet_folders_path}/nnUNet_raw/{nnunet_dataset}/imagesTs',
        input_segs_folder=f'{nnunet_folders_path}/nnUNet_raw/{nnunet_dataset}/labelsTs_1',
        folder_name='train_transport'
    )
    test_nnunet_dataset = discriminator.create_dataset_from_folder(test_nnunet, is_test_data=False)
    def exclude_center_pattern(case_id: str) -> bool:
        return 'Center_03' not in case_id
    # center 03 not in the source, I want just label effect here
    just_in_train_centers_dataset_unseen_for_transport = test_nnunet_dataset.subset_by_pattern(exclude_center_pattern) # for
    #infer_env_data_for_disc, infer_env_data_for_disc_for_infer = just_in_train_centers_dataset_unseen.random_split(0.65)

    # now he out of domain dataset
    test_path = discriminator.preprocess_custom_folder(
        input_images_folder=f'{nnunet_folders_path}/nnUNet_raw/{nnunet_dataset}/imagesTs',
        input_segs_folder=f'{nnunet_folders_path}/nnUNet_raw/{nnunet_dataset}/labelsTs_3', # just change the labe here
        folder_name='test'
    )
    out_of_dataset = discriminator.create_dataset_from_folder(test_path, is_test_data=True)
    out_of_domain_with_same_images = out_of_dataset.subset_by_pattern(exclude_center_pattern)
    test_data_for_disc, test_data_for_infer = out_of_domain_with_same_images.random_split(0.65)

    train_data, val_data = training_nnunet_dataset.merge_and_split(test_data_for_disc, split_ratio=0.7)
    discriminator.set_datasets(
        #train_dataset=training_nnunet_dataset.merge(test_data_for_disc),
        train_data=train_data,
        val_data=val_data,
        test_dataset_train_domain=just_in_train_centers_dataset_unseen_for_transport,
        test_dataset_inference_domain=test_data_for_infer
    )

    # IMPORTANT: Use separate register_keys for different feature sets
    # Train model for images+labels
    discriminator._fit_mechanism_models(
        input_features=['images', 'labels'],
        register_key='images_labels'
    )

    # Train model for images only
    discriminator._fit_mechanism_models(
        input_features=['images'],
        register_key='images'  # Use a different key!
    )

    # Get probabilities using correct keys and variables
    # For images+labels model
    probs_train = discriminator._get_probabilities(
        variables=['images', 'labels'],  # Match the input_features used in training
        register_key='images_labels',   # Match the register_key used in training
        domain='train'
    )

    probs_inference = discriminator._get_probabilities(
        variables=['images', 'labels'],  # Match the input_features used in training
        register_key='images_labels',   # Match the register_key used in training
        domain='inference'
    )

    # For images-only model
    probs_train_images = discriminator._get_probabilities(
        variables=['images'],          # Match the input_features used in training
        register_key='images',   # Match the register_key used in training
        domain='train'
    )

    probs_inference_images = discriminator._get_probabilities(
        variables=['images'],          # Match the input_features used in training
        register_key='images',   # Match the register_key used in training
        domain='inference'
    )

    print("Ratio train domain (images+labels):", probs_train[:,1]/probs_train[:,0])
    print("Ratio inference domain (images+labels):", probs_inference[:,1]/probs_inference[:,0])
    print("Ratio train domain (images only):", probs_train_images[:,1]/probs_train_images[:,0])
    print("Ratio inference domain (images only):", probs_inference_images[:,1]/probs_inference_images[:,0])