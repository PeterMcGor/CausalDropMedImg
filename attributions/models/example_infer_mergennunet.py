
# Example usage
import os
from pathlib import Path

import numpy as np
import torch
from attributions.models.base_models import InferenceConfig
from attributions.models.merge_nnunet_trainers_inferers import MergedNNUNetDataLoaderSpecs, MergedNNUNetInference
from attributions.models.monai_binary import BinaryClassifierConfig, MonaiBinaryClassifier, MonaiBinaryClassifierInference, MonaiModelType
from nnunet_utils.dataset_utils import MergerNNUNetDataset

def run_example():
    """
    An example showing how to use preprocessing_iterator_fromfiles with MergedNNUNetInference
    """

    # Data paths - update these to your actual data
    dataset_folder = '/home/jovyan/nnunet_data/nnUNet_raw/Dataset824_FLAWS-HCO'
    config_folder = '/home/jovyan/nnunet_data/nnUNet_preprocessed/Dataset824_FLAWS-HCO/'
    test_folder = os.path.join(config_folder, 'nnUNetPlans_3d_fullres_test_images')

    #test_files = [f for f in os.listdir(test_folder) if f.endswith('.nii.gz')]
    #input_folders = [[os.path.join(test_folder, f)] for f in test_files[:2]]

    # Path to your saved model
    model_path = '/workspaces/MICCAI_2025/models_save/monai_binary_classifier_test_just_images_50B/Dataset824_FLAWS-HCO_20250225_161555/best_model.pth'  # Update this to your actual model path

    # Create inference config
    config = InferenceConfig(
        model_path=Path(model_path),
        device="cuda" if torch.cuda.is_available() else "cpu",
        output_path=Path('./inference_results'),
        verbosity=1
    )

    # Create model configuration
    classifier_config = BinaryClassifierConfig(
        model_type=MonaiModelType.DENSENET121,
        num_input_channels=2,  # data + target channels
        #dropout_prob=0.2
    )
    def process_case_id(case_id: str, nnunet_dataset=None) -> int:
            return 'deployment' in case_id

    dataset2 = MergerNNUNetDataset(
            test_folder,
            additional_data={'source': test_folder, 'deployment_data': process_case_id, 'test_data': 1}
        )

    dataloader_specs = MergedNNUNetDataLoaderSpecs(
        dataset_json_path=os.path.join(config_folder, 'dataset.json'),
        dataset_plans_path=os.path.join(config_folder, 'nnUNetPlans.json'),
        dataset_train=dataset2,
        dataset_val=dataset2,
        batch_size=4,
        num_processes=12,
        unpack_data=True,
        inference=True
        )

    #inference_loader = dataloader_specs.val_loader


    print("\nRunning inference by loading model from checkpoint...")
    inference_tool = MonaiBinaryClassifierInference.from_checkpoint(
        model_class=MonaiBinaryClassifier,
        model_args={'config': classifier_config},
        config=config,
        dataloader_specs = dataloader_specs,
        output_transform=torch.nn.Softmax(dim=1),
        post_process=lambda x: x.cpu().numpy()
    )

    # Run inference
    try:
        results = inference_tool.run_inference()
        print(f"Processed {len(results)} batches successfully")
        print(results)

        # Assuming your data is in a variable called 'data_list'
        # Initialize an empty list to collect all probability arrays
        all_probabilities = []

        # Loop through each dictionary in the data list
        for item in results:
            # Extract the probabilities array from each item and add it to our collection
            all_probabilities.append(item['probabilities'])

        # Combine all probability arrays into a single 2D array
        # This will stack all the arrays vertically (along axis 0)
        combined_probabilities = np.vstack(all_probabilities)

        print(f"Combined shape: {combined_probabilities.shape}")
        print(combined_probabilities)
    except Exception as e:
        print(f"Error during inference: {e}")

if __name__ == "__main__":
    print("-----run_example-----------------------")
    run_example()