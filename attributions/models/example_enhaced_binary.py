import os
import torch
import numpy as np
from pathlib import Path
import argparse

from monai_binary import MonaiBinaryClassifier, BinaryClassifierConfig, MonaiModelType
from attributions.models.base_models import InferenceConfig
from enhanced_binary_inference import EnhancedMonaiBinaryClassifierInference

def run_inference_example(
    model_path: str,
    dataset_folder: str,
    output_folder: str,
    max_files: int = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Run inference on test files using a trained MONAI binary classifier

    Args:
        model_path: Path to model checkpoint
        dataset_folder: Path to nnUNet raw dataset folder
        output_folder: Path to save results
        max_files: Maximum number of files to process (None for all)
        device: Device to use for inference
    """
    print("=" * 50)
    print("MONAI Binary Classifier Inference")
    print("=" * 50)

    # Step 1: Setup paths
    model_path = Path(model_path)
    dataset_folder = Path(dataset_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Get preprocessed folder path
    preprocessed_folder = dataset_folder.parent / "nnUNet_preprocessed" / dataset_folder.name
    dataset_json_path = str(preprocessed_folder / 'dataset.json')
    dataset_plans_path = str(preprocessed_folder / 'nnUNetPlans.json')

    print(f"Model path: {model_path}")
    print(f"Dataset folder: {dataset_folder}")
    print(f"Preprocessed folder: {preprocessed_folder}")
    print(f"Output folder: {output_folder}")

    # Step 2: Find test files
    test_folder = dataset_folder / 'imagesTs'
    if not test_folder.exists():
        raise ValueError(f"Test folder not found: {test_folder}")

    test_files = list(test_folder.glob('*.nii.gz'))
    if not test_files:
        raise ValueError(f"No .nii.gz files found in {test_folder}")

    if max_files:
        test_files = test_files[:max_files]

    print(f"Found {len(test_files)} test files")

    # Step 3: Create input folders format for nnUNet pipeline
    input_folders = [[str(f)] for f in test_files]

    # Step 4: Create inference configuration
    inference_config = InferenceConfig(
        model_path=model_path,
        input_folders=input_folders,
        device=device,
        output_path=output_folder,
        verbosity=1,
        save_outputs=True
    )

    # Step 5: Create model configuration matching your trained model
    classifier_config = BinaryClassifierConfig(
        model_type=MonaiModelType.DENSENET121,  # Update based on your model type
        num_input_channels=2,  # data + target channels
        dropout_prob=0.2,
        out_channels=2  # Binary classification
    )

    # Step 6: Create inference runner
    print("\nInitializing inference runner...")
    inference = EnhancedMonaiBinaryClassifierInference.from_checkpoint(
        model_class=MonaiBinaryClassifier,
        model_args={'config': classifier_config},
        checkpoint_path=str(model_path),
        config=inference_config,
        dataset_plans_path=dataset_plans_path,
        dataset_json_path=dataset_json_path,
        output_transform=torch.nn.Softmax(dim=1),
        num_processes=8
    )

    # Step 7: Run inference
    print("\nRunning inference...")
    try:
        results = inference.run_inference()

        # Step 8: Display results summary
        df = results.to_dataframe()
        print("\nResults summary:")
        print(f"Total cases: {len(df)}")
        print(f"Positive predictions: {sum(df['prediction'] == 1)}")
        print(f"Negative predictions: {sum(df['prediction'] == 0)}")

        print("\nPredictions:")
        print(df.head(10))  # Show first 10 predictions

        print(f"\nResults saved to {output_folder}")
        return results

    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run MONAI Binary Classifier Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, required=True, help='Path to nnUNet raw dataset folder')
    parser.add_argument('--output', type=str, required=True, help='Path to save results')
    parser.add_argument('--max_files', type=int, default=None, help='Maximum number of files to process')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    run_inference_example(
        model_path=args.model,
        dataset_folder=args.dataset,
        output_folder=args.output,
        max_files=args.max_files,
        device=args.device
    )