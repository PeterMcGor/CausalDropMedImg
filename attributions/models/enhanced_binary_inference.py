import os
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional, Union

from attributions.models.base_models import InferenceConfig
from attributions.models.merge_nnunet_trainers_inferers import MergedNNUNetInference
from nnunetv2.inference.data_iterators import preprocessing_iterator_fromfiles
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from batchgenerators.utilities.file_and_folder_operations import load_json

class BinaryClassificationResults:
    """Helper class to process and analyze binary classification results"""
    def __init__(self, results: List[Dict[str, Any]]):
        self.results = results
        self.case_ids = [r['case_id'] for r in results]
        self.predictions = [r['prediction'] for r in results]
        self.probabilities = [r['probabilities'] for r in results]

    def to_dataframe(self):
        """Convert results to pandas DataFrame"""
        import pandas as pd

        # Process predictions and probabilities to handle different formats
        processed_preds = []
        processed_probs = []

        for pred, prob in zip(self.predictions, self.probabilities):
            # Handle array predictions
            if isinstance(pred, np.ndarray):
                if pred.size == 1:
                    processed_preds.append(int(pred[0]))
                else:
                    processed_preds.append(int(np.argmax(pred)))
            else:
                processed_preds.append(int(pred))

            # Handle probability arrays
            if isinstance(prob, np.ndarray):
                if prob.size > 1:
                    # For multi-class, get probability of positive class
                    processed_probs.append(float(prob[1]))
                else:
                    processed_probs.append(float(prob))
            else:
                processed_probs.append(float(prob))

        return pd.DataFrame({
            'case_id': self.case_ids,
            'prediction': processed_preds,
            'probability': processed_probs
        })

    def save_csv(self, output_path: Union[str, Path]):
        """Save results to CSV file"""
        df = self.to_dataframe()
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        return output_path

    def get_performance_metrics(self, true_labels: Dict[str, int]) -> Dict[str, float]:
        """Calculate performance metrics if true labels are available"""
        predictions = self.to_dataframe()
        pred_dict = dict(zip(predictions['case_id'], predictions['prediction']))

        # Get matching cases
        common_cases = set(pred_dict.keys()).intersection(true_labels.keys())
        if not common_cases:
            return {"error": "No matching case IDs between predictions and true labels"}

        y_pred = [pred_dict[case_id] for case_id in common_cases]
        y_true = [true_labels[case_id] for case_id in common_cases]

        # Calculate metrics
        tp = sum((p == 1 and t == 1) for p, t in zip(y_pred, y_true))
        tn = sum((p == 0 and t == 0) for p, t in zip(y_pred, y_true))
        fp = sum((p == 1 and t == 0) for p, t in zip(y_pred, y_true))
        fn = sum((p == 0 and t == 1) for p, t in zip(y_pred, y_true))

        accuracy = (tp + tn) / len(y_true) if y_true else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        balanced_acc = (recall + specificity) / 2

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "balanced_acc": balanced_acc,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn
        }


class EnhancedMonaiBinaryClassifierInference:
    """
    Enhanced inference class for MONAI binary classifiers that works with nnUNet preprocessing
    """
    def __init__(
        self,
        model: torch.nn.Module,
        config: InferenceConfig,
        dataset_plans_path: str,
        dataset_json_path: str,
        architecture_config: str = '3d_fullres',
        output_transform: Optional[Callable] = None,
        post_process: Optional[Callable] = None,
        cleanup_temp_files: bool = True,
        num_processes: int = max(1, os.cpu_count() - 2)
    ):
        """
        Initialize the inference class

        Args:
            model: Trained PyTorch model
            config: Inference configuration
            dataset_plans_path: Path to nnUNet plans file
            dataset_json_path: Path to nnUNet dataset.json
            architecture_config: nnUNet architecture configuration
            output_transform: Function to transform model outputs (e.g., softmax)
            post_process: Function for post-processing outputs
            cleanup_temp_files: Whether to clean up temporary files
            num_processes: Number of processes for data loading
        """
        self.model = model
        self.config = config
        self.cleanup_temp_files = cleanup_temp_files
        self.device = torch.device(config.device if isinstance(config.device, str) else config.device)

        # Move model to device and set to evaluation mode
        self.model = self.model.to(self.device)
        self.model.eval()

        # Set default transforms if not provided
        self.output_transform = output_transform or torch.nn.Softmax(dim=1)
        self.post_process = post_process or (lambda x: x)

        # Setup nnUNet components
        self.plans_manager = PlansManager(dataset_plans_path)
        self.dataset_json = load_json(dataset_json_path)
        self.config_manager = self.plans_manager.get_configuration(architecture_config)

        # Create the data iterator using preprocessing_iterator_fromfiles
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

    @classmethod
    def from_checkpoint(
        cls,
        model_class: type,
        model_args: Dict[str, Any],
        checkpoint_path: str,
        config: InferenceConfig,
        dataset_plans_path: str,
        dataset_json_path: str,
        architecture_config: str = '3d_fullres',
        output_transform: Optional[Callable] = None,
        post_process: Optional[Callable] = None,
        cleanup_temp_files: bool = True,
        num_processes: int = max(1, os.cpu_count() - 2)
    ) -> 'EnhancedMonaiBinaryClassifierInference':
        """Create binary classifier inference from checkpoint"""
        # Initialize the model
        model = model_class(**model_args)

        # Load weights from checkpoint
        device = config.device
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Try loading directly
            model.load_state_dict(checkpoint)

        return cls(
            model=model,
            config=config,
            dataset_plans_path=dataset_plans_path,
            dataset_json_path=dataset_json_path,
            architecture_config=architecture_config,
            output_transform=output_transform,
            post_process=post_process,
            cleanup_temp_files=cleanup_temp_files,
            num_processes=num_processes
        )

    def _process_batch(self, data):
        """Process a batch for the model input"""
        if isinstance(data, torch.Tensor):
            inputs = data
        elif isinstance(data, str) and os.path.exists(data):
            # Load numpy data from file
            data_np = np.load(data)
            inputs = torch.from_numpy(data_np)

            # Clean up temporary file if needed
            if self.cleanup_temp_files:
                try:
                    os.remove(data)
                except Exception as e:
                    print(f"Warning: Could not remove temporary file {data}: {e}")
        elif isinstance(data, np.ndarray):
            inputs = torch.from_numpy(data)
        else:
            raise ValueError(f"Unsupported data format: {type(data)}")

        # Move to device
        inputs = inputs.to(self.device)

        return inputs

    def run_inference(self):
        """
        Run inference using the configured dataloader

        Returns:
            BinaryClassificationResults object with inference results
        """
        self.model.eval()
        results = []

        print(f"Starting inference with {len(self.config.input_folders)} cases...")
        with torch.no_grad():
            for batch_idx, preprocessed in enumerate(self.dataloader):
                # Extract batch data
                data = preprocessed['data']
                properties = preprocessed['data_properties']

                # Get case identifier
                case_identifier = properties.get('case_identifier',
                                   os.path.basename(properties.get('list_of_data_files',
                                   ['unknown'])[0]).split('.')[0])

                print(f"Processing batch {batch_idx}, case: {case_identifier}")

                # Process batch
                inputs = self._process_batch(data)
                print(f"Input shape: {inputs.shape}")

                # Run model
                outputs = self.model(inputs)

                # Apply output transform (e.g., softmax)
                transformed_outputs = self.output_transform(outputs)

                # Apply post-processing if needed
                processed_outputs = self.post_process(transformed_outputs)

                # For binary classification, get class index with highest probability
                if processed_outputs.dim() > 1 and processed_outputs.shape[1] > 1:
                    probabilities = processed_outputs.cpu().numpy()
                    prediction = np.argmax(probabilities, axis=1)
                else:
                    # Single value output (sigmoid)
                    probabilities = processed_outputs.cpu().numpy()
                    prediction = (probabilities > 0.5).astype(int)

                # Store results
                result = {
                    'case_id': case_identifier,
                    'prediction': prediction,
                    'probabilities': probabilities,
                    'properties': properties
                }
                results.append(result)

                print(f"Finished case {case_identifier}. Prediction: {prediction}")

                # Save individual result if output path is specified
                if self.config.output_path:
                    output_file = self.config.output_path / f"{case_identifier}_result.npz"
                    np.savez(
                        output_file,
                        prediction=prediction,
                        probabilities=probabilities
                    )

        print(f"Inference completed for {len(results)} cases")

        # Save combined results if output path is specified
        if self.config.output_path and results:
            results_obj = BinaryClassificationResults(results)
            csv_path = self.config.output_path / "predictions.csv"
            results_obj.save_csv(csv_path)

            # Save all results in a single NPZ file
            combined_output = self.config.output_path / "all_results.npz"
            np.savez(
                combined_output,
                case_ids=[r['case_id'] for r in results],
                predictions=[r['prediction'] for r in results],
                probabilities=[r['probabilities'] for r in results]
            )
            print(f"Combined results saved to {combined_output}")

        return BinaryClassificationResults(results)