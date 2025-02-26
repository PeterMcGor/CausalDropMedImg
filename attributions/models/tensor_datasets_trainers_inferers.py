from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from attributions.models.base_models import BaseInferenceWithSpecs, BaseTrainerWithSpecs, DataLoaderSpecs, InferenceConfig, TrainingConfig


class TorchTensorDataLoaderSpecs(DataLoaderSpecs):
    """Specifications for PyTorch tensor-based DataLoader"""
    def __init__(
        self,
        feature_columns: List[str],  # Columns to use as features
        batch_size: int = 32,
        num_processes: int = 1  # Usually 1 for tensor datasets
    ):
        super().__init__(
            loader_type=DataLoader,
            dataset_type=TensorDataset,
            batch_keys=[],
            batch_size=batch_size,
            num_processes=num_processes
        )
        self.feature_columns = feature_columns

    def prepare_data(
        self,
        data: pd.DataFrame,
        label_column: Optional[str] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Prepare data from DataFrame to tensor format.

        Args:
            data: Input DataFrame
            label_column: Column to use as labels (if None, only features returned)

        Returns:
            If label_column is None: feature tensor
            If label_column is provided: (feature tensor, label tensor)
        """
        features = torch.FloatTensor(data[self.feature_columns].values)

        if label_column is not None:
            labels = torch.FloatTensor(data[label_column].values)
            return features, labels

        return features



class TorchTensorDiscriminatorTrainer(BaseTrainerWithSpecs):
    """Trainer for PyTorch discriminator models working with tensor data"""
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        config: TrainingConfig,
        dataloader_specs: TorchTensorDataLoaderSpecs
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
            metric_computer=None,  # We handle metrics directly
            dataloader_specs=dataloader_specs
        )

    def _process_batch(self, batch: Dict[str, Any]) -> tuple:
        """Process batch for PyTorch model"""
        inputs, labels = batch[0], batch[1]
        return inputs.to(self.device), labels.to(self.device)

    def _compute_batch_metrics(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """Compute metrics for binary classification using torch operations"""
        probs = outputs.sigmoid()

        # Brier score implementation in torch
        brier = ((probs - labels) ** 2).mean()

        # ROC AUC implementation in torch
        # Sort probabilities and corresponding labels
        sorted_probs, sort_idx = torch.sort(probs, descending=True)
        sorted_labels = labels[sort_idx]

        # Calculate TPR and FPR
        tps = torch.cumsum(sorted_labels, dim=0)
        fps = torch.cumsum(1 - sorted_labels, dim=0)

        # Get total positives and negatives
        total_pos = sorted_labels.sum()
        total_neg = len(sorted_labels) - total_pos

        # Calculate rates
        tpr = tps / total_pos
        fpr = fps / total_neg

        # Calculate AUC using trapezoidal rule
        width = fpr[1:] - fpr[:-1]
        height = (tpr[1:] + tpr[:-1]) / 2
        auc = (width * height).sum()

        return {
            'roc_auc': auc.item(),
            'brier': brier.item()
        }


class TorchTensorInference(BaseInferenceWithSpecs):
    """Inference class for models using TensorDataset"""
    def __init__(
        self,
        model: torch.nn.Module,
        config: InferenceConfig,
        dataloader_specs: 'TorchTensorDataLoaderSpecs',
        output_transform: Optional[Callable] = None,
        post_process: Optional[Callable] = None,
        id_column: Optional[str] = None
    ):
        super().__init__(
            model=model,
            config=config,
            dataloader_specs=dataloader_specs,
            output_transform=output_transform,
            post_process=post_process
        )
        self.id_column = id_column

    def run_inference_on_dataframe(self, data: 'pd.DataFrame') -> Dict[str, Any]:
        """Run inference directly on a pandas DataFrame"""
        import pandas as pd
        from torch.utils.data import DataLoader, TensorDataset

        # Prepare features
        features = torch.FloatTensor(data[self.dataloader_specs.feature_columns].values)

        # Create dataset and dataloader
        dataset = TensorDataset(features)
        dataloader = DataLoader(
            dataset,
            batch_size=self.dataloader_specs.batch_size,
            shuffle=False
        )

        # Extract case identifiers if id_column is provided
        case_identifiers = []
        if self.id_column and self.id_column in data.columns:
            case_identifiers = data[self.id_column].tolist()
        else:
            case_identifiers = [f"case_{i}" for i in range(len(data))]

        # Run inference
        self.model.eval()
        all_outputs = []

        with torch.no_grad():
            for batch_idx, (inputs,) in enumerate(dataloader):
                inputs = inputs.to(self.device)
                raw_outputs = self.model(inputs)
                transformed_outputs = self.output_transform(raw_outputs)

                # Convert to numpy for easier handling
                if isinstance(transformed_outputs, torch.Tensor):
                    batch_outputs = transformed_outputs.cpu().numpy()
                else:
                    batch_outputs = transformed_outputs

                all_outputs.append(batch_outputs)
                self._log(f"Processed batch {batch_idx+1}/{len(dataloader)}", level=2)

        # Concatenate all batch outputs
        if all(isinstance(out, np.ndarray) for out in all_outputs):
            combined_outputs = np.concatenate(all_outputs, axis=0)
        else:
            combined_outputs = all_outputs

        # Process each output individually if needed
        processed_outputs = [self.post_process(output) for output in combined_outputs]

        # Save results if configured
        if len(case_identifiers) == len(processed_outputs):
            self._save_outputs(processed_outputs, case_identifiers)

        # Return results in a structured format
        results = {
            'outputs': processed_outputs,
            'case_identifiers': case_identifiers
        }

        # If id_column was provided, create a DataFrame with results
        if self.id_column and self.config.save_outputs and self.config.output_path:
            # Convert outputs to a format suitable for DataFrame
            if isinstance(processed_outputs[0], np.ndarray) and processed_outputs[0].size > 1:
                # For multi-dimensional outputs, create multiple columns
                result_cols = {}
                for i, output in enumerate(processed_outputs):
                    flat_output = output.flatten()
                    for j, val in enumerate(flat_output):
                        col_name = f"output_{j}"
                        if col_name not in result_cols:
                            result_cols[col_name] = []
                        result_cols[col_name].append(val)

                result_df = pd.DataFrame({self.id_column: case_identifiers})
                for col_name, values in result_cols.items():
                    result_df[col_name] = values
            else:
                # For scalar outputs, create a single column
                result_df = pd.DataFrame({
                    self.id_column: case_identifiers,
                    'output': [o if np.isscalar(o) else o[0] for o in processed_outputs]
                })

            # Save to CSV
            result_path = self.config.output_path / "inference_results.csv"
            result_df.to_csv(result_path, index=False)
            self._log(f"Saved results to {result_path}")

            # Add DataFrame to results
            results['result_df'] = result_df

        return results

    def _process_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Process tensor batch for inference"""
        # For TensorDataset, batch is already a tensor
        return batch.to(self.device)

    def _extract_case_identifiers(self, batch: torch.Tensor) -> List[str]:
        """Generate default case identifiers for tensor batch"""
        # For tensor batches, we use generic identifiers
        return [f"case_{i}" for i in range(batch.shape[0])]

