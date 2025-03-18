# tests/memory_tests/test_sequential_training.py
import pytest
import torch
import gc
import numpy as np
import time
from pathlib import Path
import os

# Import your actual implementation - adjust these imports as needed
from attributions.core.mechanism_shift import CausalMechanismShift
from attributions.core.distribution_base import MechanismSpec
from attributions.models.base_models import TrainingConfig, MetricConfig, MetricGoal
from attributions.distribution_estimators.importance_sampling.nnunet_discriminators import (
    FlexibleNNunetBinaryDiscriminatorRatioEstimator
)

# Mark this file as slow tests that require more resources
pytestmark = [pytest.mark.memory, pytest.mark.slow]

@pytest.fixture
def mini_training_config():
    """Create a minimal training configuration for testing"""
    return TrainingConfig(
        num_epochs=2,  # Just 2 epochs for testing
        val_interval=1,
        num_train_iterations_per_epoch=3,
        num_val_iterations_per_epoch=2,
        metric=MetricConfig("balanced_acc", MetricGoal.MAXIMIZE),
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbosity=1,
        log_path=None,
        save_path=None,
        exp_name="Memory_Test",
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_real_discriminator_sequential_training(memory_tracker, mini_training_config, tmp_path):
    """Test memory behavior with real discriminator training sequence"""
    # Setup temporary paths for outputs
    result_folder = tmp_path / "results"
    os.makedirs(result_folder, exist_ok=True)

    # Update training config with temp paths
    mini_training_config.log_path = tmp_path / "logs"
    mini_training_config.save_path = tmp_path / "models"

    memory_tracker.snapshot("Initial state")

    # Create minimal test mechanisms
    mechanisms = [
        MechanismSpec(
            variables_key="images",
            variables=["images"],
            parents_key=None,
            parents=None,
        ),
        MechanismSpec(
            variables_key="images_labels",
            variables=["images", "labels"],
            parents_key="images",
            parents=["images"],
        )
    ]

    memory_tracker.snapshot("After creating mechanisms")

    # Mock the discriminator
    class MockDiscriminator(FlexibleNNunetBinaryDiscriminatorRatioEstimator):
        def __init__(self, *args, **kwargs):
            # Skip the real __init__ to avoid actual data loading
            self.fitted_models = {}
            self.training_config = mini_training_config

        def _fit_mechanism_models(self, train_data, inference_data, input_features, register_key):
            """Mock implementation that actually allocates memory similarly to real code"""
            memory_tracker.snapshot(f"Start fitting {register_key}")

            # Create a model that consumes a predictable amount of memory
            num_channels = len(input_features)
            model = torch.nn.Sequential(
                torch.nn.Conv3d(num_channels, 32, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool3d(kernel_size=2),
                torch.nn.Conv3d(32, 64, kernel_size=3, padding=1),
                torch.nn.ReLU()
            ).cuda()

            # Simulate training
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            for epoch in range(2):
                # Generate random batches and train
                for _ in range(3):
                    # Random 3D medical image-like data
                    inputs = torch.randn(2, num_channels, 32, 32, 32).cuda()
                    labels = torch.randint(0, 2, (2,)).cuda()

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = torch.nn.functional.cross_entropy(outputs, labels)
                    loss.backward()
                    optimizer.step()

            memory_tracker.snapshot(f"After training {register_key}")

            # Store the model path (only the path, not the model)
            dummy_path = f"{tmp_path}/models/{register_key}_dummy.pth"
            self.fitted_models[register_key] = dummy_path

            # Cleanup in different ways based on implementation being tested

            # 1. Original implementation (minimal cleanup)
            if hasattr(self, "_test_mode") and self._test_mode == "original":
                pass  # No explicit cleanup

            # 2. Basic cleanup
            elif hasattr(self, "_test_mode") and self._test_mode == "basic":
                del model
                del optimizer
                torch.cuda.empty_cache()

            # 3. Thorough cleanup (new implementation)
            else:
                del model
                del optimizer
                gc.collect()
                torch.cuda.empty_cache()

            memory_tracker.snapshot(f"After cleanup for {register_key}")

    # Test original implementation
    discriminator = MockDiscriminator()
    discriminator._test_mode = "original"

    memory_tracker.snapshot("Before original fit")
    discriminator.fit(None, None, mechanisms)
    memory_tracker.snapshot("After original fit")

    # Force cleanup
    del discriminator
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(1)

    # Test basic cleanup implementation
    discriminator = MockDiscriminator()
    discriminator._test_mode = "basic"

    memory_tracker.snapshot("Before basic fit")
    discriminator.fit(None, None, mechanisms)
    memory_tracker.snapshot("After basic fit")

    # Force cleanup
    del discriminator
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(1)

    # Test thorough cleanup implementation (new version)
    discriminator = MockDiscriminator()
    discriminator._test_mode = "thorough"

    memory_tracker.snapshot("Before thorough fit")
    discriminator.fit(None, None, mechanisms)
    memory_tracker.snapshot("After thorough fit")

    # Final cleanup
    del discriminator
    gc.collect()
    torch.cuda.empty_cache()

    memory_tracker.snapshot("Final state")
    memory_tracker.print_report()

    # Check which implementation had the best memory profile
    # Extract peaks during each implementation
    snapshots = memory_tracker.snapshots
    original_snapshots = [s for s in snapshots if "original fit" in s['label']]
    basic_snapshots = [s for s in snapshots if "basic fit" in s['label']]
    thorough_snapshots = [s for s in snapshots if "thorough fit" in s['label']]

    # Get peak RAM values
    def get_peak_ram(snapshots):
        return max(s['ram'] for s in snapshots) if snapshots else 0

    original_peak = get_peak_ram(original_snapshots)
    basic_peak = get_peak_ram(basic_snapshots)
    thorough_peak = get_peak_ram(thorough_snapshots)

    # Check how well memory was released
    assert thorough_peak <= basic_peak, "Thorough cleanup should not use more peak memory than basic cleanup"