# tests/memory_tests/conftest.py
import os
import gc
import pytest
import psutil
import torch
import numpy as np
from pathlib import Path

class MemoryTracker:
    """Utility class to track memory usage"""
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.reset()

    def reset(self):
        """Reset memory tracking"""
        self.start_ram = None
        self.current_ram = None
        self.peak_ram = 0
        self.start_gpu = None
        self.current_gpu = None
        self.peak_gpu = 0
        self.snapshots = []

        # Clear memory to start fresh
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def start(self):
        """Start memory tracking"""
        self.reset()
        self.start_ram = self.process.memory_info().rss / (1024 * 1024)
        self.current_ram = self.start_ram

        if torch.cuda.is_available():
            self.start_gpu = torch.cuda.memory_allocated() / (1024 * 1024)
            self.current_gpu = self.start_gpu

        return self

    def snapshot(self, label=""):
        """Take a memory snapshot"""
        self.current_ram = self.process.memory_info().rss / (1024 * 1024)
        self.peak_ram = max(self.peak_ram, self.current_ram)

        if torch.cuda.is_available():
            self.current_gpu = torch.cuda.memory_allocated() / (1024 * 1024)
            self.peak_gpu = max(self.peak_gpu, self.current_gpu)

        self.snapshots.append({
            'label': label,
            'ram': self.current_ram,
            'ram_delta': self.current_ram - self.start_ram,
            'gpu': self.current_gpu if torch.cuda.is_available() else 0,
            'gpu_delta': (self.current_gpu - self.start_gpu) if torch.cuda.is_available() else 0
        })

        return self

    def report(self):
        """Generate memory usage report"""
        report_str = "\n==== Memory Usage Report ====\n"
        report_str += f"Starting RAM: {self.start_ram:.2f} MB\n"

        if torch.cuda.is_available():
            report_str += f"Starting GPU: {self.start_gpu:.2f} MB\n"

        report_str += "\nSnapshots:\n"
        for i, snapshot in enumerate(self.snapshots):
            report_str += f"  [{i+1}] {snapshot['label']}\n"
            report_str += f"      RAM: {snapshot['ram']:.2f} MB (Δ: {snapshot['ram_delta']:+.2f} MB)\n"

            if torch.cuda.is_available():
                report_str += f"      GPU: {snapshot['gpu']:.2f} MB (Δ: {snapshot['gpu_delta']:+.2f} MB)\n"

        if len(self.snapshots) > 0:
            final = self.snapshots[-1]
            report_str += f"\nFinal RAM: {final['ram']:.2f} MB (Δ: {final['ram_delta']:+.2f} MB)\n"
            report_str += f"Peak RAM: {self.peak_ram:.2f} MB (Δ: {self.peak_ram - self.start_ram:+.2f} MB)\n"

            if torch.cuda.is_available():
                report_str += f"Final GPU: {final['gpu']:.2f} MB (Δ: {final['gpu_delta']:+.2f} MB)\n"
                report_str += f"Peak GPU: {self.peak_gpu:.2f} MB (Δ: {self.peak_gpu - self.start_gpu:+.2f} MB)\n"

        report_str += "============================\n"
        return report_str

    def print_report(self):
        """Print memory usage report"""
        print(self.report())
        return self
    
    def assert_cleanup_effective(self, threshold_pct=70, min_peak_mb=2.0):
        """Assert that memory cleanup was effective

        Args:
            threshold_pct: Percentage of peak memory that should be freed
            min_peak_mb: Minimum peak memory delta (in MB) to enforce the threshold
        """
        if len(self.snapshots) < 3:
            raise ValueError("Need at least 3 snapshots (start, peak, cleanup)")

        # Get peak memory usage (should be in the middle snapshots)
        mid_snapshots = self.snapshots[1:-1]

        # Check both RAM and GPU memory
        peak_ram_delta = max(s['ram_delta'] for s in mid_snapshots)
        final_ram_delta = self.snapshots[-1]['ram_delta']

        # Check GPU memory if available
        if 'gpu' in self.snapshots[0]:
            peak_gpu_delta = max(s['gpu_delta'] for s in mid_snapshots)
            final_gpu_delta = self.snapshots[-1]['gpu_delta']
        else:
            peak_gpu_delta = 0
            final_gpu_delta = 0

        # Determine which memory type had more significant usage
        if peak_gpu_delta > peak_ram_delta and peak_gpu_delta > min_peak_mb:
            # GPU memory was more significant
            peak_delta = peak_gpu_delta
            final_delta = final_gpu_delta
            memory_type = "GPU"
        elif peak_ram_delta > min_peak_mb:
            # RAM was significant
            peak_delta = peak_ram_delta
            final_delta = final_ram_delta
            memory_type = "RAM"
        else:
            # Neither memory type had significant usage, test passes
            print(f"Memory usage too small to enforce cleanup threshold: RAM peak {peak_ram_delta:.1f} MB, GPU peak {peak_gpu_delta:.1f} MB")
            return True

        # Calculate cleanup effectiveness
        pct_cleaned = (peak_delta - final_delta) / peak_delta * 100
        cleanup_msg = (f"{memory_type} memory cleanup: {pct_cleaned:.1f}% of peak returned "
                    f"({peak_delta:.1f} MB peak, {final_delta:.1f} MB final)")

        # Print info but don't assert if peak is very small
        print(cleanup_msg)

        assert pct_cleaned >= threshold_pct, f"Insufficient memory cleanup: {cleanup_msg}"
        return True

@pytest.fixture
def memory_tracker():
    """Fixture to provide memory tracking"""
    return MemoryTracker().start()

# Create mock model fixtures for testing
@pytest.fixture
def mock_model_factory():
    """Factory fixture to create test models that output the correct shape for classification"""
    def _create_model(input_channels=2, size="small"):
        """Create a model of specified size with proper output shape for classification

        Args:
            input_channels: Number of input channels
            size: Model size - "small", "medium", or "large"
        """
        if size == "small":
            return torch.nn.Sequential(
                torch.nn.Conv3d(input_channels, 8, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool3d(kernel_size=2),
                torch.nn.Conv3d(8, 16, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                # Add global pooling to reduce spatial dimensions
                torch.nn.AdaptiveAvgPool3d((1, 1, 1)),
                torch.nn.Flatten(),
                torch.nn.Linear(16, 2)  # Output 2 classes
            ).cuda()
        elif size == "medium":
            return torch.nn.Sequential(
                torch.nn.Conv3d(input_channels, 32, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv3d(32, 64, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool3d(kernel_size=2),
                torch.nn.Conv3d(64, 128, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                # Add global pooling to reduce spatial dimensions
                torch.nn.AdaptiveAvgPool3d((1, 1, 1)),
                torch.nn.Flatten(),
                torch.nn.Linear(128, 2)  # Output 2 classes
            ).cuda()
        elif size == "large":
            return torch.nn.Sequential(
                torch.nn.Conv3d(input_channels, 64, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv3d(64, 128, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool3d(kernel_size=2),
                torch.nn.Conv3d(128, 256, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv3d(256, 256, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                # Add global pooling to reduce spatial dimensions
                torch.nn.AdaptiveAvgPool3d((1, 1, 1)),
                torch.nn.Flatten(),
                torch.nn.Linear(256, 2)  # Output 2 classes
            ).cuda()
        else:
            raise ValueError(f"Unknown model size: {size}")

    return _create_model


@pytest.fixture
def mock_batch_factory():
    """Factory fixture to create test batches"""
    def _create_batch(batch_size=2, channels=2, shape=(32, 32, 32)):
        """Create a test batch of specified size"""
        data = torch.randn(batch_size, channels, *shape).cuda()
        labels = torch.randint(0, 2, (batch_size,)).cuda()
        return data, labels

    return _create_batch