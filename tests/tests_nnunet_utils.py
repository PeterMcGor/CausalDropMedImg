import os
import pytest
import numpy as np
from pathlib import Path

@pytest.fixture
def sample_data():
    """Fixture to provide sample data for testing."""
    return np.random.rand(4, 4, 4)  # Example 3D data

@pytest.mark.gpu  # Mark tests that need GPU
def test_preprocessing_pipeline(sample_data):
    """Test the preprocessing pipeline with sample data."""
    # Your test code here
    processed_data = preprocess_function(sample_data)
    assert processed_data.shape == sample_data.shape
    assert not np.array_equal(processed_data, sample_data)

@pytest.mark.slow  # Mark slow tests
def test_full_inference():
    """Test the full inference pipeline."""
    # Your test code here
    pass

def test_data_loading():
    """Test that data paths are correctly configured."""
    raw_data_path = Path(os.environ['nnUNet_raw'])
    assert raw_data_path.exists(), "nnUNet raw data path not found"