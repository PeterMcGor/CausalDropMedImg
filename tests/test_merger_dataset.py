import pytest
import os
import shutil
import numpy as np
import pickle
from pathlib import Path
from typing import List
from nnunet_utils.dataset_utils import MergerNNUNetDataset

class TestDatasetCreator:
    """Helper class to create mock nnUNet dataset structure"""
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.plans_name = "nnUNetPlans_3d_fullres"
        print(f"Creating test dataset at: {self.base_path}")

    def create_dataset_structure(self, case_identifiers: List[str]):
        """Creates a mock dataset structure with necessary files"""
        plans_dir = self.base_path / self.plans_name
        plans_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created plans directory at: {plans_dir}")

        # Create dummy files for each case
        for case_id in case_identifiers:
            # Create dummy data array
            data = np.random.rand(1, 16, 16, 16)
            seg = np.random.randint(0, 2, size=(1, 16, 16, 16))

            # Save data as npz
            npz_path = plans_dir / f"{case_id}.npz"
            np.savez_compressed(npz_path, data=data)
            print(f"Created npz file: {npz_path}")

            # Save segmentation as npy
            seg_path = plans_dir / f"{case_id}_seg.npy"
            np.save(seg_path, seg)
            print(f"Created segmentation file: {seg_path}")

            # Create pickle file with properties
            properties = {
                'spacing': [1.0, 1.0, 1.0],
                'original_size': [16, 16, 16],
                'size_after_cropping': [16, 16, 16],
                'modality': '0',
                'shape': data.shape[1:],
                'case_identifier': case_id
            }

            pkl_path = plans_dir / f"{case_id}.pkl"
            with open(pkl_path, 'wb') as f:
                pickle.dump(properties, f)
            print(f"Created properties file: {pkl_path}")

        # Print all created files
        print("\nCreated files:")
        for file in plans_dir.glob("*"):
            print(f"- {file.name}")

    def cleanup(self):
        """Removes the test dataset directory"""
        if self.base_path.exists():
            print(f"Cleaning up directory: {self.base_path}")
            shutil.rmtree(self.base_path)

@pytest.fixture(scope="session")
def test_base_path(tmp_path_factory):
    """Creates a temporary directory for test datasets"""
    path = tmp_path_factory.mktemp("nnunet_test_data")
    print(f"\nTest base path: {path}")
    return path

@pytest.fixture
def dataset_creator(test_base_path):
    """Creates and manages test dataset structure"""
    creator = TestDatasetCreator(test_base_path)
    yield creator
    # Leaving cleanup commented out for inspection
    # creator.cleanup()

@pytest.fixture
def base_dataset(dataset_creator):
    """Creates a base dataset with two cases"""
    case_ids = ["case_1", "case_2"]
    dataset_creator.create_dataset_structure(case_ids)
    dataset_path = str(dataset_creator.base_path / dataset_creator.plans_name)

    return MergerNNUNetDataset(
        folder=dataset_path,
        additional_data={'test_data': 1}
    )

@pytest.fixture
def additional_dataset(test_base_path):
    """Creates an additional dataset for merging tests"""
    creator = TestDatasetCreator(test_base_path / "additional")
    case_ids = ["case_3", "case_4"]
    creator.create_dataset_structure(case_ids)

    dataset = MergerNNUNetDataset(
        folder=str(creator.base_path / creator.plans_name),
        additional_data={'test_data': 2}
    )
    yield dataset
    creator.cleanup()

def test_initialization(base_dataset):
    """Test proper initialization of MergerNNUNetDataset"""
    assert isinstance(base_dataset, MergerNNUNetDataset)
    assert len(base_dataset.dataset) == 2

    # Check dataset properties
    for case_id in ["case_1", "case_2"]:
        assert case_id in base_dataset.dataset
        case_info = base_dataset.dataset[case_id].get(base_dataset.DATASET_INFO, {})
        assert case_info.get('test_data') == 1

def test_merge_datasets(base_dataset, additional_dataset):
    """Test merging functionality"""
    initial_length = len(base_dataset)
    base_dataset.merge(additional_dataset)

    # Check proper merging
    assert len(base_dataset) == initial_length + len(additional_dataset.dataset)
    for case_id in ["case_3", "case_4"]:
        assert case_id in base_dataset.dataset

    # Check merged datasets list
    assert len(base_dataset.get_merged_datasets()) == 1
    assert base_dataset.get_merged_datasets()[0] == additional_dataset

def test_merge_conflict(base_dataset, dataset_creator):
    """Test merging datasets with conflicting case identifiers"""
    # Create dataset with conflicting case ID
    creator = TestDatasetCreator(dataset_creator.base_path / "conflict")
    creator.create_dataset_structure(["case_1"])  # Conflicts with base_dataset

    conflict_dataset = MergerNNUNetDataset(
        folder=str(creator.base_path / creator.plans_name),
        additional_data={'test_data': 3}
    )

    with pytest.raises(ValueError, match="Found conflicting case identifiers"):
        base_dataset.merge(conflict_dataset)

def test_subset(base_dataset):
    """Test creating subset of the dataset"""
    subset_cases = ["case_1"]
    subset = base_dataset.subset(subset_cases)

    assert len(subset) == 1
    assert "case_1" in subset.dataset
    assert "case_2" not in subset.dataset

    # Check if additional data is preserved
    case_info = subset.dataset["case_1"].get(subset.DATASET_INFO, {})
    assert case_info.get('test_data') == 1
    assert len(subset) == len(subset_cases)

def test_random_split(base_dataset):
    """Test random splitting of the dataset"""
    train_set, val_set = base_dataset.random_split(split_ratio=0.5, shuffle=False)

    assert len(train_set) == 1
    assert len(val_set) == 1
    assert len(train_set) + len(val_set) == len(base_dataset)

    # Check mutual exclusivity
    train_cases = set(train_set.dataset.keys())
    val_cases = set(val_set.dataset.keys())
    assert not train_cases.intersection(val_cases)

def test_load_case(base_dataset):
    """Test loading case with additional data"""
    data, seg, properties = base_dataset.load_case("case_1")

    assert isinstance(data, np.ndarray)
    assert isinstance(seg, (np.ndarray, type(None)))
    assert isinstance(properties, dict)
    assert properties.get('test_data') == 1

def test_string_representation(base_dataset, additional_dataset):
    """Test string representation of the dataset"""
    base_dataset.merge(additional_dataset)
    str_rep = str(base_dataset)

    assert 'MergerNNUNetDataset' in str_rep
    assert str(len(base_dataset)) in str_rep
    assert str(len(base_dataset.get_merged_datasets()) + 1) in str_rep

def test_invalid_merge_type(base_dataset):
    """Test merging with invalid dataset type"""
    invalid_dataset = {"case_5": {}}
    with pytest.raises(TypeError, match="Dataset must be an instance of nnUNetDataset"):
        base_dataset.merge(invalid_dataset)