import pytest
import os
from pathlib import Path
import numpy as np
import nibabel as nib
from nnunet_utils.image_modality_utils import ImageModalityName
from nnunet_utils.nnunet_datasets import MSSEG2016Config, MSSEG2016ToNNUnetConverter, SubjectSelection

@pytest.fixture
def mock_msseg_data(tmp_path):
    """Create a mock MSSEG2016 dataset structure."""
    dataset_root = tmp_path / "MSSEG2016"

    # Create data for both Training and Testing splits
    for split in ["Training", "Testing"]:
        for center in ["01", "03"]:
            for patient in ["01", "02"]:
                # Create patient directory structure
                patient_path = dataset_root / split / f"Center_{center}" / f"Patient_{patient}"

                # Create directories
                (patient_path / "Raw_Data").mkdir(parents=True)
                (patient_path / "Preprocessed_Data").mkdir()
                (patient_path / "Masks").mkdir()

                # Create dummy image data (10x10x10)
                dummy_data = np.random.rand(10, 10, 10)
                affine = np.eye(4)

                # Save modality files
                for modality in ["FLAIR", "T1", "T2", "DP", "GADO"]:
                    # Raw data
                    nib.save(
                        nib.Nifti1Image(dummy_data, affine),
                        patient_path / "Raw_Data" / f"{modality}.nii.gz"
                    )
                    # Preprocessed data
                    nib.save(
                        nib.Nifti1Image(dummy_data, affine),
                        patient_path / "Preprocessed_Data" / f"{modality}_preprocessed.nii.gz"
                    )

                # Create individual annotator masks
                for i in range(1, 8):
                    nib.save(
                        nib.Nifti1Image((dummy_data > 0.5).astype(np.uint8), affine),
                        patient_path / "Masks" / f"ManualSegmentation_{i}.nii.gz"
                    )

                # Create consensus mask
                nib.save(
                    nib.Nifti1Image((dummy_data > 0.5).astype(np.uint8), affine),
                    patient_path / "Masks" / "Consensus.nii.gz"
                )

    return dataset_root

@pytest.fixture
def mock_nnunet_raw(tmp_path):
    """Create a mock nnUNetv2_raw directory."""
    return tmp_path / "nnUNetv2_raw"

def test_config_validation(mock_msseg_data, mock_nnunet_raw):
    """Test configuration validation."""
    # Test valid configurations
    valid_config = MSSEG2016Config(
        modalities=["FLAIR", "T1", "GADO"],
        use_preprocessed=True,
        subjects=[
            SubjectSelection(
                center="01",
                subject_id="Patient_01",
                split="train",
                original_split="Training",
                annotators="Consensus"
            )
        ]
    )
    assert valid_config is not None

    # Test invalid centers
    with pytest.raises(ValueError, match="Invalid centers"):
        MSSEG2016Config(centers=["99"])

    # Test invalid annotators
    with pytest.raises(ValueError, match="Invalid annotators"):
        MSSEG2016Config(
            subjects=[SubjectSelection(
                center="01",
                subject_id="Patient_01",
                split="train",
                original_split="Training",
                annotators=[0, 8]  # Invalid annotator numbers
            )]
        )
    # naming
    converter = MSSEG2016ToNNUnetConverter(
        dataset_path=mock_msseg_data,
        nnunetv2_raw=mock_nnunet_raw,
        dataset_name="Dataset001_MSSEG",
        config=valid_config
    )

    assert set([ImageModalityName.T1, ImageModalityName.FLAIR, ImageModalityName.T1GD]) == set(converter.channel_names.values())


def test_default_conversion(mock_msseg_data, mock_nnunet_raw):
    """Test default dataset conversion using consensus masks."""
    config = MSSEG2016Config()

    converter = MSSEG2016ToNNUnetConverter(
        dataset_path=mock_msseg_data,
        nnunetv2_raw=mock_nnunet_raw,
        dataset_name="Dataset001_MSSEG",
        config=config
    )

    converter.convert_dataset()

    # Check directory structure and files
    dataset_path = mock_nnunet_raw / "Dataset001_MSSEG"
    assert dataset_path.exists()
    assert (dataset_path / "imagesTr").exists()
    assert (dataset_path / "labelsTr").exists()

    # Check for correct number of modality files
    number_of_modalities = len(converter.channel_names)
    number_of_training_subjects = len(list((mock_msseg_data / "Training").rglob("Center_*/Patient_*")))
    number_of_test_subjects = len(list((mock_msseg_data / "Testing").rglob("Center_*/Patient_*")))
    training_images = len(list((dataset_path / "imagesTr").glob("*_000[0-4].nii.gz")))
    test_images = len(list((dataset_path / "imagesTs").glob("*_000[0-4].nii.gz")))
    training_labels = len(list((dataset_path / "labelsTr").glob("*.nii.gz")))
    test_labels = len(list((dataset_path / "labelsTs").glob("*.nii.gz")))
    assert number_of_modalities*number_of_training_subjects == training_images
    assert number_of_modalities*number_of_test_subjects == test_images
    assert number_of_training_subjects == training_labels
    assert number_of_test_subjects == test_labels

def test_mixed_annotator_selection(mock_msseg_data, mock_nnunet_raw):
    """Test conversion with different annotator selections per subject."""
    config = MSSEG2016Config(
        modalities=["FLAIR", "T1"],
        subjects=[
            # Subject with consensus mask
            SubjectSelection(
                center="01",
                subject_id="Patient_01",
                split="train",
                original_split="Training",
                annotators="Consensus"
            ),
            # Subject with specific annotators
            SubjectSelection(
                center="01",
                subject_id="Patient_02",
                split="train",
                original_split="Training",
                annotators=[1, 2, 3]
            ),
            # Test subject with different annotators
            SubjectSelection(
                center="01",
                subject_id="Patient_01",
                split="test",
                original_split="Testing",
                annotators=[4, 5, 6]
            )
        ]
    )

    converter = MSSEG2016ToNNUnetConverter(
        dataset_path=mock_msseg_data,
        nnunetv2_raw=mock_nnunet_raw,
        dataset_name="Dataset001_MSSEG",
        config=config
    )

    converter.convert_dataset()

    # Verify files
    dataset_path = mock_nnunet_raw / "Dataset001_MSSEG"

    # Check training files
    train_images = list((dataset_path / "imagesTr").glob("*_0000.nii.gz"))
    train_labels = list((dataset_path / "labelsTr").glob("*.nii.gz"))
    assert len(train_images) == 2  # Two training subjects
    assert len(train_labels) == 2

    # Check test files
    test_images = list((dataset_path / "imagesTs").glob("*_0000.nii.gz"))
    test_labels = list((dataset_path / "labelsTs").glob("*.nii.gz"))
    assert len(test_images) == 1  # One test subject
    assert len(test_labels) == 1

def test_multiple_modalities_and_annotators(mock_msseg_data, mock_nnunet_raw):
    """Test handling of multiple modalities with different annotator selections."""
    config = MSSEG2016Config(
        modalities=["FLAIR", "T1", "T2"],
        subjects=[
            SubjectSelection(
                center="01",
                subject_id="Patient_01",
                split="train",
                original_split="Training",
                annotators=[1, 2]
            )
        ]
    )

    converter = MSSEG2016ToNNUnetConverter(
        dataset_path=mock_msseg_data,
        nnunetv2_raw=mock_nnunet_raw,
        dataset_name="Dataset001_MSSEG",
        config=config
    )

    converter.convert_dataset()

    # Verify files
    dataset_path = mock_nnunet_raw / "Dataset001_MSSEG"

    # Check for all modalities
    for mod_idx in range(3):  # 3 modalities
        img_path = dataset_path / "imagesTr" / f"Training_Center_01_Patient_01_{mod_idx:04d}.nii.gz"
        assert img_path.exists(), f"Missing modality {mod_idx}"

def test_file_integrity(mock_msseg_data, mock_nnunet_raw):
    """Test integrity of converted files with different annotator selections."""
    config = MSSEG2016Config(
        modalities=["FLAIR", "T1"],
        subjects=[
            SubjectSelection(
                center="01",
                subject_id="Patient_01",
                split="train",
                original_split="Training",
                annotators=[1, 2, 3]
            )
        ]
    )

    converter = MSSEG2016ToNNUnetConverter(
        dataset_path=mock_msseg_data,
        nnunetv2_raw=mock_nnunet_raw,
        dataset_name="Dataset001_MSSEG",
        config=config
    )

    converter.convert_dataset()

    # Check image files
    dataset_path = mock_nnunet_raw / "Dataset001_MSSEG"
    subject_files = sorted(list((dataset_path / "imagesTr").glob("*_000[0-1].nii.gz")))

    # Verify data integrity
    for image_file in subject_files:
        image_data = nib.load(image_file).get_fdata()
        assert len(image_data.shape) == 3, f"Wrong dimensions in {image_file.name}"
        assert not np.any(np.isnan(image_data)), f"Found NaN values in {image_file.name}"

    # Check label file
    label_file = next((dataset_path / "labelsTr").glob("*.nii.gz"))
    label_data = nib.load(label_file).get_fdata()
    assert set(np.unique(label_data)) <= {0, 1}, "Non-binary values in labels"

def test_multiple_annotator_in_test_data(mock_msseg_data, mock_nnunet_raw):
    """Test handling of multiple annotators in test data."""
    config = MSSEG2016Config(
        modalities=["FLAIR", "T1"],
        subjects=[
            SubjectSelection(
                center="01",
                subject_id="Patient_01",
                split="test",
                original_split="Testing",
                annotators=[1, 4, 7],
                combine_annotations=None
            ),
            SubjectSelection(
                center="03",
                subject_id="Patient_01",
                split="test",
                original_split="Testing",
                annotators=[1, 4, 7],
                combine_annotations=None
            ),
            SubjectSelection(
                center="03",
                subject_id="Patient_02",
                split="train",
                original_split="Training"
            )
        ]
    )

    converter = MSSEG2016ToNNUnetConverter(
        dataset_path=mock_msseg_data,
        nnunetv2_raw=mock_nnunet_raw,
        dataset_name="Dataset001_MSSEG",
        config=config
    )

    converter.convert_dataset()

    # Verify files
    dataset_path = mock_nnunet_raw / "Dataset001_MSSEG"