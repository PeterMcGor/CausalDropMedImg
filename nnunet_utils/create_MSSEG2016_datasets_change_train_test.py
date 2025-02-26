from pathlib import Path
from itertools import combinations
from nnunet_utils.nnunet_derived_datasets import MSSEG2016Config, MSSEG2016ToNNUnetConverter, SubjectSelection

def create_flipped_dataset_single_annotator(dataset_path, nnunet_raw,
                                         dataset_name,
                                         annotator_id,
                                         config=MSSEG2016Config(["FLAIR"]),
                                         preprocess_data=True):
    """Create dataset with flipped train/test splits using single annotator labels

    Args:
        dataset_path: Path to MSSEG2016 dataset
        nnunet_raw: Path to nnUNet raw data directory
        dataset_name: Name for the new dataset
        annotator_id: ID of the annotator to use for labels (1-7)
        config: Dataset configuration
        preprocess_data: Whether to use preprocessed data
    """
    # Temporary converter to get all available subjects
    temp_converter = MSSEG2016ToNNUnetConverter(
        dataset_path=dataset_path,
        nnunetv2_raw=nnunet_raw,
        dataset_name="temp",
        config=config
    )

    all_subjects = temp_converter._get_all_available_subjects()

    # Prepare modified subjects configuration
    modified_subjects = []
    test_annotators = [i for i in range(1, 8)]

    for subj in all_subjects:
        # Center 03 always stays in test set
        if subj.center == "03":
            modified_subjects.append(SubjectSelection(
                center=subj.center,
                subject_id=subj.subject_id,
                split='test',
                original_split=subj.original_split,
                annotators=test_annotators,
                 combine_annotations=None
            ))
        # For other centers, flip the training/test assignments
        elif subj.original_split == "Training":
            # Original training becomes test
            modified_subjects.append(SubjectSelection(
                center=subj.center,
                subject_id=subj.subject_id,
                split='test',
                original_split=subj.original_split,
                annotators=[annotator_id],
                combine_annotations=None
            ))
        else:
            # Original test becomes training (except center 03)
            modified_subjects.append(SubjectSelection(
                center=subj.center,
                subject_id=subj.subject_id,
                split='train',
                original_split=subj.original_split,
                annotators=test_annotators
            ))

    # Create and run converter with modified config
    converter = MSSEG2016ToNNUnetConverter(
        dataset_path=dataset_path,
        nnunetv2_raw=nnunet_raw,
        dataset_name=dataset_name,
        config=MSSEG2016Config(
            subjects=modified_subjects,
            modalities=config.modalities,
            use_preprocessed=preprocess_data
        )
    )
    converter.convert_dataset()


if __name__ == "__main__":
    mseg_root = Path("/home/jovyan/datasets/MSSEG/MSSEG2016")  # Update this path
    nnunet_raw_root = Path("/home/jovyan/nnunet_data/nnUNet_raw")

    # Create 7 datasets, one for each annotator with flipped train/test assignments
    for dataset_idx, annotator_id in enumerate(range(1, 8), start=1):
        dataset_number = dataset_idx + 10
        dataset_name = f"Dataset{dataset_number:03d}_MSSEG_FLAIR_FlippedSplit_Annotator{annotator_id}"
        print(f"Creating {dataset_name}...")

        create_flipped_dataset_single_annotator(
            dataset_path=mseg_root,
            nnunet_raw=nnunet_raw_root,
            dataset_name=dataset_name,
            annotator_id=annotator_id
        )

    # Print summary of what was created
    print("\nDataset creation complete!")
    print("Created 7 datasets with the following properties:")
    print("- Each dataset uses labels from a single annotator (1-7)")
    print("- Original training centers are now used for testing")
    print("- Original test centers are now used for training")
    print("- Center 03 remains in the test set regardless of original assignment")