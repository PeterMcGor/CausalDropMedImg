from pathlib import Path
from nnunet_datasets import MSSEG2016Config, MSSEG2016ToNNUnetConverter, SubjectSelection

def create_annotator_specific_dataset(dataset_path, nnunet_raw,
                                      dataset_name,
                                      train_annotator_id,
                                      config = MSSEG2016Config(["FLAIR"]),
                                      preprocess_data = True):
    """Create dataset with specific train annotator and remaining for test"""
    # Temporary converter to get all available subjects
    temp_converter = MSSEG2016ToNNUnetConverter(
        dataset_path=dataset_path,
        nnunetv2_raw=nnunet_raw,
        dataset_name="temp",
        config= config)

    all_subjects = temp_converter._get_all_available_subjects()

    # Prepare modified subjects configuration
    modified_subjects = []
    test_annotators = [i for i in range(1, 8)]

    for subj in all_subjects:
        if subj.original_split == "Training":
            # Training subject: use single annotator
            modified_subjects.append(SubjectSelection(
                center=subj.center,
                subject_id=subj.subject_id,
                split='train',
                original_split=subj.original_split,
                annotators=[train_annotator_id]
            ))
        else:
            # Test subject: use all other annotators
            modified_subjects.append(SubjectSelection(
                center=subj.center,
                subject_id=subj.subject_id,
                split='test',
                original_split=subj.original_split,
                annotators=test_annotators,
                combine_annotations=None  # Save individual annotations
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

    # Create 7 datasets, one for each annotator
    for dataset_idx, annotator_id in enumerate(range(1, 8), start=1):
        dataset_name = f"Dataset{dataset_idx:03d}_MSSEG_FLAIR_Annotator{annotator_id}"
        print(f"Creating {dataset_name}...")

        create_annotator_specific_dataset(
            dataset_path=mseg_root,
            nnunet_raw=nnunet_raw_root,
            dataset_name=dataset_name,
            train_annotator_id=annotator_id
        )

    # Create default dataset (consensus)
    print("Creating default consensus dataset...")
    MSSEG2016ToNNUnetConverter(
        dataset_path=mseg_root,
        nnunetv2_raw=nnunet_raw_root,
        dataset_name="Dataset008_MSSEG_FLAIR_Consensus",
        config=MSSEG2016Config(modalities=["FLAIR"])
    ).convert_dataset()