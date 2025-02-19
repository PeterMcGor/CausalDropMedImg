from abc import ABC, abstractmethod
import warnings
import numpy as np
import shutil
from pathlib import Path
from typing import List, Union, Dict, Optional, Tuple
from batchgenerators.utilities.file_and_folder_operations import save_json, join

import nibabel as nib
from dataclasses import dataclass, field
from  nnunet_utils.image_modality_utils import StandardImageModalities

class BaseNNUnetV2Converter(ABC):
    """Abstract base class for converting datasets to nnUNetv2 format."""

    def __init__(
        self,
        dataset_path: Union[str, Path],
        nnunetv2_raw: Union[str, Path],
        dataset_name: str,
        channel_names: Dict[int, str],
        labels: Dict[str, Union[int, Tuple[int, ...]]],
        regions_class_order: Optional[Tuple[int, ...]] = None,
        reference: Optional[str] = None,
        release: Optional[str] = None,
        license: Optional[str] = None,
        description: Optional[str] = None,
        overwrite_image_reader_writer: Optional[str] = None,
    ):
        """
        Initialize the base converter.

        Args:
            dataset_path: Path to source dataset
            nnunetv2_raw: Path to nnUNetv2_raw folder
            dataset_name: Name of the dataset (e.g., "Dataset001_BrainTumor")
            channel_names: Dictionary mapping channel indices to names
            labels: Dictionary mapping label names to their indices or tuples for regions
            regions_class_order: Required for region-based training, defines order of classes
            reference: Reference to the dataset source
            release: Version or release information
            license: Dataset license information
            description: Dataset description
            overwrite_image_reader_writer: Custom image reader/writer class name
        """
        self.dataset_path = Path(dataset_path)
        self.nnunetv2_raw = Path(nnunetv2_raw)
        self.dataset_name = dataset_name
        self.channel_names = channel_names
        self.labels = labels
        self.regions_class_order = regions_class_order
        self.reference = reference
        self.release = release
        self.license = license
        self.description = description
        self.overwrite_image_reader_writer = overwrite_image_reader_writer

        # Validate region-based configuration
        self._validate_region_config()

        # Create nnUNetv2 directory structure
        self.dataset_root = self.nnunetv2_raw / self.dataset_name
        self.imagestr = self.dataset_root / "imagesTr"
        self.labelstr = self.dataset_root / "labelsTr"
        self.imagests = self.dataset_root / "imagesTs"

        self._create_directories()

    def _validate_region_config(self) -> None:
        """Validate region-based training configuration."""
        has_regions = any([isinstance(i, (tuple, list)) and len(i) > 1
                          for i in self.labels.values()])
        if has_regions and self.regions_class_order is None:
            raise ValueError("You have defined regions but regions_class_order is not set. "
                           "This is required for region-based training.")

    def _create_directories(self) -> None:
        """Create the required nnUNetv2 directory structure."""
        for path in [self.imagestr, self.labelstr, self.imagests]:
            path.mkdir(parents=True, exist_ok=True)


    def _combine_annotations(
        self,
        annotation_paths: List[Path],
        method: str = "majority_voting"
    ) -> np.ndarray:
        """Base implementation for combining multiple annotations."""
        if method == "majority_voting":
            return self._majority_voting(annotation_paths)
        raise NotImplementedError(f"Method {method} not implemented")

    def _majority_voting(self, annotation_paths: List[Path]) -> np.ndarray:
        """Default majority voting implementation."""
        annotations = []
        for path in annotation_paths:
            mask = nib.load(path).get_fdata()
            annotations.append(mask)

        annotations = np.stack(annotations, axis=-1)
        combined = np.median(annotations, axis=-1) > 0.5
        return combined.astype(np.uint8)

    @abstractmethod
    def process_subject(
        self,
        subject_path: Path,
        output_identifier: str,
        is_training: bool = True
    ) -> None:
        """
        Process a single subject.

        Args:
            subject_path: Path to the subject's data
            output_identifier: Identifier for output files
            is_training: Whether this is training data
        """
        pass

    def create_dataset_json(self, file_ending: str = ".nii.gz", **kwargs) -> None:
        """
        Create dataset.json file as required by nnUNetv2.

        Args:
            file_ending: File extension for images and labels
            **kwargs: Additional parameters to include in dataset.json
        """
        # Convert channel names keys to strings
        channel_names = {str(k): v for k, v in self.channel_names.items()}

        # Ensure label values are integers or tuples of integers
        labels = {}
        for k, v in self.labels.items():
            if isinstance(v, (tuple, list)):
                labels[k] = tuple(int(i) for i in v)
            else:
                labels[k] = int(v)

        dataset_json = {
            'channel_names': channel_names,
            'labels': labels,
            'numTraining': len(list(self.imagestr.glob(f"*{file_ending}"))) // len(self.channel_names),
            'file_ending': file_ending,
        }

        # Add optional fields if they exist
        if self.dataset_name is not None:
            dataset_json['name'] = self.dataset_name
        if self.reference is not None:
            dataset_json['reference'] = self.reference
        if self.release is not None:
            dataset_json['release'] = self.release
        if self.license is not None:
            dataset_json['licence'] = self.license
        if self.description is not None:
            dataset_json['description'] = self.description
        if self.overwrite_image_reader_writer is not None:
            dataset_json['overwrite_image_reader_writer'] = self.overwrite_image_reader_writer
        if self.regions_class_order is not None:
            dataset_json['regions_class_order'] = self.regions_class_order

        # Add any additional parameters
        dataset_json.update(kwargs)

        # Save the dataset.json file
        save_json(dataset_json, join(str(self.dataset_root), 'dataset.json'), sort_keys=False)

    def verify_dataset_integrity(self) -> bool:
        """
        Verify that the dataset is complete and properly formatted.

        Returns:
            bool: True if dataset is valid, False otherwise
        """
        try:
            # Check directory structure
            for path in [self.imagestr, self.labelstr, self.imagests]:
                if not path.exists():
                    print(f"Missing directory: {path}")
                    return False

            # Check dataset.json
            if not (self.dataset_root / "dataset.json").exists():
                print("Missing dataset.json")
                return False

            # Verify training data
            num_channels = len(self.channel_names)
            training_cases = set()

            # Get all unique case IDs from the image files
            for img_path in self.imagestr.glob("*.nii.gz"):
                case_id = img_path.name.split("_")[0]
                training_cases.add(case_id)

            # Verify each training case
            for case_id in training_cases:
                # Check channel completeness
                channels_present = len(list(self.imagestr.glob(f"{case_id}_*.nii.gz")))
                if channels_present != num_channels:
                    print(f"Incomplete channels for case {case_id}: {channels_present}/{num_channels}")
                    return False

                # Check corresponding label
                label_path = self.labelstr / f"{case_id}.nii.gz"
                if not label_path.exists():
                    print(f"Missing label file for training case: {case_id}")
                    return False

            return True

        except Exception as e:
            print(f"Verification failed with error: {str(e)}")
            return False

    @abstractmethod
    def convert_dataset(self) -> None:
        """Convert the entire dataset to nnUNetv2 format."""
        pass



@dataclass
class SubjectSelection:
    """Configuration for subject selection from MSSEG2016 dataset."""
    center: str
    subject_id: str
    split: str  # 'train' or 'test'
    original_split: str  # 'Training' or 'Testing' - original location in MSSEG dataset
    annotators: Union[str, List[int]] = "Consensus"  # Can be "Consensus" or list of annotator numbers [1-7]
    combine_annotations: str = "majority_voting"  # Method for combining multiple annotations

    def __post_init__(self):
        if self.split not in ['train', 'test']:
            raise ValueError("Invalid split. Must be 'train' or 'test'")
        if self.original_split not in ['Training', 'Testing']:
            raise ValueError("Invalid original_split. Must be 'Training' or 'Testing'")
        if self.annotators != "Consensus" and not isinstance(self.annotators, list):
            raise ValueError("Invalid annotators. Must be 'Consensus' or list of integers [1-7]")
        if self.split == 'train' and self.combine_annotations is None:
            raise ValueError("combine_annotations must be specified for training subjects")

@dataclass
class MSSEG2016Config:
    """Configuration for MSSEG2016 dataset conversion.

    Attributes:
        modalities: List of modalities to include (e.g., ["FLAIR", "T1", "T2", "DP"])
        use_preprocessed: Whether to use preprocessed data (True) or raw data (False)
        annotators: "Consensus" or list of annotator numbers [1-7]
        centers: Optional list of centers to include (e.g., ["01", "03"])
        subjects: Optional list of specific subjects with their split assignments
    """
    modalities: List[str] = field(default_factory=lambda: ["FLAIR", "T1", "T2", "DP", "GADO"])
    use_preprocessed: bool = True
    centers: Optional[List[str]] = None
    subjects: Optional[List[SubjectSelection]] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate the configuration."""
        # Validate centers if specified
        all_centers = {"01", "03", "07", "08"}
        if self.centers is not None:
            invalid_centers = set(self.centers) - all_centers
            if invalid_centers:
                raise ValueError(f"Invalid centers: {invalid_centers}. Valid centers are: {all_centers}")

        # Validate subject selections if specified
        if self.subjects is not None:
            for subject in self.subjects:
                if subject.center not in all_centers:
                    raise ValueError(f"Invalid center in subject selection: {subject.center}")
                if subject.split not in ['train', 'test']:
                    raise ValueError(f"Invalid split in subject selection: {subject.split}")
                if subject.original_split not in ['Training', 'Testing']:
                    raise ValueError(f"Invalid original_split in subject selection: {subject.original_split}")
                # Validate annotators for each subject
                if isinstance(subject.annotators, list):
                    invalid_annotators = [a for a in subject.annotators if a not in range(1, 8)]
                    if invalid_annotators:
                        raise ValueError(f"Invalid annotators for subject {subject.subject_id}: {invalid_annotators}")
                elif subject.annotators != "Consensus":
                    raise ValueError(f"Invalid annotators for subject {subject.subject_id}. Must be 'Consensus' or list of integers [1-7]")


class MSSEG2016ToNNUnetConverter(BaseNNUnetV2Converter):
    """Converter for MSSEG 2016 Challenge dataset to nnUNetv2 format."""

    def __init__(
        self,
        dataset_path: Union[str, Path],
        nnunetv2_raw: Union[str, Path],
        dataset_name: str,
        config: MSSEG2016Config
    ):
        """
        Initialize MSSEG2016 dataset converter.

        Args:
            dataset_path: Path to MSSEG dataset root
            nnunetv2_raw: Path to nnUNetv2_raw folder
            dataset_name: Name of the dataset (e.g., "Dataset001_MSSEG")
            config: Configuration for dataset conversion
        """
        self.config = config
        super().__init__(
            dataset_path=dataset_path,
            nnunetv2_raw=nnunetv2_raw,
            dataset_name=dataset_name,
            channel_names=StandardImageModalities.get_standard_channel_names(config.modalities),
            labels={"background": 0, "lesion": 1}
        )

    def _get_all_available_subjects(self, annotators: Union[str, List[int]] = "Consensus") -> List[SubjectSelection]:
        """Get list of all available subjects with their default splits."""
        subjects = []
        centers = self.config.centers or [f"{i:02d}" for i in [1, 3, 7, 8]]

        # Process both Training and Testing folders
        for split in ["Training", "Testing"]:
            for center in centers:
                center_path = self.dataset_path.joinpath(split, f"Center_{center}")
                if center_path.exists():
                    for patient_folder in center_path.glob("Patient_*"):
                        subjects.append(SubjectSelection(
                            center=center,
                            subject_id=patient_folder.name,
                            split='train' if split == "Training" else 'test',
                            original_split=split,
                            annotators=annotators
                        ))

        return subjects

    def _get_subject_path(self, subject: SubjectSelection) -> Path:
        """Get path to subject folder."""
        return self.dataset_path / subject.original_split / f"Center_{subject.center}" / subject.subject_id

    def _generate_unique_identifier(self, subject: SubjectSelection) -> str:
        """Generate unique identifier for a subject including original split information."""
        return f"{subject.original_split}_Center_{subject.center}_{subject.subject_id}"

    def _get_modality_path(self, subject_path: Path, modality: str) -> Path:
        """Get path to a specific modality file."""
        data_dir = "Preprocessed_Data" if self.config.use_preprocessed else "Raw_Data"
        suffix = "_preprocessed" if self.config.use_preprocessed else ""
        return subject_path / data_dir / f"{modality}{suffix}.nii.gz"

    def _get_label_path(self, subject_path: Path) -> Union[Path, List[Path]]:
        """Get path(s) to label file(s)."""
        if self.config.annotators == "Consensus":
            return subject_path / "Masks" / "Consensus.nii.gz"
        else:
            return [subject_path / "Masks" / f"ManualSegmentation_{i}.nii.gz"
                   for i in self.config.annotators]




    def process_subject(self, subject_path, output_identifier, subject, is_training=True):
        """Optimized subject processing with direct file copying"""
        # Process modalities - copy files directly
        for mod_idx, mod in enumerate(self.config.modalities):
            src = self._get_modality_path(subject_path, mod)
            dst = (self.imagestr if is_training else self.imagests) / f"{output_identifier}_{mod_idx:04d}.nii.gz"

            # Copy without loading if preprocessing matches requirements
            if src.exists():
                shutil.copyfile(src, dst)

        # Process labels
        try:
            if isinstance(subject.annotators, list):
                label_paths = [subject_path / "Masks" / f"ManualSegmentation_{i}.nii.gz"
                            for i in subject.annotators]

                if subject.combine_annotations is not None:
                    # Only load data when combining masks
                    combined_data = self._combine_annotations(label_paths, subject.combine_annotations)
                    img_template = nib.load(label_paths[0])  # Get affine from first file
                    output_path = (self.labelstr if is_training else
                                self.dataset_root / "labelsTs") / f"{output_identifier}.nii.gz"
                    nib.save(nib.Nifti1Image(combined_data, img_template.affine), output_path)
                else:
                    # Copy individual annotations for test set
                    for i, path in enumerate(label_paths):
                        if path.exists():
                            annotator_id = subject.annotators[i]
                            dst_folder = self.dataset_root / f"labelsTs_{annotator_id}"
                            dst_folder.mkdir(exist_ok=True)
                            shutil.copyfile(path, dst_folder / f"{output_identifier}.nii.gz")
            else:  # Consensus mask
                src = subject_path / "Masks" / "Consensus.nii.gz"
                dst_folder = (self.labelstr if is_training else
                    self.dataset_root / "labelsTs")
                dst = dst_folder / f"{output_identifier}.nii.gz"
                dst_folder.mkdir(exist_ok=True)
                if src.exists():
                    shutil.copyfile(src, dst)

        except Exception as e:
            warnings.warn(f"Could not process labels for {output_identifier}: {str(e)}")

    def convert_dataset(self) -> None:
        """Convert the MSSEG dataset according to configuration."""
        # Get subjects to process
        subjects = self.config.subjects or self._get_all_available_subjects()

        # Process each subject
        for subject in subjects:
            subject_path = self._get_subject_path(subject)
            if not subject_path.exists():
                raise ValueError(f"Subject path does not exist: {subject_path}")

            # Generate unique identifier
            output_identifier = self._generate_unique_identifier(subject)

            # Process subject
            self.process_subject(
                subject_path=subject_path,
                output_identifier=output_identifier,
                subject=subject,  # Pass the subject object
                is_training=(subject.split == 'train')
            )

        # Create dataset.json
        self.create_dataset_json()

    def _generate_dataset_description(self, subjects: List[SubjectSelection]) -> str:
        """Generate a detailed description of the dataset configuration.

        This includes:
        - Original dataset information
        - Selected subjects and their distribution
        - Modalities used
        - Annotation method
        """
        # Count subjects per split and center
        train_subjects = [s for s in subjects if s.split == 'train']
        test_subjects = [s for s in subjects if s.split == 'test']

        # Count original vs new split
        original_train = [s for s in subjects if s.original_split == 'Training']
        original_test = [s for s in subjects if s.original_split == 'Testing']

        # Group by centers
        centers_used = set(s.center for s in subjects)
        center_counts = {center: len([s for s in subjects if s.center == center])
                        for center in centers_used}

        # Get unique annotator configurations
        annotator_configs = set(str(s.annotators) for s in subjects)

        description = [
            "Dataset created from MSSEG 2016 Challenge data.",
            "",
            f"Original dataset distribution:",
            f"- Training set: {len(original_train)} subjects",
            f"- Testing set: {len(original_test)} subjects",
            "",
            f"Current nnUNet dataset distribution:",
            f"- Training set: {len(train_subjects)} subjects",
            f"- Test set: {len(test_subjects)} subjects",
            "",
            f"Centers included: {', '.join(sorted(centers_used))}",
            "Center distribution:",
        ]

        # Add center-wise counts
        for center, count in sorted(center_counts.items()):
            description.append(f"- Center_{center}: {count} subjects")

        description.extend([
            "",
            f"Modalities used: {', '.join(self.config.modalities)}",
            "",
            "Annotation information:",
        ])

        # Add annotation configurations
        if len(annotator_configs) == 1:
            ann_config = next(iter(annotator_configs))
            if ann_config == '"Consensus"':
                description.append("- Using consensus masks for all subjects")
            else:
                description.append(f"- Using combination of annotators {ann_config} for all subjects")
        else:
            description.append("Mixed annotation sources:")
            for s in subjects:
                if s.annotators == "Consensus":
                    ann_desc = "consensus mask"
                else:
                    ann_desc = f"combined annotations from raters {s.annotators}"
                description.append(f"- {s.original_split}_Center_{s.center}_{s.subject_id}: {ann_desc}")

        # Join with newlines
        return "\n".join(description)

    def create_dataset_json(self) -> None:
        """Create dataset.json file as required by nnUNetv2."""
        subjects = self.config.subjects or self._get_all_available_subjects()
        description = self._generate_dataset_description(subjects)

        super().create_dataset_json(
            dataset_description=description,
            reference="MSSEG 2016 Challenge",
            license="See MSSEG 2016 Challenge terms",
            release="1.0"
        )