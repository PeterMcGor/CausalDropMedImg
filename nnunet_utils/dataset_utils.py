import copy
import random


from typing import Any, Dict, Union, List, Tuple
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset

class MergerNNUNetDataset(nnUNetDataset):
    """
    An extension of nnUNetDataset that allows merging multiple datasets.
    This class maintains the original dataset functionality while adding
    the ability to combine data from multiple sources.
    """
    DATASET_INFO = 'dataset_info'

    def __init__(self, *args, additional_data: Dict[str, Union[str, int, float]] = None, **kwargs):
        """
        Initialize the merger dataset.

        Args:
            folder (str): Folder containing the dataset
            case_identifiers (List[str]): List of case identifiers
            num_images_properties_per_case (Dict[str, List[dict]]): Properties for each case
            additional_data (Dict[str, Union[str, int, float]], optional): Additional data to store with the dataset
        """
        super().__init__(*args, **kwargs)
        self.folder = args[0] if args else None
        self._merged_datasets: List[nnUNetDataset] = []
        self.additional_data = additional_data if additional_data is not None else {}
        """
        Initialize the merger dataset.

        Args:
            folder (str): Folder containing the dataset
            case_identifiers (List[str]): List of case identifiers
            num_images_properties_per_case (Dict[str, List[dict]]): Properties for each case
        """
        super().__init__(*args, **kwargs)
        self._merged_datasets: List[nnUNetDataset] = []
        # Add additional data to each case in the merged dataset

        if self.additional_data is not None:
            for case_id in self.dataset.keys():
                processed_data = {}
                for key, value in self.additional_data.items():
                    if callable(value):
                        processed_data[key] = value(case_id, super())
                    else:
                        processed_data[key] = value
                self.dataset[case_id][self.DATASET_INFO] = processed_data

    def merge(self, dataset: nnUNetDataset) -> None:
        """
        Merge another nnUNetDataset into this one.

        Args:
            dataset (nnUNetDataset): The dataset to merge

        Raises:
            TypeError: If the provided dataset is not an instance of nnUNetDataset
            ValueError: If there are conflicting case identifiers
        """
        if not isinstance(dataset, nnUNetDataset):
            raise TypeError(f'Dataset must be an instance of nnUNetDataset, got {type(dataset)}')

        # Check for conflicting case identifiers
        conflicting_cases = set(self.dataset.keys()) & set(dataset.dataset.keys())
        if conflicting_cases:
            raise ValueError(
                f'Found conflicting case identifiers in datasets: {conflicting_cases}. '
                'Cannot merge datasets with duplicate cases.'
            )

        # Store reference to merged dataset
        if len(self._merged_datasets) == 0:
            self._merged_datasets.append(self)
        self._merged_datasets.append(dataset)

        # Merge the datasets
        self.dataset.update(dataset.dataset) # case_identifiers this is essential
        #self.additional_data.update(dataset.additional_data)

    def load_case(self, key):
        data, seg, properties = super().load_case(key)
        if self.DATASET_INFO in self.dataset[key].keys():
            properties.update(self.dataset[key][self.DATASET_INFO])
        return data, seg, properties

    def subset(self, case_identifiers: List[str]) -> 'MergerNNUNetDataset':
        """
        Create a subset of the current dataset by filtering case identifiers,
        avoiding reinitialization through superclass and cleaning up unneeded cases.

        Args:
            case_identifiers: List of case IDs to keep in the subset

        Returns:
            A new MergerNNUNetDataset instance with only the specified cases
        """
        # Create a shallow copy of the current instance
        new_dataset = copy.deepcopy(self)

        # Filter dataset to only keep specified cases
        # First identify which cases to remove
        cases_to_remove = set(self.dataset.keys()) - set(case_identifiers)

        # Remove unwanted cases from the dataset dictionary
        for case_id in cases_to_remove:
            if case_id in new_dataset.dataset:
                del new_dataset.dataset[case_id]

        return new_dataset


    def random_split(self, split_ratio: float = 0.8, shuffle: bool = True, seed: int = 42) -> Tuple['MergerNNUNetDataset', 'MergerNNUNetDataset']:
        """
        Randomly split the dataset into two parts based on the split ratio.

        Args:
            split_ratio: Fraction of data to use for the first split (default 0.8)
            shuffle: Whether to shuffle the data before splitting (default True)
            seed: Random seed for reproducibility (default 42)

        Returns:
            Tuple of (first_split, second_split) datasets
        """

        if seed is not None:
            random.seed(seed)

        keys = list(self.dataset.keys())
        if shuffle:
            random.shuffle(keys)

        split_idx = int(len(keys) * split_ratio)
        train_keys = keys[:split_idx]
        val_keys = keys[split_idx:]

        return self.subset(train_keys), self.subset(val_keys)

    def merge_and_split(self, dataset_to_merge: nnUNetDataset,
                        split_ratio: Union[Tuple[float, float], float] = 0.8,
                        **kwargs):

        """
        Create training and validation datasets by:
        1. Splitting each folder into train/val
        2. Merging the train portions and val portions separately

        returns:  Tuple[MergerNNUNetDataset, MergerNNUNetDataset]
        """
        assert self.additional_data.keys() == dataset_to_merge.additional_data.keys()
        # Split each dataset
        if isinstance(split_ratio, float):
            split_ratio = (split_ratio, split_ratio)

        train_train, train_val = self.random_split(split_ratio=split_ratio[0], **kwargs)
        test_train, test_val = dataset_to_merge.random_split(split_ratio=split_ratio[1], **kwargs)
        print(f"From the training of nnUNet for training domain classifier: {len(train_train)}, for validation: {len(train_val)}")
        print(f"From the test of nnUNet for trainining domain classifier: {len(test_train)}, for validation {len(test_val)}")

        # Merge training sets
        train_train.merge(test_train)
        train_val.merge(test_val)

        return train_train, train_val

    def get_merged_datasets(self) -> List[nnUNetDataset]:
        """
        Get list of all merged datasets.

        Returns:
            List[nnUNetDataset]: List of merged datasets
        """
        return self._merged_datasets

    def __str__(self) -> str:
        """
        String representation of the merged dataset.

        Returns:
            str: Description of the merged dataset
        """
        return (
            f'MergerNNUNetDataset with {len(self)} total cases, '
            f'merged from {len(self._merged_datasets) + 1} datasets'
        )
