import copy
import random


from typing import Any, Callable, Dict, Union, List, Tuple
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset

class MergerNNUNetDataset(nnUNetDataset):
    """
    An extension of nnUNetDataset that allows merging multiple datasets.
    This class maintains the original dataset functionality while adding
    the ability to combine data from multiple sources.
    """
    DATASET_INFO = 'dataset_info'
    BATCH_IMAGES_KEY = 'data'
    BATCH_SEG_KEY = 'target'
    BATCH_PROPERTIES_KEY = 'properties'


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
            self.extend_dataset(self.additional_data, self.DATASET_INFO)
            """
            for case_id in self.dataset.keys():
                processed_data = {}
                for key, value in self.additional_data.items():
                    if callable(value):
                        processed_data[key] = value(case_id, super())
                    else:
                        processed_data[key] = value
                self.dataset[case_id][self.DATASET_INFO] = processed_data
            """

    def merge(self, dataset: nnUNetDataset, check_conflicting_cases = True) -> None:
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
        if check_conflicting_cases:
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

    def subset_by_pattern(self, pattern_callable: Callable[[str], bool]) -> 'MergerNNUNetDataset':
        """
        Create a subset of the current dataset by filtering cases using a callable function.

        Args:
            pattern_callable: A function that takes a case identifier and returns True if
                            the case should be included in the subset, False otherwise.

        Returns:
            A new MergerNNUNetDataset instance with only the cases that match the pattern
        """
        # Find all case identifiers that match the pattern
        matching_cases = [case_id for case_id in self.dataset.keys() if pattern_callable(case_id)]

        # Use the existing subset method to create the filtered dataset
        return self.subset(matching_cases)

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

    def stratified_split(self, split_ratio: float = 0.8, min_samples_per_group: int = 3,
                     groupby_func=None, seed: int = 42) -> Tuple['MergerNNUNetDataset', 'MergerNNUNetDataset']:
        """
        Split the dataset ensuring stratification based on a grouping factor extracted from keys.

        Args:
            split_ratio: Fraction of data to use for the first split (default 0.8)
            min_samples_per_group: Minimum number of samples per group in each split (if possible)
            groupby_func: Function that extracts group identifier from a key. If None, will try to
                        extract center ID from the format "*_Center_XX_*". Can be a callable function
                        that takes a key string and returns a group identifier.
            seed: Random seed for reproducibility (default 42)

        Returns:
            Tuple of (first_split, second_split) datasets
        """
        if seed is not None:
            random.seed(seed)

        # Default grouping function - tries to extract center ID
        def default_groupby(key):
            parts = key.split('_')
            for i, part in enumerate(parts):
                if part.lower() == "center" and i + 1 < len(parts):
                    # Return prefix + "Center" + center_id
                    return '_'.join(parts[:i+2])
            return "unknown"  # Fallback if no center is found

        # Use provided groupby function or default
        extract_group = groupby_func if groupby_func is not None else default_groupby

        # Group keys by the extracted group identifier
        group_to_keys = {}
        for key in self.dataset.keys():
            group = extract_group(key)
            if group not in group_to_keys:
                group_to_keys[group] = []
            group_to_keys[group].append(key)

        first_split_keys = []
        second_split_keys = []

        # Process each group
        for group, keys in group_to_keys.items():
            # Shuffle keys within each group
            random.shuffle(keys)

            n_samples = len(keys)
            n_first_split = int(n_samples * split_ratio)

            # Try to ensure minimum samples per group in both splits if possible
            if n_samples >= 2 * min_samples_per_group:
                # Enough samples to meet minimum for both splits
                if n_first_split < min_samples_per_group:
                    n_first_split = min_samples_per_group
                elif n_samples - n_first_split < min_samples_per_group:
                    n_first_split = n_samples - min_samples_per_group
            elif n_samples >= min_samples_per_group:
                # Can only ensure minimum for one split, prioritize based on split_ratio
                if split_ratio >= 0.5 and n_first_split < min_samples_per_group:
                    n_first_split = min(min_samples_per_group, n_samples)
                elif split_ratio < 0.5 and n_samples - n_first_split < min_samples_per_group:
                    n_first_split = max(0, n_samples - min_samples_per_group)

            # Add keys to respective splits
            first_split_keys.extend(keys[:n_first_split])
            second_split_keys.extend(keys[n_first_split:])

        # Print summary of the stratification
        #print(f"Stratified split summary:")
        #for group, keys in group_to_keys.items():
        #    group_keys_in_first = sum(1 for k in first_split_keys if k in keys)
        #    group_keys_in_second = sum(1 for k in second_split_keys if k in keys)
        #    print(f"  {group}: {len(keys)} total, {group_keys_in_first} in first split, {group_keys_in_second} in second split")

        return self.subset(first_split_keys), self.subset(second_split_keys)

    def merge_and_split(self, dataset_to_merge: nnUNetDataset,
                        split_ratio: Union[Tuple[float, float], float] = 0.8,  shuffle: bool = True,
                    seed: int = 42,
                        **merge_kwargs):

        """
        Create training and validation datasets by:
        1. Splitting each folder into train/val
        2. Merging the train portions and val portions separately
        Important: Even not checking for conflics when the dict updates with the new dataset just unique keys will be kept.

        returns:  Tuple[MergerNNUNetDataset, MergerNNUNetDataset]
        """
        assert self.additional_data.keys() == dataset_to_merge.additional_data.keys()
        # Split each dataset
        if isinstance(split_ratio, float):
            split_ratio = (split_ratio, split_ratio)

        train_train, train_val = self.random_split(split_ratio=split_ratio[0],  shuffle=shuffle, seed=seed)
        test_train, test_val = dataset_to_merge.random_split(split_ratio=split_ratio[1],  shuffle=shuffle, seed=seed)

        # Merge training sets
        train_train.merge(test_train,**merge_kwargs)
        train_val.merge(test_val,**merge_kwargs)

        return train_train, train_val

    def get_merged_datasets(self) -> List[nnUNetDataset]:
        """
        Get list of all merged datasets.

        Returns:
            List[nnUNetDataset]: List of merged datasets
        """
        return self._merged_datasets

    def extend_dataset(self, additional_data: Dict[str, Any], additional_data_key:str) -> None: # TODO for specific case_id
        for case_id in self.dataset.keys():
            processed_data = {}
            for key, value in additional_data.items():
                if callable(value):
                    processed_data[key] = value(case_id, super())
                else:
                    processed_data[key] = value
            self.dataset[case_id][additional_data_key] = processed_data

    def extract_per_case_id(self, key: str) -> Dict[str,Any]:
        return {case_id:self[case_id][key] for case_id in self.dataset.keys()}

    def transform_keys(self, key_transform_function):
        """
        Creates a new dictionary with transformed keys but same values.

        Args:
            original_dict: The dictionary to transform
            key_transform_function: A function that takes a key and returns a new key

        Returns:
            A new dictionary with transformed keys
        """
        self.dataset = {key_transform_function(key): value for key, value in self.dataset.items()}


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

