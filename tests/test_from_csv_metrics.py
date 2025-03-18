import pytest
import pandas as pd
import numpy as np

from attributions.core.metrics.metrics import compute_weighted_metrics, compute_weighted_metrics_merged_dataset

# Import the functions to test
# from your_module import compute_weighted_metrics, compute_weighted_metrics_merged_dataset

@pytest.fixture
def mock_dataframe():
    """Create a mock DataFrame for testing"""
    mock_data = {
        "dataset": ["test"] * 15,
        "id": [
            "P3_T3.nii",
            "P5_T2.nii",
            "P49_T2.nii",
            "P3_T1.nii",
            "P35_T1.nii",
            "P49_T1.nii",
            "P15_T1.nii",
            "P31_T2.nii",
            "P5_T1.nii",
            "P31_T1.nii",
            "P3_T4.nii",
            "P3_T2.nii",
            "P7_T1.nii",
            "P8_T1.nii",
            "P9_T1.nii",
        ],
        "ref": [599] * 15,
        "labelsTs": ["labelsTs"] * 15,
        "Jaccard": [
            0.534574,
            0.559676,
            0.389657,
            0.566943,
            0.633868,
            0.731755,
            0.604536,
            0.581162,
            0.218918,
            0.589146,
            0.444399,
            0.550077,
            np.nan,
            0.612345,
            0.723456,
        ],
        "Dice": [
            0.696707,
            0.717682,
            0.560796,
            0.723629,
            0.775911,
            0.845102,
            0.753534,
            0.735107,
            0.359200,
            0.741462,
            0.615341,
            0.709742,
            0.680123,
            np.nan,
            0.810234,
        ],
        "Sensitivity": [
            0.590308,
            0.721627,
            0.402808,
            0.642497,
            0.702926,
            0.756000,
            0.844957,
            0.596710,
            0.414024,
            0.665036,
            0.569393,
            0.680432,
            0.712345,
            0.653421,
            np.nan,
        ],
        "Specificity": [
            0.999970,
            0.999854,
            0.999964,
            0.999945,
            0.999821,
            0.999840,
            0.999904,
            0.999976,
            0.999502,
            0.999923,
            0.999926,
            0.999948,
            np.nan,
            0.999876,
            0.999912,
        ],
        "PPV": [
            0.849894,
            0.713781,
            0.922691,
            0.828213,
            0.865808,
            0.958014,
            0.679963,
            0.957090,
            0.317198,
            0.837735,
            0.669355,
            0.741690,
            0.801234,
            0.732156,
            np.nan,
        ],
        "F1_score": [
            0.761194,
            0.736196,
            0.645598,
            0.799154,
            0.762431,
            0.686747,
            1.000000,
            0.799695,
            0.534979,
            0.737705,
            0.740741,
            0.767932,
            np.nan,
            0.720123,
            0.812345,
        ],
    }
    return pd.DataFrame(mock_data)

@pytest.fixture
def weights_dict():
    """Create a weights dictionary for testing"""
    return {
        "P3_T3.nii": 1.5,
        "P3_T1.nii": 1.5,
        "P3_T4.nii": 1.5,
        "P3_T2.nii": 1.5,
        "P5_T2.nii": 0.8,
        "P5_T1.nii": 0.8,
        "P49_T2.nii": 1.2,
        "P49_T1.nii": 1.2,
        "P31_T2.nii": 1.0,
        "P31_T1.nii": 1.0,
        "P15_T1.nii": 0.7,
        "P35_T1.nii": 0.9,
        "P7_T1.nii": 1.1,
        "P8_T1.nii": 0.6,
        "P9_T1.nii": 0.5,
    }

@pytest.fixture
def merged_dataset():
    """Create a merged dataset dictionary for testing"""
    return {
        "P3_T3": "path/to/file1",
        "P5_T2": "path/to/file2",
        "P49_T2": "path/to/file3",
        "P3_T1": "path/to/file4",
        "P35_T1": "path/to/file5",
        "P49_T1": "path/to/file6",
    }

class TestComputeWeightedMetrics:
    """Tests for the compute_weighted_metrics function"""

    def test_basic_weighted_metrics(self, mock_dataframe, weights_dict):
        """Test basic functionality with valid weights"""
        results = compute_weighted_metrics(mock_dataframe, case_col="id", weights=weights_dict)

        # Check that we got results for all 6 metrics
        assert len(results) == 6
        assert "Dice" in results
        assert "Jaccard" in results

    def test_specific_measures(self, mock_dataframe, weights_dict):
        """Test with specific measures selected"""
        selected_measures = ["Dice", "Jaccard", "F1_score"]
        results = compute_weighted_metrics(
            mock_dataframe,
            case_col="id",
            weights=weights_dict,
            measures=selected_measures
        )

        # Check that only requested measures are returned
        assert len(results) == 3
        assert set(results.keys()) == set(selected_measures)

    def test_subset_subjects(self, mock_dataframe, weights_dict):
        """Test with a subset of subjects"""
        subset = [
            "P3_T3.nii",
            "P3_T1.nii",
            "P3_T4.nii",
            "P3_T2.nii",
            "P5_T2.nii",
            "P5_T1.nii",
        ]

        # Create a filtered weights dictionary with only the subset keys
        subset_weights = {k: weights_dict[k] for k in subset}

        # Method 1: Using subset_subjects parameter
        results_with_subset = compute_weighted_metrics(
            mock_dataframe,
            case_col="id",
            weights=subset_weights,
            subset_subjects=subset,
            measures=["Dice", "Jaccard"]
        )

        # Method 2: Pre-filtering the dataframe (alternate approach)
        filtered_df = mock_dataframe[mock_dataframe["id"].isin(subset)]
        results_with_filtered_df = compute_weighted_metrics(
            filtered_df,
            case_col="id",
            weights=subset_weights,
            measures=["Dice", "Jaccard"]
        )

        # Verify both methods produce the same results
        assert results_with_subset == results_with_filtered_df

        # Verify we get the expected number of metrics
        assert len(results_with_subset) == 2
        assert "Dice" in results_with_subset
        assert "Jaccard" in results_with_subset

        # Verify we get different results with the subset vs. full dataset
        full_results = compute_weighted_metrics(
            mock_dataframe,
            case_col="id",
            weights=weights_dict,
            measures=["Dice", "Jaccard"]
        )
        # Results should be different when using different subsets
        assert results_with_subset != full_results

    def test_missing_data_handling(self, mock_dataframe, weights_dict):
        """Test handling of missing data"""
        # Add missing data
        df = mock_dataframe.copy()
        df["CompletelyMissing"] = np.nan
        df.loc[0:3, "Jaccard"] = np.nan

        results = compute_weighted_metrics(
            df,
            case_col="id",
            weights=weights_dict,
            measures=["Jaccard", "CompletelyMissing", "Dice"]
        )

        # Check handling of partially and completely missing data
        assert not pd.isna(results["Jaccard"]), "Jaccard should have a value despite some NaNs"
        assert pd.isna(results["CompletelyMissing"]), "CompletelyMissing should be NaN"

    def test_return_raw_values(self, mock_dataframe):
        """Test returning raw values instead of averages"""
        raw_values = compute_weighted_metrics(
            mock_dataframe,
            case_col="id",
            measures=["Dice", "Jaccard"],
            return_average=False
        )

        # Verify raw values were returned
        assert isinstance(raw_values, np.ndarray)
        assert raw_values.shape[0] == len(mock_dataframe)  # Two columns for Dice and Jaccard

    def test_invalid_weights_keys(self, mock_dataframe, weights_dict):
        """Test validation of weights keys against data"""
        invalid_weights = weights_dict.copy()
        invalid_weights["NonExistent1.nii"] = 2.0
        invalid_weights["NonExistent2.nii"] = 1.0

        with pytest.raises(ValueError) as excinfo:
            compute_weighted_metrics(mock_dataframe, case_col="id", weights=invalid_weights)

        # Verify both missing keys are mentioned in the error message
        error_msg = str(excinfo.value)
        assert "NonExistent1.nii" in error_msg
        assert "NonExistent2.nii" in error_msg

    def test_duplicate_keys_in_data(self, mock_dataframe, weights_dict):
        """Test validation of duplicate keys in the data"""
        # Create DataFrame with duplicate ID
        df_with_duplicates = mock_dataframe.copy()
        duplicate_row = mock_dataframe.iloc[0].copy()  # Duplicate P3_T3.nii
        df_with_duplicates = pd.concat([df_with_duplicates, pd.DataFrame([duplicate_row])], ignore_index=True)

        with pytest.raises(ValueError) as excinfo:
            compute_weighted_metrics(df_with_duplicates, case_col="id", weights=weights_dict)

        # Verify error message mentions the duplicated key
        assert "P3_T3.nii" in str(excinfo.value)
        assert "appear multiple times" in str(excinfo.value)

    def test_equal_weights_when_none_provided(self, mock_dataframe):
        """Test that equal weights are used when none are provided"""
        # With explicit None
        results_with_none = compute_weighted_metrics(
            mock_dataframe,
            case_col="id",
            weights=None,
            measures=["Dice"]
        )

        # Without specifying weights (default=None)
        results_default = compute_weighted_metrics(
            mock_dataframe,
            case_col="id",
            measures=["Dice"]
        )

        # Results should be the same
        assert results_with_none == results_default


class TestComputeWeightedMetricsMergedDataset:
    """Tests for the compute_weighted_metrics_merged_dataset function"""

    def test_with_merged_dataset_and_list_weights(self, mock_dataframe, merged_dataset):
        """Test using merged_dataset with list weights"""
        # Define weights as list
        weights_list = [1.5, 0.8, 1.2, 1.5, 0.9, 1.2]

        results = compute_weighted_metrics_merged_dataset(
            mock_dataframe,
            merged_dataset=merged_dataset,
            weights=weights_list,
            measures=["Dice", "Jaccard"]
        )

        assert len(results) == 2
        assert isinstance(results["Dice"], float)
        assert isinstance(results["Jaccard"], float)

    def test_with_custom_key_transform(self, mock_dataframe, merged_dataset):
        """Test using a custom key_transform function"""
        weights_list = [1.5, 0.8, 1.2, 1.5, 0.9, 1.2]

        # Define custom transform
        def custom_transform(key):
            return key + ".nii"

        results = compute_weighted_metrics_merged_dataset(
            mock_dataframe,
            merged_dataset=merged_dataset,
            weights=weights_list,
            key_transform=custom_transform,
            measures=["Dice", "Jaccard"]
        )

        assert len(results) == 2

    def test_with_dict_weights_bypass_merged_dataset(self, mock_dataframe, merged_dataset, weights_dict):
        """Test bypassing merged_dataset when dict weights are provided"""
        # Define a no-op transform that doesn't modify the keys
        identity_transform = lambda k: k

        # Generate results with merged_dataset present
        results_with_merged = compute_weighted_metrics_merged_dataset(
            mock_dataframe,
            merged_dataset=merged_dataset,  # Should be ignored
            weights=weights_dict,           # Dictionary weights take precedence
            key_transform=identity_transform,  # Don't modify the keys
            measures=["Dice", "Jaccard"]
        )

        # Generate results with merged_dataset explicitly set to None
        results_without_merged = compute_weighted_metrics_merged_dataset(
            mock_dataframe,
            merged_dataset=None,           # Explicitly set to None
            weights=weights_dict,          # Same dictionary weights
            key_transform=identity_transform,  # Don't modify the keys
            measures=["Dice", "Jaccard"]
        )

        # Results should be identical, proving merged_dataset is ignored
        for metric in ["Dice", "Jaccard"]:
            assert results_with_merged[metric] == results_without_merged[metric]

        # Also verify we get the expected number of metrics
        assert len(results_with_merged) == 2

    def test_weights_length_validation(self, mock_dataframe, merged_dataset):
        """Test validation of weights list length against merged_dataset keys"""
        invalid_weights = [1.0, 2.0]  # Too short compared to merged_dataset

        with pytest.raises(ValueError) as excinfo:
            compute_weighted_metrics_merged_dataset(
                mock_dataframe,
                merged_dataset=merged_dataset,
                weights=invalid_weights
            )

        assert "Length of weights" in str(excinfo.value)

    def test_missing_merged_dataset(self, mock_dataframe):
        """Test error when merged_dataset is missing but required"""
        weights_list = [1.0, 2.0, 3.0]

        with pytest.raises(ValueError) as excinfo:
            compute_weighted_metrics_merged_dataset(
                mock_dataframe,
                merged_dataset=None,  # Missing
                weights=weights_list  # Not a dict, so merged_dataset is required
            )

        assert "merged_dataset is required" in str(excinfo.value)