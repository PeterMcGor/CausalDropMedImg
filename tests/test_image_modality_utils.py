import pytest
from nnunet_utils.image_modality_utils import ImageModalityName, StandardImageModalities

def test_modality_normalization():
    """Test normalization of different modality name formats."""
    assert StandardImageModalities.normalize_modality_name("T1w") == ImageModalityName.T1
    assert StandardImageModalities.normalize_modality_name("t1ce") == ImageModalityName.T1CE
    assert StandardImageModalities.normalize_modality_name("FLAIR") == ImageModalityName.FLAIR
    assert StandardImageModalities.normalize_modality_name("T2-FLAIR") == ImageModalityName.T2F

def test_invalid_modality_name():
    """Test handling of invalid modality names."""
    with pytest.raises(ValueError):
        StandardImageModalities.normalize_modality_name("InvalidModality")

def test_channel_names_generation():
    """Test generation of channel names dictionary."""
    modalities = ["T1w", "FLAIR", "t2"]
    channel_names = StandardImageModalities.get_standard_channel_names(modalities)

    assert channel_names == {
        0: "T1",
        2: "T2",
        1: "FLAIR"
    }

def test_standard_protocols():
    """Test standard protocol definitions."""
    brain_protocol = StandardImageModalities.get_brain_mri_standard()
    assert len(brain_protocol) == 4
    assert ImageModalityName.T1 in brain_protocol
    assert ImageModalityName.T2 in brain_protocol
    assert ImageModalityName.T2F in brain_protocol
    assert ImageModalityName.T1CE in brain_protocol

    stroke_protocol = StandardImageModalities.get_stroke_ct_standard()
    assert len(stroke_protocol) == 3
    assert ImageModalityName.NCCT in stroke_protocol
    assert ImageModalityName.CTA in stroke_protocol
    assert ImageModalityName.CT in stroke_protocol

def test_alias_consistency():
    """Test that all aliases map to valid modalities."""
    aliases = StandardImageModalities.get_all_aliases()

    for alias, modality in aliases.items():
        assert isinstance(modality, ImageModalityName)
        assert modality in ImageModalityName

def test_case_insensitivity():
    """Test that modality normalization is case-insensitive."""
    variations = ["t1", "T1", "T1W", "t1w"]

    for variant in variations:
        assert StandardImageModalities.normalize_modality_name(variant) == ImageModalityName.T1

def test_modality_enum_values():
    """Test that enum values are consistent."""
    assert ImageModalityName.T1.value == "T1"
    assert ImageModalityName.FLAIR.value == "FLAIR"
    assert ImageModalityName.CT.value == "CT"

def test_multiple_modality_sets():
    """Test handling of different modality combinations."""
    test_cases = [
        (["T1w", "T2w"], {0: "T1", 1: "T2"}),
        (["FLAIR"], {0: "FLAIR"}),
        (["t1", "t1ce", "t2", "flair"], {0: "T1", 1: "T1ce", 2: "T2", 3: "FLAIR"}),
        (["CT", "CTA"], {0: "CT", 1: "CTA"})
    ]

    for input_mods, expected_output in test_cases:
        assert StandardImageModalities.get_standard_channel_names(input_mods) == expected_output