from enum import Enum
from typing import Dict, List, Set

class ImageModalityName(str, Enum):
    """Standard names for medical image modalities."""

    # MRI Modalities
    T1 = "T1"
    T1C = "T1c"  # T1 with contrast
    T1CE = "T1ce"  # T1 contrast-enhanced
    T1GD = "T1Gd"  # T1 with gadolinium
    T2 = "T2"
    T2F = "T2-FLAIR"  # T2 FLAIR
    FLAIR = "FLAIR"
    DWI = "DWI"  # Diffusion-weighted
    ADC = "ADC"  # Apparent diffusion coefficient
    SWI = "SWI"  # Susceptibility-weighted
    DP = "DP"

    # CT Modalities
    CT = "CT"
    CTA = "CTA"  # CT angiography
    CECT = "CECT"  # Contrast-enhanced CT
    NCCT = "NCCT"  # Non-contrast CT

    # Nuclear Medicine
    PET = "PET"
    PETCT = "PET-CT"
    PETFDG = "PET-FDG"
    SPECT = "SPECT"

    # Ultrasound
    US = "US"
    CEUS = "CEUS"  # Contrast-enhanced ultrasound

    # X-ray
    XRAY = "XRAY"

    # Other
    MASK = "MASK"
    SEG = "SEG"

class StandardImageModalities:
    """Utility class for handling standard medical image modalities."""

    # Define aliases as a class variable
    _ALIASES = {
        # T1 aliases
        "t1": ImageModalityName.T1,
        "t1w": ImageModalityName.T1,
        "t1-weighted": ImageModalityName.T1,
        "t1_weighted": ImageModalityName.T1,

        # T1 with contrast aliases
        "t1c": ImageModalityName.T1C,
        "t1ce": ImageModalityName.T1CE,
        "t1-ce": ImageModalityName.T1CE,
        "t1_ce": ImageModalityName.T1CE,
        "t1gd": ImageModalityName.T1GD,
        "t1-gd": ImageModalityName.T1GD,
        "t1_gd": ImageModalityName.T1GD,
        "gado": ImageModalityName.T1GD,
        "t1+c": ImageModalityName.T1C,

        # T2 aliases
        "t2": ImageModalityName.T2,
        "t2w": ImageModalityName.T2,
        "t2-weighted": ImageModalityName.T2,
        "t2_weighted": ImageModalityName.T2,

        # FLAIR aliases
        "flair": ImageModalityName.FLAIR,
        "t2-flair": ImageModalityName.T2F,
        "t2_flair": ImageModalityName.T2F,

        # Proton Density aliases
        "dp": ImageModalityName.DP,
        "pd": ImageModalityName.DP,  # Alternative name (Proton Density)
        "proton-density": ImageModalityName.DP,

        # DWI aliases
        "dwi": ImageModalityName.DWI,
        "diffusion": ImageModalityName.DWI,

        # CT aliases
        "ct": ImageModalityName.CT,
        "cat": ImageModalityName.CT,
        "ncct": ImageModalityName.NCCT,
        "cect": ImageModalityName.CECT,

        # PET aliases
        "pet": ImageModalityName.PET,
        "pet-ct": ImageModalityName.PETCT,
        "pet/ct": ImageModalityName.PETCT,
        "pet_ct": ImageModalityName.PETCT,
        "fdg-pet": ImageModalityName.PETFDG,
        "fdg_pet": ImageModalityName.PETFDG,

        # Mask/Segmentation aliases
        "mask": ImageModalityName.MASK,
        "seg": ImageModalityName.SEG,
        "segmentation": ImageModalityName.SEG,
        "label": ImageModalityName.SEG,
    }

    @staticmethod
    def get_all_aliases() -> Dict[str, ImageModalityName]:
        """Get dictionary of all modality aliases."""
        return StandardImageModalities._ALIASES.copy()

    @staticmethod
    def normalize_modality_name(name: str) -> ImageModalityName:
        """
        Convert various modality name formats to standard format.

        Args:
            name: Input modality name

        Returns:
            Standardized ImageModalityName

        Raises:
            ValueError: If modality name is not recognized
        """
        # First try to find in aliases (case-insensitive)
        name_lower = name.lower()
        if name_lower in StandardImageModalities._ALIASES:
            return StandardImageModalities._ALIASES[name_lower]

        # Then try to match with enum values (case-sensitive)
        try:
            return ImageModalityName(name)
        except ValueError:
            # Try to match with enum values (case-insensitive)
            for modality in ImageModalityName:
                if modality.value.lower() == name_lower:
                    return modality

            raise ValueError(f"Unknown modality name: {name}. "
                           f"Known modalities: {list(ImageModalityName)}")

    @staticmethod
    def get_standard_channel_names(modalities: List[str]) -> Dict[int, str]:
        """
        Convert list of modality names to nnUNet channel names dictionary.

        Args:
            modalities: List of modality names

        Returns:
            Dictionary mapping channel indices to standardized names
        """
        return {i: StandardImageModalities.normalize_modality_name(mod).value
                for i, mod in enumerate(modalities)}

    @staticmethod
    def get_brain_mri_standard() -> List[ImageModalityName]:
        """Get standard brain MRI modalities."""
        return [
            ImageModalityName.T1,
            ImageModalityName.T2,
            ImageModalityName.T2F,
            ImageModalityName.T1CE
        ]

    @staticmethod
    def get_stroke_ct_standard() -> List[ImageModalityName]:
        """Get standard stroke CT modalities."""
        return [
            ImageModalityName.NCCT,
            ImageModalityName.CTA,
            ImageModalityName.CT
        ]