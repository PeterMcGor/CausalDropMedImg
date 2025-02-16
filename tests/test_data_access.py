# test_data_access.py
from pathlib import Path
import os

def check_data_access():
    """Verify access to all mounted data directories."""

    # Check all important data paths
    paths_to_check = {
        'Work Directory': '/home/jovyan/work',
        'nnUNet Data': '/home/jovyan/nnunet_data',
        'Additional Datasets': '/home/jovyan/datasets'
    }

    print("Checking data access points:")
    print("-" * 50)

    for name, path in paths_to_check.items():
        path_obj = Path(path)
        if path_obj.exists():
            print(f"{name}: ✓ Connected")
            print(f"Contents of {path}:")
            # List first few items in directory
            for item in list(path_obj.iterdir())[:5]:
                print(f"  - {item.name}")
        else:
            print(f"{name}: ✗ Not found")
        print("-" * 50)

if __name__ == "__main__":
    check_data_access()