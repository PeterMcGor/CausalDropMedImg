# Basic test to verify our environment setup
def test_imports():
    # Test core scientific libraries
    import numpy as np
    import scipy
    import matplotlib.pyplot as plt
    
    # Test ML libraries
    import wilds
    import timm
    import xgboost
    
    # Test nnUNet
    import nnunetv2
    
    # Test dowhy
    import dowhy
    
    print("All imports successful!")

if __name__ == "__main__":
    test_imports()
