"""
Core density ratio estimation implementations including:
- Classifier-based IS (as in Zhang et al. 2023)
- KLIEP
- Adversarial discriminators
"""

from abc import ABC, abstractmethod
import numpy as np
from sklearn.calibration import CalibratedClassifierCV

class DensityRatioEstimator(ABC):
    """Base class for density ratio estimators"""

    @abstractmethod
    def fit(self, source, target):
        """Train on source and target data"""
        pass

    @abstractmethod
    def compute_ratios(self, data):
        """Compute p_target(x)/p_source(x)"""
        pass

class ClassifierDRE(DensityRatioEstimator):
    """Zhang et al. 2023"""

    def __init__(self, classifier, calibrate=True, clip_prob=None):
        self.classifier = classifier
        self.calibrate = calibrate
        self.clip_prob = clip_prob

    def fit(self, source, target):
        X = np.vstack([source, target])
        y = np.concatenate([np.zeros(len(source)), np.ones(len(target))])

        self.classifier.fit(X, y)

        if self.calibrate:
            self.classifier = CalibratedClassifierCV(
                self.classifier, method='isotonic', cv='prefit'
            ).fit(X, y)

    def compute_ratios(self, data):
        prob = self.classifier.predict_proba(data)

        if self.clip_prob:
            prob = np.clip(prob, 1-self.clip_prob, self.clip_prob)

        return prob[:, 1] / prob[:, 0]

class KLIEPDRE(DensityRatioEstimator):
    """Kernel-based implementation (stub for future development)"""

    def fit(self, source, target):
        pass

    def compute_ratios(self, data):
        pass