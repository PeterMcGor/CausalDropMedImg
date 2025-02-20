"""
Medical imaging extensions preserving original functionality
while adding MONAI compatibility
"""

import torch
from monai.networks.nets import DenseNet

class MedicalShapleyExplainer(CausalShapley):
    """Extension for medical imaging with MONAI support"""

    def __init__(self, cgm, density_estimator,
                 image_key='image', label_key='label'):
        super().__init__(cgm, density_estimator)
        self.image_key = image_key
        self.label_key = label_key

    def _medical_metric(self, model, data_df, weights=None):
        """Adapt original metric for medical data"""
        # MONAI-specific data loading
        dataset = MedicalDataset(data_df, self.image_key, self.label_key)
        loader = DataLoader(dataset, batch_size=4)

        total = 0
        for batch in loader:
            inputs = batch[self.image_key].to(model.device)
            preds = model(inputs)
            total += dice_score(preds, batch[self.label_key]).item()

        return total / len(loader)

class MedicalDRE(ClassifierDRE):
    """Medical imaging density ratio estimator"""

    def __init__(self, feature_extractor='densenet121'):
        super().__init__(classifier=None)
        self.feature_extractor = DenseNet(spatial_dims=3, in_channels=1)

    def fit(self, source_loader, target_loader):
        # Extract features from medical images
        source_features = self._extract_features(source_loader)
        target_features = self._extract_features(target_loader)

        # Use original classifier logic on extracted features
        super().fit(source_features, target_features)