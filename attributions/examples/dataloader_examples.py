# Example usage
import os
from typing import Tuple, Union
from attributions.models.base_models import OptimizerConfig
from attributions.models.merge_nnunet_trainers_inferers import MergedNNUNetDataLoaderSpecs
from nnunet_utils.dataset_utils import MergerNNUNetDataset

from batchgenerators.dataloading.data_loader import DataLoaderFromDataset


def main():
    dataset = 'Dataset824_FLAWS-HCO'
    dataset_folder = '/home/jovyan/nnunet_data/nnUNet_preprocessed/'+dataset

    def create_split_datasets(
        train_folder: str,
        test_folder: str,
        split_ratio: Union[Tuple[float, float], float] = 0.8,
        seed: int = 42
    ) -> Tuple[MergerNNUNetDataset, MergerNNUNetDataset]:
        """
        Create training and validation datasets by:
        1. Splitting each folder into train/val
        2. Merging the train portions and val portions separately
        """
        def process_case_id(case_id: str, nnunet_dataset=None) -> int:
            return 'deployment' in case_id

        # Create datasets for each folder
        dataset1 = MergerNNUNetDataset(
            train_folder,
            additional_data={'source': train_folder, 'deployment_data': process_case_id, 'test_data': 0}
        )
        dataset2 = MergerNNUNetDataset(
            test_folder,
            additional_data={'source': test_folder, 'deployment_data': process_case_id, 'test_data': 1}
        )
        return dataset1.merge_and_split(dataset2, split_ratio, seed=seed) #train_train, train_val
    # Create configurations
    """
    classifier_config = BinaryClassifierConfig(
        model_type=MonaiModelType.DENSENET121,
        num_input_channels= 1,#2,  # data + target channels
        dropout_prob=0.2
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_config = TrainingConfig(
        num_epochs=2,
        val_interval=1,
        num_train_iterations_per_epoch=50, #250
        num_val_iterations_per_epoch=20, #150
        metric=MetricConfig('f1', MetricGoal.MAXIMIZE),
        log_path=Path(f'./logs/monay_binary_classifier_test_just_images/{dataset}_{timestamp}'),
        save_path=Path(f'./models_save/monai_binary_classifier_test_just_images/{dataset}_{timestamp}'),
        device="cuda",
        verbosity=2
    )

    # Optional custom optimizer configuration
    optimizer_config = OptimizerConfig(
        optimizer_class=torch.optim.AdamW,
        optimizer_kwargs={'lr': 1e-4}
    )

    """

    dataset_folder = '/home/jovyan/nnunet_data/nnUNet_preprocessed/Dataset824_FLAWS-HCO'
    folder1 = os.path.join(dataset_folder, 'nnUNetPlans_3d_fullres_train_images')
    folder2 = os.path.join(dataset_folder, 'nnUNetPlans_3d_fullres_test_images')

    # Create datasets
    train_ds, val_ds = create_split_datasets(folder1, folder2, split_ratio=(0.2, 0.8))
    print(f"Training dataset size: {len(train_ds)}")
    print(f"Validation dataset size: {len(val_ds)}")

    dataloader_specs = MergedNNUNetDataLoaderSpecs(
        dataset_json_path=os.path.join(dataset_folder, 'dataset.json'),
        dataset_plans_path=os.path.join(dataset_folder, 'nnUNetPlans.json'),
        dataset_train=train_ds,
        dataset_val=val_ds,
        batch_size=4,
        num_processes=12,
        unpack_data=True,
        inference=True
        )
    val_dataloader = dataloader_specs.val_loader
    print("Infinite",type(val_dataloader))
    print("Infinite",val_dataloader.infinite)
    val_dataloader.infinite = False
    print("Infinite",val_dataloader.infinite)
    #val_dataloader.generator.reset()

    for batch in val_dataloader:
        print(batch['keys'])



if __name__ == '__main__':
    main()