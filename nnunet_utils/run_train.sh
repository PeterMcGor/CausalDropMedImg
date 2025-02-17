#!/bin/bash

# Ensure symbolic links exist before training
ln -s /home/jovyan/shared/pedro-maciasgordaliza/ms-data/nnunet_folders/nnUNet_results /opt/nnunet_resources/
ln -s /home/jovyan/shared/pedro-maciasgordaliza/ms-data/nnunet_folders/nnUNet_rar /opt/nnunet_resources/
ln -s /home/jovyan/shared/pedro-maciasgordaliza/ms-data/nnunet_folders/nnUNet_preprocessed /opt/nnunet_resources/

# Run nnUNet training with dataset, fold, and trainer as parameters
nnUNetv2_train "$DATASET" 3d_fullres "$FOLD" -tr "$TRAINER" --npz
