#!/bin/bash

# Print environment variables for debugging
echo "Environment variables:"
echo "DATASET: ${DATASET}"
echo "FOLD: ${FOLD}"
echo "TRAINER: ${TRAINER}"

# Ensure variables are set
if [ -z "${DATASET}" ] || [ -z "${FOLD}" ] || [ -z "${TRAINER}" ]; then
    echo "Error: Required environment variables are not set"
    echo "DATASET: ${DATASET}"
    echo "FOLD: ${FOLD}"
    echo "TRAINER: ${TRAINER}"
    exit 1
fi

# Ensure symbolic links exist before training
ln -sf /home/jovyan/shared/pedro-maciasgordaliza/ms-data/nnunet_folders/nnUNet_results /opt/nnunet_resources/
ln -sf /home/jovyan/shared/pedro-maciasgordaliza/ms-data/nnunet_folders/nnUNet_rar /opt/nnunet_resources/
ln -sf /home/jovyan/shared/pedro-maciasgordaliza/ms-data/nnunet_folders/nnUNet_preprocessed /opt/nnunet_resources/

# Run nnUNet training with dataset, fold, and trainer as parameters
nnUNetv2_train "${DATASET}" 3d_fullres "${FOLD}" -tr "${TRAINER}" --npz