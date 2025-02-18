import os
from pathlib import Path

# Configuration
YAML_TEMPLATE = '''schema_version: 2024.04.01
type: job
spec:
  name: nnunet-train-ds{dataset}-fold{fold}
  owner: dgm-ms-brain-mri/pedro-maciasgordaliza
  description: 'Training nnUNet for dataset {dataset} fold {fold}'
  image: dgm-ms-brain-mri/pedro-maciasgordaliza/seg_model_attr:0.0.3
  instance_type: 1xA100
  environment_variables:
    DATASET: "{dataset}"
    FOLD: "{fold}"
    TRAINER: "{trainer}"
    nnUNet_raw: "/home/jovyan/shared/pedro-maciasgordaliza/ms-data/nnunet_folders/nnUNet_raw"
    nnUNet_preprocessed: "/home/jovyan/shared/pedro-maciasgordaliza/ms-data/nnunet_folders/nnUNet_preprocessed"
    nnUNet_results: "/home/jovyan/shared/pedro-maciasgordaliza/ms-data/nnunet_folders/nnUNet_results"
  working_directory: /home/jovyan/workspace/CausalDropMedImg
  command: 'nnUNetv2_train "{dataset}" 3d_fullres "{fold}" -tr "{trainer}" --npz && sc stop job nnunet-train-ds{dataset}-fold{fold}'
  scale: 1
  use_spot_instance: false
  schedule: null
  token_scope: workspace:{{self}}:dask:write
  shared_folders:
    - owner: dgm-ms-brain-mri/pedro-maciasgordaliza
      path: /home/jovyan/shared/pedro-maciasgordaliza/ms-data
      name: ms-data'''

# Parameters
DATASETS = ["1", "2", "3", "4", "5", "6", "7", "8"]
FOLDS = range(5)  # 0 to 4
TRAINERS = ["nnUNetTrainer_250epochs"]

def main():
    # Create output directory
    output_dir = Path("experiments_yamls/MSSEG2016_rater_experiments")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate YAML files for all combinations
    for dataset in DATASETS:
        for fold in FOLDS:
            for trainer in TRAINERS:
                # Generate YAML content
                yaml_content = YAML_TEMPLATE.format(
                    dataset=dataset,
                    fold=fold,
                    trainer=trainer
                )
                
                # Replace the double curly braces with single ones for {self}
                yaml_content = yaml_content.replace('{{self}}', '{self}')
                
                # Create filename
                filename = f"ds{dataset}_fold{fold}_{trainer}.yaml"
                file_path = output_dir / filename
                
                # Write YAML file
                with open(file_path, 'w') as f:
                    f.write(yaml_content)
                print(f"Created {filename}")

if __name__ == "__main__":
    main()