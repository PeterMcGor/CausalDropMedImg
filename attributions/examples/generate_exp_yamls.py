import os
from pathlib import Path
from itertools import permutations

# Configuration
YAML_TEMPLATE = '''schema_version: 2024.04.01
type: job
spec:
  name: causal-shift-train{train_annotator}-test{test_annotator}
  owner: dgm-ms-brain-mri/pedro-maciasgordaliza
  description: 'Causal mechanism shift analysis from annotator {train_annotator} to annotator {test_annotator}'
  image: dgm-ms-brain-mri/pedro-maciasgordaliza/seg_model_attr:0.0.4
  instance_type: 1xA100
  environment_variables:
    TRAIN_ANNOTATOR: "{train_annotator}"
    TEST_ANNOTATOR: "{test_annotator}"
    NNUNET_FOLDERS_PATH: "/home/jovyan/shared/pedro-maciasgordaliza/ms-data/nnunet_folders"
    NUM_EPOCHS: "1"
    DEVICE: "cuda"
    N_JOBS: "1"
    SHAPLEY_SAMPLES: "500"
  working_directory: /home/jovyan/workspace/CausalDropMedImg
  command: 'git pull origin main && python attributions/examples/medical_image_call.py --train_annotator {train_annotator} --test_annotator {test_annotator} --nnunet_folders_path $NNUNET_FOLDERS_PATH --num_epochs $NUM_EPOCHS --device $DEVICE --n_jobs $N_JOBS --shapley_samples $SHAPLEY_SAMPLES && sc stop job causal-shift-train{train_annotator}-test{test_annotator}'
  scale: 1
  use_spot_instance: false
  schedule: null
  shared_folders:
    - owner: dgm-ms-brain-mri/pedro-maciasgordaliza
      path: /home/jovyan/shared/pedro-maciasgordaliza/ms-data
      name: ms-data'''

def main():
    # Create output directory
    output_dir = Path("experiments_yamls/causal_shift_annotator_combinations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define all possible annotators
    annotators = range(1, 8)  # 1 to 7

    # Generate all valid combinations of train and test annotators
    # Using permutations to ensure we get (1,2) and (2,1) as separate combinations
    combinations = []
    for train, test in permutations(annotators, 2):
        combinations.append((train, test))

    # Generate YAML files for all combinations
    for train_annotator, test_annotator in combinations:
        # Generate YAML content
        yaml_content = YAML_TEMPLATE.format(
            train_annotator=train_annotator,
            test_annotator=test_annotator
        )

        # Create filename
        filename = f"train{train_annotator}_test{test_annotator}.yaml"
        file_path = output_dir / filename

        # Write YAML file
        with open(file_path, 'w') as f:
            f.write(yaml_content)
        print(f"Created {filename}")

    print(f"Generated {len(combinations)} YAML files in {output_dir}")

if __name__ == "__main__":
    main()