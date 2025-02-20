import os
import multiprocessing
from time import sleep

import tqdm
from batchgenerators.utilities.file_and_folder_operations import load_json, maybe_mkdir_p
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.utils import (
    get_identifiers_from_splitted_dataset_folder,
    create_lists_from_splitted_dataset_folder,
)


class AnyFolderPreprocessor(DefaultPreprocessor):
    """A preprocessor that can handle any folder structure for nnUNet preprocessing."""

    def __init__(self, input_images_folder, input_segs_folder, output_folder, plans_file, dataset_json_file):
        """
        Initialize the preprocessor with input and output paths.

        Args:
            input_images_folder (str): Path to input images
            input_segs_folder (str): Path to input segmentations
            output_folder (str): Path to output directory
            plans_file (str): Path to plans file
            dataset_json_file (str): Path to dataset JSON file
        """
        super().__init__()
        self.input_folder = input_images_folder
        self.input_segs_folder = input_segs_folder
        self.output_folder = output_folder
        self.plans_file = plans_file
        self.dataset_json_file = dataset_json_file
        self.dataset_json = load_json(self.dataset_json_file)

    def get_dataset(self):
        """
        Get dataset information from the folder structure.

        Returns:
            dict: Dictionary containing image and label paths for each identifier
        """
        identifiers = get_identifiers_from_splitted_dataset_folder(
            self.input_folder,
            self.dataset_json['file_ending']
        )
        images = create_lists_from_splitted_dataset_folder(
            self.input_folder,
            self.dataset_json['file_ending'],
            identifiers
        )
        if self.input_segs_folder is not None:
            segs = [
            os.path.join(self.input_segs_folder, i + self.dataset_json['file_ending'])
            for i in identifiers
            ]
        else:
            segs = [None] * len(identifiers)

        return {
            i: {'images': im, 'label': se}
            for i, im, se in zip(identifiers, images, segs)
        }

    def run(self, configuration_name: str, num_processes: int):
        """
        Run the preprocessing pipeline.

        Args:
            configuration_name (str): Name of the configuration to use
            num_processes (int): Number of processes to use for multiprocessing

        Raises:
            RuntimeError: If a worker process dies during execution
        """
        plans_manager = PlansManager(self.plans_file)
        configuration_manager = plans_manager.get_configuration(configuration_name)

        if self.verbose:
            print(f'Preprocessing the following configuration: {configuration_name}')
            print(configuration_manager)

        output_directory = self.output_folder
        maybe_mkdir_p(output_directory)
        dataset = self.get_dataset()

        # Set up multiprocessing
        r = []
        with multiprocessing.get_context("spawn").Pool(num_processes) as p:
            remaining = list(range(len(dataset)))
            workers = [j for j in p._pool]

            # Queue preprocessing tasks
            for k in dataset.keys():
                r.append(p.starmap_async(
                    self.run_case_save,
                    ((
                        os.path.join(output_directory, k),
                        dataset[k]['images'],
                        dataset[k]['label'],
                        plans_manager,
                        configuration_manager,
                        self.dataset_json
                    ),)
                ))

            # Monitor progress
            with tqdm.tqdm(desc=None, total=len(dataset), disable=self.verbose) as pbar:
                while len(remaining) > 0:
                    if not all(j.is_alive() for j in workers):
                        raise RuntimeError(
                            'Worker process died. This could be due to an error or '
                            'insufficient RAM. Try reducing the number of workers.'
                        )

                    done = [i for i in remaining if r[i].ready()]
                    # Get results and handle any errors
                    _ = [r[i].get() for i in done]
                    for _ in done:
                        r[_].get()
                        pbar.update()

                    remaining = [i for i in remaining if i not in done]
                    sleep(0.1)


def main():
    # Configuration
    dataset_folder = '/home/jovyan/shared/pedro-maciasgordaliza/ms-data/nnunet_folders/nnUNet_preprocessed/Dataset001_MSSEG_FLAIR_Annotator1'
    raw_dataset_folder = '/home/jovyan/shared/pedro-maciasgordaliza/ms-data/nnunet_folders/nnUNet_raw/Dataset001_MSSEG_FLAIR_Annotator1/'
    configuration = '3d_fullres'
    num_processes = 12

    # Initialize and run preprocessor
    preprocessor = AnyFolderPreprocessor(
        input_images_folder=os.path.join(raw_dataset_folder, 'imagesTs'),
        input_segs_folder=os.path.join(raw_dataset_folder, 'labelsTs_2'),
        output_folder=os.path.join(dataset_folder, 'nnUNetPlans_3d_fullres_test_images_ann2'),
        plans_file=os.path.join(dataset_folder, 'nnUNetPlans.json'),
        dataset_json_file=os.path.join(dataset_folder, 'dataset.json')
    )

    preprocessor.run(configuration, num_processes)


    # Initialize and run preprocessor
    preprocessor2 = AnyFolderPreprocessor(
        input_images_folder=os.path.join(raw_dataset_folder, 'imagesTr'),
        input_segs_folder=os.path.join(raw_dataset_folder, 'labelsTr'),
        output_folder=os.path.join(dataset_folder, 'nnUNetPlans_3d_fullres_train_images'),
        plans_file=os.path.join(dataset_folder, 'nnUNetPlans.json'),
        dataset_json_file=os.path.join(dataset_folder, 'dataset.json')
    )

    preprocessor2.run(configuration, num_processes)


if __name__ == '__main__':
    main()