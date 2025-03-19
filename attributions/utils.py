import os
import argparse
import numpy as np
import nibabel as nib
from skimage import measure, morphology


def get_available_cpus():
    """
    Determine the number of available CPUs for processing.
    Tries Kubernetes cgroup limits first, then falls back to os.cpu_count().
    """
    # Try Kubernetes cgroup CPU limit first
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r") as f:
            quota = int(f.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r") as f:
            period = int(f.read().strip())
        if quota > 0:
            return max(1, quota // period)
    except (FileNotFoundError, IOError, ValueError):
        # For newer cgroup v2 in Kubernetes
        try:
            with open("/sys/fs/cgroup/cpu.max", "r") as f:
                quota_info = f.read().strip().split()
                if quota_info[0] != "max":
                    quota = int(quota_info[0])
                    period = int(quota_info[1])
                    return max(1, quota // period)
        except (FileNotFoundError, IOError, ValueError, IndexError):
            pass

    # Fall back to os.cpu_count() with a safety margin
    cpu_count = os.cpu_count()
    if cpu_count is None:
        return 1  # Be conservative if we can't determine

    # Use most CPUs, but leave some headroom
    return max(1, cpu_count - min(2, cpu_count // 4))

def load_nifti(file_path):
    """Load a NIFTI file and return the image object and data"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    img = nib.load(file_path)
    data = img.get_fdata()
    return img, data

def save_nifti(output_path, data, reference_img):
    """Save data as a NIFTI file with the same metadata as reference_img"""
    # Create a new image with the same header and affine
    binary_data = (data > 0).astype(np.int8)
    new_img = nib.Nifti1Image(binary_data,
                              reference_img.affine,
                              reference_img.header)
    # Save the image
    nib.save(new_img, output_path)
    print(f"Saved result to {output_path}")

def count_and_reduce_blobs(data, reduction_ratio):
    """
    Count blobs in the binary mask and reduce their number by the given ratio
    """
    # Ensure the data is binary
    binary_data = (data > 0).astype(np.int8)

    # Label connected components (blobs)
    labeled_data, num_blobs = measure.label(binary_data, return_num=True)
    print(f"Found {num_blobs} blobs in the mask")

    if num_blobs <= 1:
        print("Only one or zero blobs found, nothing to reduce")
        return binary_data

    if reduction_ratio <= 0 or reduction_ratio >= 1:
        print("Warning: Reduction ratio must be between 0 and 1")
        return binary_data

    # Calculate how many blobs to keep
    num_to_keep = max(1, int(num_blobs * (1 - reduction_ratio)))
    print(f"Keeping {num_to_keep} blobs out of {num_blobs}")

    # Get blob properties
    regions = measure.regionprops(labeled_data)
    blob_sizes = [(r.label, r.area) for r in regions]

    # Sort blobs by size (largest first)
    blob_sizes.sort(key=lambda x: x[1], reverse=True)

    # Select which blobs to keep
    labels_to_keep = [b[0] for b in blob_sizes[:num_to_keep]]

    # Create a new mask with only the selected blobs
    new_mask = np.zeros_like(binary_data)
    for label in labels_to_keep:
        new_mask[labeled_data == label] = 1

    return new_mask

def dilate_mask(data, dilation_ratio):
    """
    Dilate the binary mask by the given ratio
    """
    # Ensure the data is binary
    binary_data = (data > 0).astype(np.int8)

    if dilation_ratio <= 0:
        print("Warning: Dilation ratio must be greater than 0")
        return binary_data

    # Calculate the radius of the structuring element based on the ratio
    radius = max(1, int(dilation_ratio * 3))
    print(f"Using spherical structuring element with radius {radius}")

    # Create a spherical structuring element and dilate
    struct_elem = morphology.ball(radius)
    dilated_mask = morphology.binary_dilation(binary_data, struct_elem).astype(binary_data.dtype)

    return dilated_mask

def erode_mask(data, erosion_ratio):
    """
    Erode the binary mask by the given ratio
    """
    # Ensure the data is binary
    binary_data = (data > 0).astype(np.int8)

    if erosion_ratio <= 0:
        print("Warning: Erosion ratio must be greater than 0")
        return binary_data

    # Calculate the radius of the structuring element based on the ratio
    radius = max(1, int(erosion_ratio * 3))
    print(f"Using spherical structuring element with radius {radius}")

    # Create a spherical structuring element and erode
    struct_elem = morphology.ball(radius)
    eroded_mask = morphology.binary_erosion(binary_data, struct_elem).astype(binary_data.dtype)

    # Check if erosion removed everything
    if np.sum(eroded_mask) == 0:
        print("Warning: Erosion removed all voxels. Consider using a smaller ratio.")

    return eroded_mask

def process_folder(source_folder, dest_folder, operation, ratio):
    """
    Apply the specified operation to all NIFTI files in source_folder and save results to dest_folder

    Parameters:
    source_folder (str): Path to folder containing NIFTI files to process
    dest_folder (str): Path to folder where processed files will be saved
    operation (str): Operation to perform ('reduce', 'dilate', or 'erode')
    ratio (float): Ratio for the operation

    Returns:
    int: Number of files processed
    """
    # Ensure destination folder exists
    os.makedirs(dest_folder, exist_ok=True)

    # List all NIFTI files in the source folder
    nifti_files = [f for f in os.listdir(source_folder)
                  if f.endswith('.nii') or f.endswith('.nii.gz')]

    if not nifti_files:
        print(f"No NIFTI files found in {source_folder}")
        return 0

    print(f"Found {len(nifti_files)} NIFTI files to process")

    # Process each file
    processed_count = 0
    for filename in nifti_files:
        input_path = os.path.join(source_folder, filename)
        output_path = os.path.join(dest_folder, filename)

        try:
            print(f"Processing {filename}...")
            # Load the input file
            img, data = load_nifti(input_path)

            # Apply the requested operation
            if operation == 'reduce':
                result = count_and_reduce_blobs(data, ratio)
            elif operation == 'dilate':
                result = dilate_mask(data, ratio)
            elif operation == 'erode':
                result = erode_mask(data, ratio)
            else:
                print(f"Invalid operation: {operation}")
                continue

            # Save the result
            save_nifti(output_path, result, img)
            processed_count += 1

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    print(f"Successfully processed {processed_count} out of {len(nifti_files)} files")
    return processed_count
def main():
    """Main function to parse arguments and perform the requested operation"""
    parser = argparse.ArgumentParser(description='Process 3D medical binary masks (NIFTI format)')

    # Create subparsers for the two modes
    subparsers = parser.add_subparsers(dest='mode', help='Processing mode')

    # Single file mode
    single_parser = subparsers.add_parser('file', help='Process a single file')
    single_parser.add_argument('input_file', help='Path to the input NIFTI file')
    single_parser.add_argument('output_file', help='Path for the output NIFTI file')

    # Batch processing mode
    batch_parser = subparsers.add_parser('folder', help='Process all files in a folder')
    batch_parser.add_argument('input_folder', help='Path to the folder containing input NIFTI files')
    batch_parser.add_argument('output_folder', help='Path to the folder where output NIFTI files will be saved')

    # Common arguments for both modes
    for subparser in [single_parser, batch_parser]:
        subparser.add_argument('--operation', choices=['reduce', 'dilate', 'erode'],
                              required=True, help='Operation to perform')
        subparser.add_argument('--ratio', type=float, required=True,
                              help='Ratio for the selected operation (0-1 for reduce, positive for dilate/erode)')

    args = parser.parse_args()

    # Validate that a mode was specified
    if args.mode is None:
        parser.print_help()
        return 1

    try:
        if args.mode == 'file':
            # Process a single file
            print(f"Loading {args.input_file}...")
            img, data = load_nifti(args.input_file)

            print(f"Performing {args.operation} operation with ratio {args.ratio}...")
            if args.operation == 'reduce':
                result = count_and_reduce_blobs(data, args.ratio)
            elif args.operation == 'dilate':
                result = dilate_mask(data, args.ratio)
            elif args.operation == 'erode':
                result = erode_mask(data, args.ratio)

            save_nifti(args.output_file, result, img)

        elif args.mode == 'folder':
            # Process all files in a folder
            process_folder(args.input_folder, args.output_folder, args.operation, args.ratio)

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)



