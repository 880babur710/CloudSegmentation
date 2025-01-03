import os
import numpy as np
from PIL import Image

# Set paths to specific subdirectories
project_directory = os.path.dirname(os.path.abspath(__file__))
train_val_set_dir = os.path.join(project_directory, 'train_val_set')
test_set_dir = os.path.join(project_directory, 'test_set')
train_gt_dir = os.path.join(train_val_set_dir, 'train_gt')


def analyze_folder(folder_path, is_train_gt=False):
    """
    Analyze all .TIF files in a given folder and output statistics for the folder.
    For train_gt, compute unique pixel values and specific zero-value distribution.
    """
    pixel_values = []  # To collect all pixel values in the folder

    # Count number of .TIF files
    tif_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.tif')]
    file_count = len(tif_files)

    if file_count == 0:
        print(f"Folder: {os.path.basename(folder_path)}")
        print(" - No .TIF files found.")
        return

    for file in tif_files:
        file_path = os.path.join(folder_path, file)
        with Image.open(file_path) as img:
            img_data = np.array(img)

            # Collect pixel values for min, max, mean calculations
            pixel_values.append(img_data)

    # Concatenate all pixel values from files in the folder
    all_pixels = np.concatenate([img.ravel() for img in pixel_values])

    # Calculate statistics
    min_val = all_pixels.min()
    max_val = all_pixels.max()

    # Calculate histogram for pixel value ranges (e.g., 0-50, 50-100, ..., max_val)
    histogram, bin_edges = np.histogram(all_pixels, bins=10, range=(min_val, max_val))
    histogram_percentages = (histogram / all_pixels.size) * 100

    # Output the statistics
    print(f"Folder: {os.path.basename(folder_path)}")
    print(f" - Files: {file_count}")
    print(" - Histogram (Percentage of pixels in value ranges):")
    for i in range(len(histogram)):
        print(f"    {bin_edges[i]:.2f} - {bin_edges[i + 1]:.2f}: {histogram_percentages[i]:.2f}%")


def analyze_all_subfolders(directory, is_train_gt=False):
    """
    Traverse specified subfolders in a given directory and analyze .TIF files in each.
    """
    for subfolder in os.listdir(directory):
        subfolder_path = os.path.join(directory, subfolder)
        if os.path.isdir(subfolder_path):
            analyze_folder(subfolder_path, is_train_gt=(is_train_gt and subfolder == 'train_gt'))


# Start analysis in each set
print("Analyzing train_val_set...")
analyze_all_subfolders(train_val_set_dir, is_train_gt=True)

print("\nAnalyzing test_set...")
analyze_all_subfolders(test_set_dir)
