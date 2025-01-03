import os
import numpy as np
from PIL import Image

# Set paths to specific subdirectories
project_directory = os.path.dirname(os.path.abspath(__file__))
train_val_set_dir = os.path.join(project_directory, 'train_val_set')
test_set_dir = os.path.join(project_directory, 'test_set')


def analyze_folder(folder_path):
    """
    Analyze all .TIF files in a given folder and output statistics for the folder.
    """
    dimensions = set()  # To store unique dimensions of images in the folder
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

            # Track dimensions
            dimensions.add(img_data.shape)

            # Collect pixel values for min, max, mean calculations
            pixel_values.append(img_data)

    # Concatenate all pixel values from files in the folder
    all_pixels = np.concatenate([img.ravel() for img in pixel_values])

    # Calculate statistics
    min_val = all_pixels.min()
    max_val = all_pixels.max()
    mean_val = all_pixels.mean()
    median_val = np.median(all_pixels)

    # Output the statistics
    print(f"Folder: {os.path.basename(folder_path)}")
    print(f" - Files: {file_count}")
    print(f" - Dimensions: {dimensions if len(dimensions) > 1 else next(iter(dimensions))}")
    print(f" - Min val: {min_val}")
    print(f" - Max val: {max_val}")
    print(f" - Mean val: {mean_val:.2f}")
    print(f" - Median val: {median_val}")


def analyze_all_subfolders(directory):
    """
    Traverse specified subfolders in a given directory and analyze .TIF files in each.
    """
    for subfolder in os.listdir(directory):
        subfolder_path = os.path.join(directory, subfolder)
        if os.path.isdir(subfolder_path):
            analyze_folder(subfolder_path)


# Start analysis in each set
print("Analyzing train_val_set...")
analyze_all_subfolders(train_val_set_dir)

print("\nAnalyzing test_set...")
analyze_all_subfolders(test_set_dir)
