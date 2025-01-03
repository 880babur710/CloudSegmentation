import os
import numpy as np
from PIL import Image

# Set paths to specific subdirectories
project_directory = os.path.dirname(os.path.abspath(__file__))
train_val_set_dir = os.path.join(project_directory, 'train_val_set')
train_gt_dir = os.path.join(train_val_set_dir, 'train_gt')


def analyze_folder(folder_path, ):
    """
    Analyze all .TIF files in a given folder and output statistics for the folder.
    For train_gt, compute unique pixel values and specific zero-value distribution.
    """
    pixel_values = []  # To collect all pixel values in the folder
    zero_percentage_counts = [0] * 11  # Track counts for 0-10%, 10-20%, ... 90-100%, 100%

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

            # Check for NaN values in floating-point images
            if np.isnan(img_data).any():
                print(f"NaN values found in {file_path}")

            # Check for negative values (if unexpected in image)
            if (img_data < 0).any():
                print(f"Negative values found in {file_path}")

            # If in train_gt, analyze pixel values for unique values and zero percentages
            unique_vals = np.unique(img_data)
            zero_pixel_count = np.sum(img_data == 0)
            zero_percentage = (zero_pixel_count / img_data.size) * 100
            bucket_index = min(int(zero_percentage // 10), 9)  # Cap to 9 for 90-100%
            zero_percentage_counts[bucket_index] += 1
            if zero_percentage == 100:
                zero_percentage_counts[10] += 1

    # Concatenate all pixel values from files in the folder
    all_pixels = np.concatenate([img.ravel() for img in pixel_values])

    # Additional analysis for train_gt folder
    print("\nUnique pixel values in train_gt:", np.unique(all_pixels))
    print("Percentage of files with pixel value 0 in different ranges:")
    for i in range(10):
        print(f" - {i * 10}% to {(i + 1) * 10}%: {zero_percentage_counts[i]} files")

    print(f" - Exactly 100%: {zero_percentage_counts[10]} files")


def analyze_all_subfolders(directory):
    """
    Traverse specified subfolders in a given directory and analyze .TIF files in each.
    """
    for subfolder in os.listdir(directory):
        subfolder_path = os.path.join(directory, subfolder)
        if os.path.isdir(subfolder_path) and subfolder == 'train_gt':
            analyze_folder(subfolder_path)


# Start analysis in each set
print("Analyzing train_val_set...")
analyze_all_subfolders(train_val_set_dir)
