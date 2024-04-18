import pandas as pd
import numpy as np
import tifffile
from skimage import measure, filters, exposure
from pathlib import Path
import pyclesperanto_prototype as cle
import matplotlib.pyplot as plt


def extract_image_info(image):
    """Reads a filepath and extracts sample info based on the folder structure"""
    # Extract filename, region, mouse and IHC round
    file_path = Path(image)
    file_path_parts = file_path.parts
    filename = file_path.stem
    region = file_path_parts[-2]
    mouse_id = file_path_parts[-3]
    ihc_round = file_path_parts[-4]

    return filename, region, mouse_id, ihc_round


def split_channels(image):
    """Reads a filepath containing a .lsm stack and returns each channel as a separate numpy array"""
    # Read the image file
    img = tifffile.imread(image)

    # Extract per channel info
    nuclei_img = img[0, :, :]
    h2a_img = img[1, :, :]
    cfos_img = img[2, :, :]

    return nuclei_img, h2a_img, cfos_img


def segment_nuclei(nuclei_img, gaussian_sigma, model, cellpose_nuclei_diameter):
    """Takes a numpy array as an input and preprocess the image using a gaussian filter and contrast stretching.
    Afterwards, performs nuclei segmentation using Cellpose"""
    # Might need to perform a Gaussian-blur so Cellpose does not focus in heterochromatin spots
    post_gaussian_img = filters.gaussian(nuclei_img, sigma=gaussian_sigma)

    # Apply Contrast Stretching to improve Cellpose detection of overly bright nuclei
    p2, p98 = np.percentile(post_gaussian_img, (2, 98))
    img_rescale = exposure.rescale_intensity(post_gaussian_img, in_range=(p2, p98))

    # Predict nuclei nuclei_masks using cellpose
    nuclei_masks, flows, styles, diams = model.eval(
        img_rescale,
        diameter=cellpose_nuclei_diameter,
        channels=[0, 0],
        net_avg=False,
    )

    return nuclei_masks


def extract_intensities(
    nuclei_masks, h2a_img, cfos_img, filename, region, mouse_id, ihc_round
):
    """Taking the nuclei mask and the channels of interest images it extracts morphological and intensity measurements.
    Returns a pandas dataframe containing the extracted features on a per label basis and associated with sample info
    """
    # Extract regionprops from nuclei labels and H2A channel
    h2a_props = measure.regionprops_table(
        label_image=nuclei_masks,
        intensity_image=h2a_img,
        properties=[
            "label",
            "intensity_mean",
            "intensity_max",
            "area_filled",
            "perimeter",
            "equivalent_diameter",
        ],
    )

    # Construct a dataframe
    h2a_df = pd.DataFrame(h2a_props)

    # Rename the intensity columns for further merging with cfos_df
    h2a_df.rename(columns={"intensity_mean": "h2a_intensity_mean"}, inplace=True)
    h2a_df.rename(columns={"intensity_max": "h2a_intensity_max"}, inplace=True)

    # Extract regionprops from nuclei labels and CFOS channel
    cfos_props = measure.regionprops_table(
        label_image=nuclei_masks,
        intensity_image=cfos_img,
        properties=["label", "intensity_mean", "intensity_max"],
    )

    # Construct a dataframe
    cfos_df = pd.DataFrame(cfos_props)

    # Rename the intensity columns for further merging with cfos_df
    cfos_df.rename(columns={"intensity_mean": "cfos_intensity_mean"}, inplace=True)
    cfos_df.rename(columns={"intensity_max": "cfos_intensity_max"}, inplace=True)

    merged_df = pd.merge(cfos_df, h2a_df, on="label", how="inner")

    # Create a new DataFrame with the same index as merged_df and the new columns
    new_columns_df = pd.DataFrame(
        {
            "filename": [filename] * len(merged_df),
            "region": [region] * len(merged_df),
            "mouse_id": [mouse_id] * len(merged_df),
            "ihc_round": [ihc_round] * len(merged_df),
        },
        index=merged_df.index,
    )

    # Concatenate the new columns DataFrame with the original merged_df
    # Using pd.concat and specifying axis=1 for columns, and placing the new DataFrame first
    merged_df = pd.concat([new_columns_df, merged_df], axis=1)

    return merged_df


def classify_cells(merged_df, nuclei_masks, h2a_threshold, cfos_threshold):
    """Select H2A and CFOS positive cells based on mean_intensity thresholds, return a mask of + cells"""
    # Filtering the DataFrames for values above the set threshold
    h2a_filtered_df = merged_df[merged_df["h2a_intensity_mean"] > h2a_threshold]
    cfos_filtered_df = merged_df[merged_df["cfos_intensity_mean"] > cfos_threshold]

    # Extracting the h2a and cfos label values as a list
    h2a_pos_labels = h2a_filtered_df["label"].tolist()
    cfos_pos_labels = cfos_filtered_df["label"].tolist()

    # Convert lists to NumPy arrays to leverage vectorized comparisons
    h2a_pos_labels_array = np.array(h2a_pos_labels)
    cfos_pos_labels_array = np.array(cfos_pos_labels)

    # Find common labels
    double_pos_labels = np.intersect1d(h2a_pos_labels_array, cfos_pos_labels_array)

    # Create copies of the original nuclei_masks array
    h2a_nuclei_labels = nuclei_masks.copy()
    cfos_nuclei_labels = nuclei_masks.copy()
    double_pos_nuclei_labels = nuclei_masks.copy()

    nuclei_labels_arrays = [
        h2a_nuclei_labels,
        cfos_nuclei_labels,
        double_pos_nuclei_labels,
    ]
    labels_to_keep = [h2a_pos_labels, cfos_pos_labels, double_pos_labels]

    for array, labels in zip(nuclei_labels_arrays, labels_to_keep):

        # Check if elements are in the 'values_to_keep' array
        in_values = np.isin(array, labels)

        # Set elements not in 'values_to_keep' to zero
        array[~in_values] = 0

    return (
        h2a_nuclei_labels,
        cfos_nuclei_labels,
        double_pos_nuclei_labels,
        h2a_pos_labels,
        cfos_pos_labels,
        double_pos_labels,
    )


def convert_labels_to_edges(labels, radius):
    """Shows just the outline of labels. Radius argument makes outline thicker or thinner"""
    # Push the image to GPU memory
    label_image_gpu = cle.push(labels)

    # Detect edges
    edges = cle.detect_label_edges(label_image_gpu)

    # Mask original labels with edges
    labeled_edges = label_image_gpu * edges

    # Dilate labels to make visualization easier
    dilated_labeled_edges = cle.dilate_labels(labeled_edges, radius=radius)

    return dilated_labeled_edges


def plot_segmentation(
    nuclei_img,
    nuclei_labels,
    h2a_img,
    h2a_nuclei_labels,
    cfos_img,
    cfos_nuclei_labels,
    double_pos_nuclei_labels,
):
    """Visualize the segmentation results on a per image basis"""
    plt.figure(figsize=(40, 40))
    plt.subplot(1, 7, 1)
    plt.imshow(nuclei_img, cmap="gray")
    plt.title("Input Nuclei Image")
    plt.axis("off")

    plt.subplot(1, 7, 2)
    plt.imshow(nuclei_labels, cmap="viridis")
    plt.title("Segmentation nuclei")
    plt.axis("off")

    plt.subplot(1, 7, 3)
    plt.imshow(h2a_img, cmap="gray")
    plt.title("Input H2A Image")
    plt.axis("off")

    plt.subplot(1, 7, 4)
    plt.imshow(h2a_nuclei_labels, cmap="viridis")
    plt.title("H2A + cells")
    plt.axis("off")

    plt.subplot(1, 7, 5)
    plt.imshow(cfos_img, cmap="gray")
    plt.title("Input CFOS Image")
    plt.axis("off")

    plt.subplot(1, 7, 6)
    plt.imshow(cfos_nuclei_labels, cmap="viridis")
    plt.title("CFOS + cells")
    plt.axis("off")

    plt.subplot(1, 7, 7)
    plt.imshow(double_pos_nuclei_labels, cmap="viridis")
    plt.title("Double H2A/CFOS + cells")
    plt.axis("off")

    plt.show()
