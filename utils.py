import pandas as pd
from skimage import measure
from pathlib import Path
import pyclesperanto_prototype as cle


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


def extract_intensities(nuclei_masks, h2a_img, cfos_img, filename, region, mouse_id, ihc_round):
    """Taking the nuclei mask and the channels of interest images it extracts morphological and intensity measurements.
    Returns a pandas dataframe containing the extracted features on a per label basis and associated with sample info"""
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
            "equivalent_diameter"
        ],
    )

    # Construct a dataframe
    h2a_df = pd.DataFrame(h2a_props)

    # Rename the intensity columns for further merging with cfos_df
    h2a_df.rename(columns={'intensity_mean': 'h2a_intensity_mean'}, inplace=True)
    h2a_df.rename(columns={'intensity_max': 'h2a_intensity_max'}, inplace=True)
    
    # Extract regionprops from nuclei labels and CFOS channel
    cfos_props = measure.regionprops_table(
        label_image=nuclei_masks,
        intensity_image=cfos_img,
        properties=[
            "label",
            "intensity_mean",
            "intensity_max"
        ],
    )

    # Construct a dataframe
    cfos_df = pd.DataFrame(cfos_props)

    # Rename the intensity columns for further merging with cfos_df
    cfos_df.rename(columns={'intensity_mean': 'cfos_intensity_mean'}, inplace=True)
    cfos_df.rename(columns={'intensity_max': 'cfos_intensity_max'}, inplace=True)
    
    merged_df = pd.merge(cfos_df, h2a_df, on='label', how='inner')
    
    # Create a new DataFrame with the same index as merged_df and the new columns
    new_columns_df = pd.DataFrame({
        'filename': [filename] * len(merged_df),
        'region': [region] * len(merged_df),
        'mouse_id': [mouse_id] * len(merged_df),
        'ihc_round': [ihc_round] * len(merged_df)
    }, index=merged_df.index)

    # Concatenate the new columns DataFrame with the original merged_df
    # Using pd.concat and specifying axis=1 for columns, and placing the new DataFrame first
    merged_df = pd.concat([new_columns_df, merged_df], axis=1)
    
    return merged_df


def convert_labels_to_edges(labels):
    
    # Push the image to GPU memory
    label_image_gpu = cle.push(labels)

    # Detect edges
    edges = cle.detect_label_edges(label_image_gpu)

    # Mask original labels with edges
    labeled_edges = label_image_gpu * edges

    # Dilate labels to make visualization easier
    dilated_labeled_edges = cle.dilate_labels(labeled_edges, radius=1)
    
    return dilated_labeled_edges

