import nibabel as nib
import numpy as np
import pandas as pd


def compute_label_volumes(seg_img_path, lookup_df):
    """
    Compute volumes (in mm³) for each label in a segmentation image.

    Parameters
    ----------
    seg_img_path : str or Path
        Path to the segmentation image (e.g., NIfTI file) where voxel values represent labels.
    lookup_df : pd.DataFrame
        DataFrame with label indices as index and at least an 'abbreviation' column.

    Returns
    -------
    dict
        Dictionary mapping each label abbreviation to its volume in mm³.
    """
    img_nib = nib.load(seg_img_path)
    img = img_nib.get_fdata()
    zooms = img_nib.header.get_zooms()
    voxel_mm3 = np.prod(zooms)

    volumes = {}
    for index, name in zip(lookup_df.index.to_list(), lookup_df.abbreviation.to_list()):
        volumes[name] = np.sum(img == index) * voxel_mm3

    return volumes


def generate_volume_dataframe(subject_id, seg_paths, lookup_tsv, output_tsv, hemis=["L", "R"]):
    """
    Generate a DataFrame with volumes for each label and hemisphere for a given subject.

    Parameters
    ----------
    subject_id : str
        Subject ID to include in the output table.
    seg_paths : list of str
        List of two segmentation image paths (left and right hemisphere).
    lookup_tsv : str or Path
        Path to lookup table TSV file with index as label values and 'abbreviation' column.
    output_tsv : str or Path
        Output path for the TSV file to be written.
    hemis : list of str
        List of hemisphere labels, typically ['L', 'R'].

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['subject', 'hemi', <label abbreviations>]
    """
    lookup_df = pd.read_table(lookup_tsv, index_col="index")

    all_entries = []
    for hemi, seg_path in zip(hemis, seg_paths):
        vols = compute_label_volumes(seg_path, lookup_df)
        entry = {
            "subject": f"sub-{subject_id}",
            "hemi": hemi,
            **vols,
        }
        all_entries.append(entry)
        
    df = pd.DataFrame(all_entries)

    # Ensure consistent column order (subject, hemi, then all abbreviations)
    desired_columns = ["subject", "hemi"] + lookup_df["abbreviation"].to_list()
    df = df.reindex(columns=desired_columns)
    df.to_csv(output_tsv, sep="\t", index=False)

    return df
