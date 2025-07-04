import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
import nibabel as nib
import matplotlib
import pandas as pd
import seaborn as sns
from scipy.ndimage import binary_dilation, generate_binary_structure

def concat_tsv(input, output_tsv):
    """
    Concatenates multiple TSV files into a single output TSV file.

    Parameters
    ----------
        input : list of str
            List of file paths to the input TSV files to be concatenated.
        
        output_tsv : str
            File path for the output TSV file that will contain the concatenated contents.

    Returns
    -------
        None
    """
    pd.concat([pd.read_table(in_tsv) for in_tsv in input]).to_csv(
        output_tsv, sep="\t", index=False
    )

def convert_warp_2d_to_3d(input_warp, input_ref, output_warp):
    """
    Converts a 2D warp field (used for image registration) into a 3D warp field
    by copying the 2D displacement across all slices in the Z dimension.

    Parameters
    ----------
        input_warp : str
            Path to the input 2D warp field NIfTI file. Should be shape (X, Y, 1, 1, 2) or (X, Y, 1, 2).

        input_ref : str
            Path to the 3D reference image NIfTI file. Used to determine the Z dimension.

        output_warp : str
            Path to the output 3D warp field NIfTI file. Output shape will be (X, Y, Z, 1, 3).

    Returns
    -------
        None
    """

    # Read 2D xfm nifti
    xfm2d_nib = nib.load(input_warp)
    xfm2d_vol = xfm2d_nib.get_fdata()

    # Read 3d ref nifti
    ref3d_nib = nib.load(input_ref)
    ref3d_vol = ref3d_nib.get_fdata()


    # Define the new shape
    Nx, Ny, Nz = ref3d_vol.shape[:3]

    if Nx != xfm2d_vol.shape[0] or Ny != xfm2d_vol.shape[1]:
        print(f"ref_vol: {ref3d_vol.shape}, warp_vol: {xfm2d_vol.shape}")
        raise ValueError("Ref nifti and warp nifti must have the same X and Y dimensions")


    # Create a new array initialized with zeros
    xfm3d_vol = np.zeros((Nx, Ny, Nz, 1, 3), dtype=xfm2d_vol.dtype)

    # Insert the original array, which replicates in each zslice. Leaves the z-displacement untouched (as zero)
    xfm3d_vol[..., :2] = xfm2d_vol
    xfm3d_vol[:, :, :, :, 1] = -xfm3d_vol[
        :, :, :, :, 1
    ]  # apply flip to y (seems to be needed for some reason - not sure why)..

    # Save as nifti
    nib.Nifti1Image(
        xfm3d_vol, affine=ref3d_nib.affine, header=xfm2d_nib.header
    ).to_filename(output_warp)


def dice(ref, res_mask, hipp_lbls, output_dice):
    """
    Computes the Dice similarity coefficient between a reference binary mask and a predicted mask 
    for specified hippocampal labels, and writes the result to a text file.

    Parameters
    ----------
        ref : str
            File path to the reference NIfTI image containing the ground truth binary mask.

        res_mask : str
            File path to the predicted NIfTI image, where regions of interest are labeled.

        hipp_lbls : list of int
            List of label values in `res_mask` that correspond to the hippocampal structures 
            to be included in the Dice calculation.

        output_dice : str
            File path to the output text file where the Dice score will be written.

    Returns
    -------
        None
    """
    r = nib.load(ref)
    ref_mask = r.get_fdata()

    n = nib.load(res_mask)
    nnunet_rois = n.get_fdata()
    nnunet_bin = np.zeros(nnunet_rois.shape)

    lbls = hipp_lbls
    for l in lbls:
        nnunet_bin[nnunet_rois == l] = 1

    dice = np.sum(ref_mask[nnunet_bin == 1]) * 2.0 / (np.sum(ref_mask) + np.sum(nnunet_bin))

    # write to txt file
    with open(output_dice, "w") as f:
        f.write(str(dice))


def plot_subj_subfields(input_tsv, wildcards, output_png):
    """
    Generates and saves a line plot comparing left and right hippocampal subfield volumes 
    for a given subject, using data from a TSV file.

    Parameters
    ----------
        input_tsv : str
            Path to a TSV file containing volume measurements for hippocampal subfields.
            The table must include 'subject', 'hemi', and 'Cyst' columns, which will be dropped.

        wildcards : str or object
            Identifier (e.g., subject ID or Snakemake wildcards object) used as the title of the plot.

        output_png : str
            Path to the output PNG file where the plot will be saved.

    Returns
    -------
        None

    """

    matplotlib.use("Agg")

    df = pd.read_table(input_tsv)

    subjdf = df.drop(columns=["subject", "hemi", "Cyst"]).transpose()
    subjdf.columns = ["L", "R"]

    sns_plot = sns.lineplot(data=subjdf)
    sns_plot.set_title(str(wildcards))
    sns_plot.set_ylabel("Volume (mm^3)")
    sns_plot.get_figure().savefig(output_png)

def rewrite_vertices_to_flat(input_surf_gii, coords_AP, coords_PD, vertspace, z_level, output_surf_gii):
    """
    Projects a 3D surface geometry to a flat 2D plane using anterior-posterior (AP) and 
    proximal-distal (PD) coordinate maps, and sets a constant Z level for all vertices.
    
    Parameters
    ----------
        input_surf_gii : str
            Path to the input GIFTI surface file (.surf.gii) containing 3D vertex coordinates.

        coords_AP : str
            Path to a GIFTI file containing the AP (anterior-posterior) scalar coordinates.

        coords_PD : str
            Path to a GIFTI file containing the PD (proximal-distal) scalar coordinates.

        vertspace : dict
            Dictionary with keys `"extent"` and `"origin"` that define the scaling and translation 
            of the AP/PD space. Example:
                {
                    "extent": [AP_extent, PD_extent],
                    "origin": [AP_origin, PD_origin]
                }

        z_level : float
            The fixed Z-coordinate value to assign to all vertices (used to flatten in 2D).

        output_surf_gii : str
            Path to the output GIFTI surface file (.surf.gii) where the flattened geometry will be saved.

    Returns
    -------
        None
    """
    gii = nib.load(input_surf_gii)
    vertices = gii.get_arrays_from_intent("NIFTI_INTENT_POINTSET")[0].data

    ap = nib.load(coords_AP).darrays[0].data
    pd = nib.load(coords_PD).darrays[0].data

    vertices[:, 0] = ap * -float(vertspace["extent"][0]) - float(
        vertspace["origin"][0]
    )
    vertices[:, 1] = pd * float(vertspace["extent"][1]) - float(
        vertspace["origin"][1]
    )
    vertices[:, 2] = z_level

    nib.save(gii, output_surf_gii)


def selective_dilation(
    input_nifti, output_nifti, src, sink, src_bg, sink_bg, structure_size=3
):
    """
    Perform selective binary dilation on specified labels in a NIfTI image.

    This function dilates voxels belonging to two labels (`src` and `sink`) 
    only into their respective background regions (`src_bg` and `sink_bg`) 
    using a 3D structuring element. The dilation expands each label within 
    its allowed background without overlapping other structures.

    Parameters
    ----------
        input_nifti : str
            Path to the input NIfTI image file.

        output_nifti : str
            Path where the output dilated NIfTI image will be saved.

        src : int
            Label value in the image to be selectively dilated (source label).

        sink : int
            Another label value to be selectively dilated (sink label).

        src_bg : int
            Background label value into which `src` voxels are allowed to dilate.

        sink_bg : int
            Background label value into which `sink` voxels are allowed to dilate.

        structure_size : int, optional (default=3)
            Size parameter for the structuring element used in dilation 
            (currently not used in this implementation, but can be added to 
            control dilation neighborhood).

    Returns
    -------
        None
    """
    # Load image
    img = nib.load(input_nifti)
    data = img.get_fdata().astype(np.int32)

    # Create structuring element
    structure = generate_binary_structure(3, 1)

    # Dilation: src -> src_bg
    src_mask = data == src
    src_bg_mask = data == src_bg
    src_dilated = binary_dilation(src_mask, structure=structure) & src_bg_mask
    data[src_dilated] = src

    # Dilation: sink -> sink_bg
    sink_mask = data == sink
    sink_bg_mask = data == sink_bg
    sink_dilated = binary_dilation(sink_mask, structure=structure) & sink_bg_mask
    data[sink_dilated] = sink

    # Save the modified image
    new_img = nib.Nifti1Image(data, img.affine, img.header)
    nib.save(new_img, output_nifti)

    print(f"Output saved to {output_nifti}")


def download_extract(unzip_dir, url):
    """
    Downloads a ZIP file from a URL and extracts its contents into a specified directory.

    Parameters
    ----------
        unzip_dir : str or Path
            The directory where the ZIP file contents will be extracted.

        url : str
            The URL (excluding the "https://") from which to download the ZIP file.

    Returns
    -------
        None
    """

    outdir = str(unzip_dir)
    os.makedirs(outdir, exist_ok=True)

    zip_path = os.path.join(outdir, "temp.zip")
    urllib.request.urlretrieve("https://" + url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(outdir)

    os.remove(zip_path)

def read_metric_from_gii(metric_gifti):
    """
    Load a surface metric array from a GIFTI file.

    Parameters
    ----------
        metric_gifti : str or Path
            Path to the GIFTI file.

    Returns
    -------
        np.ndarray
            The first matching metric array.
    """
    return nib.load(metric_gifti).darrays[0].data

def write_metric_gii(scalars, out_metric_gii, metadata=None):
    """
    Write a per-vertex scalar array (e.g., cortical metric) to a GIFTI (.gii) file.

    This function creates a GIFTI file containing a single data array representing
    scalar values defined on a brain surface, such as curvature, thickness, or
    functional activation. Optional metadata such as anatomical structure labels
    can be included in the GIFTI header.

    Parameters
    ----------
        scalars : np.ndarray
            A 1D array of scalar values (e.g., one value per vertex on a mesh).
            The array is cast to float32 before saving.
        out_path : str or Path
            File path where the output GIFTI file will be saved.
        metadata : dict, optional
            Optional dictionary containing metadata to add to the GIFTI image.
            If provided, keys such as 'AnatomicalStructurePrimary' will be added
            to the GIFTI meta field.

    Returns
    -------
        None

    Notes
    -----
    - Only a single data array is written to the GIFTI file.
    - This function does not require a full mesh, only scalar values.
    - Useful for writing surface-based metrics generated from neuroimaging pipelines.
    """

    # save the coordinates to a gifti file
    data_array = nib.gifti.GiftiDataArray(data=scalars.astype(np.float32))
    image = nib.gifti.GiftiImage()

    # set structure metadata
    image.meta["AnatomicalStructurePrimary"] = metadata["AnatomicalStructurePrimary"]

    image.add_gifti_data_array(data_array)
    nib.save(image, out_metric_gii)

def write_label_gii(label_scalars, out_label_gii, label_dict={}, metadata=None):
    """
    Write a label array to a GIFTI (.gii) label file with an optional label table.

    This function encodes per-vertex discrete labels (e.g., cortical parcellation)
    as a GIFTI file using the `NIFTI_INTENT_LABEL` intent. It supports assigning
    a label lookup table (LUT) and optional anatomical metadata.

    Parameters
    ----------
        label_scalars : np.ndarray
            A 1D array of integers where each value represents the label assigned to
            a vertex (e.g., 0 = unknown, 1 = hippocampus, 2 = amygdala, etc.).
        out_label_gii : str or Path
            Path to the output GIFTI label file (.gii).
        label_dict : dict, optional
            A dictionary where keys are label names (str), and values are keyword
            arguments to initialize `nibabel.gifti.GiftiLabel` (e.g., {'Red': 255}).
        metadata : dict, optional
            Optional metadata to add to the GIFTI file header, such as
            {'AnatomicalStructurePrimary': 'CortexLeft'}.

    Returns
    -------
        None

    Notes
    -----
    - This function assumes that `label_scalars` corresponds to surface vertices.
    - The label LUT is stored in the GIFTI LabelTable element.
    - Colors and keys must be set manually via `label_dict`.
    """

    # Create a GIFTI label data array
    gii_data = nib.gifti.GiftiDataArray(label_scalars, intent="NIFTI_INTENT_LABEL")

    # Create a Label Table (LUT)
    label_table = nib.gifti.GiftiLabelTable()

    for label_name, label_kwargs in label_dict.items():

        lbl = nib.gifti.GiftiLabel(**label_kwargs)
        lbl.label = label_name
        label_table.labels.append(lbl)

    # Assign label table to GIFTI image
    gii_img = nib.gifti.GiftiImage(darrays=[gii_data], labeltable=label_table)

    # set structure metadata
    if metadata is not None:
        gii_img.meta["AnatomicalStructurePrimary"] = metadata[
            "AnatomicalStructurePrimary"
        ]

    # Save the label file
    gii_img.to_filename(out_label_gii)

def get_adjacent_voxels(mask_a, mask_b):
    """
    Create a mask for voxels where label A is adjacent to label B.

    Parameters
    ----------
        mask_a :: np.ndarray
            A 3D binary mask for label A.
        mask_b :: np.ndarray
            A 3D binary mask for label B.

    Returns
    -------
        adjacency_mask :: np.ndarray
            A 3D mask where adjacent voxels for label A and label B are marked as True.
    """
    # Dilate each mask to identify neighboring regions
    dilated_a = binary_dilation(mask_a)
    dilated_b = binary_dilation(mask_b)

    # Find adjacency: voxels of A touching B and B touching A
    adjacency_mask = (dilated_a.astype("bool") & mask_b.astype("bool")) | (
        dilated_b.astype("bool") & mask_a.astype("bool")
    )

    return adjacency_mask