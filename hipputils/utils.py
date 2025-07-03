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

    Convert warp from 2 dimensions to 3 dimensions.

    Parameters
    ----------
        input_warp ::

        input_ref :: 

        output_warp ::

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

def gen_volume_tsv(lookup_tsv, segs, subjects, output_tsv):
    """
    Generates a TSV file summarizing volumes of segmented regions for each subject and hemisphere.

    Parameters
    ----------
        lookup_tsv : str
            Path to a TSV file that maps label indices to region names (with columns "index" and "abbreviation").

        segs : list of str
            List of paths to segmentation image files (one per hemisphere).

        subjects : str
            Subject identifier (used to label the output rows, e.g., "001").

        output_tsv : str
            Path to the output TSV file that will store the volume measurements.

    Returns
    -------
        None
    """

    lookup_df = pd.read_table(lookup_tsv, index_col="index")

    # get indices and names from lookup table
    indices = lookup_df.index.to_list()
    names = lookup_df.abbreviation.to_list()
    hemis = ["L", "R"]

    # create the output dataframe

    df = pd.DataFrame(columns=["subject", "hemi"] + names)

    for in_img, hemi in zip(segs, hemis):
        img_nib = nib.load(in_img)
        img = img_nib.get_fdata()
        zooms = img_nib.header.get_zooms()

        # voxel size in mm^3
        voxel_mm3 = np.prod(zooms)

        new_entry = {
            "subject": "sub-{subject}".format(subject=subjects),
            "hemi": hemi,
        }
        for index, name in zip(indices, names):
            # add volume as value, name as key
            new_entry[name] = np.sum(img == index) * voxel_mm3

        # now create a dataframe from it
        df = df.append(new_entry, ignore_index=True)

    df.to_csv(output_tsv, sep="\t", index=False)


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
