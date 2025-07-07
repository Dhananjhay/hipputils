# lib/visualization.py

import matplotlib.pyplot as plt
import numpy as np
from nilearn import plotting


def plot_segmentation_qc(seg_path, anat_path, out_png, dim: float = -0.5, offset: float = 2.0):
    """
    Generate a 3‑panel QC figure for a segmentation overlaid on anatomy:
      1) Centered cuts
      2) Shifted backward by `offset` mm
      3) Shifted forward by `offset` mm

    Parameters
    ----------
    seg_path : str or Path
        Path to the segmentation image.
    anat_path : str or Path
        Path to the anatomical (background) image.
    out_png : str or Path
        Output filepath for the saved QC PNG.
    dim : float, optional
        Background dimming value, default -0.5.
    offset : float, optional
        Millimeter shift for backward/forward views, default 2.0.
    """
    matplotlib = plt.matplotlib
    matplotlib.use("Agg")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))

    # Centered
    disp = plotting.plot_roi(
        axes=ax1,
        roi_img=seg_path,
        bg_img=anat_path,
        display_mode="ortho",
        view_type="continuous",
        alpha=0.5,
        dim=dim,
        draw_cross=False,
    )

    # Backward
    plotting.plot_roi(
        axes=ax2,
        roi_img=seg_path,
        bg_img=anat_path,
        cut_coords=[c - offset for c in disp.cut_coords],
        display_mode="ortho",
        view_type="continuous",
        alpha=0.5,
        dim=dim,
        draw_cross=False,
    )

    # Forward
    plotting.plot_roi(
        axes=ax3,
        roi_img=seg_path,
        bg_img=anat_path,
        cut_coords=[c + offset for c in disp.cut_coords],
        display_mode="ortho",
        view_type="continuous",
        alpha=0.5,
        dim=dim,
        draw_cross=False,
    )

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_surface_qc(surf_path, out_png, view: str = "dorsal"):
    """
    Generate a 3D surface QC image.

    Parameters
    ----------
    surf_path : str or Path
        Path to the surface file (e.g., GIFTI or otherwise supported by Nilearn).
    out_png : str or Path
        Where to save the surface QC PNG.
    view : str, optional
        Nilearn view angle (e.g., 'dorsal', 'lateral', etc.).
    """
    matplotlib = plt.matplotlib
    matplotlib.use("Agg")

    fig = plotting.plot_surf(surf_path, view=view)
    fig.savefig(out_png)
    plt.close(fig)


def plot_registration_qc(flo_path, ref_path, out_png, dim: float = -0.5, contour_color: str = "r"):
    """
    Generate a registration QC figure: moving image with reference contours.

    Parameters
    ----------
    flo_path : str or Path
        Path to the floating (to be registered) image.
    ref_path : str or Path
        Path to the reference image for contours.
    out_png : str or Path
        Where to save the registration QC PNG.
    dim : float, optional
        Background dimming value.
    contour_color : str, optional
        Color for the reference contours.
    """
    matplotlib = plt.matplotlib
    matplotlib.use("Agg")

    display = plotting.plot_anat(flo_path, display_mode="ortho", dim=dim)
    display.add_contours(ref_path, colors=contour_color)
    display.savefig(out_png)
    display.close()
