import re
import numpy as np
import pandas as pd
from hipputils.surface import Surface  # adjust import based on your project structure

def summarize_surface_densities(surf_giis, output_csv):
    """
    Summarize vertex counts and edge length statistics for given GIFTI surfaces.

    Parameters
    ----------
    surf_giis : list of str or Path
        List of GIFTI surface file paths.
    output_csv : str or Path
        Output CSV path to save the summary table.

    Returns
    -------
    pd.DataFrame
        DataFrame summarizing each surface's density and edge length stats.
    """

    records = []

    for surf_gii in surf_giis:
        # Extract info from filename with regex
        resample = re.search(r"_resample-(\d+)_", str(surf_gii))
        hemi = re.search(r"_hemi-([LR])_", str(surf_gii))
        label = re.search(r"_label-(hipp|dentate)_", str(surf_gii))

        resample_val = resample.group(1) if resample else "NA"
        hemi_val = hemi.group(1) if hemi else "NA"
        label_val = label.group(1) if label else "NA"

        # Load surface as Surface object
        surface = Surface.from_gifti(surf_gii)

        n_vertices = surface.mesh.n_points
        if n_vertices > 850:
            density = f"{int(np.round(n_vertices / 1000))}k"
        else:
            density = str(n_vertices)

        # Compute edge lengths using Surface method
        edge_lengths = surface.compute_edge_lengths()

        # Calculate statistics
        stats = {
            "min_spacing": np.min(edge_lengths),
            "max_spacing": np.max(edge_lengths),
            "mean_spacing": np.mean(edge_lengths),
            "median_spacing": np.median(edge_lengths),
        }

        record = {
            "surf_gii": str(surf_gii),
            "hemi": hemi_val,
            "label": label_val,
            "resample": resample_val,
            "n_vertices": n_vertices,
            "density": density,
            **stats,
        }

        records.append(record)

    # Create DataFrame and save to CSV
    df = pd.DataFrame(records)
    df = df.astype({"density": str, "resample": str})
    df.to_csv(output_csv, index=False)

    return df
