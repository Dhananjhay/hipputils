# lib/boundary_labels.py

import numpy as np
import nibabel as nib
from utils import write_label_gii
import logging


def assign_src_sink_labels(
    edge_path, ap_src_path, ap_sink_path, pd_src_path, pd_sink_path,
    min_terminal_vertices=10, max_iterations=50, shifting_epsilon=0.01,
    output_ap=None, output_pd=None
):
    """
    Assign source/sink labels to mesh edge vertices based on multiple distance transforms
    and iterative shifting to ensure minimum label counts.

    Parameters
    ----------
    edge_path : str
        Path to GIFTI label file containing edge mask (1=edge, 0=non-edge).
    ap_src_path, ap_sink_path, pd_src_path, pd_sink_path : str
        Paths to GIFTI metric files containing signed distances.
    min_terminal_vertices : int
        Minimum required vertices per label.
    max_iterations : int
        Maximum iterations for shifting distances.
    shifting_epsilon : float
        Amount to shift distances during boosting.
    output_ap : str
        Output GIFTI file for AP label.
    output_pd : str
        Output GIFTI file for PD label.
    """
    logger = logging.getLogger(__name__)

    edges = nib.load(edge_path).agg_data()
    ap_src = nib.load(ap_src_path).agg_data()
    ap_sink = nib.load(ap_sink_path).agg_data()
    pd_src = nib.load(pd_src_path).agg_data()
    pd_sink = nib.load(pd_sink_path).agg_data()

    metadata = dict(nib.load(edge_path).meta)

    distances = np.vstack(
        (ap_src[edges == 1], ap_sink[edges == 1], pd_src[edges == 1], pd_sink[edges == 1])
    ).T

    shifting_factors = np.zeros(4)
    num_labels = 4

    for i in range(max_iterations):
        shifted_distances = distances - shifting_factors
        labels = np.argmin(shifted_distances, axis=1)
        unique, counts = np.unique(labels, return_counts=True)
        label_counts = dict(zip(unique, counts))
        # Ensure all labels in range(num_labels) are present, setting missing ones to 0
        label_counts = {k: label_counts.get(k, 0) for k in range(num_labels)}
        logger.info(f"Iteration {i}: label_counts={counts}")

        if all(count >= min_terminal_vertices for count in label_counts.values()):
            break

        for k in range(num_labels):
            if label_counts[k] < min_terminal_vertices:
                shifting_factors[k] += shifting_epsilon
                logger.info(f"Increasing competitiveness of label {k}: shift = {shifting_factors[k]}")
    else:
        raise ValueError(f"Labeling failed after {max_iterations} iterations")
    
    # Ensure all labels are represented
    for k in range(num_labels):
        if label_counts[k] < min_terminal_vertices:
            raise ValueError(
                f"Label {k} has less than minimum number of vertices, {min_terminal_vertices}, label_counts={label_counts}"
            )

    logger.info(f"Final label counts: {label_counts}")

    idx_edges = np.where(edges == 1)[0]

    ap_srcsink = np.zeros(len(edges), dtype=np.int32)
    ap_srcsink[idx_edges[labels == 0]] = 1
    ap_srcsink[idx_edges[labels == 1]] = 2

    pd_srcsink = np.zeros(len(edges), dtype=np.int32)
    pd_srcsink[idx_edges[labels == 2]] = 1
    pd_srcsink[idx_edges[labels == 3]] = 2

    label_dict = {
        "Background": {"key": 0, "red": 1.0, "green": 1.0, "blue": 1.0, "alpha": 0.0},
        "Source": {"key": 1, "red": 1.0, "green": 0.0, "blue": 0.0, "alpha": 1.0},
        "Sink": {"key": 2, "red": 0.0, "green": 0.0, "blue": 1.0, "alpha": 1.0},
    }

    if output_ap:
        write_label_gii(ap_srcsink, output_ap, label_dict, metadata)
    if output_pd:
        write_label_gii(pd_srcsink, output_pd, label_dict, metadata)
