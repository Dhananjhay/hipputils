# lib/plots.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_subfield_volumes(tsv_path, output_png_path, title=None):
    """
    Plot left and right hemisphere subfield volumes from a TSV.

    Parameters
    ----------
    tsv_path : str or Path
        Path to the TSV file with subfield volumes.
    output_png_path : str or Path
        Path to save the output PNG plot.
    title : str, optional
        Title for the plot. Defaults to None.
    """
    df = pd.read_table(tsv_path)
    subjdf = df.drop(columns=["subject", "hemi", "Cyst"], errors="ignore").transpose()
    subjdf.columns = ["L", "R"]

    ax = sns.lineplot(data=subjdf)
    if title:
        ax.set_title(title)
    ax.set_ylabel("Volume (mm³)")
    ax.get_figure().savefig(output_png_path)
    plt.close(ax.get_figure())
