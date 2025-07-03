import numpy as np
import pyvista as pv
import nibabel as nib
from scipy.ndimage import binary_dilation
from collections import defaultdict
from surface import Surface
from utils import setup_logger, get_adjacent_voxels, remove_nan_vertices  # import if needed

class IsosurfaceGenerator:
    def __init__(self, coords_img_path, nan_mask_path, src_mask_path, sink_mask_path,
                 threshold, params, logger=None):
        """
        Initialize the isosurface generator with paths to input data and parameters.

        Parameters
        ----------
        coords_img_path : str or Path
            Path to the coordinates image (e.g., a NIfTI file).
        nan_mask_path : str or Path
            Path to the NaN mask volume.
        src_mask_path : str or Path
            Path to the source mask volume.
        sink_mask_path : str or Path
            Path to the sink mask volume.
        threshold : float
            Threshold value for isosurface extraction.
        params : dict
            Dictionary containing parameters for post-processing:
            - hole_fill_radius: float
            - decimate_opts: float or dict (as used in PyVista)
        logger : logging.Logger, optional
            Logger instance for logging (optional).
        """
        self.coords_img_path = coords_img_path
        self.nan_mask_path = nan_mask_path
        self.src_mask_path = src_mask_path
        self.sink_mask_path = sink_mask_path
        self.threshold = threshold
        self.params = params

        self.logger = logger or setup_logger()

        # Load images
        self.coords_img = nib.load(coords_img_path)
        self.coords = self.coords_img.get_fdata()

        self.nan_mask = nib.load(nan_mask_path).get_fdata()
        self.src_mask = nib.load(src_mask_path).get_fdata()
        self.sink_mask = nib.load(sink_mask_path).get_fdata()
        self.affine = self.coords_img.affine

    def preprocess_coords(self):
        """
        Modify coords array with nan mask, src/sink masks and adjacency logic.
        """
        self.logger.info("Preprocessing coordinate data and masks")

        # Set coords to nan where nan_mask == 1
        self.coords[self.nan_mask == 1] = np.nan
        # Set coords for source and sink masks (to create artificial boundaries)
        self.coords[self.src_mask == 1] = -0.1
        self.coords[self.sink_mask == 1] = 1.1

        # Create adjacency mask where source and sink meet
        src_sink_nan_mask = get_adjacent_voxels(self.sink_mask, self.src_mask)
        self.coords[src_sink_nan_mask == 1] = np.nan

    def build_grid(self):
        """
        Create a PyVista ImageData grid and assign cell data from coords.
        """
        self.logger.info("Building PyVista grid")
        dims = np.array(self.coords.shape) + 1  # add one for cell data grid
        grid = pv.ImageData(dimensions=dims, spacing=(1, 1, 1), origin=(0, 0, 0))
        grid.cell_data["values"] = self.coords.flatten(order="F")

        # Convert cell data to point data
        self.grid = grid.cells_to_points("values")

    def generate_isosurface(self):
        """
        Generate the isosurface mesh by contouring the grid.
        """
        self.logger.info(f"Generating isosurface at threshold {self.threshold}")
        self.surface_mesh = self.grid.contour([self.threshold], method="contour", compute_scalars=True)
        self.logger.info(f"Isosurface mesh: {self.surface_mesh}")

    def clean_surface(self):
        """
        Clean and post-process the generated surface mesh.
        """
        self.logger.info("Removing NaN-valued vertices")
        self.surface_mesh = Surface.remove_nan_vertices(self.surface_mesh)
        self.logger.info(f"Mesh after NaN removal: {self.surface_mesh}")

        self.logger.info("Cleaning surface")
        self.surface_mesh = self.surface_mesh.clean(point_merging=False)
        self.logger.info(f"Mesh after clean(): {self.surface_mesh}")

        self.logger.info("Extracting largest connected component")
        self.surface_mesh = self.surface_mesh.extract_largest()
        self.logger.info(f"Mesh after extracting largest component: {self.surface_mesh}")

        hole_fill_radius = self.params.get("hole_fill_radius")
        self.logger.info(f"Filling holes up to radius {hole_fill_radius}")
        self.surface_mesh = self.surface_mesh.fill_holes(hole_fill_radius)
        self.logger.info(f"Mesh after filling holes: {self.surface_mesh}")

        decimate_opts = self.params.get("decimate_opts")
        self.logger.info(f"Decimating surface with decimate_opts={decimate_opts}")
        self.surface_mesh = self.surface_mesh.decimate(decimate_opts)
        self.logger.info(f"Mesh after decimation: {self.surface_mesh}")

        self.logger.info("Final surface clean")
        self.surface_mesh = self.surface_mesh.clean()

    def apply_affine_transform(self):
        """
        Apply the affine matrix transformation to the mesh.
        """
        self.logger.info("Applying affine transformation to surface mesh")
        # You can either use Surface method or PyVista transform directly
        self.surface_mesh = self.surface_mesh.transform(self.affine, inplace=False)

    def build(self):
        """
        Run the full isosurface generation pipeline.

        Returns
        -------
        Surface
            A Surface instance containing the final mesh.
        """
        self.preprocess_coords()
        self.build_grid()
        self.generate_isosurface()
        self.clean_surface()
        self.apply_affine_transform()

        # Wrap in your Surface class
        return Surface(self.surface_mesh)
