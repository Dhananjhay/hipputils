import nibabel as nib
import numpy as np
from scipy.spatial import Delaunay
import pyvista as pv
from surface import Surface


class UnfoldAtlasMeshBuilder:
    """
    Builds a surface mesh by performing Delaunay triangulation on a 2D metric grid,
    transforms points to physical space, filters valid vertices, and prepares a Surface object.
    """

    def __init__(self, metric_path, z_level):
        """
        Initialize with metric file path and desired z-level.

        Parameters
        ----------
        metric_path : str or Path
            Filepath to the metric image (e.g., NIfTI or GIFTI).
        z_level : float
            The z coordinate value to set for all points after transformation.
        """
        self.metric_path = metric_path
        self.z_level = z_level

        self.metric_img = nib.load(self.metric_path)
        self.metric_data = self.metric_img.get_fdata()
        self.affine = self.metric_img.affine

    def build_mesh(self):
        """
        Build the surface mesh from metric data.

        Returns
        -------
        Surface
            A Surface object encapsulating the generated mesh.
        """
        # Create 2D meshgrid points in metric space
        x = np.linspace(0, self.metric_data.shape[0]-1, self.metric_data.shape[0])
        y = np.linspace(0, self.metric_data[1]-1, self.metric_data.shape[1])
        coords_y, coords_x = np.meshgrid(y, x)
        all_points = np.column_stack((coords_x.ravel(), coords_y.ravel()))

        # Perform Delaunay triangulation in 2D
        tri = Delaunay(all_points)

        # Add placeholder z=0 for all points
        points = np.column_stack((all_points, np.full(all_points.shape[0],0)))

        # Filter valid vertices (non-zero metric values)
        valid_mask = (self.metric_data > 0).ravel()
        filtered_points = points[valid_mask]

        # Apply affine transform to physical space (homogeneous coords)
        coords_h = np.hstack((filtered_points, np.ones((filtered_points.shape[0], 1))))
        transformed = (self.affine @ coords_h.T)
        filtered_points = transformed.T[:,:3]

        # Set all z-coordinates to specified z_level
        filtered_points[:, 2] = self.z_level

        # Remap triangle vertex indices to filtered points
        new_indices = np.full(points.shape[0], -1, dtype=int)
        new_indices[valid_mask] = np.arange(filtered_points.shape[0])
        valid_faces_mask = np.all(valid_mask[tri.simplices], axis=1)
        filtered_faces = new_indices[tri.simplices[valid_faces_mask]]

        # Format faces for PyVista: prepend 3 (triangle size) to each face
        faces_pv = np.hstack([np.full((filtered_faces.shape[0], 1), 3), filtered_faces])

        # Create PyVista mesh
        mesh = pv.PolyData(transformed, faces_pv)

        # Wrap mesh in Surface class and return
        return Surface(mesh)

    def save(self, out_path):
        """
        Build the mesh and save it to a GIFTI file.

        Parameters
        ----------
        out_path : str or Path
            Output path for the surface GIFTI file.
        """
        surface = self.build_mesh()
        surface.save(out_path)
