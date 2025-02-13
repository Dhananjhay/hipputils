import pyvista as pv
import numpy as np
import nibabel as nib
import nibabel.gifti as gifti
from collections import defaultdict
from lib.utils import setup_logger

# Setup logger
log_file = snakemake.log[0] if snakemake.log else None
logger = setup_logger(log_file)


class Surface:
    """
    Class instance for Surface objects
    """

    def __init__(self,surf_gii):
        """
        Initialize surface
        """

        #Load a surface mesh from a GIFTI file
        self.surf = nib.load(surf_gii)

    def read_surface_from_gifti(self):
        """
        Load a surface mesh from a GIFTI file

        Parameters
        ----------
            None

        Returns
        -------
            pv.PolyData(vertices, faces_pv) ::
        
        """

        vertices = self.surf.agg_data("NIFTI_INTENT_POINTSET")
        faces = self.surf.agg_data("NIFTI_INTENT_TRIANGLE")
        faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces])  # PyVista format

        return pv.PolyData(vertices, faces_pv)

    @staticmethod
    def find_boundary_vertices(mesh):
        """
        Find boundary vertices of a 3D mesh.

        Parameters
        ----------
            mesh ::

        Returns
        -------
            boundary_vertices :: numpy.array 
                Array of vertex indices that are boundary vertices, sorted in ascending order.
        """
        vertices = mesh.points
        faces = mesh.faces.reshape((-1, 4))[:, 1:4]  # Extract triangle indices

        edge_count = defaultdict(int)
        # Step 1: Count edge occurrences
        for face in faces:
            # Extract edges from the face, ensure consistent ordering (min, max)
            edges = [
                tuple(sorted((face[0], face[1]))),
                tuple(sorted((face[1], face[2]))),
                tuple(sorted((face[2], face[0]))),
            ]
            for edge in edges:
                edge_count[edge] += 1
        # Step 2: Identify boundary edges
        boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
        # Step 3: Collect boundary vertices
        boundary_vertices = set()
        for edge in boundary_edges:
            boundary_vertices.update(edge)
        # Convert the set to a sorted list (array)
        return np.array(sorted(boundary_vertices), dtype=np.int32)
    
    def get_boundary_vertices(self, output_label_gii):
        """
        **TODO**

        Parameters
        ----------
            output_label_gii ::

        Returns
        -------
            None
        
        """

        logger.info("Loading surface from GIFTI...")
        surface = self.read_surface_from_gifti()
        logger.info(f"Surface loaded: {surface.n_points} vertices, {surface.n_faces} faces.")

        logger.info("Find boundary vertices")
        boundary_indices = self.find_boundary_vertices(surface)

        boundary_scalars = np.zeros(surface.n_points, dtype=np.int32)  # Default is 0
        boundary_scalars[boundary_indices] = 1  # Set boundary vertices to 1
        logger.info(
            f"Boundary scalar array created. {np.sum(boundary_scalars)} boundary vertices marked."
        )

        logger.info("Saving GIFTI label file...")

        # Create a GIFTI label data array
        gii_data = gifti.GiftiDataArray(boundary_scalars, intent="NIFTI_INTENT_LABEL")

        # Create a Label Table (LUT)
        label_table = gifti.GiftiLabelTable()

        # Define Background label (key 0)
        background_label = gifti.GiftiLabel(
            key=0, red=1.0, green=1.0, blue=1.0, alpha=0.0
        )  # Transparent
        background_label.label = "Background"
        label_table.labels.append(background_label)

        # Define Boundary label (key 1)
        boundary_label = gifti.GiftiLabel(
            key=1, red=1.0, green=0.0, blue=0.0, alpha=1.0
        )  # Red color
        boundary_label.label = "Boundary"
        label_table.labels.append(boundary_label)

        # Assign label table to GIFTI image
        gii_img = gifti.GiftiImage(darrays=[gii_data], labeltable=label_table)

        # Save the label file
        gii_img.to_filename(output_label_gii)
        logger.info(f"GIFTI label file saved as '{output_label_gii}'.")





