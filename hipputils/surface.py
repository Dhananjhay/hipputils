import pyvista as pv
import numpy as np
import nibabel as nib
import nibabel.gifti as gifti
from collections import defaultdict
from lib.utils import setup_logger
from scipy.sparse import diags, linalg, lil_matrix
from scipy.signal import argrelextrema
from scipy.ndimage import binary_dilation


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

        Parameters
        ----------
            surf_gii :: str
                Path to the GIFTI file

        Returns
        -------
            None
        """

        #Load a surface mesh from a GIFTI file
        self.surf = nib.load(surf_gii)
        self.vertices = self.surf.agg_data("NIFTI_INTENT_POINTSET")
        self.faces = self.surf.agg_data("NIFTI_INTENT_TRIANGLE")

    def load_surface_as_pyvista(self):
        """
        Load a surface mesh from a GIFTI file

        Parameters
        ----------
            None

        Returns
        -------
            pv.PolyData(vertices, faces_pv)
                 PyVista PolyData object containing the surface mesh.
        
        """

        faces_pv = np.hstack([np.full((self.faces.shape[0], 1), 3), self.faces])  # PyVista format

        return pv.PolyData(self.vertices, faces_pv)


    @staticmethod
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

    @staticmethod
    def write_surface_to_gifti(mesh, out_surf_gii):
        """
        Writes a PyVista mesh to a GIFTI surface file.

        Parameters
        ----------
            mesh ::

            out_surf_gii ::

        Returns
        -------
            None

        """

        points = mesh.points
        faces = mesh.faces.reshape((-1, 4))[:, 1:4]  # Extract triangle indices

        points_darray = nib.gifti.GiftiDataArray(
            data=points, intent="NIFTI_INTENT_POINTSET", datatype="NIFTI_TYPE_FLOAT32"
        )

        tri_darray = nib.gifti.GiftiDataArray(
            data=faces, intent="NIFTI_INTENT_TRIANGLE", datatype="NIFTI_TYPE_INT32"
        )

        gifti = nib.GiftiImage()
        gifti.add_gifti_data_array(points_darray)
        gifti.add_gifti_data_array(tri_darray)
        gifti.to_filename(out_surf_gii)

    @staticmethod
    def apply_affine_transform(mesh, affine_matrix, inverse=False):
        """
        Applies an affine transformation to a PyVista mesh.

        Parameters
        ----------
            mesh :: 

            affine_matrix ::

        Returns
        -------
            transformed_mesh :: 

        """
        if inverse:
            affine_matrix = np.linalg.inv(affine_matrix)

        transformed_points = (
            np.hstack([mesh.points, np.ones((mesh.n_points, 1))]) @ affine_matrix.T
        )[:, :3]

        transformed_mesh = pv.PolyData(transformed_points, mesh.faces)

        return transformed_mesh
    
    @staticmethod
    def remove_nan_vertices(mesh):
        """
        Removes NaN vertices from a PyVista mesh and updates faces accordingly.

        Parameters
        ----------
            mesh :: 

        Returns
        -------
            cleaned_mesh ::
            
        """
        valid_mask = ~np.isnan(mesh.points).any(axis=1)
        new_indices = np.full(mesh.n_points, -1, dtype=int)
        new_indices[valid_mask] = np.arange(valid_mask.sum())

        faces = mesh.faces.reshape((-1, 4))[:, 1:4]  # Extract triangle indices
        valid_faces_mask = np.all(valid_mask[faces], axis=1)
        new_faces = new_indices[faces[valid_faces_mask]]
        new_faces_pv = np.hstack([np.full((new_faces.shape[0], 1), 3), new_faces])

        cleaned_mesh = pv.PolyData(mesh.points[valid_mask], new_faces_pv)
        return cleaned_mesh


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
        Identify and mark the boundary vertices of a surface mesh, then save the result as a GIFTI label file.

        Parameters
        ----------
            output_label_gii : str
                The file path where the resulting GIFTI label file will be saved.

        Returns
        -------
            None
        
        """

        logger.info("Loading surface from GIFTI...")
        surface = self.load_surface_as_pyvista()
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

    @staticmethod
    def get_terminal_indices_firstminima(
        sdt, min_vertices, boundary_mask, bins=100, smoothing_window=5
    ):
        """
        Gets the terminal (src/sink) vertex indices by determining an adaptive threshold
        using the first local minimum of the histogram of `sdt` values.

        Parameters
        ----------
            sdt :: 
                Signed distance transform array.
            min_vertices :: 
                The minimum number of vertices required.
            boundary_mask :: Boolean 
                Booleaan or binary mask indicating boundary regions.
            bins ::
                Number of bins to use in the histogram (default: 100).
            smoothing_window ::
                Window size for moving average smoothing (default: 5).

        Returns:
            indices :: 
                List of terminal vertex indices.
        """

        # Extract SDT values within the boundary mask
        sdt_values = sdt[boundary_mask == 1]

        # Compute histogram
        hist, bin_edges = np.histogram(sdt_values, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Smooth the histogram using a simple moving average
        smoothed_hist = np.convolve(
            hist, np.ones(smoothing_window) / smoothing_window, mode="same"
        )

        # Find local minima
        minima_indices = argrelextrema(smoothed_hist, np.less)[0]

        if len(minima_indices) == 0:
            raise ValueError("No local minima found in the histogram.")

        # Select the first local minimum after the first peak
        first_minimum_bin = bin_centers[minima_indices[0]]

        # Select indices where SDT is below this threshold
        indices = np.where((sdt < first_minimum_bin) & (boundary_mask == 1))[0].tolist()

        if len(indices) >= min_vertices:
            return indices

        raise ValueError(
            f"Unable to find minimum of {min_vertices} vertices on boundary within the first local minimum threshold."
        )

    @staticmethod
    def get_terminal_indices_percentile(
        sdt, min_percentile, max_percentile, min_vertices, boundary_mask
    ):
        """
        Gets the terminal (src/sink) vertex indices by sweeping a percentile-based threshold
        of the signed distance transform (sdt), ensuring at least `min_vertices` are selected.

        Instead of a fixed distance range, this function dynamically determines the threshold
        by scanning from `min_percentile` to `max_percentile`.

        Parameters
        ----------
            sdt :: ndarray
                Signed distance transform array.

            min_percentile :: float
                Starting percentile for thresholding (0-100).

            max_percentile :: float
                Maximum percentile for thresholding (0-100).

            min_vertices :: int
                The minimum number of vertices required.
                
            boundary_mask :: ndarray
                Boolean or binary mask indicating boundary regions.

        Returns:
            indices :: list of int
                List of terminal vertex indices.
        """

        for percentile in np.arange(min_percentile, max_percentile, 0.5):
            dist_threshold = np.percentile(sdt[boundary_mask == 1], percentile)
            indices = np.where((sdt < dist_threshold) & (boundary_mask == 1))[0].tolist()

            if len(indices) >= min_vertices:
                logger.info(
                    f"Using {percentile}-th percentile to obtain sdt threshold of {dist_threshold}, with {len(indices)} vertices"
                )
                return indices

        raise ValueError(
            f"Unable to find minimum of {min_vertices} vertices on boundary within the {max_percentile}th percentile of distances"
        )


    @staticmethod
    def get_terminal_indices_threshold(
    sdt, min_dist, max_dist, min_vertices, boundary_mask
    ):  
        """
        Gets the terminal (src/sink) vertex indices based on distance to the src/sink mask,
        a boundary mask, and a minumum number of vertices. The distance from the mask is
        swept from min_dist to max_dist, until the min_vertices is achieved, else an
        exception is thrown.

        Parameters
        ----------
            sdt :: ndarray
                The signed distance transform array, where each element represents the distance to the nearest boundary.

            min_dist :: float
                The minimum distance threshold to start the selection of terminal vertices.

            max_dist ::  float
                The maximum distance threshold for selecting terminal vertices.
            
            min_vertices :: int
                The minimum number of terminal vertices that must be selected. 
            
            boundary_mask :: ndarray
                A Boolean or binary mask indicating boundary regions.

        Returns
        -------
            indices :: list of int
                 A list of terminal vertex indices
        
        """

        for dist in np.linspace(min_dist, max_dist, 20):
            indices = np.where((sdt < dist) & (boundary_mask == 1))[0].tolist()
            if len(indices) >= min_vertices:
                return indices
        raise ValueError(
            f"Unable to find minimum of {min_vertices} vertices on boundary, within {max_dist}mm of the terminal mask"
        )

    @staticmethod
    def solve_laplace_beltrami_open_mesh(vertices, faces, boundary_conditions=None):
        """
        Solve the Laplace-Beltrami equation on a 3D open-faced surface mesh. No islands please!

        Parameters
        ----------
            vertices :: (np.ndarray)
                Array of shape (n_vertices, 3) containing vertex coordinates.
            faces :: (np.ndarray)
                Array of shape (n_faces, 3) containing indices of vertices forming each triangular face.
            boundary_conditions :: (dict, optional)
                Dictionary where keys are vertex indices with fixed values.

        Returns
        -------
            solution :: (np.ndarray)
                Array of shape (n_vertices,) with the solution values.
        """
        logger.info("solve_laplace_beltrami_open_mesh")
        n_vertices = vertices.shape[0]
        logger.info(f"n_vertices: {n_vertices}")
        # Step 1: Compute cotangent weights
        logger.info("Computing cotangent weights")
        weights = lil_matrix((n_vertices, n_vertices))
        for tri in faces:
            v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
            e0 = v1 - v2
            e1 = v2 - v0
            e2 = v0 - v1
            # Compute cross products and norms
            cross0 = np.cross(e1, -e2)
            cross1 = np.cross(e2, -e0)
            cross2 = np.cross(e0, -e1)
            norm0 = np.linalg.norm(cross0)
            norm1 = np.linalg.norm(cross1)
            norm2 = np.linalg.norm(cross2)
            # Avoid division by zero
            cot0 = np.dot(e1, -e2) / norm0 if norm0 > 1e-12 else 0.0
            cot1 = np.dot(e2, -e0) / norm1 if norm1 > 1e-12 else 0.0
            cot2 = np.dot(e0, -e1) / norm2 if norm2 > 1e-12 else 0.0
            weights[tri[0], tri[1]] += cot2 / 2
            weights[tri[1], tri[0]] += cot2 / 2
            weights[tri[1], tri[2]] += cot0 / 2
            weights[tri[2], tri[1]] += cot0 / 2
            weights[tri[2], tri[0]] += cot1 / 2
            weights[tri[0], tri[2]] += cot1 / 2
        logger.info("weights.tocsr()")
        weights = weights.tocsr()
        # Step 2: Handle boundaries for open meshes
        logger.info("Handle boundaries for open meshes")
        diagonal = weights.sum(axis=1).A1
        # Ensure no zero entries in diagonal to avoid singular matrix issues
        diagonal[diagonal < 1e-12] = 1e-12
        laplacian = diags(diagonal) - weights
        if boundary_conditions is None:
            boundary_conditions = {}
        boundary_indices = list(boundary_conditions.keys())
        boundary_values = np.array(list(boundary_conditions.values()))
        free_indices = np.setdiff1d(np.arange(n_vertices), boundary_indices)
        logger.info("Setting boundary conditions")
        b = np.zeros(n_vertices)
        for idx, value in boundary_conditions.items():
            laplacian[idx, :] = 0
            laplacian[idx, idx] = 1
            b[idx] = value
        # Step 3: Solve the Laplace-Beltrami equation
        logger.info("Solve the Laplace-Beltrami equation")
        solution = np.zeros(n_vertices)
        if len(free_indices) > 0:
            free_laplacian = laplacian[free_indices][:, free_indices]
            free_b = (
                b[free_indices]
                - laplacian[free_indices][:, boundary_indices] @ boundary_values
            )
            solution[boundary_indices] = boundary_values
            try:
                logger.info("about to solve")
                solution[free_indices] = linalg.spsolve(free_laplacian, free_b)
                logger.info("done solve")
            except linalg.MatrixRankWarning:
                logger.info("Warning: Laplacian matrix is singular or ill-conditioned.")
                solution[free_indices] = np.zeros(len(free_indices))
        else:
            solution[boundary_indices] = boundary_values
        return solution

    def load_boundary_and_sdt(self, boundary, src_sdt, sink_sdt):
        """
        Load the boundary mask and signed distance transforms (SDT) for the source and sink.

        Parameters
        ----------
            boundary :: str
                The file path to the GIFTI file containing the boundary mask.

            src_sdt :: str
                The file path to the GIFTI file containing the source signed distance transform (SDT). 

            sink_sdt :: str
                he file path to the GIFTI file containing the sink signed distance transform (SDT). 

        Returns
        -------
            None
        
        """

        self.boundary_mask = nib.load(boundary).agg_data()
        self.src_sdt = nib.load(src_sdt).agg_data()
        self.sink_sdt = nib.load(sink_sdt).agg_data()


    def threshold(self, threshold_method, min_dist_percentile, max_dist_percentile, min_terminal_vertices):
        """
        Apply a thresholding method to determine source and sink terminal vertices based on their distances.
        
        Parameters
        ----------
            threshold_method : str
                The method to use for thresholding. Options are "percentile" and "firstminima".
            
            min_dist_percentile : float
                The minimum percentile of distance used when thresholding (only applicable if method is "percentile").
            
            max_dist_percentile : float
                The maximum percentile of distance used when thresholding (only applicable if method is "percentile").
            
            min_terminal_vertices : int
                The minimum number of terminal vertices that should be considered as a valid thresholding outcome.

        Returns
        -------
            tuple
            A tuple containing two lists:
            - src_indices : list of int
                The indices of the source terminal vertices.
            - sink_indices : list of int
                The indices of the sink terminal vertices.

        """

        if threshold_method == "percentile":
            src_indices = self.get_terminal_indices_percentile(
                self.src_sdt,
                min_dist_percentile,
                max_dist_percentile,
                min_terminal_vertices,
                self.boundary_mask,
            )
            sink_indices = self.get_terminal_indices_percentile(
                self.sink_sdt,
                min_dist_percentile,
                max_dist_percentile,
                min_terminal_vertices,
                self.boundary_mask,
            )


        elif threshold_method == "firstminima":
            src_indices = self.get_terminal_indices_firstminima(
                self.src_sdt,
                min_terminal_vertices,
                self.boundary_mask,
            )
            sink_indices = self.get_terminal_indices_firstminima(
                self.sink_sdt,
                min_terminal_vertices,
                self.boundary_mask,
            )
        
        logger.info(f"# of src boundary vertices: {len(src_indices)}")
        logger.info(f"# of sink boundary vertices: {len(sink_indices)}")


        return src_indices, sink_indices

    def compute_laplace_beltrami(self, threshold_method, min_dist_percentile, max_dist_percentile, min_terminal_vertices, output_coords):
        """
        Compute the Laplace-Beltrami operator on the surface mesh using source and sink terminal vertices determined 
        by the specified thresholding method.

        Parameters
        ----------
            threshold_method : str
                The method to use for thresholding. Options are "percentile" and "firstminima".
            
            min_dist_percentile : float
                The minimum percentile of distance used when thresholding (only applicable if method is "percentile").
            
            max_dist_percentile : float
                The maximum percentile of distance used when thresholding (only applicable if method is "percentile").
            
            min_terminal_vertices : int
                The minimum number of terminal vertices that should be considered as a valid thresholding outcome.

        Returns
        -------
            None
        
        """

        src_indices, sink_indices = self.threshold(threshold_method, min_dist_percentile, max_dist_percentile, min_terminal_vertices)

        src_vals = [0 for i in range(len(src_indices))]
        sink_vals = [1 for i in range(len(sink_indices))]

        boundary_conditions = dict(zip(src_indices + sink_indices, src_vals + sink_vals))

        coords = self.solve_laplace_beltrami_open_mesh(self.vertices, self.faces, boundary_conditions)

        data_array = nib.gifti.GiftiDataArray(data=coords.astype(np.float32))
        image = nib.gifti.GiftiImage()
        image.add_gifti_data_array(data_array)
        nib.save(image, output_coords)






