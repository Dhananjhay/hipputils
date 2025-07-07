import pyvista as pv
import numpy as np
import nibabel as nib
import nibabel.gifti as gifti
from collections import defaultdict
from utils import setup_logger
import numpy as np
import nibabel as nib
import scipy.sparse as sp
from utils import write_metric_gii


# Setup logger
log_file = snakemake.log[0] if snakemake.log else None
logger = setup_logger(log_file)


class Surface:
    """
    Class instance for Surface objects
    """

    def __init__(self, mesh, metadata=None):
        """
        Initialize a Surface object.

        Parameters
        ----------
        mesh : pyvista.PolyData
            The surface mesh representing vertices and faces of the surface.
        metadata : dict, optional
            Optional dictionary containing metadata associated with the surface.
            Defaults to an empty dictionary if not provided.

        Returns
        -------
        None
        """
        self.mesh = mesh
        self.metadata = metadata or {}

    @classmethod
    def from_gifti(cls, surf_gii):
        """
        Load a surface mesh and associated metadata from a GIFTI (.gii) file.

        This method parses the GIFTI file using nibabel, extracting the vertex coordinates,
        face indices, and any metadata associated with the vertex data array. The faces
        are converted into a format compatible with PyVista's PolyData, which requires
        an initial count of points per face (typically 3 for triangles).

        Parameters
        ----------
        surf_gii : str or Path
            Path to the GIFTI (.gii) surface file to load.

        Returns
        -------
        tuple
            A tuple containing:
            - mesh : pyvista.PolyData
                The surface mesh represented as a PyVista PolyData object with vertices
                and faces loaded from the GIFTI file.
            - metadata : dict
                A dictionary of metadata extracted from the vertex data array in the GIFTI
                file. If no metadata is found, returns an empty dictionary.

        Notes
        -----
        - The faces are converted to PyVista format by prefixing each face with the number
        of points per face (3 for triangles).
        - This method only extracts the first vertex data array found with intent code
        'NIFTI_INTENT_POINTSET' for vertices and 'NIFTI_INTENT_TRIANGLE' for faces.
        - Additional data arrays (such as labels or shape metrics) are not extracted here.
        """
        surf = nib.load(surf_gii)
        vertices = surf.agg_data("NIFTI_INTENT_POINTSET")
        faces = surf.agg_data("NIFTI_INTENT_TRIANGLE")
        faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces])  # PyVista format

        # Find the first darray that represents vertices (NIFTI_INTENT_POINTSET)
        vertices_darray = next(
            (
                darray
                for darray in surf.darrays
                if darray.intent == intent_codes["NIFTI_INTENT_POINTSET"]
            ),
            None,
        )

        # Extract metadata as a dictionary (return empty dict if no metadata)
        metadata = dict(vertices_darray.meta) if vertices_darray else {}

        return cls(pv.PolyData(vertices, faces_pv), metadata)

    def compute_edge_lengths(self):
        """
        Compute the lengths of all edges in a given surface mesh.

        Parameters
        ----------
        surface : pyvista.PolyData (or similar)
            A mesh surface object containing points and edges.

        Returns
        -------
        np.ndarray
            Array of edge lengths (float) for all edges in the surface.
        """

        # Extract edges
        edges = self.extract_all_edges()

        # Extract individual edge segments
        edge_lengths = []
        lines = edges.lines.reshape(-1, 3)  # Each row: [2, point1, point2]

        for line in lines:
            _, p1, p2 = line  # First entry is always "2" (pairs of points)
            length = np.linalg.norm(edges.points[p1] - edges.points[p2])
            edge_lengths.append(length)

        edge_lengths = np.array(edge_lengths)

        return edge_lengths
    
    def to_gifti(self, out_surf_gii):
        """
        Write the current surface mesh (point and faces) to a GIFTI (.gii) file.

        Parameters
        ----------
            out_surf_gii : str or Path
                Path where the GIFTI file will be saved.

        Returns
        -------
            None
        """

        # Extract vertices and faces from self.mesh
        points = self.mesh.points
        # Extract triangle indices
        faces = self.mesh.faces.reshape((-1,4))[:, 1:4].astype(np.int32)

        points_darray = nib.gifti.GiftiDataArray(
            data=points, intent="NIFTI_INTENT_POINTSET", datatype="NIFTI_TYPE_FLOAT32"
        )

        tri_darray = nib.gifti.GiftiDataArray(
            data=faces, intent="NIFTI_INTENT_TRIANGLE", datatype="NIFTI_TYPE_INT32"
        )

        gifti_img = nib.GiftiImage()
        gifti_img.add_gifti_data_array(points_darray)
        gifti_img.add_gifti_data_array(tri_darray)
        gifti_img.to_filename(out_surf_gii)


    def apply_affine(self, affine_matrix, inverse=False):
        """
        Apply an affine transformation to the surface and return a new Surface.

        Parameters
        ----------
        affine_matrix : np.ndarray
            4x4 affine transformation matrix.
        inverse : bool, optional
            If True, apply the inverse of the transformation.

        Returns
        -------
        Surface
            A new Surface instance with transformed geometry and the same metadata.
        """
        if inverse:
            affine_matrix = np.linalg.inv(affine_matrix)

        points_h = np.hstack([self.mesh.points, np.ones((self.mesh.n_points, 1))])
        transformed_points = (points_h @ affine_matrix.T)[:, :3]
        transformed_mesh = pv.PolyData(transformed_points, self.mesh.faces)

        return Surface(transformed_mesh, metadata=self.metadata.copy())
    
    def remove_nan_vertices(self):
        """
        Remove vertices with NaN coordinates from the surface mesh.

        This method identifies and removes any vertices in the mesh that contain NaN
        values, and updates the face topology accordingly. Returns a new Surface
        instance with the cleaned mesh.

        Returns
        -------
        Surface
            A new Surface instance with NaN vertices removed and faces updated.
        """
        valid_mask = ~np.isnan(self.mesh.points).any(axis=1)
        new_indices = np.full(self.mesh.n_points, -1, dtype=int)
        new_indices[valid_mask] = np.arange(valid_mask.sum())

        faces = self.mesh.faces.reshape((-1, 4))[:, 1:4]  # Extract triangle indices
        valid_faces_mask = np.all(valid_mask[faces], axis=1)
        new_faces = new_indices[faces[valid_faces_mask]]
        new_faces_pv = np.hstack([np.full((new_faces.shape[0], 1), 3), new_faces])

        cleaned_mesh = pv.PolyData(self.mesh.points[valid_mask], new_faces_pv)
        return Surface(cleaned_mesh, self.metadata.copy())
    
    def find_boundary_vertices(self):
        """
        Find the boundary vertices of the surface mesh.

        A boundary vertex is one that lies on an edge used by only one face.
        This is useful for detecting open boundaries or holes in the mesh.

        Returns
        -------
        np.ndarray
            Sorted array of boundary vertex indices (int32 dtype).
        """
        faces = self.mesh.faces.reshape((-1, 4))[:, 1:4]

        edge_count = defaultdict(int)

        for face in faces:
            edges = [
                tuple(sorted((face[0], face[1]))),
                tuple(sorted((face[1], face[2]))),
                tuple(sorted((face[2], face[0]))),
            ]
            for edge in edges:
                edge_count[edge] += 1

        boundary_edges = [edge for edge, count in edge_count.items() if count == 1]

        boundary_vertices = set()
        for edge in boundary_edges:
            boundary_vertices.update(edge)

        return np.array(sorted(boundary_vertices), dtype=np.int32)
    
    def get_largest_boundary_component_label(self) -> np.ndarray:
        """
        Identify boundary vertices of the surface mesh and return a scalar array marking those that belong to the largest connected component.

        Returns
        -------
        np.ndarray
            A binary array of shape (n_points,) with 1s marking boundary vertices in the largest connected component, and 0s elsewhere.
        """

        boundary_indices = self.find_boundary_vertices()

        boundary_scalars = np.zeros(self.mesh.n_points, dtype=np.int32)
        boundary_scalars[boundary_indices] = 1  # Set boundary vertices to 1

        # Extract points that are within the boundary scalars
        sub_mesh = self.mesh.extract_points(
            boundary_scalars.astype(bool), adjacent_cells=True
        )

        # Compute connectivity to find the largest connected component
        connected_sub_mesh = sub_mesh.connectivity("largest")

        # Get indices of the largest component in the sub-mesh
        largest_component_mask = (
            connected_sub_mesh.point_data["RegionId"] == 0
        )  # Largest component has RegionId 0
        largest_component_indices = connected_sub_mesh.point_data["vtkOriginalPointIds"][
            largest_component_mask
        ]

        # Create an array for all points in the original surface
        boundary_scalars = np.zeros(self.mesh.n_points, dtype=np.int32)

        # Keep only the largest component
        boundary_scalars[largest_component_indices] = 1

        return boundary_scalars

    def solve_laplace_beltrami(self, src_indices, sink_indices):
        """
        Solve the Laplace-Beltrami equation with Dirichlet boundary conditions on an open mesh.

        Parameters
        ----------
        src_indices : array-like of int
            Indices of source boundary vertices (value = 0).
        sink_indices : array-like of int
            Indices of sink boundary vertices (value = 1).

        Returns
        -------
        np.ndarray
            Scalar solution defined at each vertex.
        """
        vertices = self.mesh.points
        faces = self.mesh.faces.reshape((-1, 4))[:, 1:4]

        n_vertices = vertices.shape[0]
        boundary_conditions = dict(
            zip(
                list(src_indices) + list(sink_indices),
                [0.0] * len(src_indices) + [1.0] * len(sink_indices),
            )
        )

        laplacian = self._cotangent_laplacian(vertices, faces)

        boundary_indices = np.array(list(boundary_conditions.keys()))
        boundary_values = np.array(list(boundary_conditions.values()))
        free_indices = np.setdiff1d(np.arange(n_vertices), boundary_indices)

        for i in boundary_indices:
            start, end = laplacian.indptr[i], laplacian.indptr[i + 1]
            laplacian.data[start:end] = 0
            laplacian[i, i] = 1

        b = np.zeros(n_vertices)
        b[boundary_indices] = boundary_values

        solution = np.zeros(n_vertices)
        if len(free_indices) > 0:
            free_laplacian = laplacian[free_indices][:, free_indices]
            free_b = (
                b[free_indices]
                - laplacian[free_indices][:, boundary_indices] @ boundary_values
            )
            try:
                solution[boundary_indices] = boundary_values
                solution[free_indices] = sp.linalg.spsolve(free_laplacian, free_b)
            except sp.linalg.MatrixRankWarning:
                logger.warning("Laplacian matrix is singular or ill-conditioned.")
                solution[free_indices] = np.zeros(len(free_indices))
        else:
            solution[boundary_indices] = boundary_values

        return solution

    def _cotangent_laplacian(self, vertices, faces):
        """
        Compute the cotangent-weighted Laplace matrix for the surface mesh.

        Parameters
        ----------
        vertices : np.ndarray
            Array of shape (n_vertices, 3) containing vertex coordinates.
        faces : np.ndarray
            Array of shape (n_faces, 3) with triangle indices.

        Returns
        -------
        scipy.sparse.csr_matrix
            Cotangent Laplacian matrix in CSR format.
        """
        n_vertices = vertices.shape[0]
        weights = sp.coo_matrix((n_vertices, n_vertices), dtype=np.float64).tocsr()

        for tri in faces:
            v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
            e0 = v1 - v2
            e1 = v2 - v0
            e2 = v0 - v1

            # Compute norms of cross products for angle calculation
            cross0 = np.cross(e1, -e2)
            cross1 = np.cross(e2, -e0)
            cross2 = np.cross(e0, -e1)

            norm0 = np.linalg.norm(cross0)
            norm1 = np.linalg.norm(cross1)
            norm2 = np.linalg.norm(cross2)

            cot0 = np.dot(e1, -e2) / norm0 if norm0 > 1e-12 else 0.0
            cot1 = np.dot(e2, -e0) / norm1 if norm1 > 1e-12 else 0.0
            cot2 = np.dot(e0, -e1) / norm2 if norm2 > 1e-12 else 0.0

            # Accumulate symmetric cotangent weights
            weights[tri[0], tri[1]] += cot2 / 2
            weights[tri[1], tri[0]] += cot2 / 2

            weights[tri[1], tri[2]] += cot0 / 2
            weights[tri[2], tri[1]] += cot0 / 2

            weights[tri[2], tri[0]] += cot1 / 2
            weights[tri[0], tri[2]] += cot1 / 2

        weights = weights.tocsr()
        diagonal = weights.sum(axis=1).A1
        diagonal[diagonal < 1e-12] = 1e-12  # avoid singularity
        laplacian = sp.diags(diagonal) - weights
        return laplacian

    def map_to_flat(self, ap: np.ndarray, pd: np.ndarray, 
                    extent: tuple[float, float], 
                    origin: tuple[float, float], 
                    z_level: float) -> "Surface":
        """
        Reposition vertices onto a flat 2D grid using two per-vertex metrics.

        Parameters
        ----------
        ap : np.ndarray
            AP-coordinate metric per vertex, in [0,1].
        pd : np.ndarray
            PD-coordinate metric per vertex, in [0,1].
        extent : tuple of float
            The full size of the target plane in the AP and PD directions,
            e.g. (AP_extent, PD_extent).
        origin : tuple of float
            The offset of the plane origin in the AP and PD directions.
        z_level : float
            Constant z-coordinate to assign to all vertices.

        Returns
        -------
        Surface
            A new Surface instance whose mesh has been flattened.
        """
        # Copy mesh so original isn’t mutated
        new_mesh = self.mesh.copy()

        # Map AP → x, PD → y, constant z
        new_mesh.points[:, 0] = -ap * float(extent[0]) - float(origin[0])
        new_mesh.points[:, 1] = pd  * float(extent[1]) - float(origin[1])
        new_mesh.points[:, 2] = float(z_level)

        # Preserve metadata
        return Surface(new_mesh, metadata=self.metadata.copy())

