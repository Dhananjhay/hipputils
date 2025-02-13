import nibabel as nib
import numpy as np
import pyvista as pv
import pygeodesic.geodesic as geodesic
from scipy.ndimage import binary_dilation
from lib.utils import setup_logger

log_file = snakemake.log[0] if snakemake.log else None
logger = setup_logger(log_file)

class IsoSurface:
    """
    Generate iso surface
    """

    def __init__(self, coords, nan_mask, sink_mask, src_mask, clean_method):

        # Load the coords image
        self.coords_img = nib.load(coords)
        self.coords = self.coords_img.get_fdata()

        # Load the nan mask
        self.nan_mask = nib.load(nan_mask).get_fdata()

        # Load the sink mask
        self.sink_mask = nib.load(sink_mask).get_fdata()

        # Load the src mask
        self.src_mask = nib.load(src_mask).get_fdata() 

        self.clean_method = clean_method

        self.affine = self.coords_img.affine

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

    
    def create_grid(self):
        """
        Create a PyVista grid

        Parameters
        ----------
            coords :: numpy.ndarray
                From coordinates NIfTI file to Numpy array 

        Returns
        -------
            tfm_grid :: pyvista.ImageData
                A structured PyVista ImageData grid created from the provided coordinates.
        """

        # create a PyVista grid
        grid = pv.ImageData(
            dimensions=np.array(self.coords.shape) + 1,
            spacing=(1,1,1),
            origin=(0,0,0)
        )

        self.coords[self.nan_mask == 1] = np.nan
        self.coords[self.src_mask == 1] = -0.1
        self.coords[self.sink_mask == 1] = 1.1

        # we also need to use a nan mask for the voxels where src and sink meet directly
        # (since this is another false boundary)..
        if self.clean_method == "cleanAK":

            logger.info("Using cleanAK method")
            src_sink_nan_mask = self.get_adjacent_voxels(self.sink_mask, self.src_mask)
            self.coords[src_sink_nan_mask == 1] = np.nan

        # Add the scalar field
        grid.cell_data["values"] = self.coords.flatten(order="F")
        grid = grid.cells_to_point("values")
        tfm_grid = grid.transform(self.affine, inplace=False)

        return tfm_grid

    
    def generate(self, threshold, hole_fill_radius, decimate_opts, coords_epsilon, morph_openclose_dist, output_surf_gii):
        """
        Generate isosurface

        Parameters
        ----------
            

        Return
        ------
            None
        """
        logger.info("Generating isosurface")
        # create a PyVista grid
        tfm_grid = self.create_grid()

        # the contour function produces the isosurface
        surface = tfm_grid.contour(
            [threshold], 
            method="contour", 
            compute_scalars=True
            ) 
        
        logger.info(surface)

        # remove the nan-valued vertices - this isn't handled in PolyData.clean()
        logger.info("Removing nan-valued vertices")
        surface = self.remove_nan_vertices(surface)
        logger.info(surface)

        logger.info("Cleaning surface")
        surface = surface.clean(point_merging=False)
        logger.info(surface)

        logger.info("Extracting largest connected component")
        surface = surface.extract_largest()
        logger.info(surface)

        """
        #still experimenting with this..

        logger.info("Applying surface smoothing")  
        surface = surface.smooth_taubin(#normalize_coordinates=True,
                                        #boundary_smoothing=True,
                                        #feature_smoothing=True,
                                        n_iter=20,
                                        pass_band=0.1)  #n_iter=30, pass_band=0.1)
        logger.info(surface)
        """

        logger.info(f"Filling holes up to radius {hole_fill_radius}")
        surface = surface.fill_holes(hole_fill_radius)
        logger.info(surface)

        # reduce # of vertices with decimation
        logger.info(f"Decimating surface with {decimate_opts}")
        surface = surface.decimate_pro(**decimate_opts)
        logger.info(surface)

        if self.clean_method == "cleanJD":

            ## JD clean - instead of trimming surfaces with a nan mask, we
            # keep vertices that overlap with good coord values. We then apply
            # some surface-based morphological opening and closing to keep
            # vertices along holes in the dg
            logger.info("Using cleanJD method")

            # this is equivalent to wb_command -volume-to-surface-mapping -enclosing
            # apply inverse affine to surface to get back to matrix space
            xfm_surface = self.apply_affine_transform(surface, self.affine, inverse=True)

            V = np.round(xfm_surface.points).astype("int") - 1
            # sample coords
            coord_at_V = self.coords[V[:, 0], V[:, 1], V[:, 2]]

            # keep vertices that are in a nice coordinate range
            epsilon = coords_epsilon
            good_v = np.where((coord_at_V < (1 - epsilon)) & (coord_at_V > epsilon))[0]

            geoalg = geodesic.PyGeodesicAlgorithmExact(
                surface.points, surface.faces.reshape((-1, 4))[:, 1:4]
            )
            maxdist, _ = geoalg.geodesicDistances(good_v, None)
            bad_v = np.where(maxdist > morph_openclose_dist)[0]
            maxdist, _ = geoalg.geodesicDistances(bad_v, None)
            bad_v = np.where(maxdist < morph_openclose_dist)[0]

            surface.points[bad_v, :] = np.nan

            logger.info("Removing nan-valued vertices")
            surface = self.remove_nan_vertices(surface)
            logger.info(surface)

            logger.info("Extracting largest connected component")
            surface = surface.extract_largest()
            logger.info(surface)

        # Save the final mesh
        self.write_surface_to_gifti(surface, output_surf_gii)
                        


        




    

