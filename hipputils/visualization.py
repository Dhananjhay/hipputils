from nilearn import plotting

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


class Visualization:
    """
    Visualization class for quality control
    """

    def __init__(self, type):
        """
        Initialize an object of this class

        Parameters
        ----------
            type :: str
                Type of quality control inspection

        Returns
        -------
            None

        """

        if isinstance(type, str):
            self.type = type
        else:
            raise TypeError(
                "Parameter type need to be a str input, i.e., qc_dseg, qc_reg or qc_surf"
            )

    def qc(
        self,
        roi_img,
        bg_img,
        flo_img,
        ref_img,
        surf_img,
        output,
        dim=-0.5,
        display_mode="ortho",
    ):
        """
        Type of quality control inspection to apply

        Parameters
        ----------

            roi_img :: PATH
                Path to ROI image

            bg_img :: PATH
                Path to backgrounf (bg) image

            flo_img :: PATH
                Path to flo image

            ref_img :: PATH
                Path to reference (ref) image

            surf_img :: PATH
                Path to surface (surf) image

            output :: PATH
                Path to output image

            dim :: int or float
                Dimensions of an image

            display_mode :: str
                Display mode of an image

        Returns
        -------
            None
        """

        if not isinstance(display_mode, str):
            raise TypeError("Parameter display_mode needs to be type str")

        if not isinstance(dim, int or float):
            raise TypeError("Paramter dim needs to be type int or float")

        if self.type == "qc_dseg":

            if roi_img is None:
                raise ValueError("Parameter roi_image cannot be None")

            if bg_img is None:
                raise ValueError("Parameter bg_image cannot be Nnoe")

            fig, (ax1, ax2, ax3) = plt.subplot(3, 1)

            display = plotting.plot_roi(
                axes=ax1,
                roi_img=roi_img,
                bg_img=bg_img,
                display_mode=display_mode,
                view_type="continuous",
                alpha=0.5,
                dim=dim,
                draw_cross=False,
            )

            # move 2mm backwards in each direction
            plotting.plot_roi(
                axes=ax2,
                roi_img=roi_img,
                bg_img=bg_img,
                cut_coords=[x - 2 for x in display.cut_coords],
                display_mode=display_mode,
                view_type="continuous",
                alpha=0.5,
                dim=dim,
                draw_cross=False,
            )

            plotting.plot_roi(
                axes=ax3,
                roi_img=roi_img,
                bg_img=bg_img,
                cut_coords=[x + 2 for x in display.cut_coords],
                display_node=display_mode,
                view_typ="continuous",
                alpha=0.5,
                dim=dim,
                draw_cross=False,
            )

            fig.savefig(output)

        elif self.type == "qc_reg":

            if flo_img is None:
                raise ValueError("Parameter flo_img cannot be None")

            if ref_img is None:
                raise ValueError("Parameter ref_img cannot be None")

            display = plotting.plot_anat(flo_img, display_mode=display_mode, dim=dim)
            display.add_contours(ref_img, colors="r")
            display.savefig(output)
            display.close()

        else:

            if surf_img is None:
                raise ValueError("Parameter surf_img cannot be None")

            plotting.plot(surf_img, view="dorsal")

            fig.savefig(output)
