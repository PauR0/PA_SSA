###
### Script to align all VascularEncodings and save them as case/Encoding/encoding_aligned.vtm
###

import os
from multiprocessing import Pool

import pyvista as pv

from vef_scripts import align_encodings
from vef_scripts.case_io import load_vascular_encoding, save_vascular_encoding

from configuration import (cohort_dir,
                           n_cores,
                           case_directories,
                           align_params,
                           exclude)


def align_to_junction(case):
    """
    Translate the VascularEncoding such that the junction between LPA and RPA is set to the origin.

    The junction is estimated as the mid-point between the RPA inlet, and its projection on the LPA
    branch.

    The translated VascularEncoding will be saved using the vef convention at Encoding/encoding_aligned_junction.vtm

    """

    case_dir = os.path.join(cohort_dir, case)

    vsc_enc = load_vascular_encoding(case_dir, suffix="")

    rpa_origin = vsc_enc['RPA'].vcs_to_cartesian(tau=0.0, theta=0.0, rho=0.0).ravel()
    projection = vsc_enc['LPA'].centerline.get_projection_point(rpa_origin)
    midpoint = (rpa_origin + projection)/2

    vsc_enc.translate(-midpoint)

    save_vascular_encoding(case_dir, vsc_enc, suffix="_aligned_junction", binary=True, overwrite=True)

    #p = pv.Plotter()
    #p.add_mesh(vsc_enc.to_multiblock(), multi_colors=True, opacity=0.5)
    #p.add_mesh(rpa_origin, color='k', render_points_as_spheres=True, point_size=15)
    #p.add_mesh(projection, color='g', render_points_as_spheres=True, point_size=15)
    #p.add_mesh(midpoint, color='b', render_points_as_spheres=True, point_size=15)
    #p.show()
#


def main():

    print(f"Aligning using GPA over the centerline")
    align_encodings(cohort_dir, params=align_params, exclude=exclude, overwrite=True)

    with Pool(n_cores) as pool:
        pool.map(align_to_junction, case_directories)


if __name__ == '__main__':
    main()