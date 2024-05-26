###
### Script to encode all geometries.
###

import os
from multiprocessing import Pool

import numpy as np
import pyvista as pv

from vef_scripts import compute_centerline, encode

from configuration import (n_cores,
                           cohort_dir,
                           case_directories,
                           cl_params,
                           ec_params)


def compute_centerline_and_encoding(case_name):

    case_dir = os.path.join(cohort_dir, case_name)

    #Compute the centerline if it does not exist.
    if not os.path.exists(os.path.join(case_dir, 'Centerline', 'centerline.vtm')):
        compute_centerline(case_dir,
                        params=cl_params,
                        binary=True,
                        overwrite=True,
                        force=False,
                        debug=False)

    #Compute the encoding if it does not exist.
    if not os.path.exists(os.path.join(case_dir, 'Encoding', 'encoding.vtm')):
        encode(case_dir,
               params=ec_params,
               binary=True,
               overwrite=True,
               debug=False)
#


def main():

    print(f"Running encoding in {n_cores} cores")
    pool = Pool(n_cores)
    pool.map(compute_centerline_and_encoding, case_directories)



if __name__ == '__main__':
    main()