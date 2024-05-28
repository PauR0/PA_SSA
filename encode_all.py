###
### Script to encode all geometries.
###

import os
from multiprocessing import Pool

import numpy as np
import pyvista as pv

import vascular_encoding_framework.messages as msg
from vef_scripts import compute_centerline, encode

from configuration import (n_cores,
                           cohort_dir,
                           case_directories,
                           cl_params,
                           ec_params)


def compute_centerline_and_encoding(case_name):

    case_dir = os.path.join(cohort_dir, case_name)

    msg.computing_message(f"{case_name} centerline")
    #Compute the centerline if it does not exist.
    if not os.path.exists(os.path.join(case_dir, 'Centerline', 'centerline.vtm')):
        compute_centerline(case_dir,
                        params=cl_params,
                        binary=True,
                        overwrite=True,
                        force=False,
                        debug=False)
    msg.done_message(f"{case_name} centerline")

    msg.computing_message(f"{case_name} encoding")
    #Compute the encoding if it does not exist.
    if not os.path.exists(os.path.join(case_dir, 'Encoding', 'encoding.vtm')):
        encode(case_dir,
               params=ec_params,
               binary=True,
               overwrite=True,
               debug=False)
    msg.done_message(f"{case_name} encoding")
#


def main():

    print(f"Running encoding in {n_cores} cores")
    pool = Pool(n_cores)
    pool.map(compute_centerline_and_encoding, case_directories)



if __name__ == '__main__':
    main()