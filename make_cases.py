#!/usr/bin/env python3

import os
import sys
from multiprocessing import Pool


import numpy as np
import pyvista as pv
import vascular_encoding_framework as vef
from vef_scripts import make_case

from configuration import (input_dir,
                           cohort_dir,
                           id_parser,
                           input_fnames,
                           hierarchy,
                           cl_params,
                           ec_params,
                           n_cores)


def update_ids(vmesh):
    """
    This sets the boundaries ids as PT, LPA and RPA according to the following logic:
    1) The boundary whose center have smaller value in the x coordinate is RPA
    2) From the boundaries left, that whose center have smaller value in the y coordinate is PT
    3) The boundary left is the LPA.

    Then sets the hierarchy:
    PT - LPA
    LPA - RPA

    Arguments
    ---------
        vmesh : vef:VascularMesh
    """

    ids = vmesh.boundaries.enumerate()
    i = ids[np.argmin([vmesh.boundaries[j].center[0] for j in ids])]
    vmesh.boundaries.change_node_id(old_id=i, new_id='RPA')

    ids.remove(i)
    i = ids[np.argmin([vmesh.boundaries[j].center[1] for j in ids])]
    vmesh.boundaries.change_node_id(old_id=i, new_id='PT')

    ids.remove(i)
    i = ids[0]
    vmesh.boundaries.change_node_id(old_id=i, new_id='LPA')
#

def set_v1(vmesh):
    """
    Given a vascular Mesh objects with Boundaries with ids PT and RPA. This function
    sets the initial angular origin (i.e. v1 vector) as the projection of the RPA center
    on the PT plane.

    Arguments
    ---------

        vmesh : VascularMesh

    """

    v1 = vmesh.boundaries['RPA'].center
    vmesh.boundaries['PT'].v1 = v1
#

def case_maker(fname):

    #Build VascularMesh from a file
    inp_mesh_fname = os.path.join(input_dir, fname)
    print(f"Working with {inp_mesh_fname}")
    inp_mesh = pv.read(inp_mesh_fname)
    vmesh = vef.VascularMesh(inp_mesh)

    #Setting the boundary ids according to the "heuristic"
    update_ids(vmesh=vmesh)

    #Set v1 as the projection of RPA otulet center onto the PT boundary plane
    set_v1(vmesh=vmesh)

    #Setting an arbtrary but fixed hierarchy"
    vmesh.set_boundary_data(hierarchy)

    #Finally let us make the case with the vmesh already prepared.
    make_case(case_dir=os.path.join(cohort_dir, id_parser(fname)),
              mesh_fname=None,
              vmesh=vmesh,
              show_boundaries=False,
              overwrite=True,
              cl_params=cl_params,
              ec_params=ec_params)

    print(f"........{inp_mesh_fname} done!")
#

def main():

    print(f"Running case creation in {n_cores} cores")
    pool = Pool(n_cores)
    pool.map(case_maker, input_fnames)
#


if __name__ == '__main__':
    main()