#!/usr/bin/env python3

#####
# Script with general stuff
#####


import os
from vef_scripts.vef_cohort import get_case_directories

input_dir = ""
input_suffix = "_SSA_surf.vtk"
n_cores = 4
input_fnames = sorted([f for f in os.listdir(input_dir) if f.endswith(input_suffix)])

cohort_dir = ""
os.makedirs(cohort_dir, exist_ok=True) #Create in case it does not exist.


exclude = []
case_directories = get_case_directories(cohort_dir, exclude=exclude, required="mesh", suffix="_input", cohort_relative=True)

def id_parser(s):
    return s.replace(input_suffix, "") #Extract the id from filename


hierarchy = {
    "PT"  : {"parent" : None,  "children" : {"LPA"}},
    "LPA" : {"parent" : "PT",  "children" : {"RPA"}},
    "RPA" : {"parent" : "LPA", "children" : []},
}
#

cl_params = {
    "metadata" : { "type" : "centerline", "version" : "0.0.1"},
    "params_domain" : {"method" : "flux"}, #This means use the default params params for flux method in centerline domain extraction method.
    "params_path"   : { "mode"              : "j2o",
                        "reverse"           : False,
                        "adjacency_factor"  : 0.25,
                        "pass_pointid"      : True},
    "n_knots"           : 10,
    "curvature_penalty" : 1,
    "graft_rate"        : 0.5,
    "force_extremes"    : True,
    #"LPA" : {},
    "RPA" : {"n_knots" : 5}
}
#

ec_params = {
    "metadata" : { "type" : "encoding", "version" : "0.0.1"},
    "method"            : "decoupling",
    "tau_knots"         : 15,
    "theta_knots"       : 15,
    "laplacian_penalty" : 1e-2,
    "insertion"         : 1.0,
    #"LPA" : {},
    "RPA" : {"tau_knots" : 5, "theta_knots" : 7}
}
#

align_params = {
    "metadata" : { "type" : "alignment", "version" : "0.0.1"},
    "alignment_method" : "procrustes",
    "alignment_params" : None,
    "n_iters"          : 3,
    "reference_id"     : 0
}
#
