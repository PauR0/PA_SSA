
import os
from multiprocessing import Pool

import pyvista as pv

from configuration import input_dir, input_fnames, n_cores



def load_convert_and_save(fname):
    ffname = os.path.join(input_dir, fname)
    mesh = pv.read(ffname).connectivity(extraction_mode='largest').extract_surface()
    mesh.save(ffname)
#

def main():

    pool = Pool(processes=n_cores)
    pool.map(load_convert_and_save, input_fnames)
#

if __name__ == '__main__':
    main()