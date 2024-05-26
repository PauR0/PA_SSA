SSA on Pulmonary Arteries
-------------------------

The configuration.py script can be used to set the paths according to your specific system and choose the parameter configuration that will be used during the SSA computation.

The steps to carry the study out are as follows:

1) Ensure all the vtk files contain a PolyData. This is achieved by running the convert_input_to_polydata.py script. WARNING: This script will overwrite the vtk files at ```configuration.input_dir``` ! Make sure you have a copy, just in case...
```bash
python convert_input_to_polydata.py
```

2) Use all the