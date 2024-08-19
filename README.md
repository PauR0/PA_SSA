SSA on Pulmonary Arteries
-------------------------

The configuration.py script can be used to set the paths according to your specific system and choose the parameter configuration that will be used during the SSA computation.

The steps to carry the study out are as follows:

1) Ensure all the vtk files contain a PolyData. This is achieved by running the convert_input_to_polydata.py script. **WARNING**: This script will overwrite the vtk files at `configuration.input_dir` ! Make sure you have a copy, just in case...
```sh
python convert_input_to_polydata.py
```

2) Set the variable `configuration.cohort_dir` (it is created if it doesn't exist) to store all the VEF cases. Then run the make_cases script.
```sh
python make_cases.py
```

3) Now, to compute encoding of all the cases with the pararmeter configuration from the __configuration.py__ module, run the encode_all.py script. This is likely take hours depending on the amount of cases. This proces can be run in parallel, since there are no dependencies on the encoding process. This results in a direct speed up depending on the number of cores available. To control the parallelism number of processes, set the variable `configuration.n_cores` as you wish.
```sh
python encode_all.py
```

4) With all the cases in `configuration.cohort_dir` having its vascular encoding stored in the subdirectory _cases/encoding_ now it time to align the shapes. To do so, run the align.py script:
```sh
python align.py
```

5) Finally, to run the actual SSA to assess the different proposed scenarios, run the SSA.py script. However, I'd suggest to open the SSA.py module, and comment the lines to run the desired scenrio at script part, i.e. below the part : `if __name__ == "__main__"`
```sh
python SSA.py
```
