# RigidBody Function Approximation Example

This example demonstrates the use of Smoothed Particle Hydrodynamics
(RigidBody) for function approximation. Uses Cabana, and runs across all the
architechtures.

We have three parts:

1. Implement it in Cabana and run the executable from the terminal directly
2. Use additional `python` file to run the code
3. Use automan and the python file created in step 2 reproduce the entire results

# Implement it in Cabana and run the executable from the terminal directly

The main purpose of this program is to approximate the sine function using RigidBody. The process involves:

Once the code is written, we need a way to execute it and post-process the results:

1. Write the code
   1. Read the inputs.
   2. Generating particles.
   3. Create the neighbours.
   4. Approximating the sine function.
2. Execute the program
3. Post-processing


## Write the code

### Read the inputs.
### Generating particles.
### Create the neighbours.
### Approximating the sine function.

## Execute the program

Run the compiled program by:
```sh
$ cd build
$ ./examples/Problem01RigidBodyFunctionApproximation 7. 0.1 1
```

The args of the program are as follows:
```sh
./examples/Problem01RigidBodyFunctionApproximation <length> <spacing> <hdx>
```

This will generate the output files:
```sh
$ ls particles*
particles_0.h5  particles_0.xmf
```
Using the output files, we will plot our approximated `sine` function in the next section using `python`.

## Post-processing
Inorder to plot the function, we need to read the `h5` file, we use `python` to do this. Assuming we are
inside the `build` directory, or `particles_0.h5` file directory.
```python
import h5py
import matplotlib.pyplot as plt

data = h5py.File("particles_0.h5", "r")
cabana_x = data['positions'][:, 0]
cabana_sin_appr = data['pressure'][:]
cabana_sin_analytical = data['wij'][:]
plt.plot(cabana_x, cabana_sin_analytical, label="Analytical")
plt.scatter(cabana_x, cabana_sin_appr, label="RigidBody approximation")
plt.legend()
plt.show()
```




## Notes

### `ViewVectorType`

A type to define arrays on the `device`.

We declare the type as follows:

```cpp
  typedef Kokkos::View<double*>   ViewVectorType;
```

An example is:
```cpp
  ViewVectorType b_rho0_p0( "b_rho0_p0", 3 );
  ViewVectorType gravity( "gravity", 3 );
```

To initialize these device arrays, we can not directly index them, so
we create host copies as follows:

```cpp
  ViewVectorType::HostMirror h_b_rho0_p0 = Kokkos::create_mirror_view( b_rho0_p0 );
  ViewVectorType::HostMirror h_gravity = Kokkos::create_mirror_view( gravity );


  h_b_rho0_p0[0] = 1.;
  h_b_rho0_p0[1] = 1000.;
  h_b_rho0_p0[2] = 0.;

  h_gravity[0] = 0.;
  h_gravity[1] = -9.81;
  h_gravity[2] = 0.;
```

Copy the host array back to the device array by,

```cpp
  // Deep copy host views to device views.
  Kokkos::deep_copy( b_rho0_p0, h_b_rho0_p0 );
  Kokkos::deep_copy( gravity, h_gravity );
```


### Create equispaced points on device directly

We create equispaced points on a line directly on the device. Please
refer to the example for details.


# Use additional `python` file to run the code

To eliminate running it manually and post-processing it seperately, we
can use a seperate python file to execute it. This is saved in the
same folder `examples` with same name but a `python` extension.

```python
import shutil
import sys
import os
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import h5py

from utils import make_directory_if_not_exists

# =============================================
# 1. compile the code the code
# =============================================
os.system('cd build' + '&& make -j 12 ')
os.system('cd ../')

# =====================================================
# 2. copy the executable to the output directory and run
# =====================================================
# create the output directory if it doesn't exists
output_dir = sys.argv[1]
make_directory_if_not_exists(output_dir)

shutil.copy('./build/examples/Problem01RigidBodyFunctionApproximation', output_dir)

# executable args
cli_args = ' '.join(element.split("=")[1] for element in sys.argv[2:])
os.system('cd ' + output_dir + '&& ./Problem01RigidBodyFunctionApproximation ' + cli_args)

# =====================================================
# 3. post process the output file and plot
# =====================================================
cabana = h5py.File(output_dir + "/particles_0.h5", "r")

cabana_x = cabana['positions'][:, 0]
cabana_sin_appr = cabana['pressure'][:]
cabana_sin_analytical = cabana['wij'][:]

res_npz = os.path.join(output_dir, "results.npz")
np.savez(res_npz,
         x=cabana_x,
         sin_appr=cabana_sin_appr,
         sin_analytical=cabana_sin_analytical)

# only plot some of the points
step = int(len(cabana_x) / 20)
cabana_x_plot = cabana_x[::step]
cabana_sin_appr_plot = cabana_sin_appr[::step]
cabana_sin_analytical_plot = cabana_sin_analytical[::step]

plt.plot(cabana_x_plot, cabana_sin_analytical_plot, "^-", label="Analytical")
plt.scatter(cabana_x_plot, cabana_sin_appr_plot, label="RigidBody approximation")
plt.legend()
res_plot = os.path.join(output_dir, "sin_appr.pdf")
plt.savefig(res_plot)
# plt.show()
```

In this code, we explicitly copy the executable to the folder we want
to save the output, and run the executable from the python file, after
the results, we use the output files to run the post-processing, and
we save the plot and the data which are used in the plot. In order to
execute this python file, we use the following command in the command line

```sh
$ pwd
CabanaRigidBody
$ python examples/problem_01_RigidBody_function_approximation.py junk length=6.4 spacing=0.01 hdx=1.
```
remember to execute this command from the root of the package.

For the python command, the second argument, here in the example it is
`junk`, is the output folder we want to save our output data. Further,
we explicity state the arguments with names. After running, in the
output folder we see the following contents,

```sh
$ ls
particles_0.h5  particles_0.xmf  problem01sphfunctionapproximation  results.npz  sin_appr.pdf
```


# Use automan and the python file created in step 2 reproduce the entire results
Finally, for different values of spacing, or different values of
length, we want to compare our results, this is done with automan, very conviniently.


```sh
$ python automate.py Problem01RigidBodyFunctionApproximation -f
```

This would run three cases with different spacings and compare the
`sine` function, and plot it. The resulting figure can be found at
`manuscript/figures/problem_01_RigidBody_function_approximation/x_vs_sin.pdf`
