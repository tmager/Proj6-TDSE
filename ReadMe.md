# The Time-Dependent Schrodinger Equation
###
#### Computational Physics, Tufts University, Spring 2015


### Overview


### Usage

This solver was written using and has been tested for Python 2.7.x with Numpy 1.8
and Scipy 

The simple finite-difference solver in `ObviousFiniteDifference/` can be run using

`python FiniteDifferenceWrapper.py TIMESTEPS STEP\_INTERVAL GRID\_POINTS
   BOUNDARY\_CONDITIONS`

where `TIMESTEPS` is the number of timesteps to be run, `STEP_INTERVAL` is the
length of each timestep, `GRID_POINTS` is the number of points in space that the
wave is sampled at, and `BOUNDARY_CONDITIONS` is a value, either `zero` or
`periodic`, which describes how the waveform acts at the edges of the range of
positions being used.

