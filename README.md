# Neural Pyro Pressure Prediction
### A full data pipeline and neural training setup to enhance the pressure solve in a Houdini pyro solver.

This is a research project started at Untold Studios in London back in 2022 where we deemed it not worth further pursuing, but with more dedicated time in the current year, the outcome was improved to a point where results are feasible. 
The main development which made it reasonable was the addition of the H20 **ONNX SOP** which allows the evaluation of a model within Houdini without exporting the data for each frame. Previously, this was a bottleneck and would have required a custom C++ plugin to handle I/O of the data not through python or be slowed down by disk writing speeds.

## Requirements
`environment.yml` file is included in root directory and is to be installed with any Conda install. 
The `environment_wsl.yml` is for Windows Subsystem for Linux, where I made it run on Ubuntu 22.04 (wsl2)

## Requirements (GPU)
