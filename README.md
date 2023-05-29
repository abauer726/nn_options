# CFMAR Lab - Neural Nets for American Option Pricing

These files are used to price Bermudan options (discrete time steps), with the ultimate goal of pricing American options (not included yet).

## Folders
The folder **tf** contains the neural network architecture for various options in Tensorflow. The folder **torch** contains files that currently do not run without error. 


## Branches
We've combined all of the code from all previous branches into the **main** branch. You should start off by creating a new branch for future work based on the the main branch. **June2023** is a duplicate of the main branch and can be used as a reference point for future versions. **stage1and2_stuff** is a branch that we used before compiling code. If any portion of the code in main is giving errors, try switching to this branch. 


## Testing
The code in main.py can be used to test various neural networks hyperparameters and option models for each of the three stages.  

You can change the parameters (and dimensions) of the puts and calls directly in main.py, or choose from the list of puts and calls in model_parameters.py. The definitions in model_parameters.py are the same models tested in the paper. 

### Stage 1
Stage 1 is a sequential neural network (establishing a new neural network for each time step) that follows the Longstaff-Schwartz Algorithm (LSM). Stage 1 only takes in the locations as x inputs (an array of x-dimensional locations).

### Stage 2
Stage 2 (nn_aggregate_neo.py) is an aggregated neural network (1 neural network that runs through each step of the option). Stage 2 takes in both the locations and the time (y) as x inputs.

### Stage 3
This stage inludes reinforcenment learning. It uses 10000 simulated validation paths which are used as feedback for the reinforcement learning. This code is more complicated and we only include Stage 1 and 2 in the paper.






