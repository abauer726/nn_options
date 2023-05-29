# CFMAR Lab - Neural Nets for American Option Pricing

These files are used to price Bermudan options (discrete time steps), with the ultimate goal of pricing American options (not included yet).

The folder "tf" contains the neural network architecture for various options in Tensorflow. The folder "torch" contains files that currently do not run without error. 

You can change the parameters (and dimensions) of the puts and calls using bput1 and mcall2 in main.py, or choose from the list of puts and calls in model_parameters.py.
## Branches
## Testing
### Stage 1
The branch "stage1and2_stuff" is the most straightforward version of the code. Stage 1 is a sequential neural network (establishing a new neural network for each time step) that follows the Longstaff-Schwartz Algorithm (LSM) and Stage 2 (nn_aggregate_neo.py) is an aggregated neural network (1 neural network that runs through each step of the option). Stage 1 only takes in the locations as x inputs (an array of x-dimensional locations) and Stage 2 takes in both the locations and the time (y).


### Stage 2


### Stage 3






The branches "main" and "Anna" are from stage three of the code. This stage inludes reinforcenment learning. It uses 10000 simulated validation paths which are used as feedback for the reinforcement learning. This code is more complicated and we only include Stage 1 and 2 in the paper.

Try not to merge stage1and2_stuff with the other branches - they have the exact same titles for files, but different contents. Stage 3 code should be separate from stages 1 and 2.
