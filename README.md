# CFMAR Lab - Neural Nets for American Option Pricing


These files are used to price Bermudan options (discrete time steps), which, when given enough time steps, can be used to approximate American options. 

The folder "tf" contains the neural network architecture for various options in Tensorflow. The folder "torch" is in-progress; we are working to switch the code from Tensoflow to PyTorch to see if there is a change in price. The files under "torch" currently do not run without error. You can change the parameters (and dimensions) of the puts and calls using bput1 and mcall2 in main.py.

The branch "stage1and2_stuff" is the most straightforward version of the code. Stage 1 is a sequential neural network (establishing a new neural network for each time step) that follows the Longstaff-Schwartz Algorithm (LSM) and Stage 2 (nn_aggregate_neo.py) is an aggregated neural network (1 neural network that runs through each step of the option). Stage 1 only takes in the locations as x inputs (an array of x-dimensional locations) and Stage 2 takes in both the locations and the time (y).

The branches "main" and "Anna" are from stage three of the code. This stage inludes reinforcenment learning. It uses 10000 simulated validation paths which are used as feedback for the reinforcement learning. This code is more complicated and we plan to use only Stage 1 and Stage 2 for the report.
