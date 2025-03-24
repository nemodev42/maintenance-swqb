# maitenance-swqb
Reinforcment Learning with Foirier/Polynomial State Weighted Q Basis for Homogeus Multi Component Maitenance
By Joseph Wittrock

Enviroment/Training code adapted from:
https://pytorch.org/rl/stable/tutorials/torchrl_envs.html
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html


Note that SWQB methods are called by a legacy name AK methods (action kernel) in some of the scripts.



    FILE                      |        DESCRIPTION
----------------------------------------------------------------------
maitenance_util_cost_env.py   |  This script has classes for the maitenance enviroment which can be run in paralell on GPU
deep_q_target_training.py     |  This script has classes for training a policy with deep q and swqb methods in the maitenance enviroment accelerated by GPU
training_example.ipynb        |  This notebook has an example training usage
hyperparameter_tuning.ipynb   |  This notebook tests for optimal hyperparameter of different models
score_cache*.txt              |  Hyperparameter test results
hyperparameter_plotting.ipynb |  This notebook plots the hyperparameter test results
pareto_fronts.ipynb           |  This notebook plots the optimal 
torch_tests.ipynb             |  This notebook tests the advantage of GPU acceleration on matrix multiplication






------------------------
-- Summary of Methods --
------------------------

The enviroment is composed of n components with m condition states. Each component degrades each step according to its degredation transition matrix. 
The actions to the enviroment are 0 for do nothing, or 1 for repair.
Only k components can be repaired at a time, and the repair cost is proportional to the number of components repaired.
If a component is in a failed state, it will have a failure cost no matter what action is taken. (Though I want to change how this works for multiperiod adaptation)
The goal is to maximize the reward over a fixed number of steps, reward is negative for repair costs and failure costs.

The enviroment is written using TorchRL and TensorDict for efficient paralell computation on a cuda enabled GPU.

The state space is reduced by considering the distrobution of the components in each condition state, rather than the condition state of each component.
This condenses the state space to m states.

To generalize the action space we search for an "equitable" policy. i.e. a component in worse condition is always repaired before a component in better condition.
This reduces the action space to k actions.

For large values of m, a traditional deep Q network would not generalize well, as there are too many output neurons.
Instead, we use a basis of continuous valued functions over [0,1], and map the action space to i \mapsto i/k for i in [0,k]. 
Note when i=0, no components are repaired, and when i=k, all components are repaired, so there are k+1 actions in total.

The degrees of approximation determine the output nodes for the neural network, then the output is dot producted with the basis functions evaluated at the precalculated action domain points.
This allows for scale free computation for increasing values of the repair constraint.