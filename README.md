# Low-Discrepancy Action Selection (LDAS) for Reinforcement Learning

# Project Purpose

The purpose of this project is to create a unique method of action selection for a reinfocement learning agent. The method will entail selecting actions during the exploration phase in a low discrepancy manner. This will fill the state-action space in such a way that every new action selected will be as dissimilar from previous actions as possible. This will force the agent to try actions that is has not seen before when it explores, akin to a without replacement form of selecting.  

# Project Utility

During the exploration phase the agent will see more of it's actions more quickly then say, uniform selection. Since this is the case it should increase its efficiency during the learning phase. The project has been tested on various reinfocement learning environments. A conclusion drawn from that is that the utility of the project is minimized when the state-action space is very large. The reason is that if the state-action space is large, say thousands or tens of thousands of dimensions, then all actions selected will likely be dissimilar to previous actions regarless of selection method. In cases like this, LDAS is not strongly recommended. However, in cases of a more moderate state-action space, LDAS is highly recommended as a possible selection method.

# How to get started with the project

LDAS can be used just like uniform selection or Ornstein-Uhlenbeck selection. There are several files of code, some from R and some from python. We will detail what each of the files is used for.

# LDAS functions.R

This is the core code for the method in R. It will be detailed here. going in order of the code as it appears in the file. There is also a core code for the method in python, that will not be detailed as it follows the same methods detailed here with different python specific syntax. Before detailing the code specifically, the overall idea of the code will be explained. As a new action is generated it is compared to previous actions. If it is the first action it is just placed psuedo-randomly. If it is the second action and onward it is compared to all previous actions. The closest previous action is determined and the new action is moved away from the closest previous action (only movement in the acion dimension/dimensions is allowed). Once it has an updated position the calculation is done again to find a closest previouos action, and the process is repeated. To end the process a decay rate is applyied to the movement. When a movement is beneath a threshold then the current position is considered to be the final position. The other method to end the process is to check if the current action is similar to the postion of the action five iterations ago. If the the distance between these two actions is less then a threshold then the current position is considred to be the final position. The reasoning behind this is that if the distance is small then the new action must be bouncing back and forth instead of making imparative movements.       

### pause

This simply is used to pause a process in R. It will ask the user to hit enter to continue with every iteration. This can be useful when graphing the simulations, but only for few dimensions since it is harder to visualize higher dimensions. With how the code is set up this should only be run if plotstuff is TRUE (meaning a plot will be generated).

### get_closest_point

This function will sweep through a matrix given to it to find the distance from a current point to the nearest previous point. It will also return the closest previous point as well as the distance to that point.

### get_norm

A simple function to find the norm of an input argument.

### get_phantom_matrix

This is a unique 


# exploration_lunar.py

This is used to generate data from the LunarLanderContinuous-v2 enviornment. It has three main chunks of code to genereate the data using a uniform selection method, an Ornstein-Uhlenbeck selection method, and the LDAS method. The output of the code is a csv file of the state-action spaces. The parameters of the code could be altered if desired. 

# exploration_mcar.py

This is used to generate data from the MountainCarContinuous-v0 enviornment. It has three main chunks of code to genereate the data using a uniform selection method, an Ornstein-Uhlenbeck selection method, and the LDAS method. The output of the code is a csv file of the state-action spaces. The parameters of the code could be altered if desired. 

# exploration_rcar.py

This is used to generate data from the CarRacing-v0 enviornment. It has three main chunks of code to genereate the data using a uniform selection method, an Ornstein-Uhlenbeck selection method, and the LDAS method. The output of the code is a csv file of the state-action spaces. The parameters of the code could be altered if desired.

Where users can get help with your project
Who maintains and contributes to the project


