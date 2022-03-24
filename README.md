# Low-Discrepancy Action Selection (LDAS) for Reinforcement Learning

# Project Purpose

The purpose of this project is to create a unique method of action selection for a reinforcement learning agent. The method will entail selecting actions during the exploration phase in a low discrepancy manner. This will fill the state-action space in such a way that every new action selected will be as dissimilar from previous actions as possible. This will force the agent to try actions that is has not seen before when it explores, akin to a without replacement form of selecting.  

# Project Utility

During the exploration phase the agent will see more of its actions more quickly than say, uniform selection. Since this is the case, it should increase its efficiency during the learning phase. The project has been tested on various reinforcement learning environments. A conclusion drawn from that is that the utility of the project is minimized when the state-action space is very large. The reason is that if the state-action space is large, say thousands or tens of thousands of dimensions, then all actions selected will likely be dissimilar to previous actions regardless of selection method. In cases like this, LDAS is not strongly recommended. However, in cases of a more moderate state-action space, LDAS is highly recommended as a possible selection method.

# How to get started with the project

LDAS can be used just like uniform selection or Ornstein-Uhlenbeck selection. There are several files of code, some from R and some from python. We will detail what each of the files is used for.

# LDAS functions.R

This is the core code for the method in R. It will be detailed here. going in order of the code as it appears in the file. There is also a core code for the method in python, that will not be detailed as it follows the same methods detailed here with different python specific syntax. Before detailing the code specifically, the overall idea of the code will be explained. 

As a new action is generated it is compared to previous actions. If it is the first action it is just placed pseudo-randomly. If it is the second action and onward it is compared to all previous actions. The closest previous action is determined and the new action is moved away from the closest previous action (only movement in the action dimension/dimensions is allowed). Once it has an updated position the calculation is done again to find a closest previous action, and the process is repeated. Since old actions move the current action away the process would overrepresent the extreme values. To correct for this phantom point are created only during the iterations that are just beyond the boundaries in the action dimension. As the current action approaches a boundary then it will get nearer to a phantom point, and if the phantom point becomes the closest previous action, it will move the current action away. 

There are three ways to end the process. One is if the current action is equal to or greater than a boundary value. Then the current action is placed on the boundary and this is considered to be its final position. The second method to end the process uses a decay rate applied to the movement. When a movement is beneath a threshold then the current position is considered to be the final position. The last method to end the process is to check if the current action is similar to the position of the action five iterations ago. If the distance between these two actions is less than a threshold then the current position is considered to be the final position. The reasoning behind this is that if the distance is small then the new action must be bouncing back and forth instead of making imperative movements.       

### pause

This simply is used to pause a process in R. It will ask the user to hit enter to continue with every iteration. This can be useful when graphing the simulations, but only for few dimensions since it is harder to visualize higher dimensions. With how the code is set up this should only be run if plotstuff is TRUE (meaning a plot will be generated).

### get_closest_point

This function will sweep through a matrix given to it to find the distance from a current point to the nearest previous point. It will also return the closest previous point as well as the distance to that point.

### get_norm

A simple function to find the norm of an input argument.

### get_phantom_matrix

This function creates the phantom points that will be just beyond the boundaries of the action dimension to ensure the extreme values are not overrepresented. It takes an input of the current state and action (these can be vectors depending on the number of dimensions of each) and a small value epsilon. This epsilon will be the distance beyond the boundary that the phantom point goes. The output of this function will be the state space with these phantom points appended.

### get_ld_action

This is the core of the core code. It shows the details of the code and how it pulls in all of these other functions to generate a low discrepancy action. There is a component called ldas_config the must be specified when this function is called. Here is an example of the inputs and good default values for them.    

ldas_config <- list(
      "state_dim" = d_s,
      "action_dim" = d_a,
      "learning_rate" = .001,
      "learning_rate_decay" = .95,
      "num_recent_actions" = 5,
      "num_candidate_actions" = 10,
      "rel_threshold" = .001
    )

### estimate_discrepancy_CV

This function is used for testing purposes. It calculates the discrepancy using in a method other than the classic method. This method uses the idea of the coefficient of variation (CV). In our experiments it proved to be more suitable than the classic method of discrepancy calculation, and aligned more closely with the visual results.

### estimate_discrepancy_classic
 
This function is also used for testing purposes. It calculates the discrepancy as well, but in the classic manner rather the CV manner.

### simulate_ldas_phantom

This function uses the method to create a state-action space. It is useful for visualizing changes to the method and how different parameters affect the discrepancy. 

# Epsilon_LDAS.py

This is the core code created in python. It is adapted to the EpsilonLDASGreedyPolicy (ExplorationPolicy) from the Gym library. Whenever the agent explores it will use the LDAS method rather than the default of Uniform selection. The details of this code are similar to the ones described in LDAS functions.R except for python specific syntax.   

# exploration_lunar.py

This is used to generate data from the LunarLanderContinuous-v2 environment. It has three main chunks of code to generate the data using a uniform selection method, an Ornstein-Uhlenbeck selection method, and the LDAS method. The output of the code is a csv file of the state-action spaces. The parameters of the code could be altered if desired. 

# exploration_mcar.py

This is used to generate data from the MountainCarContinuous-v0 environment. It has three main chunks of code to generate the data using a uniform selection method, an Ornstein-Uhlenbeck selection method, and the LDAS method. The output of the code is a csv file of the state-action spaces. The parameters of the code could be altered if desired. 

# exploration_rcar.py

This is used to generate data from the CarRacing-v0 environment. It has three main chunks of code to generate the data using a uniform selection method, an Ornstein-Uhlenbeck selection method, and the LDAS method. The output of the code is a csv file of the state-action spaces. The parameters of the code could be altered if desired.

# gym_env_disc_analysis.R

This code is used to calculate discrepancy based off of the csv files created from the different environments. It can also create visuals for analysis.

# make_figures.R

This code is used to make several figures related to discrepancy. Some are just comparisons between selection using the Halton sequence (a known low discrepancy sequence) and a pseudo-random form of selection. Other useful figures it creates is multiple dimension comparison between the LDAS method and Uniform. This has been done for three state and three action dimensions and in all cases LDAS has a superior discrepancy. However, it does take a while to generate the plot for higher dimension.
