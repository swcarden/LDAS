# Low-Discrepancy Action Selection (LDAS) for Reinforcement Learning

# Project Purpose

The purpose of this project is to create a unique method of action selection for a reinfocement learning agent. The method will entail selecting actions during the exploration state in a low discrepancy manner. This will fill the state-action space in such a way that every new action selected will be as dissimilar from previous actions as possible. This will force the agent to try actions that is has not seen before when it explores, akin to a without replacement form of selecting.  

# Project Utility

During the exploration phase the agent will see more of it's actions more quickly then say, uniform selection. Since this is the case it should increase its efficiency during the learning phase. The project has been tested on various reinfocement learning environments. A conclusion drawn from that is that the utility of the project is minimized when the state-action space is very large. The reason is that if the state-action space is large, say thousands or tens of thousands of dimensions, then all actions selected will likely be dissimilar to previous actions regarless of selection method. In cases like this, LDAS is not strongly recommended. However, in cases of a more moderate state-action space LDAS is highly recommended as a possible selection method.

# How to get started with the project

LDAS can be used just like uniform selection or Ornstein-Uhlenbeck selection. 


Where users can get help with your project
Who maintains and contributes to the project


