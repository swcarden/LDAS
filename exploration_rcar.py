#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 14:37:36 2021

@author: garage-user
"""

import garage
import gym

import random
import numpy as np

import time

# Local file
from Epsilon_LDAS import *


# Gather stuff for experiment
from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.np.policies import UniformRandomPolicy
from garage.np.exploration_policies import EpsilonGreedyPolicy
from garage.np.exploration_policies import AddOrnsteinUhlenbeckNoise
from garage.np.exploration_policies.exploration_policy import ExplorationPolicy

#%% RaceCar environment initialization
env_id = 'CarRacing-v0'
gym_env = gym.make(env_id)
env = GymEnv(gym_env)

#%% RaceCar discrepancy experiment
M = 100 # Number of repetitions
N = 100

#%%% uniform
# States, Actions, Run, Timestep, Rime
rcar_uniform_results = np.zeros((N*M, 96*96*3 + 3 + 1 + 1 + 1))
exploration_policy = UniformRandomPolicy(env_spec = env.spec)

for m in range(M):
    env.reset()    
    for n in range(N):
        start = time.time()
        state = env.state        
        action = exploration_policy.get_action(state)[0]
        step_id = env.step(action).step_type                         
        end = time.time()
        elapsed_time = end - start
        rcar_uniform_results[(m-1)*N + n,:] = np.concatenate((state.flatten(), action, np.array((m+1, n+1, elapsed_time))))
        if (not((step_id == 0) | (step_id == 1))):
            env.reset()  
        print((m,n))
        
np.savetxt('data/rcar_uniform_disc.csv', rcar_uniform_results, delimiter=',', fmt='%s')


#%%% RCar LDAS discrepancy
# States, Actions, Run, Timestep, Time
rcar_ldas_results = np.zeros((N*M, 96*96*3 + 3 + 1 + 1 + 1)) 

#%%% RCar Random discrepancy
policy = UniformRandomPolicy(env_spec = env.spec)
             
for m in range(M):
    env.reset()
    exploration_policy = EpsilonLDASGreedyPolicy(env.spec,
                                                      policy,
                                                      total_timesteps = 100,
                                                      min_epsilon=1,
                                                      max_epsilon=1,
                                                      ldas_capacity=50,
                                                      learning_rate = .001,
                                                      learning_rate_decay = .95)
    
    for n in range(N):
        start = time.time()
        state = env.state        
        action = exploration_policy.get_action(env.state.flatten())[0]
        step_id = env.step(action).step_type        
        end = time.time()
        elapsed_time = end - start
        rcar_ldas_results[(m-1)*N + n,:] = np.concatenate((state.flatten(), action, np.array((m+1, n+1, elapsed_time))))
        if (not((step_id == 0) | (step_id == 1))):
            env.reset()  
        print((m,n))

np.savetxt('data/rcar_LDAS_disc.csv', rcar_ldas_results, delimiter=',', fmt='%s')

#%%% RCar OU discrepancy
# States, Actions, Run, Timestep, Time
rcar_OU_results = np.zeros((N*M, 96*96*3 + 3 + 1 + 1 + 1)) 
#%%% RCar Random discrepancy
policy = UniformRandomPolicy(env_spec = env.spec)
             
for m in range(M):
    env.reset()
    exploration_policy = AddOrnsteinUhlenbeckNoise(env.spec,
                                                       policy,
                                                       theta=.2)    
    for n in range(N):
        start = time.time()
        state = env.state        
        action = exploration_policy.get_action(state)[0]
        step_id = env.step(action).step_type        
        end = time.time()
        elapsed_time = end - start
        rcar_OU_results[(m-1)*N + n,:] = np.concatenate((state.flatten(), action, np.array((m+1, n+1, elapsed_time))))
        if (not((step_id == 0) | (step_id == 1))):
            env.reset()  
        print((m,n))

np.savetxt('data/rcar_OU_disc.csv', rcar_OU_results, delimiter=',', fmt='%s')
