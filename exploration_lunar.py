#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 12:32:18 2021

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

#%% LunarLander environment initialization
env_id = 'LunarLanderContinuous-v2'
gym_env = gym.make(env_id)
env = GymEnv(gym_env)

#%% LunarLander discrepancy experiment
M = 100 # Number of repetitions
N = 100

#%%% uniform
# States, Actions, Run, Timestep, Rime
lunar_uniform_results = np.zeros((N*M, 8 + 2 + 1 + 1 + 1))
exploration_policy = UniformRandomPolicy(env_spec = env.spec)

for m in range(M):
    state = env.reset()[0]    
    for n in range(N):
        start = time.time()
        #state = env.state        
        action = exploration_policy.get_action(state)[0]
        env_step = env.step(action)
        step_id = env_step.step_type
        next_state = env_step.observation
        #step_id = env.step(action).step_type                
        end = time.time()
        elapsed_time = end - start
        lunar_uniform_results[(m-1)*N + n,:] = np.concatenate((state, action, np.array((m+1, n+1, elapsed_time))))
        state = next_state
        if (not((step_id == 0) | (step_id == 1))):
            state = env.reset()[0]                    
        print((m,n))
        
np.savetxt('data/lunar_uniform_disc.csv', lunar_uniform_results, delimiter=',', fmt='%s')

#%%% LDAS
# States, Actions, Run, Timestep, Rime
lunar_ldas_results = np.zeros((N*M, 8 + 2 + 1 + 1 + 1))
policy = UniformRandomPolicy(env_spec = env.spec)

for m in range(M):
    exploration_policy = EpsilonLDASGreedyPolicy(env.spec,
                                                 policy,
                                                 total_timesteps = 100,
                                                 min_epsilon=1,
                                                 max_epsilon=1,
                                                 ldas_capacity=50,
                                                 learning_rate = .001,
                                                 learning_rate_decay = .95)
    
    state = env.reset()[0]
    for n in range(N):
        start = time.time()
        #state = env.state        
        action = exploration_policy.get_action(state)[0]
        env_step = env.step(action)
        step_id = env_step.step_type
        next_state = env_step.observation
        #step_id = env.step(action).step_type                
        end = time.time()
        elapsed_time = end - start
        lunar_ldas_results[(m-1)*N + n,:] = np.concatenate((state, action, np.array((m+1, n+1, elapsed_time))))
        state = next_state
        if (not((step_id == 0) | (step_id == 1))):
            state = env.reset()[0]                    
        print((m,n))
        
np.savetxt('data/lunar_LDAS_disc.csv', lunar_ldas_results, delimiter=',', fmt='%s')
        
#%% OU 
lunar_OU_results = np.zeros((N*M, 8 + 2 + 1 + 1 + 1))
policy = UniformRandomPolicy(env_spec = env.spec)

for m in range(M):
    exploration_policy = AddOrnsteinUhlenbeckNoise(env.spec,
                                                   policy)    
    state = env.reset()[0]
    for n in range(N):
        start = time.time()
        #state = env.state        
        action = exploration_policy.get_action(state)[0]
        env_step = env.step(action)
        step_id = env_step.step_type
        next_state = env_step.observation
        #step_id = env.step(action).step_type                
        end = time.time()
        elapsed_time = end - start
        lunar_OU_results[(m-1)*N + n,:] = np.concatenate((state, action, np.array((m+1, n+1, elapsed_time))))
        state = next_state
        if (not((step_id == 0) | (step_id == 1))):
            state = env.reset()[0]                    
        print((m,n))
        
np.savetxt('data/lunar_OU_disc.csv', lunar_OU_results, delimiter=',', fmt='%s')
   
