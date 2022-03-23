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

#%% MountainCar environment initialization
env_id = 'MountainCarContinuous-v0'
gym_env = gym.make(env_id)
max_episode_length = 10000               
gym_env._max_episode_steps = max_episode_length
gym_env.spec.max_episode_steps = max_episode_length
#env = normalize(GymEnv(gym_env))
env = GymEnv(gym_env)

#%% MountainCar discrepancy experiment
M = 100  # Number of repetitions
N = 100  # Number of time-steps per repetition

# States, Actions, Run, Timestep, Time
mcar_uniform_results = np.zeros((N*M, 2 + 1 + 1 + 1 + 1))  
#%%% MCar Random discrepancy
exploration_policy = UniformRandomPolicy(env_spec = env.spec)
             
for m in range(M):
    env.reset()    
    for n in range(N):
        start = time.time()
        state = env.state        
        action = exploration_policy.get_action(state)[0]
        step_id = env.step(action).step_type        
        if (not((step_id == 0) | (step_id == 1))):
            env.reset()                    
        end = time.time()
        elapsed_time = end - start
        mcar_uniform_results[(m-1)*N + n,:] = np.concatenate((state, action, np.array((m+1, n+1, elapsed_time))))
        print((m,n))

np.savetxt('data/mcar_uniform_disc.csv', mcar_uniform_results, delimiter=',', fmt='%s')

#%%% MCar LDAS discrepancy
# States, Actions, Run, Timestep, Time
mcar_ldas_results = np.zeros((N*M, 2 + 1 + 1 + 1 + 1))  
#%%% MCar Random discrepancy
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
        action = exploration_policy.get_action(state)[0]
        step_id = env.step(action).step_type        
        if (not((step_id == 0) | (step_id == 1))):
            env.reset()                           
        end = time.time()
        elapsed_time = end - start
        mcar_ldas_results[(m-1)*N + n,:] = np.concatenate((state, action, np.array((m+1, n+1, elapsed_time))))
        print((m,n))

np.savetxt('data/mcar_LDAS_disc.csv', mcar_ldas_results, delimiter=',', fmt='%s')


#%%% MCar OU discrepancy
# States, Actions, Run, Timestep, Time
mcar_OU_results = np.zeros((N*M, 2 + 1 + 1 + 1 + 1))  
#%%% MCar Random discrepancy
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
        if (not((step_id == 0) | (step_id == 1))):
            env.reset()            
        end = time.time()
        elapsed_time = end - start
        mcar_uniform_results[(m-1)*N + n,:] = np.concatenate((state, action, np.array((m+1, n+1, elapsed_time))))
        print((m,n))

np.savetxt('data/mcar_OU_disc.csv', mcar_uniform_results, delimiter=',', fmt='%s')
