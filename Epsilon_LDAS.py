from dowel import tabular
import numpy as np

from garage.np.exploration_policies.exploration_policy import ExplorationPolicy

class EpsilonLDASGreedyPolicy(ExplorationPolicy):#ExplorationPolicy
    """ϵ-greedy exploration strategy.
    Select action based on the value of ϵ. ϵ will decrease from
    max_epsilon to min_epsilon within decay_ratio * total_timesteps.
    At state s, with probability
    1 − ϵ: select action = argmax Q(s, a)
    ϵ    : select a random action from an uniform distribution.
    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        policy (garage.Policy): Policy to wrap.
        total_timesteps (int): Total steps in the training, equivalent to
            max_episode_length * n_epochs.
        max_epsilon (float): The maximum(starting) value of epsilon.
        min_epsilon (float): The minimum(terminal) value of epsilon.
        decay_ratio (float): Fraction of total steps for epsilon decay.
    """

    def __init__(self,
                 env_spec,
                 policy,
                 *,
                 total_timesteps,
                 ldas_capacity = 500,
                 learning_rate = .1, # function of dimensions?
                 learning_rate_decay = .99,
                 num_candidate_actions = 5,
                 num_recent_actions = 5,
                 threshold = .001,
                 max_epsilon=1.0,
                 min_epsilon=0.02,
                 decay_ratio=0.1):   # Epsilon decay, not gradient
        super().__init__(policy)
        self._storage_index = 0
        self._is_buffer_full = False
        self._ldas_capacity = ldas_capacity
        self._learning_rate = learning_rate
        self._learning_rate_decay = learning_rate_decay
        self._num_candidate_actions = num_candidate_actions
        self._num_recent_actions = num_recent_actions
        self._threshold = threshold
        self._env_spec = env_spec
        self._max_epsilon = max_epsilon
        self._min_epsilon = min_epsilon
        self._decay_period = int(total_timesteps * decay_ratio)
        self._action_space = env_spec.action_space
        self._action_dim = self._action_space.flat_dim
        self._observation_space = env_spec.observation_space
        self._observation_dim = self._observation_space.flat_dim
        self._decrement = (self._max_epsilon -
                           self._min_epsilon) / self._decay_period
        self._total_env_steps = 0
        self._last_total_env_steps = 0
        self._ldas_buffer = np.zeros((self._ldas_capacity, self._observation_dim + self._action_dim))
        
    def _get_phantom_matrix(self, observation, action, eps):
        d_s = self._observation_dim
        d_a = self._action_dim
        phantom = np.zeros((2 * d_a, d_s + d_a))
        for i in range(d_a):
            phantom[2*i, 0:d_s] = observation
            phantom[2*i+1, 0:d_s] = observation
            phantom[2*i, d_s:(d_s + d_a)] = action
            phantom[2*i+1, d_s:(d_s + d_a)] = action
            phantom[2*i, d_s + i] = -eps
            phantom[2*i+1, d_s + i] = 1 + eps
        return(phantom)
            
    def _get_closest_point(self, X, x0):
        #print("Inside get_closest_distance")
        #print(len(X))
        #print(len(x0))
        distance = np.sum(np.abs(X - x0), axis=1)
        closest_distance = min(distance)
        closest_point = X[distance.argmin(),:]
        return([closest_distance, closest_point])            
    
    def ldas_sample(self, observation):
        if (not self._is_buffer_full):
            ldas_buffer = self._ldas_buffer[0:self._storage_index,:]
        else:
            ldas_buffer = self._ldas_buffer
        if (self._total_env_steps == 0):
            # Change this so ldas only deals with actions in unit hypercube, 
            # And translated back at the end
            best_action = self._action_space.low + np.random.uniform(size = self._action_dim)*(self._action_space.high - self._action_space.low)
        else:            
            best_distance = 0
            for i in range(self._num_candidate_actions):
                keep_going = True
                #print(self._learning_rate)
                temp_lr = self._learning_rate
                #print(temp_lr)
                action = self._action_space.low + np.random.uniform(size = self._action_dim)*(self._action_space.high - self._action_space.low)
                recent_actions = [''] * self._num_recent_actions
                recent_actions[0] = action
                for j in range(1, self._num_recent_actions):
                    #print("Inside for loop over recent actions")
                    phantom = self._get_phantom_matrix(observation, action, 1 / len(ldas_buffer) ** (1 / self._action_dim))
                    #print(type(ldas_buffer))
                    #print(ldas_buffer)
                    #print(type(phantom))
                    #print(phantom)
                    ldas_buffer_temp = np.vstack((ldas_buffer, phantom))            
                    # list(a) + [b]
                    observation_action = np.ndarray((1, self._observation_dim + self._action_dim))
                    observation_action[0, 0:self._observation_dim] = observation
                    observation_action[0, self._observation_dim:(self._observation_dim + self._action_dim)] = action                    
                    closest_point = self._get_closest_point(ldas_buffer_temp, observation_action)[1]
                    diff = action - closest_point[self._observation_dim:(self._observation_dim + self._action_dim)]
                    action = action + temp_lr*diff/(sum(diff**2))**.5
                    temp_lr = temp_lr * self._learning_rate_decay
                                
                    # Check scale of action space
                    action = np.where(action>self._action_space.high,self._action_space.high, action)
                    action = np.where(action<self._action_space.low,self._action_space.low, action)                
                
                    if (all((action==self._action_space.low)|(action==self._action_space.high))):
                        keep_going = False
                        #print("Breaking inside recent action loop cause borders")
                        break
                    elif (temp_lr < self._threshold):                        
                        keep_going = False
                        #print(temp_lr)
                        #print(self._threshold)
                        #print("Breaking inside recent action loop cause threshold")
                        break
                    else:
                        recent_actions[j] = action
                while(keep_going):
                    #print("Inside while loop to stabilize action")                    
                    phantom = self._get_phantom_matrix(observation, action, 1 / len(ldas_buffer) ** (1 / self._action_dim))
                    #print("ldas_buffer info")
                    #print(type(ldas_buffer))
                    #print(ldas_buffer)
                    #print("phantom info")
                    #print(type(phantom))
                    #print(phantom)
                    ldas_buffer_temp = np.vstack((ldas_buffer, phantom))
                    observation_action = np.ndarray((1, self._observation_dim + self._action_dim))
                    observation_action[0, 0:self._observation_dim] = observation
                    observation_action[0, self._observation_dim:(self._observation_dim + self._action_dim)] = action                    
                    closest_point = self._get_closest_point(ldas_buffer_temp, observation_action)[1]
                    diff = action - closest_point[self._observation_dim:(self._observation_dim + self._action_dim)]
                    action = action + temp_lr*diff/(np.sum(diff**2))**.5
                    temp_lr = temp_lr * self._learning_rate_decay
                                
                    # Check scale of action space
                    action = np.where(action>self._action_space.high,self._action_space.high, action)
                    action = np.where(action<self._action_space.low,self._action_space.low, action)                
                
                    if (all((action==self._action_space.low)|(action==self._action_space.high))):
                        keep_going = False
                        #print("Breaking inside while loop")
                        break
                    elif (temp_lr < self._threshold):
                        keep_going = False
                        #print("Breaking inside while loop")
                        break
                    elif (np.sum(np.abs(action - recent_actions[0])) < self._threshold):
                        keep_going = False
                        #print("Breaking inside while loop")
                        break             
                    else:
                        #print("Updating recent actions")
                        recent_actions = recent_actions[1:self._num_recent_actions] + [action]
                observation_action = np.ndarray((1, self._observation_dim + self._action_dim))
                observation_action[0, 0:self._observation_dim] = observation
                observation_action[0, self._observation_dim:(self._observation_dim + self._action_dim)] = action
                closest_info = self._get_closest_point(ldas_buffer, observation_action)
                #print("checking if this action is best")
                if (closest_info[0] >= best_distance):
                    #print("updating best action")
                    best_distance = closest_info[0]
                    best_action = action
                #print(closest_info[0])
                #print(best_distance)
        if (not 'best_action' in locals()):
            print(closest_info[0])
            print(best_distance)
        return(best_action)
    
    def get_action(self, observation):
        """Get action from this policy for the input observation.
        Args:
            observation (numpy.ndarray): Observation from the environment.
        Returns:
            np.ndarray: An action with noise.
            dict: Arbitrary policy state information (agent_info).
        """
        opt_action, _ = self.policy.get_action(observation)
        if np.random.random() < self._epsilon():
            opt_action = self.ldas_sample(observation=observation)
            #opt_action = self._action_space.sample()#will change to ldas_sampler   
        self._total_env_steps += 1
        #self._ldas_buffer[self._storage_index,0:(self._observation_dim)] = observation
        #self._ldas_buffer[self._storage_index,self._observation_dim:(self._observation_dim + self._action_dim)] = opt_action
        self._ldas_buffer[self._storage_index,:] = np.concatenate((observation,opt_action))
        self._storage_index += 1
        if ((self._storage_index + 1) > self._ldas_capacity):
            self._storage_index = 0
            self._is_buffer_full = True
        #store observation in buffer / could create this elsewhere
        
        return opt_action, dict()

    def get_actions(self, observations):
        """Get actions from this policy for the input observations.
        Args:
            observations (numpy.ndarray): Observation from the environment.
        Returns:
            np.ndarray: Actions with noise.
            List[dict]: Arbitrary policy state information (agent_info).
        """
        opt_actions, _ = self.policy.get_actions(observations)
        for itr, _ in enumerate(opt_actions):
            if np.random.random() < self._epsilon():
                opt_actions[itr] = self._action_space.sample()
            self._total_env_steps += 1
            #store in buffer
        return opt_actions, dict()

    def _epsilon(self):
        """Get the current epsilon.
        Returns:
            double: Epsilon.
        """
        if self._total_env_steps >= self._decay_period:
            return self._min_epsilon
        return self._max_epsilon - self._decrement * self._total_env_steps

    def update(self, episode_batch):
        """Update the exploration policy using a batch of trajectories.
        Args:
            episode_batch (EpisodeBatch): A batch of trajectories which
                were sampled with this policy active.
        """
        self._total_env_steps = (self._last_total_env_steps +
                                 np.sum(episode_batch.lengths))
        self._last_total_env_steps = self._total_env_steps
        tabular.record('EpsilonGreedyPolicy/Epsilon', self._epsilon())

    def get_param_values(self):
        """Get parameter values.
        Returns:
            list or dict: Values of each parameter.
        """
        return {
            'total_env_steps': self._total_env_steps,
            'inner_params': self.policy.get_param_values()
        }

    def set_param_values(self, params):
        """Set param values.
        Args:
            params (np.ndarray): A numpy array of parameter values.
        """
        self._total_env_steps = params['total_env_steps']
        self.policy.set_param_values(params['inner_params'])
        self._last_total_env_steps = self._total_env_steps
