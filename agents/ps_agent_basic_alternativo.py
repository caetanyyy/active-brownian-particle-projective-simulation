import numpy as np

class Agent(object):
	
	def __init__(self, num_actions, num_percepts_list, gamma_damping, eta_glow_damping, policy_type, beta_softmax, num_reflections = 0):
		"""Initialize the basic PS agent. Arguments: 
            - num_actions: integer >=1, 
            - num_percepts_list: list of integers >=1, not nested, representing the cardinality of each category/feature of percept space.
            - gamma_damping: float between 0 and 1, controls forgetting/damping of h-values
            - eta_glow_damping: float between 0 and 1, controls the damping of glow; setting this to 1 effectively switches off glow
            - policy_type: string, 'standard' or 'softmax'; toggles the rule used to compute probabilities from h-values
            - beta_softmax: float >=0, probabilities are proportional to exp(beta*h_value). If policy_type != 'softmax', then this is irrelevant.
		"""
		self.rng = np.random.RandomState(None)
		
		self.num_actions = num_actions
		self.num_percepts_list = num_percepts_list
		self.gamma_damping = gamma_damping
		self.eta_glow_damping = eta_glow_damping
		self.policy_type = policy_type
		self.beta_softmax = beta_softmax
		self.num_reflections = num_reflections

		self.num_percepts = int(np.prod(np.array(self.num_percepts_list).astype(np.float64))) # total number of possible percepts
		
		self.h_matrix = np.ones((self.num_actions, self.num_percepts), dtype=np.float64) #Note: the first index specifies the action, the second index specifies the percept.
		self.h0_matrix = np.ones((self.num_actions, self.num_percepts), dtype=np.float64) #Note: the first index specifies the action, the second index specifies the percept.
		self.g_matrix = np.zeros((self.num_actions, self.num_percepts), dtype=np.float64) #glow matrix, for processing delayed rewards

		if num_reflections > 0:
			self.last_percept_action = None  #stores the last realized percept-action pair for use with reflection. If reflection is deactivated, all necessary information is encoded in g_matrix.
			self.e_matrix = np.ones((self.num_actions, self.num_percepts), dtype=np.bool_) # emoticons

	def load_matrix(self, path):
		self.h_matrix = np.loadtxt(f'{path}/h_matrix.txt')
		self.h0_matrix = np.loadtxt(f'{path}/h0_matrix.txt')
		self.g_matrix = np.loadtxt(f'{path}/h_matrix.txt')
		self.e_matrix = np.loadtxt(f'{path}/e_matrix.txt')

	def percept_preprocess(self, observation): # preparing for creating a percept
		percept = 0
		for which_feature in range(len(observation)):
			percept += int(observation[which_feature] * np.prod(self.num_percepts_list[:which_feature]))
		return percept
		
	def learn(self, reward):
		#self.h_matrix =  self.h_matrix - self.gamma_damping * (self.h_matrix - 1.) + self.g_matrix * reward # learning and forgetting
		self.h_matrix =  self.h_matrix*(1. - self.gamma_damping) + self.gamma_damping * self.h0_matrix + self.g_matrix * reward # learning and forgetting

		if (self.num_reflections > 0) and (self.last_percept_action != None) and (reward <= 0): # reflection update
			self.e_matrix[self.last_percept_action] = 0

	def deliberate(self, observation):
		
		percept = self.percept_preprocess(observation) 
		action = self.rng.choice(self.num_actions, p=self.probability_distr(percept)) #deliberate once	
		
		for i_counter in range(self.num_reflections):  #if num_reflection >=1, repeat deliberation if indicated
			if self.e_matrix[action, percept]:
				break
			action = self.rng.choice(self.num_actions, p=self.probability_distr(percept))		

		self.g_matrix = (1 - self.eta_glow_damping) * self.g_matrix
		self.g_matrix[action, percept] += 1 #record latest decision in g_matrix
		
		if self.num_reflections > 0:
			self.last_percept_action = action, percept	#record latest decision in last_percept_action
		
		return action
		
	def probability_distr(self, percept):        
		if self.policy_type == 'standard':
			h_vector = self.h_matrix[:, percept]
			probability_distr = h_vector / np.sum(h_vector)
		elif self.policy_type == 'softmax':
			h_vector = self.beta_softmax * self.h_matrix[:, percept]
			h_vector_mod = h_vector - np.max(h_vector)
			probability_distr = np.exp(h_vector_mod) / np.sum(np.exp(h_vector_mod))
		return probability_distr
	
	def reset_glow_matrix(self):
		self.g_matrix = np.zeros((self.num_actions, self.num_percepts), dtype=np.float64) #glow matrix, for processing delayed rewards