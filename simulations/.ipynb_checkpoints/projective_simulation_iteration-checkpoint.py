import numpy as np
import os
import pickle

class ProjectiveSimulation(object):
    """
    Class representing the Projective Simulation algorithm.

    Parameters:
    - agent: An instance of the agent class.
    - environment: An instance of the environment class.

    Methods:
    - run_learning_step(): Runs a single learning step of the algorithm.
    - run_episode(max_steps_per_episode, reset_env=True, reset_agent=True): Runs a complete episode of the algorithm.
    - fit(num_episodes, max_steps_per_episode): Runs multiple episodes of the algorithm.
    - reset_environment(): Resets the environment to its initial state.
    - reset_agent(): Resets the agent to its initial state.
    - set_agent_attribute(attribute, value): Sets the value of a specific attribute of the agent.
    - set_environment_attribute(attribute, value): Sets the value of a specific attribute of the environment.
    - get_agent_attribute(attribute): Retrieves the value of a specific attribute of the agent.
    - get_environment_attribute(attribute): Retrieves the value of a specific attribute of the environment.
    - h_matrix(): Returns the h-matrix of the agent.
    - g_matrix(): Returns the g-matrix of the agent.
    - ho_matrix(): Returns the ho-matrix of the agent.
    - e_matrix(): Returns the e-matrix of the agent, if available.
    - save(path): Saves the agent and environment objects to the specified path.
    - load(path): Loads the agent and environment objects from the specified path.
    """

    def __init__(self, agent, environment):
        self.agent = agent
        self.env = environment
    
    def run_learning_step(self):
        observation = self.env.state_observation() # Observação: [Estado (0,1), Quantidade de passos no estado (0,...,tao-1)]
        action = self.agent.deliberate(observation) # Agente toma uma ação (0 = Manter estado, 1 = Trocar de estado)
        reward, done = self.env.update_environment(action) # Atualiza o ambiente de acordo com a ação do agente
        self.agent.learn(reward) # Realiza aprendizado do agente de acordo com a recompensa recebida
        return done

    def run_episode(self, max_steps_per_episode, reset_env=True, reset_agent=True):   
        """
        Runs a complete episode of the algorithm.

        Parameters:
        - max_steps_per_episode: The maximum number of steps allowed per episode.
        - reset_env: Whether to reset the environment at the beginning of each episode (default: True).
        - reset_agent: Whether to reset the agent at the beginning of each episode (default: True).

        Returns:
        - The number of steps taken in the episode.
        """
        if reset_env:
            self.reset_environment()
        
        if reset_agent:
            self.reset_agent()

        for step in range(max_steps_per_episode):
            done = self.run_learning_step()
            if done:
                break

        return step

    def fit(self, num_episodes, max_steps_per_episode):
        """
        Runs multiple episodes of the algorithm.

        Parameters:
        - num_episodes: The number of episodes to run.
        - max_steps_per_episode: The maximum number of steps allowed per episode.

        Returns:
        - An array containing the number of steps taken in each episode.
        """
        learning_process = np.zeros(num_episodes)
        for ep in range(num_episodes):
            step = self.run_episode(max_steps_per_episode)
            learning_process[ep] = step / self.env.max_steps_per_trial
        return learning_process
    
    def reset_environment(self):
        """
        Resets the environment to its initial state.
        """
        self.env.reset_target()
        self.env.reset_agent_state(1)

    def reset_agent(self):
        """
        Resets the agent to its initial state.
        """
        self.agent.reset_glow_matrix()

    def set_agent_attribute(self, attribute, value):
        """
        Sets the value of a specific attribute of the agent.

        Parameters:
        - attribute: The name of the attribute to set.
        - value: The value to assign to the attribute.
        """
        setattr(self.agent, attribute, value)
    
    def set_environment_attribute(self, attribute, value):
        """
        Sets the value of a specific attribute of the environment.

        Parameters:
        - attribute: The name of the attribute to set.
        - value: The value to assign to the attribute.
        """
        setattr(self.env, attribute, value)

    def get_agent_attribute(self, attribute):
        """
        Retrieves the value of a specific attribute of the agent.

        Parameters:
        - attribute: The name of the attribute to retrieve.

        Returns:
        - The value of the attribute.
        """
        return getattr(self.agent, attribute)
    
    def get_environment_attribute(self, attribute):
        """
        Retrieves the value of a specific attribute of the environment.

        Parameters:
        - attribute: The name of the attribute to retrieve.

        Returns:
        - The value of the attribute.
        """
        return getattr(self.env, attribute)

    def h_matrix(self):
        """
        Returns the h-matrix of the agent.

        Returns:
        - The h-matrix.
        """
        return self.agent.h_matrix

    def g_matrix(self):
        """
        Returns the g-matrix of the agent.

        Returns:
        - The g-matrix.
        """
        return self.agent.g_matrix

    def ho_matrix(self):
        """
        Returns the ho-matrix of the agent.

        Returns:
        - The ho-matrix.
        """
        return self.agent.h0_matrix

    def e_matrix(self):
        """
        Returns the e-matrix of the agent, if available.

        Returns:
        - The e-matrix, or None if not available.
        """
        try:
            return self.agent.e_matrix
        except:
            return None

    def save(self, path):
        """
        Saves the agent and environment objects to the specified path.

        Parameters:
        - path: The path to save the objects to.
        """
        if not os.path.exists(path):
            os.makedirs(path)
            
        with open(path + '/agent', 'wb') as f:
            pickle.dump(self.agent, f, pickle.HIGHEST_PROTOCOL)

        with open(path + '/environment', 'wb') as f:
            pickle.dump(self.env, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path):
        """
        Loads the agent and environment objects from the specified path.

        Parameters:
        - path: The path to load the objects from.

        Returns:
        - An instance of the ProjectiveSimulation class with the loaded agent and environment.
        """
        if not os.path.exists(path):
            raise Exception("Directory does not exist")
                            
        else:
            with open(path + '/agent', 'rb') as f:
                agent = pickle.load(f)

            with open(path + '/environment', 'rb') as f:
                env = pickle.load(f)

            return ProjectiveSimulation(agent, env)