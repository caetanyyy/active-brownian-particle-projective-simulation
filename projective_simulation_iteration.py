import numpy as np

class ProjectiveSimulation(object): 
    def __init__(self, agent, environment):
        self.agent = agent
        self.env = environment
    
    def run_learning_step(self):
        observation = self.env.state_observation() # Observação: [Estado (0,1), Quantidade de passos no estado (0,...,tao-1)]
        action = self.agent.deliberate(observation) # Agente toma uma ação (0 = Manter estado, 1 = Trocar de estado)
        reward, done = self.env.update_environment(action) # Atualiza o ambiente de acordo com a ação do agente
        self.agent.learn(reward) # Realiza aprendizado do agente de acordo com a recompensa recebida
        return done

    def run_episode(self, max_steps_per_episode):   

        self.env.reset_target() #Resta a posição do target
        self.env.reset_agent_state(1) # Ao inicio de cada episódio, reseta o agente no estado ABP (1)
        self.agent.reset_glow_matrix() # Reseta a matriz de glow, conforme o artigo

        # Inicia passos de aprendizado do episódio
        for step in range(max_steps_per_episode):
            done = self.run_learning_step()
            if done:
                break
        return step

    def fit(self, num_episodes, max_steps_per_episode):

        learning_process = np.zeros(num_episodes)
        for ep in range(num_episodes):
            step = self.run_episode(max_steps_per_episode) # Roda o episódio
            learning_process[ep] = step/self.env.max_steps_per_trial # Salva o passo do fim do episódio

        return learning_process

    def h_matrix(self):
        return self.agent.h_matrix

    def g_matrix(self):
        return self.agent.g_matrix

    def ho_matrix(self):
        return self.agent.h0_matrix

    def e_matrix(self):
        try:
            return self.agent.e_matrix
        except:
             return None