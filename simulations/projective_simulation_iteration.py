import numpy as np
import os
import pickle
import time

class ProjectiveSimulation(object):
    """
    Classe que gerencia o ciclo de aprendizado do agente PS em um ambiente ABP.

    Parâmetros:
    - agent: Instância da classe do agente.
    - environment: Instância da classe do ambiente.

    Métodos:
    - run_learning_step(): Executa um passo de aprendizado (observa, delibera, atualiza ambiente e aprende).
    - run_episode(max_steps_per_episode, reset_env=True, reset_agent=True): Executa um episódio completo.
    - fit(num_episodes, max_steps_per_episode): Executa múltiplos episódios e retorna a curva de aprendizado.
    - reset_environment(): Reseta o ambiente para o estado inicial.
    - reset_agent(): Reseta o agente para o estado inicial.
    - set_agent_attribute(attribute, value): Define o valor de um atributo do agente.
    - set_environment_attribute(attribute, value): Define o valor de um atributo do ambiente.
    - get_agent_attribute(attribute): Retorna o valor de um atributo do agente.
    - get_environment_attribute(attribute): Retorna o valor de um atributo do ambiente.
    - h_matrix(): Retorna a matriz h do agente.
    - g_matrix(): Retorna a matriz g do agente.
    - ho_matrix(): Retorna a matriz h0 do agente.
    - e_matrix(): Retorna a matriz e do agente, se disponível.
    - save(path): Salva agente e ambiente no caminho especificado.
    - load(path): Carrega agente e ambiente do caminho especificado.

    Principais atributos:
    - agent: Instância do agente PS.
    - env: Instância do ambiente ABP.
    """

    def __init__(self, agent, environment):
        """
        Inicializa a simulação projetiva com um agente e um ambiente.
        """
        self.agent = agent
        self.env = environment

    def run_learning_step(self):
        """
        Executa um passo de aprendizado: observa, delibera, atualiza ambiente e aprende.
        Retorna True se o episódio terminou.
        """
        observation = self.env.state_observation()  # Observação: [estado, passos, colisão]
        action = self.agent.deliberate(observation)  # Ação do agente
        reward, done = self.env.update_environment(action)  # Atualiza ambiente
        self.agent.learn(reward)  # Aprendizado do agente
        return done

    def run_episode(self, max_steps_per_episode, reset_env=True, reset_agent=True):
        """
        Executa um episódio completo da simulação.

        Parâmetros:
        - max_steps_per_episode: Número máximo de passos por episódio.
        - reset_env: Se True, reseta o ambiente no início.
        - reset_agent: Se True, reseta o agente no início.

        Retorno:
        - Número de passos realizados no episódio.
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
        Executa múltiplos episódios da simulação.

        Parâmetros:
        - num_episodes: Número de episódios.
        - max_steps_per_episode: Máximo de passos por episódio.

        Retorno:
        - Array com o número de passos normalizado por episódio.
        """
        learning_process = np.zeros(num_episodes)
        for ep in range(num_episodes):
            step = self.run_episode(max_steps_per_episode)
            learning_process[ep] = step / self.env.max_steps_per_trial
        return learning_process

    def reset_environment(self):
        """
        Reseta o ambiente para o estado inicial.
        """
        self.env.reset_target()
        self.env.reset_agent_state(1)

    def reset_agent(self):
        """
        Reseta o agente para o estado inicial.
        """
        self.agent.reset_glow_matrix()

    def set_agent_attribute(self, attribute, value):
        """
        Define o valor de um atributo do agente.
        """
        setattr(self.agent, attribute, value)

    def set_environment_attribute(self, attribute, value):
        """
        Define o valor de um atributo do ambiente.
        """
        setattr(self.env, attribute, value)

    def get_agent_attribute(self, attribute):
        """
        Retorna o valor de um atributo do agente.
        """
        return getattr(self.agent, attribute)

    def get_environment_attribute(self, attribute):
        """
        Retorna o valor de um atributo do ambiente.
        """
        return getattr(self.env, attribute)

    def h_matrix(self):
        """
        Retorna a matriz h do agente.
        """
        return self.agent.h_matrix

    def g_matrix(self):
        """
        Retorna a matriz g do agente.
        """
        return self.agent.g_matrix

    def ho_matrix(self):
        """
        Retorna a matriz h0 do agente.
        """
        return self.agent.h0_matrix

    def e_matrix(self):
        """
        Retorna a matriz e do agente, se disponível.
        """
        try:
            return self.agent.e_matrix
        except:
            return None

    def save(self, path):
        """
        Salva agente e ambiente no caminho especificado.
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
        Carrega agente e ambiente do caminho especificado.
        Retorna uma instância de ProjectiveSimulation.
        """
        if not os.path.exists(path):
            raise Exception("Diretório inexistente")
        else:
            with open(path + '/agent', 'rb') as f:
                agent = pickle.load(f)
            with open(path + '/environment', 'rb') as f:
                env = pickle.load(f)
            return ProjectiveSimulation(agent, env)