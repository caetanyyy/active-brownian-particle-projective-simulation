import numpy as np
import pickle
import os

class PsAgent(object):
    """
    Agente de Simulação Projetiva (PS) para ambientes de busca de alvo.

    Parâmetros:
    - num_actions: Número de ações possíveis (>=1).
    - num_percepts_list: Lista de inteiros representando a cardinalidade de cada categoria do espaço de percepção.
    - gamma_damping: Fator de esquecimento/damping dos valores h (float entre 0 e 1).
    - eta_glow_damping: Fator de damping da matriz glow (float entre 0 e 1).
    - policy_type: Tipo de política ('standard' ou 'softmax').
    - beta_softmax: Parâmetro β da softmax (>=0).
    - num_reflections: Número de reflexões para deliberar ações.

    Métodos principais:
    - deliberate(observation): Decide a ação do agente com base na observação do ambiente.
    - learn(reward): Atualiza os pesos do agente conforme a recompensa recebida.
    - probability_distr(percept): Calcula a distribuição de probabilidade das ações.
    - reset_glow_matrix(): Reseta a matriz de glow (memória de ações).
    - save(path)/load(path): Salva ou carrega o agente em formato binário.

    Principais atributos:
    - h_matrix: Matriz de pesos de decisão.
    - g_matrix: Matriz de glow para aprendizado com atraso.
    - policy_type: Tipo de política de decisão ('standard' ou 'softmax').
    - num_reflections: Número de reflexões para deliberar ações.
    """
    def __init__(self, num_actions, num_percepts_list, gamma_damping, eta_glow_damping, policy_type, beta_softmax, num_reflections=0):
        """
        Inicializa o agente PS básico.

        Parâmetros:
        - num_actions: inteiro >=1, número de ações possíveis.
        - num_percepts_list: lista de inteiros >=1, representando a cardinalidade de cada categoria do espaço de percepção.
        - gamma_damping: float entre 0 e 1, controla o esquecimento/damping dos valores h.
        - eta_glow_damping: float entre 0 e 1, controla o damping da matriz glow; 1 desativa o glow.
        - policy_type: string, 'standard' ou 'softmax'; define a regra de probabilidade.
        - beta_softmax: float >=0, parâmetro β da softmax. Ignorado se policy_type != 'softmax'.
        - num_reflections: inteiro >=0, número de reflexões para deliberar ações.
        """
        
        self.rng = np.random.RandomState(None)
        
        self.num_actions = num_actions
        self.num_percepts_list = num_percepts_list
        self.gamma_damping = gamma_damping
        self.eta_glow_damping = eta_glow_damping
        self.policy_type = policy_type
        self.beta_softmax = beta_softmax
        self.num_reflections = num_reflections
        self.num_percepts = int(
            np.prod(
                np.array(self.num_percepts_list).astype(np.float64)
            )
        )  # Número total de percepções possíveis
        
        self.h_matrix = np.ones(
            (self.num_actions, self.num_percepts),
            dtype=np.float64
        )  # Matriz de pesos h (ação x percepção)

        self.h0_matrix = np.ones(
            (self.num_actions, self.num_percepts),
            dtype=np.float64
        )  # Matriz h0 inicial (referência)

        self.g_matrix = np.zeros(
            (self.num_actions, self.num_percepts),
            dtype=np.float64
        )  # Matriz glow para aprendizado com atraso

        if num_reflections > 0:
            self.last_percept_action = None  # Guarda o último par percepção-ação para reflexão
            self.e_matrix = np.ones(
                (self.num_actions, self.num_percepts),
                dtype=np.bool_
            )  # Matriz de emoticons (para reflexão)

    def load_matrix(self, path):
        """
        Carrega as matrizes do agente a partir de arquivos texto.
        """
        self.h_matrix = np.loadtxt(f'{path}/h_matrix.txt')
        self.h0_matrix = np.loadtxt(f'{path}/h0_matrix.txt')
        self.g_matrix = np.loadtxt(f'{path}/h_matrix.txt')
        self.e_matrix = np.loadtxt(f'{path}/e_matrix.txt')

    def percept_preprocess(self, observation):
        """
        Pré-processa a observação para gerar um índice de percepção único.
        """
        percept = 0
        for which_feature in range(len(observation)):
            percept += int(observation[which_feature] * np.prod(self.num_percepts_list[:which_feature]))
        return percept

    def learn(self, reward):
        """
        Atualiza a matriz de pesos h do agente conforme a recompensa recebida.
        Também atualiza a matriz de emoticons se reflexão estiver ativa.
        """
        self.h_matrix = self.h_matrix * (1. - self.gamma_damping) + self.gamma_damping * self.h0_matrix + self.g_matrix * reward
        if (self.num_reflections > 0) and (self.last_percept_action is not None) and (reward <= 0):
            self.e_matrix[self.last_percept_action] = 0

    def deliberate(self, observation):
        """
        Delibera uma ação a partir de uma observação do ambiente.
        Se reflexão está ativa, repete a deliberação conforme necessário.
        Atualiza a matriz glow e armazena o último par ação-percepção.
        """
        percept = self.percept_preprocess(observation)
        action = self.rng.choice(self.num_actions, p=self.probability_distr(percept))
        for _ in range(self.num_reflections):
            if self.e_matrix[action, percept]:
                break
            action = self.rng.choice(self.num_actions, p=self.probability_distr(percept))
        self.g_matrix = (1 - self.eta_glow_damping) * self.g_matrix
        self.g_matrix[action, percept] += 1
        if self.num_reflections > 0:
            self.last_percept_action = action, percept
        return action
        
    def probability_distr(self, percept):
        """
        Calcula a distribuição de probabilidade das ações para uma percepção.
        """
        if self.policy_type == 'standard':
            h_vector = self.h_matrix[:, percept]
            probability_distr = h_vector / np.sum(h_vector)
        elif self.policy_type == 'softmax':
            h_vector = self.beta_softmax * self.h_matrix[:, percept]
            h_vector_mod = h_vector - np.max(h_vector)
            probability_distr = np.exp(h_vector_mod) / np.sum(np.exp(h_vector_mod))
        return probability_distr
    
    def reset_glow_matrix(self):
        """
        Reseta a matriz glow (memória de ações recentes).
        """
        self.g_matrix = np.zeros((self.num_actions, self.num_percepts), dtype=np.float64)

    def save(self, path):
        """
        Salva o agente em arquivo binário.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path + '/agent', 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path):
        """
        Carrega o agente de um arquivo binário.
        """
        if not os.path.exists(path):
            raise Exception("Diretório inexistente")
        else:
            with open(path + '/agent', 'rb') as f:
                return pickle.load(f)