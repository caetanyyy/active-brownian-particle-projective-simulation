import numpy as np
import pickle
import os
import time

class PsEnvironment(object):
    """
    Ambiente de simulação para partículas Brownianas ativas (ABP) com busca de alvo.

    Parâmetros:
    - L: Tamanho da caixa 2D.
    - Pe: Número de Péclet.
    - l: Escala de persistência do movimento.
    - tao: Número máximo de passos por rodada.
    - dt: Intervalo de tempo da simulação.
    - colision_reward: Recompensa por aprendizado de colisão.
    - allow_colision: Permite colisão com as paredes (True/False).

    Métodos principais:
    - state_observation(): Retorna o estado observável do ambiente para o agente.
    - update_environment(action): Atualiza o ambiente conforme a ação do agente.
    - reset_target(): Reseta a posição do alvo.
    - reset_agent_state(new_state): Reseta o estado do agente.
    - save(path)/load(path): Salva ou carrega o ambiente em formato binário.

    Principais atributos:
    - state: Estado atual do agente (ativo/passivo).
    - colision: Indica colisão com a parede.
    - reward: Recompensa atual do agente.
    - target_position: Posição do alvo.
    """
    """
    Represents an environment for an active brownian particle target finding simulation.

    Attributes:
        L (float): Dimension of the space.
        Pe (float): Péclet number.
        l (float): Length scale.
        tao (int): Maximum steps per trial.
        dt (float): Time step size.
        rng (numpy.random.RandomState): Random number generator.
        max_steps_per_trial (int): Maximum steps per trial.
        num_states (int): Number of states (1 = active, 0 = passive).
        colision_state (int): Numero de estados de colisão (1 = colidiu, 0 = não colidiu).
        num_actions (int): Number of actions (1 = state change, 0 = state maintain).
        num_percepts_list (list): Size of the observables.
        allow_colision (bool): Indica se permite colisão com as paredes ou não.
        colision (int): Indica se houve colisão (1) ou não (0) no último movimento.
        reward (float): Current reward.
        trial_finished (bool): Flag indicating if the trial has finished.
        r (numpy.ndarray): Current position of the agent.
        state (int): Current state of the agent (0 or 1).
        timer (int): Number of steps the agent has been in the current state.
        distance (float): Distance between the agent and the target.
        target_radius (float): Radius of the target.
        target_position (numpy.ndarray): Current position of the target.
        v (float): Translational velocity.
        D (float): Translational coefficient.
        D_theta (float): Rotational coefficient.
        dt (float): Time step size.
        dr_theta (float): Component of ABP movement.
        theta_t (float): Rotation of the movement.
        n_t (float): Scalar noise.
        u_t (numpy.ndarray): Orientation of the active movement.
        dr (float): Passive translation.
        E_t (numpy.ndarray): Noise vector.
        dr_dt (float): Sum of the movements.
    """

    def __init__(
        self,
        L: float = 100,
        Pe: float = 100,
        l: float = 1,
        tao: float = 1e+4,
        dt: float = 1,
        colision_reward: float = 0.05,
        allow_colision: bool = False,
        collision_type: str = 'specular'
    ):
        """
        Inicializa o objeto do ambiente ABP.

        Parâmetros:
        - L (float): Tamanho da caixa 2D.
        - Pe (float): Número de Péclet.
        - l (float): Escala de persistência do movimento.
        - tao (float): Número máximo de passos por rodada (>1).
        - dt (float): Intervalo de tempo da simulação.
        - colision_reward (float): Recompensa por aprendizado de colisão.
        - allow_colision (bool): Permite colisão com as paredes (True/False).
        """
        # Gerador aleatório da classe:
        self.rng = np.random.RandomState(None)

        # Inicia variáveis do ambiente
        # Estados
        self.max_steps_per_trial = tao # tempo máximo de uma rodada
        self.num_states = 2 # 1 = ativo ou 0 = passivo
        self.colision_state = 2 # colisão ou não

        #Ações
        self.num_actions = 2 # 1 = troca de estado, 0 = mantem estado

        self.allow_colision = allow_colision
        self.collision_type = collision_type

        if allow_colision:
            self.num_percepts_list = [self.num_states, self.max_steps_per_trial, self.colision_state] # Tamanho dos observaveis
        else:
            self.num_percepts_list = [self.num_states, self.max_steps_per_trial] # Tamanho dos observaveis
        
        # Observáveis
        self.state = 0 #0 ou 1
        self.prev_state = 0 #0 ou 1 # guarda o estado anterior

        self.timer = 0 #inteiro que contabiliza a quantidade de rodadas que o agente está em um estado
        #self.timer_colision = 0 #inteiro que contabiliza a quantidade de rodadas que o agente está em colisão

        self.colision = 0 #0 ou 1, mapeia se o agente teve colisão ou não com a parede
        self.prev_colision = 0 #0 ou 1, mapeia se o agente teve colisão no passo anterior

        #Recompensa
        self.reward = 0 # Inicia a recompensa como zero
        self.trial_finished = colision_reward # Inicia o episódio
        self.colision_reward = 0.005 # Recompensa por sair do estado ativo em uma colisão

        # Espaço
        self.L = L # Dimensão do espaço

        # Estado inicial do agente no Espaço
        self.r = np.array([L/2,L/2]) #keeps track of where the agent is located
        self.distance = L #Distancia do agente para o target (inicialização)

        # Estado inicial do target
        self.target_radius = 0.05*L #tamanho do target
        self.target_position = np.array([
            self.rng.rand()*self.L, 
            self.rng.rand()*self.L
        ]) #posição do target [x,y]
        
        # Parametros de movimento do agente
        self.v = Pe*L/(tao) #Velocidade translacional
        self.D = (L*L)/(4*tao) #Coeficiente translacional
        self.D_theta = self.v/(l*L) #Coeficiente rotacional
        self.dt = dt # Período de tempo

        # Movimento ABP
        self.dr_theta = 0 
        self.theta_t = 2*np.pi*self.rng.rand() #Rotação do movimento
        self.n_t = self.rng.normal() #ruído escalar
        self.u_t = np.array([np.cos(self.theta_t), np.sin(self.theta_t)]) #orientação do movimento ativo
        
        # Movimento BP
        self.dr = 0 #translação passiva
        self.E_t = np.array([self.rng.normal(),self.rng.normal()]) #vetor ruído

        # Soma dos movimentos:
        self.dr_dt = self.dr + self.dr_theta

        if self.collision_type == 'repulsive':
            self.repulsive_k = 0.3  # Constante de força (dureza da parede)
            self.repulsive_sigma = 1*self.target_radius # Largura da zona de repulsão
            self.repulsive_cutoff = self.target_radius # Força é zero além desta distância
            self.repulsive_power = 3
        
    def reset_rng(self, seed=None):
        """
        Reseta o gerador de números aleatórios do ambiente.

        Parâmetros:
        - seed (int): Semente para o gerador. Se None, usa semente aleatória.
        """
        self.rng = np.random.RandomState(seed)

    def reset_target(self):
        """
        Reseta a posição do alvo.
        Sempre chamada ao final de uma rodada, independente do agente encontrar o alvo.
        """
        self.target_position = np.array([
            self.rng.rand() * self.L,
            self.rng.rand() * self.L
        ])

    def reset_agent_state(self, new_state):
        """
        Reseta o estado do agente para um estado específico.

        Parâmetros:
        - new_state (int): Novo estado do agente (0 ou 1).
        """
        self.timer = 0
        if new_state == 1:
            self.reset_agent_ABP()
        self.state = new_state

    def reset_agent_ABP(self):
        """
        Reseta o agente para o estado ABP (ativo).
        Chamado quando o agente muda do estado passivo para ativo.
        """
        self.theta_t = 2 * np.pi * self.rng.rand()  # Orientação inicial aleatória
        self.u_t = np.array([
            np.cos(self.theta_t),
            np.sin(self.theta_t)
        ])  # Projeta orientação nos eixos X e Y
        self.dr_theta = self.v * self.u_t * self.state * self.dt  # Componente de movimento ABP

    def update_agent_ABP(self):
        """
        Atualiza os parâmetros do estado ABP.
        Chamado quando o agente permanece no estado ativo.
        """
        self.n_t = self.rng.normal()  # Ruído do movimento ABP
        self.theta_t = self.theta_t + np.sqrt(2 * self.D_theta * self.dt) * self.n_t  # Atualiza orientação
        self.u_t = np.array([
            np.cos(self.theta_t),
            np.sin(self.theta_t)
        ])  # Projeta orientação para eixo x e y
        self.dr_theta = self.v * self.u_t * self.state * self.dt  # Componente de movimento ABP

    def wall_reflection(self, x):
        """
        Reflete a posição do agente na parede, se necessário.
        Atualiza o estado de colisão.
        """
        if x < 0:
            x = -x
            self.colision = 1
        elif x > self.L:
            x = 2 * self.L - x
            self.colision = 1
        return x
    
    def _handle_specular_collision(self):
        """
        Interação de colisão especular ("quicar").
        A posição é refletida, simulando um quique elástico.
        A orientação do movimento ativo não é afetada pela colisão.
        """
        self.colision = 0
        # A função wall_reflection já implementa a lógica de espelho
        self.r[0] = self.wall_reflection(self.r[0])
        self.r[1] = self.wall_reflection(self.r[1])

    def _handle_diffusive_collision(self):
        """
        Interação de colisão difusiva ("parar e reorientar").
        A partícula para na parede e sua orientação de movimento ativo é 
        sorteada aleatoriamente, simulando um "tumble" biológico.
        """
        self.colision = 0
        
        # Armazena o estado de colisão após a verificação do primeiro eixo
        # para que a reorientação aleatória ocorra apenas uma vez.
        colision_after_x = 0
        
        # Eixo X
        if self.r[0] < 0:
            self.r[0] = 0
            colision_after_x = 1
        elif self.r[0] > self.L:
            self.r[0] = self.L
            colision_after_x = 1

        # Eixo Y
        colision_after_y = 0
        if self.r[1] < 0:
            self.r[1] = 0
            colision_after_y = 1
        elif self.r[1] > self.L:
            self.r[1] = self.L
            colision_after_y = 1

        # A colisão final é se ocorreu em qualquer um dos eixos
        self.colision = colision_after_x or colision_after_y
        
        # Se uma colisão de fato ocorreu, reorienta o agente
        if self.colision:
            self.theta_t = 2 * np.pi * self.rng.rand()
            self.u_t = np.array([np.cos(self.theta_t), np.sin(self.theta_t)])
            # Garante que o deslocamento ativo do PRÓXIMO passo já use a nova direção
            if self.state == 1:
                self.dr_theta = self.v * self.u_t * self.state * self.dt
                
    def _handle_repulsive_collision(self):
        """
        Interação de colisão repulsiva ("sentir e desviar").
        Uma força de parede suave é adicionada ao movimento da partícula,
        simulando interações hidrodinâmicas ou eletrostáticas.
        """
        dr_repulsive = np.zeros(2)
        distances = [self.r[0], self.L - self.r[0], self.r[1], self.L - self.r[1]]
        force_directions = [np.array([1, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([0, -1])]

        self.colision = 0
        
        for i in range(4):
            d = distances[i]
            if d < self.repulsive_cutoff:
                if d < 1e-6: d = 1e-6
                
                ratio_inv = self.repulsive_sigma / d
                magnitude = self.repulsive_k * (pow(ratio_inv, self.repulsive_power))
                
                displacement_vector =  magnitude * self.dt
                if displacement_vector > self.repulsive_cutoff:
                    displacement_vector =  force_directions[i]*self.repulsive_cutoff
                else:
                    displacement_vector =  force_directions[i]*displacement_vector
                dr_repulsive += displacement_vector 

                # Define o estado de "colisão" para o agente se a força for significativa
                if magnitude > 1e-6:
                    self.colision = 1
        
        # Adiciona o deslocamento repulsivo ao deslocamento total do passo
        self.dr_dt += dr_repulsive    
    
    def target_distance(self):
        """
        Calcula a distância entre o agente e o alvo.
        Considera condições de contorno periódicas se não houver colisão.
        """
        diff = np.abs(self.r - self.target_position)
        if not self.allow_colision:
            diff = np.minimum(diff, self.L - diff)
        self.distance = np.linalg.norm(diff)

    def action(self):
        """
        Realiza a ação de troca de estado do agente.
        Troca o estado e reseta o timer.
        """
        self.prev_state = self.state
        self.state = 1 - self.state
        self.timer = 0

    def state_observation(self):
        """
        Retorna o estado observável do sistema para o agente.

        Retorno:
        - list: Estado atual do sistema (observável).
        """
        if self.allow_colision:
            return [self.state, self.timer, self.colision]
        return [self.state, self.timer]

    def update_reward(self):
        """
        Atualiza a recompensa do agente.
        Verifica se o agente encontrou o alvo e atualiza a recompensa.
        """
        self.trial_finished = False
        self.reward = 0
        # Recompensa por sair do estado ativo em colisão
        if self.allow_colision:
            if (self.prev_colision) & (self.prev_state) & (not self.state):
                self.reward = self.reward + self.colision_reward
        # Recompensa por encontrar o alvo (apenas no estado BP)
        if (self.distance < self.target_radius) & (not self.state):
            self.reward = self.reward + 1
            self.trial_finished = True

    def update_environment(self, action):
        """
        Atualiza o ambiente conforme a ação do agente.

        Parâmetros:
        - action (bool): Indica se o agente deve trocar de estado.

        Retorno:
        - tuple: (recompensa, flag de término da rodada)
        """
        self.timer += 1  # Atualiza timer do estado
        self.dr_theta = 0  # Não há movimento ABP
        self.prev_colision = self.colision

        # Se houver ação de troca de estado
        if action:
            self.action()  # Troca o estado e reseta o timer
            if self.state == 1:
                self.reset_agent_ABP()  # Inicia variáveis aleatórias do estado
        # Se não houver ação mas o estado for ABP
        elif self.state == 1:
            self.update_agent_ABP()  # Atualiza parâmetros do movimento ABP

        self.update_agent_position()  # Calcula posição final do agente
        self.update_reward()  # Calcula recompensa

        # Se atingiu limite de tempo e não finalizou, troca de estado
        if (self.timer == self.max_steps_per_trial - 1) and (not self.trial_finished):
            self.action()
            if self.state == 1:
                self.reset_agent_ABP()

        return self.reward, self.trial_finished
    
    def update_agent_position(self):
        """
        Atualiza a posição do agente.
        Calcula o movimento base e delega o tratamento da colisão para o método específico.
        """
        # 1. Calcula o deslocamento base (Browniano + Ativo)
        self.E_t = np.array([self.rng.normal(), self.rng.normal()])
        self.dr = np.sqrt(2 * self.D * self.dt) * self.E_t
        self.dr_dt = self.dr_theta + self.dr

        # 2. Lida com as fronteiras do ambiente
        if self.allow_colision:
            # Roteador: chama a função de colisão correta
            if self.collision_type == 'repulsive':
                self._handle_repulsive_collision()
            
            # Adiciona o deslocamento base à posição
            self.r = (self.r + self.dr_dt)

            if self.collision_type == 'specular':
                self._handle_specular_collision()
            elif self.collision_type == 'diffusive':
                self._handle_diffusive_collision()
                
            # vvvv BLOCO DE SEGURANÇA ADICIONADO AQUI vvvv
            # Garante que a partícula nunca escape, independentemente do tipo de colisão
            self.r[0] = np.clip(self.r[0], 0, self.L)
            self.r[1] = np.clip(self.r[1], 0, self.L)

        else: # Caso sem colisão (condições de contorno periódicas)
            self.r = (self.r + self.dr_dt) % self.L
            
        # 3. Calcula a distância final ao alvo
        self.target_distance()



    def save(self, path):
        """
        Salva o objeto do ambiente em arquivo binário.

        Parâmetros:
        - path (str): Caminho para salvar o ambiente.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path + '/environment', 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path):
        """
        Carrega o objeto do ambiente de um arquivo binário.

        Parâmetros:
        - path (str): Caminho do arquivo salvo.

        Retorno:
        - Ambiente carregado.
        """
        if not os.path.exists(path):
            raise Exception("Diretório inexistente")
        else:
            with open(path + '/environment', 'rb') as f:
                return pickle.load(f)