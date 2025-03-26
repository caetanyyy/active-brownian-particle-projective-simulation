import numpy as np
import pickle
import os
import time

class PsEnvironment(object):
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

    def __init__ (self, L:float, Pe:float, l:float, tao:int, dt:float, allow_colision:bool = False):
        """
        Initializes the Environment object.

        Args:
            L (float): Dimension of the space.
            Pe (float): Péclet number.
            l (float): Length scale.
            tao (int): Maximum steps per trial.
            dt (float): Time step size.
            allow_colision (bool): Allow colision (true) or not (false)

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

        if allow_colision:
            self.num_percepts_list = [self.num_states, self.max_steps_per_trial, self.colision_state] # Tamanho dos observaveis
        
        else:
            self.num_percepts_list = [self.num_states, self.max_steps_per_trial] # Tamanho dos observaveis
        
        # Observáveis
        self.state = 0 #0 ou 1
        self.prev_state = 0 #0 ou 1 # guarda o estado anterior

        self.timer = 0 #inteiro que contabiliza a quantidade de rodadas que o agente está em um estado

        self.colision = 0 #0 ou 1, mapeia se o agente teve colisão ou não com a parede
        self.prev_colision = 0 #0 ou 1, mapeia se o agente teve colisão no passo anterior

        #Recompensa
        self.reward = 0 # Inicia a recompensa como zero
        self.trial_finished = False # Inicia o episódio
        self.colision_reward = 0.01
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

    def reset_rng(self, seed = None):
        """
        Resets the random number generator.

        Args:
            seed (int): Seed for the random number generator. If None, a random seed is used.
        """
        # Gerador aleatório da classe:
        self.rng = np.random.RandomState(seed)

    def reset_target(self):
        """
        Resets the target position.

        Always called when a trial finishes, whether the agent found the target or not.
        """
        self.target_position = np.array([
            self.rng.rand()*self.L, 
            self.rng.rand()*self.L
        ])

    def reset_agent_state(self, new_state):
        """
        Resets the agent state to a specific state.

        Args:
            new_state (int): New state for the agent (0 or 1).
        """
        self.timer = 0
        if new_state == 1:
            self.reset_agent_ABP()
        self.state = new_state

    def reset_agent_ABP(self):
        """
        Resets the agent state to ABP.

        Called when the agent changes from passive to ABP state.
        """
        self.theta_t = 2*np.pi*self.rng.rand() #Inicia a magnetude da orientação aleatória do ABP
        self.u_t = np.array([
            np.cos(self.theta_t), 
            np.sin(self.theta_t)]) # Projeta a magnetude nos eixos X e Y
        self.dr_theta = self.v*self.u_t*self.state*self.dt # Calcula componente e movimento ABP

    def update_agent_ABP(self):
        """
        Updates the ABP state parameters.

        Called when the agent remains in the ABP state.
        """
        self.n_t = self.rng.normal() #Calcula ruído do movimento ABP
        self.theta_t = self.theta_t + np.sqrt(2*self.D_theta*self.dt)*self.n_t #Atualiza magnetude da orentação aleatória do ABP
        self.u_t = np.array([
            np.cos(self.theta_t), 
            np.sin(self.theta_t)]) #projeta a orientação para eixo x e y
        self.dr_theta = self.v*self.u_t*self.state*self.dt # Calcula componente de movimento ABP

    def update_agent_position(self):
        """
        Updates the agent's position.

        Calculates the final position of the agent based on its current state and movement parameters.
        """
        self.E_t = np.array([
            self.rng.normal(), 
            self.rng.normal()
        ]) # Calcula o ruído do movimento BP
        self.dr = np.sqrt(2*self.D*self.dt)*self.E_t # Calcula a componente de movimento BP
        self.dr_dt = self.dr_theta + self.dr
        self.r = (self.r + self.dr_dt)

        if self.allow_colision: # Se pode haver colisão
            self.colision = 0
            self.r[0] = self.wall_reflection(self.r[0]) # Detecta colisão no eixo X
            self.r[1] = self.wall_reflection(self.r[1]) # Detecta colisão no eixo Y 
            
        else: # Se o agente não colide na parede, as condições são periódicas
            self.r = self.r%self.L # Atualiza posição do agente de acordo com as condições periódicas

        # Calcula a distância entre o agente e o alvo
        self.target_distance()


    def wall_reflection(self, x):
        if x < 0:
            x = -x
            self.colision = 1

        elif x > self.L:
            x = 2*self.L - x
            self.colision = 1
        return x
    
    def target_distance(self):
        diff = np.abs(self.r - self.target_position)
        if not self.allow_colision: #Condições de contorno periódicas
            diff = np.minimum(diff, self.L - diff)

        self.distance = np.linalg.norm(diff)

    def action(self):
        """
        Performs an action of changing the agent's state.

        Changes the agent's state and resets the timer.
        """
        self.prev_state = self.state
        self.state = 1 - self.state # Troca de estado
        self.timer = 0 # Reseta o timer

    def state_observation(self):
        """
        Returns the current state of the system.

        Returns:
            list: Current state of the system (observable).
        """
        if self.allow_colision:
            return [self.state, self.timer, self.colision]
        
        return [self.state, self.timer]

    def update_reward(self):
        """
        Updates the agent's reward.

        Checks if the agent has found the target and updates the reward accordingly.
        """
        self.trial_finished = False 
        self.reward = 0
        
        ## Criar uma recompensa para quando o target sai da colisão
        if self.allow_colision:
            if (self.prev_colision) & (self.prev_state) & (not self.state):
            #if (self.prev_colision) & (not self.state):
                self.reward = self.reward + self.colision_reward

        # se encontrou o target (apenas no estado BP), ganha a recompensa e reseta a posição do target
        if (self.distance < self.target_radius) & (not self.state): 
            self.reward = self.reward + 1
            self.trial_finished = True

    def update_environment(self, action): 
        """
        Updates the environment based on the agent's action.

        Performs the necessary updates to the environment based on the agent's action.

        Args:
            action (bool): Flag indicating if the agent should change its state.

        Returns:
            tuple: A tuple containing the reward and a flag indicating if the trial has finished.
        """

        self.timer += 1 #atualiza timer do estado
        self.dr_theta = 0 #Não há movimento ABP

        #Se houver ação de troca de estado
        if action: 
            self.action() # Troca o estado e reseta o timer
            if self.state == 1: # Se trocou para o estado ABP
                self.reset_agent_ABP() # Inicia variáveis aleatórias do estado
                
        # Se não houver ação mas o estado for ABP
        elif self.state == 1: # se está em ABP
            self.update_agent_ABP() # Atualiza parâmetros do movimento ABP                
        
        # Calcula a posição final do agente
        self.update_agent_position()

        # Calcula a recompensa
        self.update_reward()
        
        # Se alcançou o limite de tempo em um estado e o episódio de treino não foi finalizado, troca de estado
        if (self.timer == self.max_steps_per_trial - 1) & (not self.trial_finished):
            self.action()
            if self.state == 1: # Se o no estado é o estado ABP
                self.reset_agent_ABP()

        self.prev_colision = self.colision
        
        return self.reward, self.trial_finished
    
    def save(self, path):
        """
        Saves the environment object to a file.

        Args:
            path (str): Path to the directory where the environment object will be saved.
        """
        if not os.path.exists(path):
            os.makedirs(path)

        with open(path + '/environment', 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path):
        """
        Loads the environment object from a file.

        Args:
            path (str): Path to the directory where the environment object is saved.

        Returns:
            Environment: The loaded environment object.
        """
        if not os.path.exists(path):
            raise Exception("Diretório inexistente")
        else:
            with open(path + '/environment', 'rb') as f:
                return pickle.load(f)