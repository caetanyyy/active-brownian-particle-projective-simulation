import numpy as np
import pickle

class Environment(object):
    def __init__ (self, L, Pe, l, tao, dt):
        # Gerador aleatório da classe:
        self.rng = np.random.RandomState(None)

        # Inicia variáveis do ambiente
        self.max_steps_per_trial = tao # tempo máximo de uma rodada
        self.num_states = 2 # 1 = ativo ou 0 = passivo
        self.num_actions = 2 # 1 = troca de estado, 0 = mantem estado
        self.num_percepts_list = [self.num_states, self.max_steps_per_trial] # Tamanho dos observaveis
        
        self.reward = 0 # Inicia a recompensa como zero
        self.trial_finished = False # Inicia o episódio

        self.L = L # Dimensão do espaço

        # Estado inicial do agente
        self.r = np.array([0, 0]) #keeps track of where the agent is located
        self.state = 1 #0 ou 1
        self.timer = 0 #inteiro que contabiliza a quantidade de rodadas que o agente está em um estado
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

    def reset_rng(self, seed = None):
        # Gerador aleatório da classe:
        self.rng = np.random.RandomState(seed)

    # Reseta estado do target
    # Sempre que um episódio finalizar, seja porque o agente encontrou o target ou porque não encontrou, reinicia a posição do target
    def reset_target(self):
        self.target_position = np.array([
            self.rng.rand()*self.L, 
            self.rng.rand()*self.L])

    # Reinicia estado do agente para um estado específico
    def reset_agent_state(self, new_state):
        self.timer = 0
        if new_state == 1:
            self.reset_agent_ABP()
        self.state = new_state

    # Reseta estado do agente caso ele mude de passivo para ABP
    def reset_agent_ABP(self):
        self.theta_t = 2*np.pi*self.rng.rand() #Inicia a magnetude da orientação aleatória do ABP
        self.u_t = np.array([
            np.cos(self.theta_t), 
            np.sin(self.theta_t)]) # Projeta a magnetude nos eixos X e Y
        self.dr_theta = self.v*self.u_t*self.state*self.dt # Calcula componente e movimento ABP

    # Atualiza parâmetros do estado ABP caso ele se mantenha no estado ABP
    def update_agent_ABP(self):
        self.n_t = self.rng.normal() #Calcula ruído do movimento ABP
        self.theta_t = self.theta_t + np.sqrt(2*self.D_theta*self.dt)*self.n_t #Atualiza magnetude da orentação aleatória do ABP
        self.u_t = np.array([
            np.cos(self.theta_t), 
            np.sin(self.theta_t)]) #projeta a orientação para eixo x e y
        self.dr_theta = self.v*self.u_t*self.state*self.dt # Calcula componente de movimento ABP

    # Atualiza posição final do agente 
    def update_agent_position(self):
        self.E_t = np.array([
            self.rng.normal(), 
            self.rng.normal()]) # Calcula o ruído do movimento BP
        self.dr = np.sqrt(2*self.D*self.dt)*self.E_t # Calcula a componente de movimento BP
        self.r = (self.r + self.dr_theta + self.dr)%self.L # Atualiza posição do agente
        self.distance = np.linalg.norm(self.r - self.target_position) # Calcula distância do target

    # Troca o estado do agente caso seja tomada a ação
    def action(self):
        self.state = 1 - self.state # Troca de estado
        self.timer = 0 # Reseta o timer

    # Retorna estado atual do sistema (observável)
    def state_observation(self):
        return [self.state, self.timer]

    # Calcula a recompensa do agente
    def update_reward(self):
        self.trial_finished = False 
        self.reward = 0
        
        # se encontrou o target (apenas no estado BP), ganha a recompensa e reseta a posição do target
        if (self.distance < self.target_radius) & (not self.state): 
            self.reward = 1
            self.trial_finished = True

    # Realiza mudanças do ambiente
    def update_environment(self, action): 

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

        return self.reward, self.trial_finished
    
    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        with open(path + '/environment', 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path):
        if not os.path.exists(path):
            raise Exception("Diretório inexistente")
        else:
            with open(path + '/environment', 'rb') as f:
                return pickle.load(f)