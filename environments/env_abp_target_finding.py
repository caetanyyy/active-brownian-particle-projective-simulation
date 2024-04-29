# -*- coding: utf-8 -*-
"""
Copyright 2018 Alexey Melnikov and Katja Ried.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.

Please acknowledge the authors when re-using this code and maintain this notice intact.
Code written by Katja Ried, implementing ideas from 

'Projective simulation applied to the grid-world and the mountain-car problem'
Alexey A. Melnikov, Adi Makmal & Hans J. Briegel
arXiv:1405.5459 (2014)

and 

'Meta-learning within projective simulation'
Adi Makmal, Alexey A. Melnikov, Vedran Dunjko & Hans J. Briegel
IEEE Access 4, pp. 2110-2122 (2016) doi:10.1109/ACCESS.2016.2556579
"""

#This code requires the following packages
import numpy as np

class TaskEnvironment(object):
    def __init__ (self, L, Pe, l, tao, dt):
        self.max_steps_per_trial = tao # tempo máximo de uma rodada
        self.num_states = 2 # 1 = ativo ou 0 = passivo
        self.num_actions = 2 # 1 = troca de estado, 0 = mantem estado
        
        self.num_percepts_list = [self.num_states, self.max_steps_per_trial]
        self.L = L

        self.r = np.array([0, 0]) #keeps track of where the agent is located
        self.state = 0 #0 ou 1
        self.timer = 0 #inteiro que contabiliza a quantidade de rodadas que o agente está em um estado

        self.target_radius = 0.05*L #tamanho do target
        self.target_position = np.array([
            np.random.rand()*self.L, 
            np.random.rand()*self.L
        ]) #posição do target [x,y]
        
        self.distance = L
        
        # Parametros de movimento
        self.v = Pe*L/(tao) #Velocidade translacional
        self.D = (L*L)/(4*tao) #Coeficiente translacional
        self.D_theta = self.v/(l*L) #Coeficiente rotacional
        self.dt = dt # Período de tempo

        self.theta_t = 2*np.pi*np.random.rand() #Rotação do movimento
        self.n_t = np.random.normal() #ruído escalar
        self.u_t = np.array([np.cos(self.theta_t), np.sin(self.theta_t)]) #orientação do movimento ativo
        self.E_t = np.array([np.random.normal(),np.random.normal()]) #vetor ruído

        self.dr_theta = 0 # translação ativa
        self.dr = 0 #translação passiva

    def reset(self):

        # reseta posição do target
        self.target_position = np.array([np.random.rand()*self.L, np.random.rand()*self.L])
        self.state = 0 #reseta o estado da partícula
        self.timer = 0 #reseta o timer de estado

        return [self.state, self.timer]
    
    def move(self, action): 

        if action: #Se a ação é trocar de estado
            self.state = 1 - self.state #troca de estado
            self.timer = 0 # reseta o timer

            if self.state == 1: # se o estado é 1
                self.theta_t = 2*np.pi*np.random.rand() #inicia a orientação aleatória
                self.u_t = np.array([np.cos(self.theta_t), np.sin(self.theta_t)]) #projeta a orientação para eixo x e y
                
        else: # Se o estado se manter
            if self.state == 1: # se está em ABP
                self.n_t = np.random.normal() #calcula o ruído da orientação
                self.theta_t = self.theta_t + np.sqrt(2*self.D_theta*self.dt)*self.n_t #atualiza a orientação
                self.u_t = np.array([np.cos(self.theta_t), np.sin(self.theta_t)]) #projeta a orientação para eixo x e y
            self.timer += 1 #atualiza timer do estado
        
        self.E_t = np.array([np.random.normal(), np.random.normal()]) #calcula o ruído do movimento BP
        
        self.dr_theta = self.v*self.u_t*self.state*self.dt #calcula o movimento ABP
        self.dr = np.sqrt(2*self.D*self.dt)*self.E_t #calcula o movimento BP
        self.r = (self.r + self.dr_theta + self.dr)%self.L #atualiza posição
        
        self.distance = np.linalg.norm(self.r - self.target_position) #calcula distância do target
        trial_finished = False 
        reward = 0
        
        # se encontrou o target (apenas no estado BP), ganha a recompensa e reseta a posição do target
        if (self.distance < self.target_radius) & (not self.state): 
            reward = 1
            trial_finished = True
            self.reset()

        # se alcançou o limite de tempo em um movimento, troca de movimento
        elif self.timer == self.max_steps_per_trial - 1:
            self.state = 1 - self.state
            self.timer = 0

            if self.state == 1: # se o estado é 1
                self.theta_t = 2*np.pi*np.random.rand() #inicia a orientação aleatória
                self.u_t = np.array([np.cos(self.theta_t), np.sin(self.theta_t)]) #projeta a orientação para eixo x e y

        return [self.state, self.timer], reward, trial_finished