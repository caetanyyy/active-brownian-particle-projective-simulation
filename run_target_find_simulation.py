import sys
import numpy as np
import argparse
import datetime
import contextlib
import joblib
from tqdm import tqdm
from joblib import Parallel, delayed
import gc
import time

sys.path.insert(0, 'agents')
sys.path.insert(0, 'environments')

import env_abp_target_finding as env_class
import ps_agent_basic_alternativo as agent_class

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

def float_range(mini,maxi):
    """Return function handle of an argument type function for 
       ArgumentParser checking a float range: mini <= arg <= maxi
         mini - minimum acceptable argument
         maxi - maximum acceptable argument"""

    # Define the function with default arguments
    def float_range_checker(arg):
        """New Type function for argparse - a float within predefined range."""

        try:
            f = float(arg)
        except ValueError:    
            raise argparse.ArgumentTypeError("must be a floating point number")
        if f < mini or f > maxi:
            raise argparse.ArgumentTypeError("must be in range [" + str(mini) + " .. " + str(maxi)+"]")
        return f

    # Return function handle to checking function
    return float_range_checker

def float_min_range(mini):
    """Return function handle of an argument type function for 
       ArgumentParser checking a float range: mini <= arg
         mini - minimum acceptable argument
         """

    # Define the function with default arguments
    def float_range_checker(arg):
        """New Type function for argparse - a float within predefined range."""

        try:
            f = float(arg)
        except ValueError:    
            raise argparse.ArgumentTypeError("must be a floating point number")
        if f < mini:
            raise argparse.ArgumentTypeError("must be in range [" + str(mini) + " ... inf]")
        return f

    # Return function handle to checking function
    return float_range_checker

def read_args():
    parser = argparse.ArgumentParser(
        prog='run_target_find_simulation',
        description='Realiza a simulação de busca de alvo por uma partícula Browniana de estados ativos e passivos através de aprendizado pro reforço com simulação projetiva',
        #epilog='Text at the bottom of help'
    )
    
    parser.add_argument(
        "--filename","--f",
        help="Nome do arquivo final", 
        type = str,
        default='target_find_simulation'
    )
    
    parser.add_argument(
        "--box_size", '--L',
        help="Tamanho L da caixa.", 
        type = float_min_range(0),
        default=100
    )

    parser.add_argument(
        "--peclet_number", '--p',
        help = "Péclet number - Pe := (v*tao)/L", 
        type = float_min_range(0),
        default = 100
    )

    parser.add_argument(
        "--persistence", "--l*", '--l',
        help = "Persistência do movimento do estado ABP - l* := v/(D_theta*L)", 
        type = float_min_range(0),
        default = 1
    )

    parser.add_argument(
        "--dt", "--t",
        help="Intervalo de tempo de cada passo da simulação. Deve estar no intervalo [1e-4 e 1] Por padrão, o tempo típico (Tao) := dt x 1e4", 
        type=float_range(1e-4, 1),
        default=1
    )

    parser.add_argument(
        "--policy", '--prob',
        help = "Tipo de cálculo da matriz de probabilidade: standard ou softmax", 
        type = str,
        default = 'standard',
        choices=['stadard','softmax']
    )

    parser.add_argument(
        "--beta_softmax", '--b',
        help = "Parâmetros β da função softmax.", 
        type = float,
        default = 1
    )

    parser.add_argument(
        "--num_episodes", '--n',
        help = "Número de episódios de treino. Deve ser maior que zero e inteiro.", 
        type = int,
        default = 100
    )

    parser.add_argument(
        "--damping_flag", '--d',
        help = "Utilizar valores otimizados de damping para o modelo de simulação projetiva mapeados pelo artigo? Padrão: True",
        type = bool,
        default = True,
        choices = [True, False]
    )

    parser.add_argument(
        "--eta_damping", '--e',
        help="Parâmetro η de esquecimento da matriz de glow do agente. Apenas se --damping_param == False", 
        type = float_range(0,1),
        default=1e-3
    )

    parser.add_argument(
        "--gamma_damping", '--g',
        help="Parâmetro γ de esquecimento da matriz de pesos do agente. Apenas se --damping_param == False", 
        type = float_range(0,1),
        default=1e-3
    )

    parser.add_argument(
        "--num_reflections", '--r',
        help="Quantidade de reflexões do agente", 
        type = int,
        default=0
    )

    parser.add_argument(
        "--n_jobs", "--nj",
        help="Quantidade de jobs", 
        type = int,
        default=1
    )

    parser.add_argument(
        "--n_sim", "--ns",
        help="Quantidade de simulações", 
        type = int,
        default=1
    )

    args = parser.parse_args()

    if args.damping_flag:
        args = damping_params(args)

    # De acordo com o artigo
    args.tao = int(args.dt/1e-4)
    args.max_steps_per_episode = int(20*args.tao)

    return args

def damping_params(args):
  
    # Parâmetros de damping em função de Pe, de acordo com o artigo (seção de Métodos)
    damping_param = {
        5:{
            'gamma_damping' : 1e-7,
            'eta_damping' : 1e-2
            },
        10:{
            'gamma_damping' : 1e-6,
            'eta_damping' : 1e-3
            },
        20:{
            'gamma_damping' : 1e-6,
            'eta_damping' : 1e-3
            },
        50:{
            'gamma_damping' : 1e-6,
            'eta_damping' : 1e-2
            },
        100:{
            'gamma_damping' : 1e-5,
            'eta_damping' : 1e-2
            }
    }

    idx = np.argmin([abs(args.peclet_number - pe) for pe in damping_param.keys()])
    idx = list(damping_param.keys())[idx]
    args.gamma_damping = damping_param[idx]['gamma_damping']
    args.eta_damping = damping_param[idx]['eta_damping']
    
    return args

def create_models(args):

    # Inicia ambiente
    env = env_class.Environment(
        args.box_size, 
        args.peclet_number, 
        args.persistence, 
        args.tao, 
        args.dt
    )
    
    # Inicia agente
    agent = agent_class.Agent(
        env.num_actions, 
        env.num_percepts_list, 
        args.gamma_damping, 
        args.eta_damping, 
        args.policy, 
        args.beta_softmax,
        args.num_reflections
    )

    # Inicia matriz de probabilidade de acordo com o artigo
    for step in range(env.max_steps_per_trial):
        for state in range(env.num_states):
            percept = agent.percept_preprocess([state,step])
            if state == 0:
                agent.h_matrix[1, percept] = 1e-2
                agent.h_matrix[0, percept] = 1 - 1e-2

            elif state == 1:
                agent.h_matrix[1, percept] = 1e-3
                agent.h_matrix[0, percept] = 1 - 1e-3

    # Matriz h0 inicial é estática
    agent.h0_matrix = agent.h_matrix

    return agent, env

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

    def fit(self, num_episodes, max_steps_per_episode, job_id = None):

        learning_process = np.zeros(num_episodes)
        #for ep in tqdm(range(num_episodes), desc = f"Total de episódios do job {job_id}", position = job_id+1):
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
        
def save_file(filename, array, args, filename_time = ''):

    filename_id = '__'.join([f'{value}' for key, value in vars(args).items()])
    filename = filename + "___" + filename_id + "____" + filename_time 
    path = 'target_find_data/' + filename + '.txt'

    np.savetxt(path, array, fmt='%.4f', delimiter=',')

def main(args, sim_id):
    # Gera os modelos
    agent, env = create_models(args)
    # Gera a classe de simulação
    model = ProjectiveSimulation(agent, env)
    # Treina os modelos
    learning_process = model.fit(args.num_episodes, args.max_steps_per_episode, sim_id)
    # Salva os dados
    filename_time = '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.datetime.now())
    save_file('learning_process', learning_process, args, filename_time)
    save_file('h_matrix', model.h_matrix(), args, filename_time)
    #save_file('g_matrix', model.g_matrix(), args, filename_time)
    #save_file('ho_matrix', model.ho_matrix(), args, filename_time)

    del agent
    del model
    del learning_process
    del filename_time
    gc.collect()
    
    return 0

if __name__ == "__main__":
    args = read_args()
    # Monitorar threads: https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution
    # Thread x Process: https://stackoverflow.com/questions/3044580/multiprocessing-vs-threading-python
    # Comportamento de np.random com cada tipo de backend: https://joblib.readthedocs.io/en/latest/auto_examples/parallel_random_state.html
    # Paraleliza se necessário https://stackoverflow.com/questions/9786102/how-do-i-parallelize-a-simple-python-loop
    start_time = time.time()

    # Se for realizada a paralelização
    if (args.n_jobs != 1) & (args.n_sim > 1):
        print('Iniciando paralelização:')

        with tqdm_joblib(tqdm(desc="Simulações finalizadas:", total=args.n_sim, position = 0)) as progress_bar:
            Parallel(
                n_jobs = args.n_jobs,
                #verbose = 10,
                backend = "multiprocessing"
            )(delayed(main)(args, sim) for sim in range(args.n_sim))
    
    # Se for sequencial
    else:
        print('Iniciando simulações:')
        for sim in tqdm(range(args.n_sim)):
            main(args, sim)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    gc.collect()
    sys.exit(0)