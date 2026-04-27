import sys
import numpy as np
import argparse
import datetime
import contextlib
import json
import joblib
from tqdm import tqdm
from joblib import Parallel, delayed
import gc
import time
import os
import pandas as pd # <-- ADICIONADO

# ==============================================================================
#      MUDANÇA 1: Importar as novas classes
# ==============================================================================
from environments.maze_environment import MazeEnvironment
from simulations.projective_simulation_iteration import ProjectiveSimulation
from agents.ps_agent import PsAgent

env_class = MazeEnvironment
agent_class = PsAgent
ps_model = ProjectiveSimulation

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager para integrar o progresso do joblib ao tqdm.
    """
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

def float_range(mini, maxi):
    """
    Retorna uma função para validar argumentos float em um intervalo: mini <= arg <= maxi.
    """
    def float_range_checker(arg):
        """
        Função de validação para argparse - float dentro do intervalo definido.
        """
        try:
            f = float(arg)
        except ValueError:
            raise argparse.ArgumentTypeError("deve ser um número real")
        if f < mini or f > maxi:
            raise argparse.ArgumentTypeError(f"deve estar no intervalo [{mini} .. {maxi}]")
        return f
    return float_range_checker

def float_min_range(mini):
    """
    Retorna uma função para validar argumentos float: mini <= arg.
    """
    def float_range_checker(arg):
        """
        Função de validação para argparse - float acima de um mínimo.
        """
        try:
            f = float(arg)
        except ValueError:
            raise argparse.ArgumentTypeError("deve ser um número real")
        if f < mini:
            raise argparse.ArgumentTypeError(f"deve ser maior ou igual a {mini}")
        return f
    return float_range_checker

def read_args():
    """
    Lê os argumentos da linha de comando para a simulação no labirinto.
    """
    parser = argparse.ArgumentParser(
        prog='run_maze_simulation',
        description='Executa uma simulação de agente em um labirinto definido por arquivo.'
    )
    
    # ==============================================================================
    #      MUDANÇA 2: Atualizar os argumentos
    # ==============================================================================
    parser.add_argument(
        "--maze_file",
        help="Caminho para o arquivo .txt que define o layout do labirinto. (Obrigatório)",
        type=str,
        required=True
    )
    parser.add_argument("--box_size", '--L', type=float_min_range(0), default=100)
    parser.add_argument("--peclet_number", '--p', type=float_min_range(0), default=100)
    parser.add_argument("--persistence", '--l', type=float_min_range(0), default=1)
    parser.add_argument("--dt", "--t", type=float_range(1e-4, 1), default=1)
    parser.add_argument("--num_episodes", '--n', type=int, default=100)
    parser.add_argument("--num_reflections", '--r',help="Quantidade de reflexões do agente", type = int,default=0)
    parser.add_argument("--n_jobs", "--nj", type=int, default=1)
    parser.add_argument("--n_sim", "--ns", type=int, default=1)
    parser.add_argument("--save_path", type=str, default='')
    parser.add_argument("--load_path", type=str, default='')
    parser.add_argument("--save_trajectory", help="Salva a trajetória (x, y) do agente em um arquivo trajectory.txt.",action='store_true')
    parser.add_argument("--policy", type=str, default='standard', choices=['standard','softmax'])
    parser.add_argument("--beta_softmax", '--b', type=float, default=1)
    parser.add_argument("--gamma_damping", '--g', type=float_range(0,1), default=1e-3)
    parser.add_argument("--eta_damping", '--e', type=float_range(0,1), default=1e-3)
    parser.add_argument("--damping_flag", type=bool, default=True)

    parser.add_argument("--target_reward", type=float_range(0,1), default=1)
    
    args = parser.parse_args()
    if args.damping_flag:
        args = damping_params(args)
    
    args.tao = int(args.dt/1e-4)
    args.max_steps_per_episode = int(20 * args.tao)
    
    return args

def damping_params(args):
    """
    Define os parâmetros de damping de acordo com o número de Péclet.
    """
  
    # Parâmetros de damping em função de Pe, de acordo com o artigo (seção de Métodos)
    damping_param = {
        0:{
            'gamma_damping' : 0,
            'eta_damping' : 0
        },        
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

    args.gamma_damping = damping_param[args.peclet_number]['gamma_damping']
    args.eta_damping = damping_param[args.peclet_number]['eta_damping']
    
    return args

def create_models(args):
    """
    Cria e retorna as instâncias do agente e do ambiente de labirinto.
    """
    # 1. Inicia o ambiente de labirinto
    env = env_class(
        maze_file=args.maze_file,
        L=args.box_size, 
        Pe=args.peclet_number, 
        l=args.persistence, 
        tao=args.tao, 
        dt=args.dt,
        target_reward=args.target_reward,
    )
    
    # 2. Inicia o agente
    agent = agent_class(
        env.num_actions, 
        env.num_percepts_list, 
        args.gamma_damping, 
        args.eta_damping, 
        args.policy, 
        args.beta_softmax,
        num_reflections=args.num_reflections# num_reflections pode não fazer mais sentido neste contexto
    )
    
    for timer_val in range(env.max_steps_per_trial):
        for colision_val in range(env.colision_state):
            for state_val in range(env.num_states):
                observation = [state_val, timer_val, colision_val]
                percept = agent.percept_preprocess(observation)
                if state_val == 0:
                    agent.h_matrix[1, percept] = 1e-2 # Prob. de se tornar ativo
                else: # state_val == 1
                    agent.h_matrix[1, percept] = 1e-3 # Prob. de se tornar passivo (menor)
    
    agent.h_matrix[0, :] = 1 - agent.h_matrix[1, :]
    agent.h0_matrix = np.copy(agent.h_matrix)
    
    return agent, env

def save_data(model, args, learning_process, ep, filename_time, prev_episodes, load_path):
    """
    Salva os dados do modelo, argumentos e curva de aprendizado.
    """
    if len(load_path) > 0:
        args['num_episodes'] = ep + prev_episodes
        model.save(load_path)
        with open(load_path + '/args.json', 'w') as fp:
            json.dump(args, fp)
        np.savetxt(load_path + '/learning_process.txt', learning_process, fmt='%.4f', delimiter=',')
        np.savetxt(load_path + '/h_matrix.txt', model.h_matrix(), fmt='%.2f', delimiter=',')

    else:
        args['num_episodes'] = ep + prev_episodes
        model.save(args['save_path'] + '/' + filename_time)
        with open(args['save_path'] + '/' + filename_time +'/args.json', 'w') as fp:
            json.dump(args, fp)
        np.savetxt(args['save_path'] + '/' + filename_time +'/learning_process.txt', learning_process, fmt='%.4f', delimiter=',')
        np.savetxt(args['save_path'] + '/' + filename_time +'/h_matrix.txt', model.h_matrix(), fmt='%.2f', delimiter=',')

def main(args, sim, load_path=''):
    """
    Função principal para executar a simulação.
    """
    # Gera o nome do arquivo para novos modelos
    filename_time = '{date:%Y-%m-%d_%H-%M-%S.%f}'.format(date=datetime.datetime.now()) + f'__{sim}'
    
    # Define o número de episódios a serem executados nesta chamada
    # CORREÇÃO: Acessa 'args' com ponto (objeto Namespace)
    num_episodes_to_run = args.num_episodes

    save_directory = ''
    if len(load_path) > 0:
        model = ps_model.load(load_path)
        
        try:
            learning_process = np.loadtxt(f'{load_path}/learning_process.txt')
            if learning_process.ndim == 0: learning_process = np.array([learning_process])
            prev_episodes = len(learning_process)
        except (IOError, ValueError):
            learning_process = np.array([])
            prev_episodes = 0
            
        with open(f'{load_path}/args.json', 'r') as file:
            # Carrega os argumentos salvos como um dicionário
            current_args_dict = json.load(file)
        
        save_directory = load_path
        max_steps_per_episode = current_args_dict['max_steps_per_episode']

    else:
        agent, env = create_models(args)
        model = ps_model(agent, env)
        learning_process = np.array([])
        prev_episodes = 0
        # CORREÇÃO: Converte o objeto 'args' em um dicionário para uso consistente
        current_args_dict = vars(args)
        max_steps_per_episode = args.max_steps_per_episode
        if len(args.save_path) > 0:
            save_directory = os.path.join(args.save_path, filename_time)

    # Configuração do arquivo de trajetória
    trajectory_file = None
    if len(save_directory) > 0 and args.save_trajectory:
        trajectory_filepath = os.path.join(save_directory, 'trajectory.txt')
        os.makedirs(save_directory, exist_ok=True)
        trajectory_file = open(trajectory_filepath, 'ab')

    # Laço de treinamento
    for ep in range(num_episodes_to_run):
        step, trajectory_data = model.run_episode(max_steps_per_episode, track_trajectory=args.save_trajectory)
        
        learning_process = np.append(learning_process, step / model.env.max_steps_per_trial)

        if trajectory_file and trajectory_data:
            np.savetxt(trajectory_file, np.array(trajectory_data), fmt='%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f', delimiter=',')

        if len(save_directory) > 0:
            # Passa o dicionário 'current_args_dict' para a função de salvamento
            save_data(model, current_args_dict, learning_process, ep + 1, filename_time, prev_episodes, load_path)

    if trajectory_file:
        trajectory_file.close()

    if len(load_path) == 0 and 'agent' in locals():
        del agent
        del env
    del model
    del learning_process
    gc.collect()
    
    return 0

if __name__ == "__main__":
    args = read_args()
    start_time = time.time()
    n_jobs = args.n_jobs
    n_sim = args.n_sim
    if len(args.save_path):
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path, exist_ok = True)

    if len(args.load_path) > 0:
        args.load_path = args.load_path.split(",")

    # Se for realizada a paralelização:
    # Se tiver mais de um arquivo para carregar
    if (n_jobs != 1):
        if len(args.load_path) > 0:
            n_sim = len(args.load_path)
            with tqdm_joblib(
                tqdm(
                    desc = "Simulações finalizadas:", 
                    total = n_sim, 
                    position = 0
                )
            ) as progress_bar:
                Parallel(
                    n_jobs = n_jobs,
                    backend = "multiprocessing"
                )(delayed(main)(args, sim, args.load_path[sim]) for sim in range(n_sim))

        elif (n_sim > 1):
            with tqdm_joblib(
                tqdm(
                        desc = "Simulações finalizadas:", 
                        total = n_sim, 
                        position = 0
                    )
            ) as progress_bar:
                Parallel(
                    n_jobs = n_jobs,
                    backend = "multiprocessing"
                )(delayed(main)(args, sim) for sim in range(n_sim))
    
    # Se for execução sequencial:
    else:
        if len(args.load_path) > 0:
            for sim in tqdm(range(len(args.load_path)), position = 0):
                main(args, sim, args.load_path[sim])
        else:
            for sim in tqdm(range(args.n_sim), position = 0):
                main(args, 0)

    print("--- %s seconds ---" % (time.time() - start_time))

    gc.collect()
    sys.exit(0)

# Monitorar threads: https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution
# Thread x Process: https://stackoverflow.com/questions/3044580/multiprocessing-vs-threading-python
# Comportamento de np.random com cada tipo de backend: https://joblib.readthedocs.io/en/latest/auto_examples/parallel_random_state.html
# Paralelição: https://stackoverflow.com/questions/9786102/how-do-i-parallelize-a-simple-python-loop