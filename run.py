"""
Projeto: active-brownian-particle-projective-simulation

Este script executa a simulação de busca de alvo por partículas Brownianas com estados ativos e passivos, utilizando aprendizado por reforço via Simulação Projetiva.

Principais Classes:

1. PsAgent (agents/ps_agent.py):
   - Implementa o agente de Simulação Projetiva (PS).
   - Principais métodos:
     * deliberate(observation): Decide a ação do agente com base na observação.
     * learn(reward): Atualiza os pesos do agente conforme a recompensa.
     * probability_distr(percept): Calcula a distribuição de probabilidade das ações.
     * reset_glow_matrix(): Reseta a matriz de glow.
     * save(path)/load(path): Salva/carrega o agente.

2. PsEnvironment (environments/ps_env_abp_target_find.py):
   - Simula o ambiente 2D para o agente ABP encontrar o alvo.
   - Principais métodos:
     * state_observation(): Retorna o estado observável do ambiente.
     * update_environment(action): Atualiza o ambiente conforme a ação do agente.
     * reset_target()/reset_agent_state(): Reseta o alvo ou o estado do agente.
     * save(path)/load(path): Salva/carrega o ambiente.

3. ProjectiveSimulation (simulations/projective_simulation_iteration.py):
   - Gerencia o ciclo de aprendizado do agente no ambiente.
   - Principais métodos:
     * run_learning_step(): Executa um passo de aprendizado.
     * run_episode(max_steps_per_episode): Executa um episódio completo.
     * fit(num_episodes, max_steps_per_episode): Executa múltiplos episódios.
     * save(path)/load(path): Salva/carrega agente e ambiente.

4. run.py (este arquivo):
   - Gerencia a leitura de argumentos, criação dos modelos, execução das simulações (paralelas ou sequenciais), e salvamento dos resultados.
   - Principais funções:
     * read_args(): Lê e processa os argumentos da linha de comando.
     * create_models(args): Instancia agente e ambiente.
     * main(args, sim, load_path): Executa uma simulação.
     * save_data(...): Salva resultados e modelos.

Para detalhes de parâmetros e outputs, consulte o README.
"""
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

from environments import PsEnvironment
from agents import PsAgent
from simulations import ProjectiveSimulation

env_class = PsEnvironment
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
    Lê os argumentos da linha de comando e retorna o objeto de argumentos.
    """
    parser = argparse.ArgumentParser(
        prog='run_target_find_simulation',
        description='Realiza a simulação de busca de alvo por uma partícula Browniana de estados ativos e passivos através de aprendizado pro reforço com simulação projetiva',
        #epilog='Text at the bottom of help'
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
        default = 1
    )

    parser.add_argument(
        "--policy", '--prob',
        help = "Tipo de cálculo da matriz de probabilidade: standard ou softmax", 
        type = str,
        default = 'standard',
        choices=['standard','softmax']
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

    parser.add_argument(
        "--colision",
        help = "Define condições decontorno periódicas (False) ou fechadas (True).", 
        type = int,
        choices = [0, 1]
    )

    parser.add_argument(
        "--colision_reward",
        help = "Recompensa para o aprendizado da colisão", 
        type = float_range(0,1),
        default = 0.005
    )

    parser.add_argument(
        "--save_path",
        help = "Caminho para salvar modelos", 
        type = str,
        default=''
    )

    parser.add_argument(
        "--load_path",
        help = "Lista de modelos para carregar", 
        type = str,
        default = ''
    )

    parser.add_argument(
        "--agent_type",
        help="Define o tipo de comportamento do agente: 'active_passive' ou 'passive_only'.",
        type=str,
        default='active_passive',
        choices=['active_passive', 'passive_only']
    )
    
    parser.add_argument(
        "--collision_type",
        help="Define o tipo de interação com a parede: 'specular', 'diffusive' ou 'repulsive'. Apenas se --colision=1.",
        type=str,
        default='specular',
        choices=['specular', 'diffusive', 'repulsive']
    )

    parser.add_argument(
        "--save_trajectory",
        help="Salva a trajetória (x, y, ângulo) do agente em um arquivo trajectory.txt.",
        action='store_true'
    )

    args = parser.parse_args()

    if args.damping_flag:
        args = damping_params(args)

    # De acordo com o artigo
    args.tao = int(args.dt/1e-4)
    args.max_steps_per_episode = int(20*args.tao)

    return args

def damping_params(args):
    """
    Define os parâmetros de damping de acordo com o número de Péclet.
    """
  
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
    """
    Cria e retorna as instâncias do agente e do ambiente.
    """
    # Inicia ambiente
    env = env_class(
        args.box_size, 
        args.peclet_number, 
        args.persistence, 
        args.tao, 
        args.dt,
        args.colision,
        allow_colision=bool(args.colision),
        collision_type=args.collision_type
    )
    
    # Inicia agente
    agent = agent_class(
        env.num_actions, 
        env.num_percepts_list, 
        args.gamma_damping, 
        args.eta_damping, 
        args.policy, 
        args.beta_softmax,
        args.num_reflections
    )

    # --- LÓGICA DE INICIALIZAÇÃO CORRIGIDA (BASEADA NO SEU CÓDIGO ORIGINAL) ---

    if args.colision:
        # CASO COM COLISÃO
        for timer_val in range(env.max_steps_per_trial):
            for colision_val in range(env.colision_state):
                if args.agent_type == 'passive_only':
                    # Para o baseline, só nos importamos com state=0
                    # A observação para o seu percept_preprocess é [state, timer, colision]
                    observation = [0, timer_val, colision_val]
                    percept = agent.percept_preprocess(observation)
                    agent.h_matrix[0, percept] = 1.0
                    agent.h_matrix[1, percept] = 0.0
                
                else: # 'active_passive'
                    # Esta é a sua lógica original, preservada
                    for state_val in range(env.num_states):
                        observation = [state_val, timer_val, colision_val]
                        percept = agent.percept_preprocess(observation)
                        if state_val == 0:
                            agent.h_matrix[1, percept] = 1e-2
                            agent.h_matrix[0, percept] = 1 - 1e-2
                        elif state_val == 1:
                            agent.h_matrix[1, percept] = 1e-3
                            agent.h_matrix[0, percept] = 1 - 1e-3
    else:
        # CASO SEM COLISÃO
        for timer_val in range(env.max_steps_per_trial):

            if args.agent_type == 'passive_only':
                observation = [0, timer_val]
                percept = agent.percept_preprocess(observation)
                agent.h_matrix[0, percept] = 1.0
                agent.h_matrix[1, percept] = 0.0

            else: # 'active_passive'
                for state_val in range(env.num_states):
                    observation = [state_val, timer_val]
                    percept = agent.percept_preprocess(observation)
                    if state_val == 0:
                        agent.h_matrix[1, percept] = 1e-2
                        agent.h_matrix[0, percept] = 1 - 1e-2
                    elif state_val == 1:
                        agent.h_matrix[1, percept] = 1e-3
                        agent.h_matrix[0, percept] = 1 - 1e-3

    # Matriz h0 inicial é estática
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

# Em run.py

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
            header = f"Episode {prev_episodes + ep + 1}, Steps {len(trajectory_data)}"
            np.savetxt(trajectory_file, np.array(trajectory_data), fmt='%.4f,%.4f,%.8f,%d', delimiter=',', header=header)

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