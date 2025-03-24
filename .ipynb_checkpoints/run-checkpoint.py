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

from environments import PsEnvironment
from agents import PsAgent
from simulations import ProjectiveSimulation

env_class = PsEnvironment
agent_class = PsAgent
ps_model = ProjectiveSimulation

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
    """Read command line arguments and return the parsed arguments"""
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

    parser.add_argument(
        "--colision",
        help="Define condições decontorno periódicas (False) ou fechadas (True).", 
        type = int,
        choices = [0, 1]
    )

    parser.add_argument(
        "--save_path",
        help="Caminho para salvar modelos", 
        type = str,
        default=''
    )

    parser.add_argument(
        "--load_path",
        help="Caminho para carregar modelo", 
        type = str,
        default=''
    )

    args = parser.parse_args()

    if args.damping_flag:
        args = damping_params(args)

    # De acordo com o artigo
    args.tao = int(args.dt/1e-4)
    args.max_steps_per_episode = int(20*args.tao)

    return args

def damping_params(args):
    """Set damping parameters based on the Péclet number"""
  
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
    """Create the agent and environment models"""
    # Inicia ambiente
    env = env_class(
        args.box_size, 
        args.peclet_number, 
        args.persistence, 
        args.tao, 
        args.dt,
        args.colision
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

def main(args):
    """Main function to run the simulation"""
    # Gera os modelos
    if len(args.load_path) > 0:
        model = ps_model.load(args.load_path)

    else:
        agent, env = create_models(args)
        # Gera a classe de simulação
        model = ps_model(agent, env)

    # Treina os modelos
    learning_process = model.fit(args.num_episodes, args.max_steps_per_episode)
    
    if len(args.save_path) > 0:
        filename_time = '{date:%Y-%m-%d_%H-%M-%S.%f}'.format(date=datetime.datetime.now())
        model.save(args.save_path + '/' + filename_time)

        with open(args.save_path + '/' + filename_time +'/args.json', 'w') as fp:
            json.dump(vars(args), fp)

        np.savetxt(args.save_path + '/' + filename_time +'/learning_process.txt', learning_process, fmt='%.4f', delimiter=',')
        np.savetxt(args.save_path + '/' + filename_time +'/h_matrix.txt', model.h_matrix(), fmt='%.2f', delimiter=',')

    if len(args.load_path) > 0:
        del agent
        del model
        
    del learning_process
    del filename_time
    gc.collect()
    
    return 0

if __name__ == "__main__":
    args = read_args()
    start_time = time.time()

    # Se for realizada a paralelização:
    if (args.n_jobs != 1) & (args.n_sim > 1):
        with tqdm_joblib(tqdm(desc="Simulações finalizadas:", total=args.n_sim, position = 0)) as progress_bar:
            Parallel(
                n_jobs = args.n_jobs,
                backend = "multiprocessing"
            )(delayed(main)(args) for sim in range(args.n_sim))
    
    # Se for execução sequencial:
    else:
        for sim in tqdm(range(args.n_sim), position = 0):
            main(args)

    print("--- %s seconds ---" % (time.time() - start_time))

    gc.collect()
    sys.exit(0)

# Monitorar threads: https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution
# Thread x Process: https://stackoverflow.com/questions/3044580/multiprocessing-vs-threading-python
# Comportamento de np.random com cada tipo de backend: https://joblib.readthedocs.io/en/latest/auto_examples/parallel_random_state.html
# Paralelição: https://stackoverflow.com/questions/9786102/how-do-i-parallelize-a-simple-python-loop