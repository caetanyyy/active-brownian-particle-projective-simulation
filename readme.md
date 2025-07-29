
# Active Brownian Particle Projective Simulation

Repositório para reprodução dos experimentos do artigo:
**Caraglio, Michele, et al. "Learning how to find targets in the micro-world: the case of intermittent active Brownian particles." Soft Matter (2024)** ([arXiv link](https://arxiv.org/abs/2311.16692))

---

## Sumário
- [Descrição](#descrição)
- [Instalação](#instalação)
- [Como usar](#como-usar)
- [Parâmetros](#parâmetros)
- [Output](#output)
- [Estrutura de Pastas](#estrutura-de-pastas)
- [Dependências](#dependências)
- [Exemplo de Execução](#exemplo-de-execução)
- [Contato](#contato)

---

## Descrição
Este projeto implementa um ambiente de simulação para agentes do tipo "Active Brownian Particle" (ABP) com aprendizado por simulação projetiva (Projective Simulation). O objetivo é estudar como partículas ativas podem aprender a encontrar alvos em ambientes 2D, reproduzindo e estendendo resultados do artigo citado.

## Instalação
Clone o repositório e instale as dependências:

```bash
git clone https://github.com/caetanyyy/active-brownian-particle-projective-simulation
cd active-brownian-particle-projective-simulation
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Como usar
Para treinar um modelo:

```bash
python run.py --save_path "models"
```

Outros parâmetros podem ser passados via linha de comando. Para ver todos os parâmetros e valores padrões:

```bash
python run.py --help
```

## Parâmetros
Principais parâmetros disponíveis:

- `n`: Quantidade de episódios para o treinamento (default: 100)
- `L`: Dimensão da caixa 2D (default: 100)
- `p`: Número de Péclet, Pe := (v*tao)/L (default: 100)
- `l`: Persistência do movimento direto, l* := v/(D_theta*L) (default: 1)
- `dt`: Intervalo de tempo da simulação (default: 1)
- `policy`: Probabilidade de seleção do movimento: "standard" ou "softmax" (default: "standard")
- `b`: Beta SoftMax (default: 1, se policy=softmax)
- `n_sim`: Quantidade de agentes a serem treinados (default: 1)
- `n_jobs`: Processos paralelos para treinar cada modelo (default: 1)
- `num_reflections`: Quantidade de reflexões do agente para tomar decisão (default: 0)
- `num_episodes`: Número de episódios de treino do agente (default: 100)
- `save_path`: Caminho de salvamento
- `load_path`: Caminho de leitura de um modelo salvo

Os valores padrões seguem o artigo. Outros parâmetros podem ser consultados com o comando de ajuda.


## Classes Principais

### PsAgent (`agents/ps_agent.py`)
Agente de Simulação Projetiva (PS) para ambientes de busca de alvo.

**Parâmetros:**
- `num_actions`: Número de ações possíveis (>=1).
- `num_percepts_list`: Lista de inteiros representando a cardinalidade de cada categoria do espaço de percepção.
- `gamma_damping`: Fator de esquecimento/damping dos valores h (float entre 0 e 1).
- `eta_glow_damping`: Fator de damping da matriz glow (float entre 0 e 1).
- `policy_type`: Tipo de política ('standard' ou 'softmax').
- `beta_softmax`: Parâmetro β da softmax (>=0).
- `num_reflections`: Número de reflexões para deliberar ações.

**Métodos principais:**
- `deliberate(observation)`: Decide a ação do agente com base na observação do ambiente.
- `learn(reward)`: Atualiza os pesos do agente conforme a recompensa recebida.
- `probability_distr(percept)`: Calcula a distribuição de probabilidade das ações.
- `reset_glow_matrix()`: Reseta a matriz de glow (memória de ações).
- `save(path)` / `load(path)`: Salva ou carrega o agente em formato binário.

**Principais atributos:**
- `h_matrix`: Matriz de pesos de decisão.
- `g_matrix`: Matriz de glow para aprendizado com atraso.
- `policy_type`: Tipo de política de decisão ('standard' ou 'softmax').
- `num_reflections`: Número de reflexões para deliberar ações.

### PsEnvironment (`environments/ps_env_abp_target_find.py`)
Ambiente de simulação para partículas Brownianas ativas (ABP) com busca de alvo.

**Parâmetros:**
- `L`: Tamanho da caixa 2D.
- `Pe`: Número de Péclet.
- `l`: Escala de persistência do movimento.
- `tao`: Número máximo de passos por rodada.
- `dt`: Intervalo de tempo da simulação.
- `colision_reward`: Recompensa por aprendizado de colisão.
- `allow_colision`: Permite colisão com as paredes (True/False).

**Métodos principais:**
- `state_observation()`: Retorna o estado observável do ambiente para o agente.
- `update_environment(action)`: Atualiza o ambiente conforme a ação do agente.
- `reset_target()`: Reseta a posição do alvo.
- `reset_agent_state(new_state)`: Reseta o estado do agente.
- `save(path)` / `load(path)`: Salva ou carrega o ambiente em formato binário.

**Principais atributos:**
- `state`: Estado atual do agente (ativo/passivo).
- `colision`: Indica colisão com a parede.
- `reward`: Recompensa atual do agente.
- `target_position`: Posição do alvo.

### ProjectiveSimulation (`simulations/projective_simulation_iteration.py`)
Gerencia o ciclo de aprendizado do agente PS em um ambiente ABP.

**Parâmetros:**
- `agent`: Instância da classe do agente.
- `environment`: Instância da classe do ambiente.

**Métodos principais:**
- `run_learning_step()`: Executa um passo de aprendizado (observa, delibera, atualiza ambiente e aprende).
- `run_episode(max_steps_per_episode, reset_env=True, reset_agent=True)`: Executa um episódio completo.
- `fit(num_episodes, max_steps_per_episode)`: Executa múltiplos episódios e retorna a curva de aprendizado.
- `reset_environment()`: Reseta o ambiente para o estado inicial.
- `reset_agent()`: Reseta o agente para o estado inicial.
- `set_agent_attribute(attribute, value)`: Define o valor de um atributo do agente.
- `set_environment_attribute(attribute, value)`: Define o valor de um atributo do ambiente.
- `get_agent_attribute(attribute)`: Retorna o valor de um atributo do agente.
- `get_environment_attribute(attribute)`: Retorna o valor de um atributo do ambiente.
- `h_matrix()`: Retorna a matriz h do agente.
- `g_matrix()`: Retorna a matriz g do agente.
- `ho_matrix()`: Retorna a matriz h0 do agente.
- `e_matrix()`: Retorna a matriz e do agente, se disponível.
- `save(path)`: Salva agente e ambiente no caminho especificado.
- `load(path)`: Carrega agente e ambiente do caminho especificado.

**Principais atributos:**
- `agent`: Instância do agente PS.
- `env`: Instância do ambiente ABP.

## Matemática das Funções Principais do Agente

### Aprendizado (`learn`)
O aprendizado do agente é realizado pela atualização da matriz de pesos $h$ conforme a equação:

$$
h_{a,s} \leftarrow (1 - \gamma) h_{a,s} + \gamma h^0_{a,s} + g_{a,s} \cdot r
$$

Onde:
- $h_{a,s}$: peso da ação $a$ para a percepção $s$.
- $\gamma$: parâmetro de esquecimento (damping).
- $h^0_{a,s}$: valor inicial de referência da matriz $h$.
- $g_{a,s}$: valor da matriz glow (memória de ações recentes).
- $r$: recompensa recebida.

Se o agente utiliza reflexão e a recompensa é negativa ou nula, a matriz de emoticons $e_{a,s}$ é atualizada para evitar repetições de ações malsucedidas.

### Deliberação (`deliberate`)
Para decidir a ação, o agente calcula a distribuição de probabilidade das ações para uma percepção $s$:

- **Política Standard:**

$$
P(a|s) = \frac{h_{a,s}}{\sum_{a'} h_{a',s}}
$$

- **Política Softmax:**

$$
P(a|s) = \frac{\exp(\beta \cdot h_{a,s})}{\sum_{a'} \exp(\beta \cdot h_{a',s})}
$$

Onde $\beta$ é o parâmetro de temperatura da softmax.

O agente então amostra uma ação dessa distribuição. Se reflexão está ativa, repete a amostragem até encontrar uma ação permitida por $e_{a,s}$.


## Matemática da Função `update_environment` (Ambiente)

A função `update_environment` implementa a dinâmica do agente ABP e do ambiente a cada passo. O fluxo matemático é:

1. **Atualização do Timer:**

$$
\text{timer} \leftarrow \text{timer} + 1
$$

2. **Troca de Estado:**
- Se o agente decide trocar de estado ($a=1$):
$$
\text{state} \leftarrow 1 - \text{state}
$$
O timer é resetado e, se for para o estado ativo, a orientação é sorteada aleatoriamente.

3. **Movimento do Agente:**
- Se no estado ativo (ABP):
  - Atualiza orientação:
$$
\theta_t \leftarrow \theta_t + \sqrt{2 D_\theta \Delta t} \cdot \xi_t
$$
onde $\xi_t \sim \mathcal{N}(0,1)$
  - Vetor de orientação:
$$
\vec{u}_t = (\cos\theta_t, \sin\theta_t)
$$
  - Deslocamento ativo:
$$
\vec{dr}_\theta = v \cdot \vec{u}_t \cdot \Delta t
$$
- Movimento browniano (sempre):
$$
\vec{dr} = \sqrt{2 D \Delta t} \cdot \vec{\eta}_t
$$
onde $\vec{\eta}_t$ é vetor de ruído normal.
- Posição final:
$$
\vec{r} \leftarrow \vec{r} + \vec{dr}_\theta + \vec{dr}
$$

- **Caso com colisão ativa:**
  - Para cada coordenada $x$:
$$
x \leftarrow \begin{cases}
    -x, & x < 0 \\
    2L - x, & x > L \\
    x, & \text{caso contrário}
\end{cases}
$$
  - Se houve reflexão, define $\text{colision} = 1$.

- **Caso sem colisão (condições periódicas):**
$$
\vec{r} \leftarrow \vec{r} \bmod L
$$

4. **Cálculo da Distância ao Alvo:**
- Com colisão:
$$
d = \|\vec{r} - \vec{r}_\text{alvo}\|
$$
- Sem colisão (periódico):
$$
d = \|\min(|\vec{r} - \vec{r}_\text{alvo}|, L - |\vec{r} - \vec{r}_\text{alvo}|)\|
$$

5. **Recompensa:**
- Se o agente encontra o alvo ($d < r_\text{alvo}$ e estado passivo): recompensa $+1$ e episódio termina.
- Se o agente sai de colisão no estado ativo para passivo: recompensa $+R_c$.
  - $R_c$ é o parâmetro de recompensa de colisão.

6. **Troca automática de estado:**
- Se o timer atinge o máximo e o episódio não terminou, força troca de estado.

O retorno da função é $(\text{recompensa}, \text{episódio finalizado})$.

## Output
Após o treinamento de um agente, são salvos:
- Arquivos binários "agent" e "environment" (pickle) em `save_path/{date:%Y-%m-%d_%H-%M-%S}`
- Medida de aprendizado por episódio: `learning_process.txt`
- Matriz H: `h_matrix.txt`
- Argumentos do modelo: `args.json`

## Estrutura de Pastas
```
active-brownian-particle-projective-simulation/
│   readme.md
│   requirements.txt
│   run.py
│
├── agents/                # Implementação dos agentes PS
│   └── ps_agent.py
│
├── environments/          # Ambientes de simulação ABP
│   └── ps_env_abp_target_find.py
│
├── simulations/           # Scripts de simulação e iteração
│   └── projective_simulation_iteration.py
│
├── analises/              # Notebooks e scripts de análise
│   └── analise_learning_curve.ipynb
│
├── data/                  # Dados gerados e modelos salvos
│
├── docs/                  # Documentação e imagens
│
└── ...
```

## Dependências
As principais dependências estão listadas em `requirements.txt`. Exemplos:
- numpy
- matplotlib
- pandas
- tqdm
- pickle (builtin)

## Exemplo de Execução
Treinando um agente padrão e salvando na pasta `models`:

```bash
python run.py --save_path "models" --n 200 --L 100 --policy softmax --b 2
```

## Contato
Para dúvidas, sugestões ou colaborações, entre em contato com o mantenedor do repositório via [GitHub](https://github.com/caetanyyy).