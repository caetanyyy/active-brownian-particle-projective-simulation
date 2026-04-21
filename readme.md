# Active Brownian Particle Projective Simulation

Repositório para reprodução dos experimentos do artigo:

\*\*Caraglio, Michele, et al. "Learning how to find targets in the micro-world: the case of intermittent active Brownian particles." Soft Matter (2024)\*\* ([arXiv link](https://arxiv.org/abs/2311.16692))

---

## Sumário

- [Descrição](#descrição)
- [Instalação](#instalação)
- [Como usar](#como-usar)
- [Parâmetros](#parâmetros)
- [Classes Principais](#classes-principais)
- [Matemática das Funções Principais](#matemática-das-funções-principais)
- [Output](#output)
- [Estrutura de Pastas](#estrutura-de-pastas)
- [Dependências](#dependências)
- [Exemplos de Execução](#exemplos-de-execução)
- [Contato](#contato)

---

## Descrição

Este projeto implementa um ambiente de simulação para agentes do tipo "Active Brownian Particle" (ABP) com aprendizado por simulação projetiva (Projective Simulation). O objetivo é estudar como partículas ativas podem aprender a encontrar alvos em ambientes 2D, reproduzindo e estendendo resultados do artigo citado.

O framework agora suporta múltiplos tipos de interação com as paredes (especular, difusiva e repulsiva) e diferentes estratégias de agente, permitindo uma análise comparativa robusta.

## Instalação

Clone o repositório e instale as dependências:

```bash
git clone https://github.com/caetanyyy/active-brownian-particle-projective-simulation
cd active-brownian-particle-projective-simulation
python -m venv .venv
..venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Como usar

Para treinar um modelo padrão:

```bash
python run.py --save_path "models"
```
Outros parâmetros podem ser passados via linha de comando para explorar diferentes cenários físicos e de aprendizado. Para ver todos os parâmetros e valores padrões:

```bash
python run.py --help
```


## Parâmetros

Principais parâmetros disponíveis:

- `n` ou `--num_episodes`: Quantidade de episódios para o treinamento (default: 100).
- `L` ou `--box_size`: Dimensão da caixa 2D (default: 100).
- `p` ou `--peclet_number`: Número de Péclet, Pe := (v\*tao)/L (default: 100).
- `l` ou `--persistence`: Persistência do movimento direto, l\* := v/(D_theta\*L) (default: 1).
- `dt`: Intervalo de tempo da simulação (default: 1).
- `policy`: Política de seleção de ação: "standard" ou "softmax" (default: "standard").
- `b` ou `--beta_softmax`: Parâmetro Beta da política SoftMax (default: 1).
- `n_sim`: Quantidade de agentes a serem treinados em paralelo (default: 1).
- `n_jobs`: Número de processos paralelos para o treinamento (default: 1).
- `colision`: Ativa condições de contorno fechadas (`1`) ou periódicas (`0`) (default: `0`).
- `agent_type`: Define a estratégia do agente.
    - `active_passive` (default): O agente pode aprender a alternar entre os estados ativo e passivo.
    - `passive_only`: O agente é forçado a permanecer sempre no estado passivo (usado para baseline).
- `collision_type`: Se `colision=1`, define o tipo de interação com a parede.
    - `specular` (default): Reflexão especular, como um espelho.
    - `diffusive`: Reflexão difusiva, onde a partícula é reorientada aleatoriamente após a colisão.
    - `repulsive`: Parede "suave" que aplica uma força repulsiva para evitar que a partícula a atravesse.
- `save_path`: Caminho para salvar os modelos treinados.
- `load_path`: Caminho para carregar um modelo e continuar o treinamento.

Outros parâmetros podem ser consultados com o comando `python run.py --help`.

## Classes Principais

### PsAgent (`agents/ps_agent.py`)
Agente de Simulação Projetiva (PS). O cérebro da operação, responsável por aprender e tomar decisões.

### PsEnvironment (`environments/ps_env_abp_target_find.py`)
Ambiente de simulação para a partícula Browniana ativa (ABP). Gerencia a "física" do mundo, incluindo o movimento da partícula, a posição do alvo e as interações com as paredes.

### ProjectiveSimulation (`simulations/projective_simulation_iteration.py`)
Classe orquestradora que gerencia o ciclo de aprendizado, conectando o `PsAgent` ao `PsEnvironment` a cada passo da simulação.

## Matemática das Funções Principais

### Aprendizado do Agente (`learn`)
O aprendizado é realizado pela atualização da matriz de pesos $h$ conforme a equação:
$$
h_{a,s} \leftarrow (1 - \gamma) h_{a,s} + \gamma h^0_{a,s} + g_{a,s} \cdot r
$$
Onde $h_{a,s}$ é o peso da ação $a$ para a percepção $s$, $\gamma$ é o fator de esquecimento, $h^0$ é a matriz de referência, $g$ é a matriz *glow* (memória de ações recentes) e $r$ é a recompensa.

### Deliberação do Agente (`deliberate`)
A probabilidade de uma ação $P(a|s)$ é calculada com base nos pesos $h_{a,s}$ usando uma política standard (proporcional) ou softmax.

### Física do Ambiente (`update_environment`)
A função `update_environment` implementa a dinâmica do agente a cada passo.

1.  Movimento da Partícula: A posição final $\vec{r}$ é a soma dos deslocamentos do movimento ativo (se `state=1`) e do movimento Browniano (sempre presente):
    $$ \vec{r}_{t+\Delta t} \leftarrow \vec{r}_t + \vec{dr}_{\theta} + \vec{dr} $$
    Onde $\vec{dr}_{\theta}$ é o deslocamento ativo e $\vec{dr}$ é o deslocamento Browniano.

2.  Interação com a Parede (se `colision=1`):
    -   Reflexão Especular (`collision_type="specular"`): A componente da posição perpendicular à parede é invertida, simulando um reflexo perfeito.
        $$ x \leftarrow -x \quad (\text{se } x<0) $$
    -   Reflexão Difusiva (`collision_type="diffusive"`): Ao colidir, a partícula é reposicionada na fronteira e sua orientação de movimento ativo $\theta_t$ é sorteada aleatoriamente.
        $$ \theta_t \leftarrow \text{Uniform}(0, 2\pi) $$
    -   Força Repulsiva (`collision_type="repulsive"`): Uma força de "parede suave" é aplicada quando a partícula entra numa zona de repulsão perto da fronteira. Um deslocamento adicional $\vec{dr}_{\text{repulsivo}}$ é adicionado, derivado de um potencial repulsivo.
        $$ \vec{dr}_{\text{repulsivo}} \propto \left( \frac{\sigma}{d} \right)^7 \cdot \Delta t $$
        Onde $d$ é a distância à parede. O estado de colisão para o agente é ativado quando a força excede um limiar.

3.  Recompensa: A recompensa principal (+1) é dada se o agente encontra o alvo enquanto está no estado passivo (`state=0`).

## Output

Após o treinamento de um agente, os seguintes arquivos são salvos na pasta definida em `save_path`:
-   `agent` e `environment`: Arquivos binários (pickle) com o estado final do modelo.
-   `learning_process.txt`: Curva de aprendizado (passos por episódio).
-   `h_matrix.txt`: Matriz de decisão final do agente.
-   `args.json`: Arquivo JSON com todos os parâmetros usados na simulação.

## Estrutura de Pastas

## Estrutura de Pastas
```
active-brownian-particle-projective-simulation/
│ readme.md
│ requirements.txt
│ run.py
│
├── agents/
│ └── ps_agent.py
│
├── environments/
│ └── ps_env_abp_target_find.py
│
├── simulations/
│ └── projective_simulation_iteration.py
│
└── ...
``` 


**## Dependências**

-   numpy
-   matplotlib
-   pandas
-   tqdm
-   joblib

**## Exemplos de Execução**

**Treinar o agente padrão (ativo-passivo) com colisão specular:**

```bash
python run.py --save_path "models/specular" --n 200 --colision 1 --collision_type specular
```

**Treinar um agente com colisão difusiva:**
```bash
python run.py --save_path "models/diffusive" --n 200 --colision 1 --collision_type diffusive
```

**Treinar um agente com parede repulsiva:**
```bash
python run.py --save_path "models/repulsive" --n 200 --colision 1 --collision_type repulsive
```

**Executar uma simulação de baseline com um agente puramente passivo:**
```bash
python run.py --save_path "models/passive_only" --n 200 --agent_type passive_only
```