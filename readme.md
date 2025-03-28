# Sobre o repositório
Reprodução do artigo Caraglio, Michele, et al. "Learning how to find targets in the micro-world: the case of intermittent active Brownian particles." Soft Matter (2024) (https://arxiv.org/abs/2311.16692)

# Instalação
git clone https://github.com/caetanyyy/active-brownian-particle-projective-simulation

<code>
python -m venv .venv
</code>

<code>
python -m ./.venv/Scripts/activate
</code>

<code>
python -m pip install -r requirements.txt
</code>

# Como usar:

## Treinamento
Para realizar o treinamento de um modelo:

<code>python run.py --save_path "models" </code>

Inúmeros parâmetros podem ser repassados pela linha de comando. Alguns dos principais são:

### n: Quantidade de episódios para o treinamento 

### L: Dimensão da caixa 2D 
Padrão é L = 100

### p: Número de Péclet 
Pe := (v*tao)/L

Padrão é p = 100

### l: Persistência do movimento direto
l* := v/(D_theta*L)

Padrão é l = 1

### dt: Intervalo de tempo da simulação. 
Pelo artigo, o tempo característico do movimento direto é dt*1e4

O padrão é 1

O tempo típico é Tao := dt x 1e4

### policy: 
Cálculo da probabilidade para selecionar o movimento: "standard" ou "softmax".

O padrão é "standard"


### b: Beta SoftMax
Se a probabilidade for softmax, o parametro beta padrão é 1

### n_sim: Quantidade de agentes que vão ser treinados

Padrão é 1

### n_jobs: Quantidade de processos paralelos para treinar cada modelo

Padrão é 1

### num_reflections: Quantiade de reflexões do agente para tomar uma decisão

Padrão é 0

### num_episodes: Número de episódios de treino do agente

Padrão é 100

### save_path: 
Caminho de salvamento

### load_path: 
Caminho de leitura de um modelo salvo.

Os padrâmetros padrões são os do artigo. Outros parâmetros podem ser repassados e seus valores padrões, basta rodar.

<code>python run.py --help </code>

## Output:

Após o treinamento de um agente, é salva a classe com dois arquivos: "agent" e "environment" em formato binário gerado pela biblioteca "pickle" no caminho "save_path/{date:%Y-%m-%d_%H-%M-%S}". Também é salvo a medida de aprendizado por episódio no arquivo "learning_process.txt", a matriz H em "h_matrix.txt" e os demais argumentos do modelo em "args.json".