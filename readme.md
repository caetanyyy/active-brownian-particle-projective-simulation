# Sobre o repositório
Reprodução do artigo Caraglio, Michele, et al. "Learning how to find targets in the micro-world: the case of intermittent active Brownian particles." Soft Matter (2024) (https://arxiv.org/abs/2311.16692)

# Como usar:

## Treinamento
Para realizar o treinamento de um modelo:

<code>python run_target_find_simulation --save_path "models" </code>

Inúmeros parâmetros podem ser repassados pela linha de comando. Alguns dos principais são:

--n: Quantidade de episódios para o treinamento

--L: Dimensão da caixa 2D de contorno periódio.

--p: Número de Péclet

--l: Persistência do movimento direto

--dt: Intervalo de tempo da simulação. Pelo artigo, o tempo característico do movimento direto é dt*1e4

--policy: "standard" ou "softmax". Tipo de cálculo da probabilidade

--n_sim: Quantidade de agentes que vão ser treinados

--n_jobs: Quantidade de processos paralelos para treinar cada modelo

--save_path: Caminho de salvamento

--load_path: Caminho de leitura de um modelo salvo.

Os padrâmetros padrões são os do artigo. Outros parâmetros podem ser repassados e seus valores padrões, basta rodar.

<code>python run_target_find_simulation --help </code>

## Output:

Após o treinamento de um agente, é salva a classe com dois arquivos: "agent" e "environment" em formato binário gerado pela biblioteca "pickle" no caminho "save_path/{date:%Y-%m-%d_%H-%M-%S}". Também é salvo a medida de aprendizado por episódio no arquivo "learning_process.txt", a matriz H em "h_matrix.txt" e os demais argumentos do modelo em "args.json".