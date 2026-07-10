# %% [markdown]
# # Simulação Comparativa de Algoritmos de Enxame
# ## Otimização de Cobertura de Área por VANTs
# 
# **Autor:** Renan Sousa Freire  
# **Curso:** Engenharia da Computação — UFMA  
# **Orientador:** Prof. Dr. Pedro Baptista Fernandes
# 
# ---
# 
# Esta célula instala automaticamente as dependências do projeto a partir do
# arquivo `requirements.txt`. O kernel *venv_tcc* deve estar selecionado no
# VSCode para que os pacotes sejam instalados no ambiente virtual correto.

# %%
# ── Instalação automática das dependências ────────────────────
# Garante que o notebook funcione em qualquer máquina
# sem necessidade de instalação manual prévia.
import subprocess, sys

with open('requirements.txt') as f:
    pacotes = [linha.strip() for linha in f if linha.strip()]

print("Verificando dependências...\n")
for pkg in pacotes:
    resultado = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', pkg, '-q'],
        capture_output=True
    )
    status = "✓" if resultado.returncode == 0 else "✗ ERRO"
    print(f"  {status}  {pkg}")

print("\nAmbiente pronto.")

# %% [markdown]
# ## Estrutura do Notebook
# 
# O notebook está organizado em etapas sequenciais que refletem
# o delineamento experimental descrito na metodologia do TCC:
# 
# 1. Importações e ambiente
# 2. Parâmetros da simulação
# 3. Grade de discretização e função de cobertura
# 4. Função objetivo normalizada
# 5. Implementação dos algoritmos: PSO, GWO, WOA, SSA, CS
# 6. Configuração dos cenários experimentais e dos perfis de peso
# 7. Execução do experimento (20 runs × 9 cenários × 3 perfis de peso × 6 condições)
# 8. Análise estatística — média, DP, MSE, ranking de robustez e Wilcoxon
# 9. Visualizações — curvas de convergência e mapas de cobertura

# %% [markdown]
# ## 1. Importações
# 
# Bibliotecas utilizadas ao longo da simulação:
# 
# - **numpy** — operações matemáticas e cálculos vetorizados sobre arrays
# - **matplotlib** — geração de gráficos de convergência e mapas de cobertura
# - **scipy** — teste estatístico de Wilcoxon e função Gamma para o CS
# - **pandas** — organização e exportação dos resultados em formato tabular
# - **joblib** — paralelização das execuções independentes no experimento oficial

# %%
import numpy as np                          # operações matemáticas e arrays vetorizados
import matplotlib.pyplot as plt              # geração dos gráficos de convergência e cobertura
import matplotlib.patches as patches         # desenho de círculos, retângulos e formas geométricas
from scipy.stats import wilcoxon             # teste estatístico de Wilcoxon (seção 3.5 do TCC)
from scipy.special import gamma              # função Gamma matemática, usada no voo de Lévy do CS
from collections import Counter              # conta o tamanho de cada grupo na rede de conectividade
from joblib import Parallel, delayed         # paralelização das execuções independentes
import pandas as pd                          # organização e exportação dos resultados em tabelas
import time                                  # medição de tempo de execução
import pickle, os                            # salvamento/carregamento de resultados em disco
import warnings
warnings.filterwarnings('ignore')            # suprime avisos não-críticos do NumPy/SciPy

print("Ambiente carregado com sucesso.")

# %% [markdown]
# ## 2. Parâmetros da Simulação
# 
# Os parâmetros centrais do experimento estão definidos aqui em um único bloco.
# Isso inclui as dimensões das áreas, os parâmetros da câmera, a configuração
# da grade de discretização e os pesos da função objetivo.
# 
# **Câmera de referência:** HFOV = 75° (compatível com câmeras de mapeamento
# embarcadas em VANTs comerciais, como a linha DJI Phantom).
# 
# O raio de cobertura de cada drone no solo é calculado pela relação
# fotogramétrica `rs = h × tan(HFOV/2) ≈ 0,77 × h`, onde h é a altitude
# operacional. Para cada escala de área, a altitude foi ajustada
# proporcionalmente, mantendo os mesmos parâmetros ópticos da câmera
# (Luhmann et al., 2013; Tsouros et al., 2019):
# 
# |    Área     | Altitude | Raio de cobertura |
# |-------------|----------|-------------------|
# | 100 × 100 m |   20 m   |       15 m        |
# | 250 × 250 m |   50 m   |       38 m        |
# | 500 × 500 m |   100 m  |       75 m        |
# 
# **Sobre os pesos abaixo:** W_COVERAGE/W_OVERLAP/W_CONNECTIVITY usam o
# valor do Perfil A (cobertura prioritária) e servem apenas para os
# testes rápidos e isolados de cada algoritmo na Seção 5 — não são mais
# usados no experimento principal. No experimento real (Seção 7 em
# diante), os pesos vêm dos 3 perfis definidos em PESOS_PERFIS e são
# passados como parâmetro explícito para objective_fast(), nunca lidos
# de variável global — isso é o que permite alternar entre perfis sem
# qualquer risco de um perfil "vazar" para o cálculo de outro.

# %%
"""
Valores usados nos testes individuais de cada algoritmo.
No experimento completo, cada cenário define seus próprios valores.
"""
AREA_WIDTH  = 100
AREA_HEIGHT = 100
N_DRONES    = 10
START_POINT = np.array([0.0, 0.0])  # ponto de partida fixo — usado só para visualização

"""
HFOV = 75° → rs = h × tan(37,5°) ≈ 0,767 × h
Os raios abaixo resultam das altitudes operacionais por escala de área
(seção 3.4.1 do TCC).
"""
HFOV_HALF_TAN = np.tan(np.radians(37.5))   # ≈ 0.767
ALTITUDES     = {100: 20,  250: 50,  500: 100}  # altitude por largura de área (m)
RADII         = {100: 15,  250: 38,  500: 75}   # raio resultante por área (m)
RADIUS        = RADII[AREA_WIDTH]               # raio do cenário base

"""
A rede de comunicação tem alcance maior que o sensoriamento da
câmera. R_comm = k × raio_de_cobertura, escalando proporcionalmente
com cada cenário. Garante conectividade entre drones para
retransmissão de dados até a base (Tossa et al., 2022).
"""
COMM_RADIUS_FACTOR = 2.5   # k: múltiplo do raio de cobertura (calibrado — seção 3.4.1 do TCC)

GRID_RESOLUTION = {100: 1, 250: 2, 500: 5}   # resolução da grade por área (metros entre pontos)

# ── Pesos da função objetivo (só para os testes isolados da Seção 5) ──
W_COVERAGE     = 0.7   # cobertura de área — objetivo principal da missão
W_OVERLAP      = 0.2   # penalidade por sobreposição entre zonas de cobertura
W_CONNECTIVITY = 0.1   # penalidade por fragmentação da rede de comunicação

# ── Limites do espaço de busca (cenário base) ────────────────
BOUNDS = np.tile(
    [[0, AREA_WIDTH],
     [0, AREA_HEIGHT]],
    (N_DRONES, 1)
)

# ── Verificação ──────────────────────────────────────────────
print(f"Área base          : {AREA_WIDTH}×{AREA_HEIGHT} m")
print(f"Altitude           : {ALTITUDES[AREA_WIDTH]} m")
print(f"Raio de cobertura  : {RADIUS} m")
print(f"Raio de comunicação: {COMM_RADIUS_FACTOR * RADIUS} m (k={COMM_RADIUS_FACTOR})")
print(f"Drones (base)      : {N_DRONES}")
print(f"Espaço de busca    : {BOUNDS.shape}  → esperado ({2*N_DRONES}, 2)")

# %% [markdown]
# ## 3. Grade de Discretização e Função de Cobertura
# 
# Para quantificar a cobertura de uma área contínua, a região de interesse
# é discretizada em uma grade regular de pontos espaçados entre si por uma
# resolução definida em metros. Cada ponto representa uma posição física
# no solo que pode ou não estar dentro do alcance de algum drone.
# 
# A cobertura é calculada pela proporção de pontos cobertos por pelo menos
# um drone — implementação direta do modelo de Disco Binário da Equação 2.1:
# 
# > **f(si, p) = 1** — quando a distância entre drone i e ponto p for menor ou igual ao raio rs
# 
# > **f(si, p) = 0** — caso contrário
# 
# **Taxa de cobertura** = pontos cobertos ÷ total de pontos da grade
# 
# Quanto maior a resolução (menor o espaçamento), mais precisa é a estimativa
# — mas maior o custo computacional. A resolução foi calibrada por escala de
# área para equilibrar precisão e tempo de execução.

# %%
def build_grid(width, height, resolution):
    """
    Discretiza a área de interesse em uma grade regular de pontos 2D.
    Cada ponto representa uma coordenada (x, y) no plano horizontal.

    A grade cobre todo o intervalo [0, width] × [0, height], incluindo
    as bordas, com pontos espaçados pela resolução definida em metros.
    """
    xs = np.arange(0, width  + resolution, resolution)
    ys = np.arange(0, height + resolution, resolution)
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel()])


GRID         = build_grid(AREA_WIDTH, AREA_HEIGHT, GRID_RESOLUTION[AREA_WIDTH])
TOTAL_POINTS = len(GRID)


def compute_coverage(drone_positions, radius, grid):
    """
    Calcula a taxa de cobertura da área pelo enxame de drones.

    Para cada drone, todos os pontos da grade dentro do raio rs são
    marcados como cobertos. A cobertura final corresponde à união dos
    discos individuais — pontos cobertos por mais de um drone são
    contados apenas uma vez (Equação 2.1 do TCC).
    """
    covered = np.zeros(len(grid), dtype=bool)
    for pos in drone_positions:
        distancias = np.linalg.norm(grid - pos, axis=1)
        covered |= (distancias <= radius)
    return covered.sum() / len(grid), covered


pos_teste = np.array([[50.0, 50.0]])
ratio, _  = compute_coverage(pos_teste, RADIUS, GRID)

print(f"Total de pontos na grade     : {TOTAL_POINTS}")
print(f"Cobertura com 1 drone central: {ratio:.4f}  (esperado ≈ 0.0707)")

# %% [markdown]
# ## 4. Função Objetivo Normalizada
# 
# A função objetivo traduz a qualidade de um posicionamento de drones
# em um único valor numérico a ser minimizado pelos algoritmos.
# 
# Ela é composta por três componentes, cada um normalizado para o
# intervalo [0, 1] antes da agregação ponderada (seção 3.3 do TCC):
# 
# | Componente            | O que representa                                 |
# |-----------------------|--------------------------------------------------|
# | Área não coberta      | 1 − taxa de cobertura                            |
# | Penalidade overlap    | pares de drones com discos sobrepostos (Eq. 2.3) |
# | Fragmentação da rede  | fração de drones fora do maior grupo conectado   |
# 
# Os pesos de cada componente não são fixos no código: o experimento
# avalia 3 perfis de peso diferentes (Seção 3.4.3 do TCC), e a versão usada no
# experimento real (objective_fast) recebe os pesos como parâmetro
# explícito, não de variável global — ver Seção 3.3 do TCC para a
# justificativa completa.
# 
# > **Sobre a conectividade:** dois drones são considerados conectados
# > se a distância entre eles for menor ou igual ao raio de comunicação
# > (R_comm = 2,5 × raio de cobertura). O custo mede a fração de drones que
# > ficam isolados do maior grupo conectado da rede — quanto maior essa
# > fração, menos dados a missão consegue efetivamente recuperar
# > (Tossa et al., 2022).

# %%
"""
Constante de normalização do overlap (cenário base): representa o
pior caso possível — todos os drones sobrepostos no mesmo ponto,
formando N×(N-1)/2 pares, cada um com penalidade máxima = dmin = 2×raio.
Usada para normalizar o componente c2 para o intervalo [0, 1].
"""
_DMIN        = 2 * RADIUS
_MAX_OVERLAP = (N_DRONES * (N_DRONES - 1) / 2) * _DMIN

"""
Raio de comunicação do cenário base — escala com o raio de cobertura
(seção 3.4.1 do TCC).
"""
COMM_RADIUS = COMM_RADIUS_FACTOR * RADIUS


def compute_largest_component(n_drones, idx_i, idx_j):
    """
    Calcula o tamanho do maior grupo de drones mutuamente conectados,
    usando a estrutura Union-Find (Disjoint Set Union).

    Recebe os pares de drones já identificados como conectados
    (distância ≤ raio de comunicação) e agrupa drones ligados direta
    ou indiretamente — dois drones que não se enxergam diretamente
    ainda podem estar no mesmo grupo se conectados via um terceiro,
    formando uma rede de retransmissão em múltiplos saltos.
    """
    parent = list(range(n_drones))

    def find(x):
        """Sobe na árvore até achar o representante do grupo de x,
        comprimindo o caminho percorrido para acelerar buscas futuras."""
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        """Funde os grupos de x e y, ligando um representante ao outro."""
        root_x, root_y = find(x), find(y)
        if root_x != root_y:
            parent[root_x] = root_y

    for i, j in zip(idx_i, idx_j):
        union(int(i), int(j))

    grupos = Counter(find(i) for i in range(n_drones))
    return max(grupos.values())


def objective(flat_positions):
    """
    Função custo multi-objetivo normalizada para o cenário base (seção 3.3).
    Utilizada SOMENTE nos testes individuais dos algoritmos (Seção 5).
    Usa os pesos globais W_COVERAGE/W_OVERLAP/W_CONNECTIVITY (Perfil A).

    No experimento completo, objective_fast() substitui esta função,
    recebendo os pesos de cada perfil como parâmetro explícito.
    """
    positions = flat_positions.reshape((N_DRONES, 2))

    coverage_ratio, _ = compute_coverage(positions, RADIUS, GRID)
    c1 = 1.0 - coverage_ratio

    overlap_sum = 0.0
    for i in range(N_DRONES):
        for j in range(i + 1, N_DRONES):
            dist         = np.linalg.norm(positions[i] - positions[j])
            overlap_sum += max(0.0, _DMIN - dist)
    c2 = overlap_sum / _MAX_OVERLAP if _MAX_OVERLAP > 0 else 0.0

    pares_i, pares_j = [], []
    for i in range(N_DRONES):
        for j in range(i + 1, N_DRONES):
            if np.linalg.norm(positions[i] - positions[j]) <= COMM_RADIUS:
                pares_i.append(i)
                pares_j.append(j)

    maior_grupo = compute_largest_component(N_DRONES, pares_i, pares_j)
    c3 = 1.0 - (maior_grupo / N_DRONES)

    return W_COVERAGE * c1 + W_OVERLAP * c2 + W_CONNECTIVITY * c3


def get_metrics(flat_positions):
    """
    Extrai métricas detalhadas de um posicionamento para análise e relatório.
    Separada da objective() para não impactar o tempo das iterações.
    """
    positions  = flat_positions.reshape((N_DRONES, 2))
    ratio, _   = compute_coverage(positions, RADIUS, GRID)

    n_overlaps = sum(
        1 for i in range(N_DRONES)
        for j in range(i + 1, N_DRONES)
        if np.linalg.norm(positions[i] - positions[j]) < _DMIN
    )

    pares_i, pares_j = [], []
    for i in range(N_DRONES):
        for j in range(i + 1, N_DRONES):
            if np.linalg.norm(positions[i] - positions[j]) <= COMM_RADIUS:
                pares_i.append(i)
                pares_j.append(j)

    maior_grupo    = compute_largest_component(N_DRONES, pares_i, pares_j)
    frac_conectada = maior_grupo / N_DRONES

    return {
        'taxa_cobertura'        : round(ratio, 4),
        'area_coberta_m2'       : round(ratio * AREA_WIDTH * AREA_HEIGHT, 2),
        'n_overlaps'            : n_overlaps,
        'maior_grupo_conectado' : maior_grupo,
        'fracao_conectada'      : round(frac_conectada, 4),
    }


np.random.seed(42)
pos_teste = np.random.uniform(0, 100, N_DRONES * 2)
custo     = objective(pos_teste)
metricas  = get_metrics(pos_teste)

print(f"Custo (solução aleatória) : {custo:.4f}")
print(f"Cobertura                 : {metricas['taxa_cobertura']:.4f}")
print(f"Overlaps                  : {metricas['n_overlaps']}")
print(f"Maior grupo conectado     : {metricas['maior_grupo_conectado']}/{N_DRONES}")
print(f"Fração conectada          : {metricas['fracao_conectada']:.4f}")

# %% [markdown]
# ## 5. Algoritmos de Otimização por Enxame
# 
# Os cinco algoritmos implementados a seguir representam paradigmas
# distintos de inteligência de enxame, selecionados conforme justificado
# na seção 1 do TCC. Cada um é testado individualmente com 20 agentes
# e 50 iterações sobre o cenário base antes do experimento completo.
# 
# ---
# 
# ### 5.1 PSO — Particle Swarm Optimization

# %%
def pso(objective_func, bounds, n_particles=30, n_iterations=100,
        w_max=0.9, w_min=0.4, c1=1.5, c2=1.5):
    """
    Particle Swarm Optimization (Kennedy & Eberhart, 1995).

    Cada partícula representa um posicionamento completo dos drones.
    A velocidade é atualizada pelas Equações 2.5 e 2.6 do TCC,
    com inércia decrescente para transição de exploração a explotação.
    """
    dim = bounds.shape[0]

    positions  = np.random.uniform(bounds[:, 0], bounds[:, 1],
                                   (n_particles, dim))
    velocities = np.zeros((n_particles, dim))

    pbest_pos   = positions.copy()
    pbest_score = np.array([objective_func(p) for p in positions])

    gbest_idx   = np.argmin(pbest_score)
    gbest_pos   = pbest_pos[gbest_idx].copy()
    gbest_score = pbest_score[gbest_idx]

    history = [gbest_score]

    for t in range(n_iterations):
        w = w_max - (w_max - w_min) * (t / n_iterations)

        for i in range(n_particles):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)

            velocities[i] = (w  * velocities[i]
                           + c1 * r1 * (pbest_pos[i] - positions[i])
                           + c2 * r2 * (gbest_pos   - positions[i]))

            positions[i] += velocities[i]
            positions[i]  = np.clip(positions[i], bounds[:, 0], bounds[:, 1])

            score = objective_func(positions[i])
            if score < pbest_score[i]:
                pbest_score[i] = score
                pbest_pos[i]   = positions[i].copy()
            if score < gbest_score:
                gbest_score = score
                gbest_pos   = positions[i].copy()

        history.append(gbest_score)

    return gbest_pos, gbest_score, history


print("Testando PSO...")
np.random.seed(0)
pos, score, hist = pso(objective, BOUNDS, n_particles=20, n_iterations=50)
metricas = get_metrics(pos)

print(f"  Custo final     : {score:.4f}")
print(f"  Cobertura       : {metricas['taxa_cobertura']:.4f}")
print(f"  Overlaps        : {metricas['n_overlaps']}")
print(f"  PSO OK ✓")

# %% [markdown]
# ### 5.2 GWO — Grey Wolf Optimizer

# %%
def gwo(objective_func, bounds, n_wolves=30, n_iterations=100):
    """
    Grey Wolf Optimizer (Mirjalili, Mirjalili & Lewis, 2014).

    A posição final de cada lobo omega é a média das três posições
    sugeridas pelos líderes alpha, beta e delta — Equação 2.14 do TCC.
    """
    dim    = bounds.shape[0]
    wolves = np.random.uniform(bounds[:, 0], bounds[:, 1],
                               (n_wolves, dim))

    alpha_pos, alpha_score = np.zeros(dim), float('inf')
    beta_pos,  beta_score  = np.zeros(dim), float('inf')
    delta_pos, delta_score = np.zeros(dim), float('inf')

    history = []

    for t in range(n_iterations):
        for wolf in wolves:
            score = objective_func(wolf)
            if score < alpha_score:
                delta_score, delta_pos = beta_score,  beta_pos.copy()
                beta_score,  beta_pos  = alpha_score, alpha_pos.copy()
                alpha_score, alpha_pos = score, wolf.copy()
            elif score < beta_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score,  beta_pos  = score, wolf.copy()
            elif score < delta_score:
                delta_score, delta_pos = score, wolf.copy()

        a = 2 - 2 * (t / n_iterations)

        for i in range(n_wolves):
            new_pos = np.zeros(dim)
            for leader in [alpha_pos, beta_pos, delta_pos]:
                r1 = np.random.rand(dim)
                r2 = np.random.rand(dim)
                A = 2 * a * r1 - a
                C = 2 * r2
                D       = np.abs(C * leader - wolves[i])
                new_pos += leader - A * D
            wolves[i] = np.clip(new_pos / 3, bounds[:, 0], bounds[:, 1])

        history.append(alpha_score)

    return alpha_pos, alpha_score, history


print("Testando GWO...")
np.random.seed(0)
pos, score, hist = gwo(objective, BOUNDS, n_wolves=20, n_iterations=50)
metricas = get_metrics(pos)

print(f"  Custo final     : {score:.4f}")
print(f"  Cobertura       : {metricas['taxa_cobertura']:.4f}")
print(f"  Overlaps        : {metricas['n_overlaps']}")
print(f"  GWO OK ✓")

# %% [markdown]
# ### 5.3 WOA — Whale Optimization Algorithm

# %%
def woa(objective_func, bounds, n_agents=30, n_iterations=100):
    """
    Whale Optimization Algorithm (Mirjalili & Lewis, 2016).

    Alterna entre cerco direto e movimento espiral logarítmico,
    controlado pelo número aleatório p sorteado a cada iteração.
    """
    dim    = bounds.shape[0]
    agents = np.random.uniform(bounds[:, 0], bounds[:, 1],
                               (n_agents, dim))

    scores     = np.array([objective_func(a) for a in agents])
    best_idx   = np.argmin(scores)
    best_pos   = agents[best_idx].copy()
    best_score = scores[best_idx]

    history = [best_score]

    for t in range(n_iterations):
        a = 2 - 2 * (t / n_iterations)

        for i in range(n_agents):
            r = np.random.rand(dim)
            A = 2 * a * r - a
            C = 2 * np.random.rand(dim)
            p = np.random.rand()
            b = 1.0
            l = np.random.uniform(-1, 1, dim)

            if p < 0.5:
                if np.all(np.abs(A) < 1):
                    D         = np.abs(C * best_pos - agents[i])
                    agents[i] = best_pos - A * D
                else:
                    rand_idx  = np.random.randint(n_agents)
                    D         = np.abs(C * agents[rand_idx] - agents[i])
                    agents[i] = agents[rand_idx] - A * D
            else:
                D_prime   = np.abs(best_pos - agents[i])
                agents[i] = (D_prime * np.exp(b * l)
                             * np.cos(2 * np.pi * l) + best_pos)

            agents[i] = np.clip(agents[i], bounds[:, 0], bounds[:, 1])

            score = objective_func(agents[i])
            if score < best_score:
                best_score = score
                best_pos   = agents[i].copy()

        history.append(best_score)

    return best_pos, best_score, history


print("Testando WOA...")
np.random.seed(0)
pos, score, hist = woa(objective, BOUNDS, n_agents=20, n_iterations=50)
metricas = get_metrics(pos)

print(f"  Custo final     : {score:.4f}")
print(f"  Cobertura       : {metricas['taxa_cobertura']:.4f}")
print(f"  Overlaps        : {metricas['n_overlaps']}")
print(f"  WOA OK ✓")

# %% [markdown]
# ### 5.4 SSA — Salp Swarm Algorithm

# %%
def ssa(objective_func, bounds, n_salps=30, n_iterations=100):
    """
    Salp Swarm Algorithm (Mirjalili et al., 2017).

    A líder atualiza sua posição pela Equação 2.16 do TCC.
    As seguidoras seguem a Lei de Newton — Equação 2.17.
    """
    dim = bounds.shape[0]
    lb  = bounds[:, 0]
    ub  = bounds[:, 1]

    salps      = np.random.uniform(lb, ub, (n_salps, dim))
    scores     = np.array([objective_func(s) for s in salps])
    best_idx   = np.argmin(scores)

    food_pos   = salps[best_idx].copy()
    food_score = scores[best_idx]

    history = [food_score]

    for t in range(n_iterations):
        c1 = 2 * np.exp(-((4 * t / n_iterations) ** 2))

        c2 = np.random.rand(dim)
        c3 = np.random.rand(dim)
        step = c1 * ((ub - lb) * c2 + lb)

        salps[0] = np.where(c3 >= 0.5, food_pos + step, food_pos - step)
        salps[0] = np.clip(salps[0], lb, ub)

        for i in range(1, n_salps):
            salps[i] = 0.5 * (salps[i] + salps[i - 1])
            salps[i] = np.clip(salps[i], lb, ub)

        for i in range(n_salps):
            score = objective_func(salps[i])
            if score < food_score:
                food_score = score
                food_pos   = salps[i].copy()

        history.append(food_score)

    return food_pos, food_score, history


print("Testando SSA...")
np.random.seed(0)
pos, score, hist = ssa(objective, BOUNDS, n_salps=20, n_iterations=50)
metricas = get_metrics(pos)

print(f"  Custo final     : {score:.4f}")
print(f"  Cobertura       : {metricas['taxa_cobertura']:.4f}")
print(f"  Overlaps        : {metricas['n_overlaps']}")
print(f"  SSA OK ✓")

# %% [markdown]
# ### 5.5 CS — Cuckoo Search

# %%
def levy_flight(beta, dim):
    """
    Gera passos de voo de Lévy via algoritmo de Mantegna.
    Produz distribuição de cauda pesada (Equações 2.20–2.22 do TCC).
    """
    numerador   = gamma(1 + beta) * np.sin(np.pi * beta / 2)
    denominador = gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma_u     = (numerador / denominador) ** (1 / beta)

    u = np.random.normal(0, sigma_u, dim)
    v = np.random.normal(0, 1,       dim)

    return u / (np.abs(v) ** (1 / beta))


def cs(objective_func, bounds, n_nests=30, n_iterations=100,
       pa=0.25, beta=1.5, alpha=0.01):
    """
    Cuckoo Search via Lévy Flights (Yang & Deb, 2009).
    """
    dim = bounds.shape[0]
    lb  = bounds[:, 0]
    ub  = bounds[:, 1]

    nests  = np.random.uniform(lb, ub, (n_nests, dim))
    scores = np.array([objective_func(n) for n in nests])

    best_idx   = np.argmin(scores)
    best_nest  = nests[best_idx].copy()
    best_score = scores[best_idx]

    history = [best_score]

    for t in range(n_iterations):
        for i in range(n_nests):
            passo      = alpha * levy_flight(beta, dim) * (nests[i] - best_nest)
            novo_ninho = np.clip(nests[i] + passo, lb, ub)
            novo_score = objective_func(novo_ninho)

            j = np.random.randint(n_nests)
            if novo_score < scores[j]:
                nests[j]  = novo_ninho
                scores[j] = novo_score

        n_abandon  = max(1, int(pa * n_nests))
        piores_idx = np.argsort(scores)[-n_abandon:]
        for idx in piores_idx:
            nests[idx]  = np.random.uniform(lb, ub, dim)
            scores[idx] = objective_func(nests[idx])

        melhor_idx = np.argmin(scores)
        if scores[melhor_idx] < best_score:
            best_score = scores[melhor_idx]
            best_nest  = nests[melhor_idx].copy()

        history.append(best_score)

    return best_nest, best_score, history


print("Testando CS...")
np.random.seed(0)
pos, score, hist = cs(objective, BOUNDS, n_nests=20, n_iterations=50)
metricas = get_metrics(pos)

print(f"  Custo final     : {score:.4f}")
print(f"  Cobertura       : {metricas['taxa_cobertura']:.4f}")
print(f"  Overlaps        : {metricas['n_overlaps']}")
print(f"  CS OK ✓")

# %% [markdown]
# ## 6. Otimização de Performance — Cobertura Vetorizada e Função Objetivo Paramétrica
# 
# A versão vetorizada `compute_coverage_fast` calcula todas as distâncias
# de todos os drones para todos os pontos da grade em uma única operação
# matricial via broadcasting do NumPy.
# 
# `objective_fast` é a versão usada no experimento real: recebe os
# parâmetros do cenário E os pesos do perfil ativo como argumentos
# explícitos — nada de variável global. Isso é o que torna seguro
# alternar entre os 3 perfis de peso sem qualquer chance de
# contaminação entre eles.

# %%
def compute_coverage_fast(drone_positions, radius, grid):
    """
    Versão vetorizada de compute_coverage — resultado idêntico,
    execução mais rápida via broadcasting NumPy.
    """
    diff  = drone_positions[:, np.newaxis, :] - grid[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))

    covered = np.any(dists <= radius, axis=0)

    return covered.sum() / len(grid), covered


def objective_fast(flat_positions, n_drones, radius, grid,
                   comm_radius, max_overlap, w_cobertura, w_overlap, w_conectividade):
    """
    Função objetivo parametrizada para uso no experimento completo.

    Diferente da versão anterior, os PESOS também são parâmetros
    explícitos (w_cobertura, w_overlap, w_conectividade) — não dependem
    de W_COVERAGE/W_OVERLAP/W_CONNECTIVITY globais. Isso permite que o
    runner alterne entre os perfis de peso (Seção 3.4.3) sem qualquer
    risco de um perfil vazar para o cálculo de outro.

    Componentes (seção 3.3 do TCC):
      c1 — área não coberta
      c2 — penalidade de overlap
      c3 — fragmentação da rede de comunicação
    """
    positions = flat_positions.reshape((n_drones, 2))
    dmin      = 2 * radius

    coverage_ratio, _ = compute_coverage_fast(positions, radius, grid)
    c1 = 1.0 - coverage_ratio

    """
    Calculamos as distâncias entre todos os pares de drones uma única
    vez, e reutilizamos esse mesmo resultado tanto para o overlap (c2)
    quanto para a conectividade (c3) — evita repetir o mesmo cálculo
    duas vezes.
    """
    if n_drones > 1:
        idx_i, idx_j = np.triu_indices(n_drones, k=1)
        dists        = np.linalg.norm(positions[idx_i] - positions[idx_j],
                                      axis=1)
    else:
        idx_i, idx_j, dists = (np.array([], dtype=int),
                               np.array([], dtype=int), np.array([]))

    if n_drones > 1:
        overlap_sum = np.maximum(0.0, dmin - dists).sum()
    else:
        overlap_sum = 0.0
    c2 = overlap_sum / max_overlap if max_overlap > 0 else 0.0

    if n_drones > 1:
        conectados  = dists <= comm_radius
        maior_grupo = compute_largest_component(
            n_drones, idx_i[conectados], idx_j[conectados]
        )
        c3 = 1.0 - (maior_grupo / n_drones)
    else:
        c3 = 0.0

    return w_cobertura * c1 + w_overlap * c2 + w_conectividade * c3


# ── Verificação de equivalência da cobertura vetorizada ───────
np.random.seed(0)
pos_2d = np.random.uniform(0, 100, (N_DRONES, 2))

t0 = time.time()
r1, _ = compute_coverage(pos_2d, RADIUS, GRID)
t_original = time.time() - t0

t0 = time.time()
r2, _ = compute_coverage_fast(pos_2d, RADIUS, GRID)
t_rapido = time.time() - t0

print(f"Resultado original   : {r1:.6f}  ({t_original*1000:.2f} ms)")
print(f"Resultado vetorizado : {r2:.6f}  ({t_rapido*1000:.2f} ms)")
print(f"Resultados idênticos : {np.isclose(r1, r2)}")

# %% [markdown]
# ## 7. Configuração Experimental e Perfis de Peso
# 
# Os cenários do experimento foram definidos com base em três escalas
# de área e três densidades de drones por escala (seção 3.4.2 do TCC):
# 
# - **Escasso** (~30% de cobertura teórica): recursos limitados,
#   avalia qual algoritmo extrai maior cobertura com menos drones
# - **Suficiente** (~65% de cobertura teórica): recursos adequados,
#   avalia eficiência na cobertura alvo com menor desperdício
# - **Redundante** (~100% de cobertura teórica): recursos em excesso,
#   avalia qual algoritmo minimiza sobreposição
# 
# **Perfis de peso:** em vez de um único vetor de pesos, o experimento
# avalia os algoritmos sob 3 perfis fixos representando diferentes
# prioridades operacionais de missão — decisão fundamentada na
# constatação de que o método da soma ponderada é sensível ao vetor de
# pesos escolhido (Marler & Arora, 2010). Os 3 perfis são fixos e
# compartilhados entre todos os algoritmos e cenários (ver Seção 3.4.3
# do TCC para a discussão completa). O peso do overlap permanece
# constante (0,2) nos três; cobertura e conectividade variam
# simetricamente em passos de 0,2.
# 
# `DEMO_MODE = True` executa os 3 perfis de peso completos sobre um
# único cenário, com agentes/iterações/runs reduzidos — pensado para
# rodar em qualquer máquina (ex. notebook sem múltiplos núcleos) e
# ainda assim validar a integridade do pipeline completo, incluindo o
# loop de pesos. `DEMO_MODE = False` executa o experimento oficial: os
# 9 cenários × os 3 perfis × N_RUNS execuções por algoritmo, com
# paralelização via joblib.

# %%
# ── Modo de execução ─────────────────────────────────────────
# True → verificação rápida (3 perfis, 1 cenário) | False → experimento completo
DEMO_MODE = False

DEMO_CENARIO_NOME = '100x100_escasso' # opções: '100x100_escasso', '100x100_suficiente', '100x100_redundante', '250x250_escasso', '250x250_suficiente', '250x250_redundante', '500x500_escasso', '500x500_suficiente', '500x500_redundante'

"""
Parâmetros por modo. O DEMO foi calibrado para ficar um pouco mais
robusto que a versão anterior (mais agentes/iterações/runs), mas ainda
rápido o suficiente pra rodar em qualquer máquina, como notebooks sem
múltiplos núcleos — continua testando os 3 perfis de peso completos
sobre 1 único cenário.
"""
if DEMO_MODE:
    N_RUNS       = 6
    N_AGENTS     = 25
    N_ITERATIONS = 75
else:
    N_RUNS       = 20
    N_AGENTS     = 30
    N_ITERATIONS = 100

# ── Perfis de peso (Seção 3.4.3 do TCC) ────────────────────────
"""
Três perfis fixos, compartilhados entre todos os algoritmos e
cenários — nenhum algoritmo ou cenário "escolhe" seu próprio peso
(a Seção 3.4.3 do TCC discute por que essas duas alternativas foram
descartadas). Overlap fica constante (0,2); cobertura e conectividade
variam em passos simétricos de 0,2.
"""
PESOS_PERFIS = {
    'A_cobertura_prioritaria': {'cobertura': 0.7, 'overlap': 0.2, 'conectividade': 0.1},
    'B_transicao':             {'cobertura': 0.5, 'overlap': 0.2, 'conectividade': 0.3},
    'C_rede_prioritaria':      {'cobertura': 0.3, 'overlap': 0.2, 'conectividade': 0.5},
}

# Verificação de sanidade: cada perfil precisa somar 1.0 — protege
# contra erro de digitação ao editar os pesos no futuro.
for _nome_perfil, _pesos in PESOS_PERFIS.items():
    _soma = sum(_pesos.values())
    assert abs(_soma - 1.0) < 1e-9, f"Perfil '{_nome_perfil}' não soma 1.0 (soma={_soma})"

# ── Paralelização ──────────────────────────────────────────────
"""
A paralelização via joblib nunca é usada por padrão em modo DEMO —
precisa continuar leve em qualquer máquina (ex. notebook sem múltiplos
núcleos, útil para apresentações fora do PC principal). Só o modo
oficial paraleliza por padrão.

Para testar manualmente a paralelização sem disparar o experimento
oficial completo, ative FORCAR_PARALELO_EM_DEMO (só faz sentido numa
máquina com vários núcleos disponíveis).
"""
FORCAR_PARALELO_EM_DEMO = False
USAR_PARALELO = (not DEMO_MODE) or FORCAR_PARALELO_EM_DEMO
N_JOBS = -1   # -1 = usa todos os núcleos disponíveis (joblib)

# ── Matriz completa de cenários (sempre definida) ─────────────
CENARIOS_OFICIAL = [
    {'nome': '100x100_escasso',    'area': (100, 100), 'n_drones':  5, 'radius': RADII[100]},
    {'nome': '100x100_suficiente', 'area': (100, 100), 'n_drones': 10, 'radius': RADII[100]},
    {'nome': '100x100_redundante', 'area': (100, 100), 'n_drones': 15, 'radius': RADII[100]},
    {'nome': '250x250_escasso',    'area': (250, 250), 'n_drones':  5, 'radius': RADII[250]},
    {'nome': '250x250_suficiente', 'area': (250, 250), 'n_drones':  9, 'radius': RADII[250]},
    {'nome': '250x250_redundante', 'area': (250, 250), 'n_drones': 14, 'radius': RADII[250]},
    {'nome': '500x500_escasso',    'area': (500, 500), 'n_drones':  5, 'radius': RADII[500]},
    {'nome': '500x500_suficiente', 'area': (500, 500), 'n_drones': 10, 'radius': RADII[500]},
    {'nome': '500x500_redundante', 'area': (500, 500), 'n_drones': 15, 'radius': RADII[500]},
]

if DEMO_MODE:
    CENARIOS = [c for c in CENARIOS_OFICIAL if c['nome'] == DEMO_CENARIO_NOME]
    if not CENARIOS:
        raise ValueError(
            f"Cenário '{DEMO_CENARIO_NOME}' não encontrado. "
            f"Opções válidas: {[c['nome'] for c in CENARIOS_OFICIAL]}"
        )
else:
    CENARIOS = CENARIOS_OFICIAL

GRID_RES = GRID_RESOLUTION

def aleatorio(objective_func, bounds, **kwargs):
    """
    Baseline sem otimização: posiciona os drones aleatoriamente.
    """
    dim   = bounds.shape[0]
    pos   = np.random.uniform(bounds[:, 0], bounds[:, 1], dim)
    score = objective_func(pos)
    return pos, score, [score]

ALGORITMOS = {
    'Aleatório': lambda obj, b, n, it: aleatorio(obj, b),
    'PSO'      : lambda obj, b, n, it: pso(obj, b, n_particles=n, n_iterations=it),
    'GWO'      : lambda obj, b, n, it: gwo(obj, b, n_wolves=n,    n_iterations=it),
    'WOA'      : lambda obj, b, n, it: woa(obj, b, n_agents=n,    n_iterations=it),
    'SSA'      : lambda obj, b, n, it: ssa(obj, b, n_salps=n,     n_iterations=it),
    'CS'       : lambda obj, b, n, it: cs( obj, b, n_nests=n,     n_iterations=it),
}

modo = "DEMO" if DEMO_MODE else "OFICIAL"
print(f"Modo          : {modo}")
if DEMO_MODE:
    print(f"Cenário demo  : {DEMO_CENARIO_NOME}")
print(f"Runs          : {N_RUNS}")
print(f"Agentes       : {N_AGENTS}")
print(f"Iterações     : {N_ITERATIONS}")
print(f"Cenários      : {len(CENARIOS)}")
print(f"Perfis de peso: {list(PESOS_PERFIS.keys())}")
print(f"Paralelização : {('ativa (' + str(N_JOBS) + ' jobs)') if USAR_PARALELO else 'desativada'}")
print(f"Algoritmos    : {list(ALGORITMOS.keys())}")
print(f"Total runs    : {len(CENARIOS) * len(ALGORITMOS) * N_RUNS * len(PESOS_PERFIS)}")

# %% [markdown]
# ## 8. Runner Experimental
# 
# Para cada cenário e cada perfil de peso, o runner reconstrói os
# parâmetros específicos e executa cada algoritmo N_RUNS vezes de
# forma completamente independente.
# 
# **Seeds:** cada run usa `np.random.seed(run)` antes de chamar o
# algoritmo — o índice da run (0, 1, 2...) é o próprio seed. Isso tem
# dois efeitos importantes: (1) torna cada run reprodutível
# individualmente, mesmo em paralelo, onde cada processo roda de forma
# independente; (2) garante que a run nº 5 do PSO use o MESMO ponto de
# partida aleatório nos 3 perfis de peso — isolando o efeito do peso da
# variação aleatória normal do algoritmo (ver Seção 3.4.5 do TCC).
# 
# **Paralelização:** quando `USAR_PARALELO=True`, as N_RUNS execuções
# de cada algoritmo são despachadas via joblib, aproveitando os
# múltiplos núcleos da máquina. Em modo sequencial (DEMO), o
# comportamento é idêntico, só sem paralelismo.
# 
# **Robustez:** se uma run específica falhar por algum erro numérico
# inesperado, ela não derruba o experimento inteiro — é registrada com
# um aviso e recebe um custo de pior caso (1.0), para não perder horas
# de execução por uma falha isolada.

# %%
def get_metrics_cenario(flat_positions, n_drones, radius, grid, comm_radius):
    """
    Calcula métricas detalhadas de um posicionamento para um cenário
    específico. Versão parametrizada de get_metrics().
    """
    positions  = flat_positions.reshape((n_drones, 2))
    ratio, _   = compute_coverage_fast(positions, radius, grid)
    dmin       = 2 * radius
    area_total = grid[:, 0].max() * grid[:, 1].max()

    n_overlaps = sum(
        1 for i in range(n_drones)
        for j in range(i + 1, n_drones)
        if np.linalg.norm(positions[i] - positions[j]) < dmin
    )

    pares_i, pares_j = [], []
    for i in range(n_drones):
        for j in range(i + 1, n_drones):
            if np.linalg.norm(positions[i] - positions[j]) <= comm_radius:
                pares_i.append(i)
                pares_j.append(j)

    maior_grupo    = (compute_largest_component(n_drones, pares_i, pares_j)
                      if n_drones > 1 else 1)
    frac_conectada = maior_grupo / n_drones

    return {
        'taxa_cobertura'       : round(ratio, 4),
        'area_coberta_m2'      : round(ratio * area_total, 1),
        'n_overlaps'           : n_overlaps,
        'maior_grupo_conectado': maior_grupo,
        'fracao_conectada'     : round(frac_conectada, 4),
    }


def executar_run_unica(seed, algo_func, obj, bounds, n_agents, n_iterations,
                        n_drones, radius, grid, comm_radius):
    """
    Executa uma única run de um algoritmo com seed fixo, retornando
    posição final, custo e métricas detalhadas. Isolada em função
    própria para poder ser despachada tanto sequencialmente quanto em
    paralelo (joblib) sem duplicar lógica.

    Em caso de erro numérico inesperado, não propaga a exceção — registra
    um aviso e devolve um resultado de pior caso (custo=1.0), para que
    uma falha isolada não derrube um experimento de horas.
    """
    try:
        np.random.seed(seed)
        pos, custo, _ = algo_func(obj, bounds, n_agents, n_iterations)
        m = get_metrics_cenario(pos, n_drones, radius, grid, comm_radius)
        return pos, custo, m
    except Exception as e:
        print(f"\n  ⚠ ERRO na run seed={seed}: {e} — usando resultado de pior caso")
        pos_fallback = np.random.uniform(bounds[:, 0], bounds[:, 1])
        m_fallback   = get_metrics_cenario(pos_fallback, n_drones, radius, grid, comm_radius)
        return pos_fallback, 1.0, m_fallback


def run_experiment(cenarios, algoritmos, n_runs, n_agents, n_iterations,
                    pesos, paralelo=False, n_jobs=-1):
    """
    Executa o experimento para UM perfil de peso sobre os cenários
    dados, retornando todos os resultados.

    Parâmetros
    ----------
    pesos    : dict com chaves 'cobertura', 'overlap', 'conectividade'
    paralelo : se True, despacha as N_RUNS de cada algoritmo via joblib
    n_jobs   : número de processos paralelos (-1 = todos os núcleos)

    Retorna
    -------
    dict aninhado: {nome_cenario: {nome_algoritmo: {
        'costs', 'coberturas', 'overlaps', 'conectividades',
        'best_pos', 'best_cost'
    }}}
    """
    todos_resultados = {}
    t_inicio_total   = time.time()

    for cenario in cenarios:
        nome           = cenario['nome']
        area_w, area_h = cenario['area']
        n_drones       = cenario['n_drones']
        radius         = cenario['radius']
        res            = GRID_RES[area_w]

        comm_radius = COMM_RADIUS_FACTOR * radius

        grid   = build_grid(area_w, area_h, res)
        bounds = np.tile([[0, area_w], [0, area_h]], (n_drones, 1))

        max_ov = (n_drones * (n_drones - 1) / 2) * (2 * radius)

        """
        make_obj() encapsula os parâmetros do cenário E os pesos do
        perfil ativo (w=pesos) como argumentos padrão — "congela" os
        valores no momento da criação, evitando o late binding clássico
        de closures em loops Python.
        """
        def make_obj(nd=n_drones, r=radius, g=grid, cr=comm_radius, mo=max_ov, w=pesos):
            return lambda x: objective_fast(x, nd, r, g, cr, mo,
                                             w['cobertura'], w['overlap'], w['conectividade'])
        obj = make_obj()

        print(f"\n{'='*60}")
        print(f" Cenário : {nome}")
        print(f" Área    : {area_w}×{area_h}m | Drones: {n_drones}"
              f" | Raio: {radius}m | R_comm: {comm_radius}m | Grid: {res}m")
        print(f"{'='*60}")

        todos_resultados[nome] = {}

        for algo_nome, algo_func in algoritmos.items():
            t_algo = time.time()

            if paralelo:
                print(f"  {algo_nome:10s} | {n_runs} runs em paralelo...", end='', flush=True)
                resultados_runs = Parallel(n_jobs=n_jobs)(
                    delayed(executar_run_unica)(
                        run, algo_func, obj, bounds, n_agents, n_iterations,
                        n_drones, radius, grid, comm_radius
                    )
                    for run in range(n_runs)
                )
            else:
                print(f"  {algo_nome:10s} | runs: ", end='', flush=True)
                resultados_runs = []
                for run in range(n_runs):
                    resultados_runs.append(
                        executar_run_unica(
                            run, algo_func, obj, bounds, n_agents, n_iterations,
                            n_drones, radius, grid, comm_radius
                        )
                    )
                    print(f"{run+1} ", end='', flush=True)

            custos, coberturas, overlaps, conectividades = [], [], [], []
            best_pos, best_cost = None, float('inf')

            for pos, custo, m in resultados_runs:
                custos.append(custo)
                coberturas.append(m['taxa_cobertura'])
                overlaps.append(m['n_overlaps'])
                conectividades.append(m['fracao_conectada'])
                if custo < best_cost:
                    best_cost = custo
                    best_pos  = pos.copy()

            tempo  = time.time() - t_algo
            custos = np.array(custos)
            print(f" | média: {custos.mean():.4f} ± {custos.std():.4f}  [{tempo:.1f}s] ✓")

            todos_resultados[nome][algo_nome] = {
                'costs'         : custos,
                'coberturas'    : np.array(coberturas),
                'overlaps'      : np.array(overlaps),
                'conectividades': np.array(conectividades),
                'best_pos'      : best_pos,
                'best_cost'     : best_cost,
            }

    t_total = time.time() - t_inicio_total
    print(f"\n{'='*60}")
    print(f" Cenários deste perfil concluídos em {t_total:.1f}s")
    print(f"{'='*60}")
    return todos_resultados

# %% [markdown]
# ## 8b. Execução do Experimento — Loop sobre os Perfis de Peso
# 
# Repete `run_experiment` uma vez para cada perfil de peso definido em
# PESOS_PERFIS, mantendo os resultados de cada perfil completamente
# separados em `resultados_todos_perfis[nome_perfil]` — nenhum dado de
# um perfil se mistura com outro.

# %%
resultados_todos_perfis = {}
t_inicio_geral = time.time()

for nome_perfil, pesos in PESOS_PERFIS.items():
    print(f"\n{'#'*70}")
    print(f" PERFIL DE PESO: {nome_perfil}  →  {pesos}")
    print(f"{'#'*70}")
    resultados_todos_perfis[nome_perfil] = run_experiment(
        CENARIOS, ALGORITMOS, N_RUNS, N_AGENTS, N_ITERATIONS,
        pesos=pesos, paralelo=USAR_PARALELO, n_jobs=N_JOBS
    )

print(f"\n{'#'*70}")
print(f" TODOS OS PERFIS CONCLUÍDOS em {time.time() - t_inicio_geral:.1f}s")
print(f"{'#'*70}")

# %% [markdown]
# ## Salvamento e Recuperação de Resultados
# 
# Cada perfil de peso é salvo em uma subpasta própria
# (`resultados_tcc/perfil_<nome>/`), com os mesmos três formatos de
# antes — garante que os resultados de perfis diferentes nunca se
# misturem em disco.
# 
# No modo DEMO, o salvamento é pulado — os resultados são descartáveis
# por definição, servindo apenas para validação rápida do notebook.

# %%
def salvar_resultados(resultados, df_raw, df_resumo, pasta='resultados_tcc'):
    """
    Exporta os resultados de UM perfil em três formatos complementares.
    """
    os.makedirs(pasta, exist_ok=True)

    with open(f'{pasta}/resultados_completos.pkl', 'wb') as f:
        pickle.dump(resultados, f)

    df_raw.to_csv(f'{pasta}/matriz_bruta.csv',       index=False)
    df_resumo.to_csv(f'{pasta}/resumo_estatistico.csv', index=False)

    print(f"Resultados salvos em '{pasta}/':")
    print(f"  • resultados_completos.pkl")
    print(f"  • matriz_bruta.csv")
    print(f"  • resumo_estatistico.csv")


def carregar_resultados(pasta='resultados_tcc'):
    """
    Recupera os resultados de UM perfil de uma sessão anterior.
    """
    with open(f'{pasta}/resultados_completos.pkl', 'rb') as f:
        resultados = pickle.load(f)
    df_raw    = pd.read_csv(f'{pasta}/matriz_bruta.csv')
    df_resumo = pd.read_csv(f'{pasta}/resumo_estatistico.csv')
    print(f"Resultados carregados de '{pasta}/'.")
    return resultados, df_raw, df_resumo

# %%
resultados_todos_perfis = {}
for perfil in ['A_cobertura_prioritaria', 'B_transicao', 'C_rede_prioritaria']:
    pasta = f'resultados_tcc/perfil_{perfil}'
    resultados, _, _ = carregar_resultados(pasta=pasta)
    resultados_todos_perfis[perfil] = resultados

print("resultados_todos_perfis reconstruído a partir dos arquivos salvos.")

# %% [markdown]
# ## 9. Análise Estatística dos Resultados
# 
# Para cada combinação (perfil, cenário, algoritmo), as N_RUNS
# execuções independentes produzem uma distribuição de custos finais.
# `calcular_estatisticas()` agora recebe o nome do perfil e marca cada
# linha com ele — isso permite consolidar os 3 perfis num único
# DataFrame (`df_resumo_geral`) sem perder a separação entre eles.

# %%
def calcular_estatisticas(resultados, nome_perfil):
    """
    Calcula média, DP e MSE para cada par (cenário, algoritmo) de UM
    perfil de peso, marcando cada linha com o nome do perfil.

    Retorna
    -------
    df_raw    : DataFrame, uma linha por execução
    df_resumo : DataFrame, uma linha por cenário (deste perfil)
    """
    nomes_algo  = list(list(resultados.values())[0].keys())
    raw_rows    = []
    resumo_rows = []

    for cenario, algos in resultados.items():
        melhor_global = min(algos[a]['best_cost'] for a in algos)
        n_runs        = len(list(algos.values())[0]['costs'])

        for run_idx in range(n_runs):
            row = {'perfil': nome_perfil, 'cenario': cenario, 'run': run_idx + 1}
            for algo in nomes_algo:
                d = algos[algo]
                row[f'{algo}_custo']         = d['costs'][run_idx]
                row[f'{algo}_cobertura']     = d['coberturas'][run_idx]
                row[f'{algo}_overlaps']      = d['overlaps'][run_idx]
                row[f'{algo}_conectividade'] = d['conectividades'][run_idx]
            raw_rows.append(row)

        row_resumo = {'perfil': nome_perfil, 'cenario': cenario}
        for algo in nomes_algo:
            d      = algos[algo]
            custos = d['costs']

            row_resumo[f'{algo}_media_custo'] = round(custos.mean(), 4)
            row_resumo[f'{algo}_dp']          = round(custos.std(), 4)
            row_resumo[f'{algo}_mse']         = round(
                ((custos - melhor_global) ** 2).mean(), 6
            )
            row_resumo[f'{algo}_cobertura_media'] = round(
                d['coberturas'].mean() * 100, 1)
            row_resumo[f'{algo}_overlaps_medio'] = round(
                d['overlaps'].mean(), 1)
            row_resumo[f'{algo}_conectividade_media'] = round(
                d['conectividades'].mean() * 100, 1)
        resumo_rows.append(row_resumo)

    df_raw    = pd.DataFrame(raw_rows)
    df_resumo = pd.DataFrame(resumo_rows)

    return df_raw, df_resumo


def imprimir_resumo(df_resumo_geral, nomes_algo):
    """
    Exibe o resumo estatístico formatado por perfil × cenário,
    destacando o melhor algoritmo em custo, estabilidade e cobertura.
    """
    print(f"\n{'RESUMO ESTATÍSTICO COMPLETO':^76}")

    for _, row in df_resumo_geral.iterrows():
        print(f"\n{'='*76}")
        print(f" Perfil: {row['perfil']}  |  Cenário: {row['cenario']}")
        print(f"{'='*76}")
        print(f"  {'Algo':<10} {'Custo':>8} {'DP':>7} {'MSE':>10} "
              f"{'Cobert%':>9} {'Overlaps':>9} {'Conect%':>9}")
        print(f"  {'-'*68}")

        for algo in nomes_algo:
            print(f"  {algo:<10} "
                  f"{row[f'{algo}_media_custo']:>8.4f} "
                  f"{row[f'{algo}_dp']:>7.4f} "
                  f"{row[f'{algo}_mse']:>10.6f} "
                  f"{row[f'{algo}_cobertura_media']:>8.1f}% "
                  f"{row[f'{algo}_overlaps_medio']:>9.1f} "
                  f"{row[f'{algo}_conectividade_media']:>8.1f}%")

        medias     = {a: row[f'{a}_media_custo']     for a in nomes_algo}
        dps        = {a: row[f'{a}_dp']              for a in nomes_algo}
        coberturas = {a: row[f'{a}_cobertura_media'] for a in nomes_algo}

        print(f"\n  → Menor custo    : {min(medias, key=medias.get)}"
              f" ({min(medias.values()):.4f})")
        print(f"  → Mais estável   : {min(dps, key=dps.get)}"
              f" (DP={min(dps.values()):.4f})")
        print(f"  → Maior cobertura: {max(coberturas, key=coberturas.get)}"
              f" ({max(coberturas.values()):.1f}%)")


def tabela_resumo_legivel(df_resumo_geral, nomes_algo):
    """
    Transforma o resumo (formato largo — uma coluna por algoritmo ×
    métrica) em formato longo (uma linha por perfil × cenário ×
    algoritmo). Muito mais fácil de ler com 3 perfis × 9 cenários ao
    mesmo tempo do que o df_resumo.to_string() bruto.
    """
    linhas = []
    for _, row in df_resumo_geral.iterrows():
        for algo in nomes_algo:
            linhas.append({
                'perfil'         : row['perfil'],
                'cenario'        : row['cenario'],
                'algoritmo'      : algo,
                'custo_medio'    : row[f'{algo}_media_custo'],
                'dp'             : row[f'{algo}_dp'],
                'cobertura_%'    : row[f'{algo}_cobertura_media'],
                'overlaps'       : row[f'{algo}_overlaps_medio'],
                'conectividade_%': row[f'{algo}_conectividade_media'],
            })
    return pd.DataFrame(linhas).sort_values(['perfil', 'cenario', 'custo_medio'])


def calcular_ranking_robustez(df_resumo_geral, nomes_algo):
    """
    Métrica de robustez definida na Seção 3.4.3 do TCC: para cada
    combinação (perfil, cenário), calcula o ranking de cada algoritmo
    pelo custo médio (1 = melhor), depois tira a média desses rankings
    por algoritmo. O algoritmo mais robusto às diferentes prioridades
    operacionais é o de menor ranking médio — não necessariamente o de
    menor custo bruto em um perfil isolado.

    Também retorna, como destaque narrativo extra (não estatístico),
    quantas vezes cada algoritmo ficou em 1º lugar isolado.
    """
    linhas = []
    for _, row in df_resumo_geral.iterrows():
        custos    = {a: row[f'{a}_media_custo'] for a in nomes_algo}
        ordenado  = sorted(custos, key=custos.get)
        for posicao, algo in enumerate(ordenado, start=1):
            linhas.append({'perfil': row['perfil'], 'cenario': row['cenario'],
                            'algoritmo': algo, 'ranking': posicao})

    df_rank       = pd.DataFrame(linhas)
    ranking_medio = df_rank.groupby('algoritmo')['ranking'].mean().sort_values()
    vitorias      = df_rank[df_rank['ranking'] == 1]['algoritmo'].value_counts()

    return df_rank, ranking_medio, vitorias


def comparar_vs_baseline(df_resumo_geral, nomes_algo, baseline='Aleatório'):
    """
    Para cada algoritmo (exceto o baseline), calcula o ganho de
    cobertura em pontos percentuais em relação ao Aleatório — quantifica
    o valor agregado pela otimização frente a um posicionamento sem
    inteligência, em cada perfil × cenário.
    """
    linhas = []
    for _, row in df_resumo_geral.iterrows():
        cob_base = row[f'{baseline}_cobertura_media']
        for algo in nomes_algo:
            if algo == baseline:
                continue
            cob_algo = row[f'{algo}_cobertura_media']
            linhas.append({
                'perfil'              : row['perfil'],
                'cenario'             : row['cenario'],
                'algoritmo'           : algo,
                'cobertura_%'         : cob_algo,
                'ganho_vs_aleatorio_pp': round(cob_algo - cob_base, 1),
            })
    return pd.DataFrame(linhas)


# ── Execução da análise (consolidando os 3 perfis) ─────────────
nomes_algo = list(ALGORITMOS.keys())

_df_raw_lista, _df_resumo_lista = [], []
for nome_perfil, resultados in resultados_todos_perfis.items():
    df_raw_p, df_resumo_p = calcular_estatisticas(resultados, nome_perfil)
    _df_raw_lista.append(df_raw_p)
    _df_resumo_lista.append(df_resumo_p)

    """
    Salva imediatamente após calcular as estatísticas DESTE perfil —
    antes das visualizações — protegendo os dados numéricos mesmo que
    algo falhe nas células de plot a seguir. Só fora do modo DEMO.
    """
    if not DEMO_MODE:
        salvar_resultados(resultados, df_raw_p, df_resumo_p,
                           pasta=f'resultados_tcc/perfil_{nome_perfil}')

df_raw_geral    = pd.concat(_df_raw_lista,    ignore_index=True)
df_resumo_geral = pd.concat(_df_resumo_lista, ignore_index=True)

imprimir_resumo(df_resumo_geral, nomes_algo)

print(f"\n\n{'TABELA RESUMO (formato legível)':^76}")
print(tabela_resumo_legivel(df_resumo_geral, nomes_algo).to_string(index=False))

print(f"\n\n{'RANKING DE ROBUSTEZ ENTRE PERFIS':^76}")
df_rank, ranking_medio, vitorias = calcular_ranking_robustez(df_resumo_geral, nomes_algo)
print("\nRanking médio (menor = mais robusto às diferentes prioridades):")
print(ranking_medio.to_string())
print("\nVitórias isoladas (1º lugar em quantas combinações perfil×cenário):")
print(vitorias.to_string())

print(f"\n\n{'GANHO DE COBERTURA VS. BASELINE ALEATÓRIO':^76}")
df_comparacao = comparar_vs_baseline(df_resumo_geral, nomes_algo)
print(df_comparacao.to_string(index=False))

# %% [markdown]
# ## 10. Teste Estatístico de Wilcoxon
# 
# Aplicado dentro de cada combinação (perfil, cenário) — comparar
# algoritmos sob o MESMO perfil de peso, nunca entre perfis diferentes
# (isso não faria sentido estatístico, já que mudam a própria definição
# do problema sendo otimizado).
# 
# O teste requer mínimo de 6 amostras. No modo DEMO (N_RUNS reduzido),
# os resultados são apenas indicativos.

# %%
def teste_wilcoxon(resultados_todos_perfis, alpha=0.05):
    """
    Aplica o teste de Wilcoxon signed-rank entre todos os pares de
    algoritmos, para cada cenário, dentro de cada perfil de peso
    separadamente (Derrac et al., 2011).
    """
    primeiro_perfil    = list(resultados_todos_perfis.values())[0]
    nomes_algo         = list(list(primeiro_perfil.values())[0].keys())
    pares = [
        (nomes_algo[i], nomes_algo[j])
        for i in range(len(nomes_algo))
        for j in range(i + 1, len(nomes_algo))
    ]

    for nome_perfil, resultados in resultados_todos_perfis.items():
        print(f"\n{'#'*70}")
        print(f" PERFIL: {nome_perfil}")
        print(f"{'#'*70}")

        for cenario, algos in resultados.items():
            n_runs = len(list(algos.values())[0]['costs'])
            print(f"\nCenário: {cenario}  (n={n_runs} runs)")

            if n_runs < 6:
                print(f"  ⚠ Wilcoxon requer mínimo 6 runs — "
                      f"resultados abaixo são apenas indicativos.")

            print(f"  {'Par':<20} {'W':>8} {'p-valor':>10} {'Significativo':>14}")
            print(f"  {'-'*54}")

            for a, b in pares:
                custos_a = algos[a]['costs']
                custos_b = algos[b]['costs']
                try:
                    stat, p = wilcoxon(custos_a, custos_b)
                    sig     = "✓ SIM" if p < alpha else "✗ NÃO"
                    print(f"  {a} vs {b:<10} {stat:>8.2f} {p:>10.4f} {sig:>14}")
                except Exception as e:
                    print(f"  {a} vs {b:<10}  —  {e}")

    print(f"\n  α = {alpha} | p < α → diferença estatisticamente significativa")


teste_wilcoxon(resultados_todos_perfis)

# %% [markdown]
# ## 10a. Correção de Holm-Bonferroni (múltiplas comparações)
# 
# Com 6 condições (5 algoritmos + baseline), cada combinação cenário×perfil gera C(6,2) = 15 comparações par-a-par. Testar 15 hipóteses simultaneamente sem correção infla a taxa de erro tipo I (chance de encontrar "diferença significativa" por acaso). A correção de Holm-Bonferroni ajusta o limiar de significância de forma sequencial: ordena os p-valores do menor para o maior e exige um alpha cada vez mais permissivo à medida que se avança na lista — menos conservadora que a Bonferroni simples, mas igualmente válida estatisticamente (HOLM, 1979).
# 
# Esta célula não re-executa o experimento — reprocessa os mesmos p-valores já calculados por teste_wilcoxon(), então pode ser adicionada a qualquer momento após a Seção 10, sem custo computacional adicional.

# %%
def holm_bonferroni(p_valores, alpha=0.05):
    """
    Aplica a correção de Holm-Bonferroni a uma lista de p-valores de
    uma mesma família de testes (ex.: as 15 comparações par-a-par de
    um único cenário×perfil).
 
    Procedimento (Holm, 1979): ordena os p-valores do menor para o
    maior; para o k-ésimo menor (k=1,2,...,m), o limiar exigido é
    alpha / (m - k + 1). Um teste é significativo somente se seu
    p-valor for menor que o limiar correspondente E todos os testes
    com p-valor menor também foram significativos — na primeira falha
    todos os restantes são rejeitados automaticamente.
 
    Retorna
    -------
    lista de tuplas (índice_original, p_valor, limiar_exigido, significativo)
    ordenada pelo índice original (não pelo p-valor), de modo que o
    chamador pode fazer zip com a lista original de pares sem
    precisar reordenar.
    """
    m = len(p_valores)
 
    """
    Ordena os índices dos p-valores do menor para o maior, sem
    modificar a lista original — guardamos os índices porque
    precisamos devolver os resultados na ordem original dos pares,
    não na ordem crescente de p-valor.
    """
    indices_ordenados = sorted(range(m), key=lambda i: p_valores[i])
 
    resultados = [None] * m   # pré-aloca com None para preencher na ordem de índice original
    parou = False             # flag de parada: uma vez que um teste falha, todos os seguintes falham
 
    for k, idx in enumerate(indices_ordenados, start=1):
        # limiar decresce conforme k cresce: quanto mais alto o p-valor, mais fácil de ser significativo
        limiar = alpha / (m - k + 1)
 
        # só é significativo se: (1) ainda não houve parada E (2) p-valor abaixo do limiar ajustado
        sig = (not parou) and (p_valores[idx] < limiar)
 
        if not sig:
            parou = True   # a partir daqui, nenhum outro teste pode ser significativo — parada sequencial
 
        resultados[idx] = (idx, p_valores[idx], limiar, sig)   # devolve na posição original do índice
 
    return resultados
 
 
def teste_wilcoxon_com_correcao(resultados_todos_perfis, alpha=0.05):
    """
    Repete a lógica de teste_wilcoxon() (Seção 10), mas aplica a
    correção de Holm-Bonferroni às 15 comparações de cada combinação
    cenário×perfil antes de reportar significância.
 
    Produz dois outputs:
    - Tabela impressa por combinação perfil×cenário, mostrando o
      p-valor bruto, o limiar ajustado e os dois vereditos lado a lado.
    - df_wilcoxon_corrigido: DataFrame com uma linha por comparação,
      útil para análise posterior ou exportação para o TCC.
    """
    primeiro_perfil = list(resultados_todos_perfis.values())[0]
    nomes_algo      = list(list(primeiro_perfil.values())[0].keys())
 
    """
    Gera todas as combinações par-a-par de algoritmos sem repetição.
    Com 6 algoritmos (incluindo o baseline aleatório), isso produz
    C(6,2) = 15 pares — exatamente a família de testes que a correção
    de Holm-Bonferroni vai avaliar em conjunto.
    """
    pares = [
        (nomes_algo[i], nomes_algo[j])
        for i in range(len(nomes_algo))
        for j in range(i + 1, len(nomes_algo))
    ]
 
    linhas_resumo = []   # acumula uma linha por comparação para montar o DataFrame no final
 
    for nome_perfil, resultados in resultados_todos_perfis.items():
        for cenario, algos in resultados.items():
            n_runs = len(list(algos.values())[0]['costs'])
 
            """
            Coleta os p-valores de todos os 15 pares nesta combinação
            perfil×cenário. Em caso de falha (ex.: todos os custos
            idênticos — impossível com algoritmos estocásticos, mas
            defensivo), substitui por p=1.0 (trata como não-significativo
            em vez de propagar exceção e interromper a análise inteira).
            """
            p_valores = []
            for a, b in pares:
                try:
                    _, p = wilcoxon(algos[a]['costs'], algos[b]['costs'])
                except Exception:
                    p = 1.0   # segurança: sem significância se o teste não puder ser calculado
                p_valores.append(p)
 
            corrigidos = holm_bonferroni(p_valores, alpha=alpha)
 
            print(f"\nPerfil: {nome_perfil}  |  Cenário: {cenario}  (n={n_runs} runs)")
            if n_runs < 6:
                print(f"  ⚠ Wilcoxon requer mínimo 6 runs — resultados abaixo são apenas indicativos.")
            print(f"  {'Par':<20} {'p-valor':>10} {'limiar Holm':>12} {'Sig. bruto':>12} {'Sig. Holm':>11}")
            print(f"  {'-'*68}")
 
            """
            zip(corrigidos, pares) emparelha cada resultado corrigido
            com o nome do par correspondente. Funciona porque
            holm_bonferroni() devolve os resultados na ordem original
            dos índices — mesma ordem em que os p-valores foram
            calculados, que é a mesma ordem da lista 'pares'.
            """
            for (idx, p, limiar, sig_holm), (a, b) in zip(corrigidos, pares):
                sig_bruto = "✓" if p < alpha else "✗"   # significância sem correção (alpha simples)
                sig_h     = "✓" if sig_holm else "✗"    # significância após correção de Holm-Bonferroni
 
                print(f"  {a} vs {b:<10} {p:>10.4f} {limiar:>12.4f} {sig_bruto:>12} {sig_h:>11}")
 
                linhas_resumo.append({
                    'perfil'              : nome_perfil,
                    'cenario'             : cenario,
                    'par'                 : f'{a} vs {b}',
                    'p_valor'             : p,
                    'significativo_bruto' : p < alpha,    # True/False sem correção
                    'significativo_holm'  : sig_holm,     # True/False após correção de Holm
                })
 
    df_wilcoxon_corrigido = pd.DataFrame(linhas_resumo)
 
    """
    Resumo agregado: mostra quantas comparações perderam significância
    após a correção — ou seja, eram aparentemente significativas pelo
    alpha simples, mas não resistiram ao ajuste para múltiplos testes.
    Esse número é o impacto real da correção de Holm-Bonferroni.
    """
    n_perde_significancia = (
        (df_wilcoxon_corrigido['significativo_bruto']) &   # era significativo sem correção
        (~df_wilcoxon_corrigido['significativo_holm'])     # mas NÃO É após correção
    ).sum()
 
    print(f"\n{'='*68}")
    print(f"Total de comparações           : {len(df_wilcoxon_corrigido)}")
    print(f"Significativas sem correção    : {df_wilcoxon_corrigido['significativo_bruto'].sum()}")
    print(f"Significativas com Holm        : {df_wilcoxon_corrigido['significativo_holm'].sum()}")
    print(f"Perderam significância (Holm)  : {n_perde_significancia}")
    print(f"{'='*68}")
 
    return df_wilcoxon_corrigido
 
 
# ── Execução ──────────────────────────────────────────────────
executar_teste_wilcoxon_corrigido = True  # True para aplicar Holm-Bonferroni, False para apenas imprimir Wilcoxon simples
if executar_teste_wilcoxon_corrigido:
    df_wilcoxon_corrigido = teste_wilcoxon_com_correcao(resultados_todos_perfis)
else:
    print("Correção de Holm-Bonferroni carregada. Acione a linha acima para aplicar (True).")

# %% [markdown]
# ## 11. Visualizações
# 
# ### 11.1 Curvas de Convergência
# 
# Ilustra a convergência de cada algoritmo sob o Perfil A (cobertura
# prioritária) por padrão — passe outro dicionário de pesos em `pesos=`
# para ilustrar um perfil diferente.

# %%
def plot_convergencia(cenario_nome, cenario_params, algoritmos,
                      n_agents, n_iterations, pesos=None):
    """
    Executa cada algoritmo uma vez com seed fixo e plota a evolução
    do melhor custo por iteração — Figura 2 do TCC.
    """
    if pesos is None:
        pesos = PESOS_PERFIS['A_cobertura_prioritaria']

    area_w, area_h = cenario_params['area']
    n_drones       = cenario_params['n_drones']
    radius         = cenario_params['radius']
    comm_radius    = COMM_RADIUS_FACTOR * radius
    res            = GRID_RES[area_w]
    grid           = build_grid(area_w, area_h, res)
    bounds         = np.tile([[0, area_w], [0, area_h]], (n_drones, 1))

    max_ov = (n_drones * (n_drones - 1) / 2) * (2 * radius)

    def make_obj(nd=n_drones, r=radius, g=grid, cr=comm_radius, mo=max_ov, w=pesos):
        return lambda x: objective_fast(x, nd, r, g, cr, mo,
                                         w['cobertura'], w['overlap'], w['conectividade'])
    obj = make_obj()

    cores = {
        'Aleatório': '#808080', 'PSO': '#1f77b4', 'GWO': '#2ca02c',
        'WOA': '#d62728', 'SSA': '#ff7f0e', 'CS': '#9467bd',
    }

    plt.figure(figsize=(10, 5))
    np.random.seed(42)

    for algo_nome, algo_func in algoritmos.items():
        _, _, history = algo_func(obj, bounds, n_agents, n_iterations)

        if algo_nome == 'Aleatório':
            plt.axhline(y=history[0], color=cores['Aleatório'],
                        linewidth=1.5, linestyle='--',
                        label='Aleatório (baseline)')
        else:
            plt.plot(history, label=algo_nome,
                     color=cores.get(algo_nome), linewidth=2)

    plt.xlabel('Iteração', fontsize=12)
    plt.ylabel('Melhor custo encontrado', fontsize=12)
    plt.title(f'Convergência dos Algoritmos — {cenario_nome}', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'convergencia_{cenario_nome}.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Salvo: convergencia_{cenario_nome}.png")


cenario_vis = CENARIOS[0]
plot_convergencia(cenario_vis['nome'], cenario_vis, ALGORITMOS, N_AGENTS, N_ITERATIONS)

# %% [markdown]
# ### 11.2 Mapa de Posicionamento Final
# 
# Visualiza o resultado do Perfil A por padrão — troque
# `resultados_todos_perfis['A_cobertura_prioritaria']` por outro perfil
# na chamada abaixo para visualizar um perfil diferente.

# %%
def plot_cobertura(cenario_nome, cenario_params, resultados):
    """
    Plota o posicionamento final de cada algoritmo lado a lado —
    Figura 3 do TCC. `resultados` aqui é o dicionário de UM perfil
    específico (ex. resultados_todos_perfis['A_cobertura_prioritaria']).
    """
    area_w, area_h = cenario_params['area']
    n_drones       = cenario_params['n_drones']
    radius         = cenario_params['radius']
    comm_radius    = COMM_RADIUS_FACTOR * radius
    res            = GRID_RES[area_w]
    grid           = build_grid(area_w, area_h, res)
    algos          = list(resultados[cenario_nome].keys())

    cores = {
        'Aleatório': '#808080', 'PSO': '#1f77b4', 'GWO': '#2ca02c',
        'WOA': '#d62728', 'SSA': '#ff7f0e', 'CS': '#9467bd',
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for ax, algo in zip(axes, algos):
        best_pos    = resultados[cenario_nome][algo]['best_pos']
        positions   = best_pos.reshape((n_drones, 2))
        ratio, mask = compute_coverage_fast(positions, radius, grid)
        cor         = cores.get(algo, '#333333')

        pts_cobertos = grid[mask]
        ax.scatter(pts_cobertos[:, 0], pts_cobertos[:, 1],
                   c=cor, alpha=0.12, s=1)

        for pos in positions:
            circulo = patches.Circle(
                pos, radius, fill=False,
                edgecolor=cor, linewidth=1.5, alpha=0.85
            )
            ax.add_patch(circulo)

        for i in range(n_drones):
            for j in range(i + 1, n_drones):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist <= comm_radius:
                    ax.plot([positions[i, 0], positions[j, 0]],
                           [positions[i, 1], positions[j, 1]],
                           color=cor, linewidth=0.8, alpha=0.4, zorder=3)

        ax.scatter(positions[:, 0], positions[:, 1],
                   c=cor, s=60, zorder=5,
                   edgecolors='black', linewidth=0.8)

        ax.scatter(*START_POINT, c='black', s=100,
                   marker='*', zorder=6)

        borda = patches.Rectangle(
            (0, 0), area_w, area_h,
            fill=False, edgecolor='black', linewidth=1.5
        )
        ax.add_patch(borda)

        pares_i, pares_j = [], []
        for i in range(n_drones):
            for j in range(i + 1, n_drones):
                if np.linalg.norm(positions[i] - positions[j]) <= comm_radius:
                    pares_i.append(i)
                    pares_j.append(j)
        maior_grupo = (compute_largest_component(n_drones, pares_i, pares_j)
                      if n_drones > 1 else 1)
        frac = maior_grupo / n_drones

        ax.set_xlim(-5, area_w + 5)
        ax.set_ylim(-5, area_h + 5)
        ax.set_aspect('equal')
        ax.set_title(f'{algo}\nCobert: {ratio:.1%} | Conect: {frac:.1%}',
                     fontsize=10)
        ax.set_xlabel('x (m)', fontsize=10)
        ax.set_ylabel('y (m)', fontsize=10)
        ax.grid(True, alpha=0.2)

    plt.suptitle(f'Posicionamento Final dos Drones — {cenario_nome}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'cobertura_{cenario_nome}.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Salvo: cobertura_{cenario_nome}.png")


plot_cobertura(cenario_vis['nome'], cenario_vis,
               resultados_todos_perfis['A_cobertura_prioritaria'])

# %%
import pandas as pd

dfs = []
for perfil in ['A_cobertura_prioritaria', 'B_transicao', 'C_rede_prioritaria']:
    caminho = f'resultados_tcc/perfil_{perfil}/resumo_estatistico.csv'
    dfs.append(pd.read_csv(caminho))

df_consolidado = pd.concat(dfs, ignore_index=True)
df_consolidado.to_csv('resumo_estatistico_consolidado.csv', index=False)
print(f"Salvo: resumo_estatistico_consolidado.csv ({len(df_consolidado)} linhas)")

# %%
# Salva os 4 resultados adicionais que hoje só existem impressos no console
os.makedirs('resultados_tcc/analises', exist_ok=True)

# 1. Wilcoxon com correção de Holm-Bonferroni (célula 10a)
df_wilcoxon_corrigido.to_csv('resultados_tcc/analises/wilcoxon_holm.csv', index=False)

# 2. Ranking de robustez entre perfis
df_rank.to_csv('resultados_tcc/analises/ranking_por_combinacao.csv', index=False)
ranking_medio.to_csv('resultados_tcc/analises/ranking_medio.csv', header=['ranking_medio'])
vitorias.to_csv('resultados_tcc/analises/vitorias_isoladas.csv', header=['vitorias'])

# 3. Ganho de cobertura vs. baseline aleatório
df_comparacao.to_csv('resultados_tcc/analises/ganho_vs_baseline.csv', index=False)

print("Salvos em 'resultados_tcc/analises/':")
print("  • wilcoxon_holm.csv")
print("  • ranking_por_combinacao.csv")
print("  • ranking_medio.csv")
print("  • vitorias_isoladas.csv")
print("  • ganho_vs_baseline.csv")

# %%
import pandas as pd

# Consolida os 3 resumo_estatistico.csv num só
dfs = []
for perfil in ['A_cobertura_prioritaria', 'B_transicao', 'C_rede_prioritaria']:
    caminho = f'resultados_tcc/perfil_{perfil}/resumo_estatistico.csv'
    dfs.append(pd.read_csv(caminho))
df_consolidado = pd.concat(dfs, ignore_index=True)
df_consolidado.to_csv('resumo_estatistico_consolidado.csv', index=False)
print(f"resumo_estatistico_consolidado.csv salvo ({len(df_consolidado)} linhas, esperado 27)")

# Confirma os 5 arquivos de análise
import os
for arq in ['wilcoxon_holm.csv', 'ranking_por_combinacao.csv', 'ranking_medio.csv',
            'vitorias_isoladas.csv', 'ganho_vs_baseline.csv']:
    caminho = f'resultados_tcc/analises/{arq}'
    status = "✓" if os.path.exists(caminho) else "✗ FALTANDO"
    print(f"  {status}  {caminho}")

# %% [markdown]
# ## 11a. Redefinição das Funções de Plot (com sufixo funcionando)
# 
# Redefine plot_convergencia e plot_cobertura (definidas originalmente na Seção 11) com dois ajustes:
# - parâmetro `sufixo`, para diferenciar o nome do arquivo salvo quando o mesmo cenário é plotado sob perfis de peso diferentes (sem isso, o mapa do perfil B sobrescreveria o do perfil A, já que os dois teriam o mesmo nome de arquivo);
# - plt.close() após plt.show(), para liberar a memória de cada figura depois de salva — gerando dezenas de imagens em sequência sem fechar, o matplotlib acumula todas em memória e emite avisos de "too many open figures".
# 
# Redefinir uma função em uma célula posterior é uma prática normal em notebooks: a versão nova simplesmente substitui a anterior a partir daqui, sem precisar editar a Seção 11 original.

# %%
import os
 
def plot_convergencia(cenario_nome, cenario_params, algoritmos,
                      n_agents, n_iterations, pesos=None, sufixo=''):
    """
    Versão com sufixo (para diferenciar arquivos ao rodar por perfil)
    e plt.close() (para não acumular figuras em memória num loop).
    Lógica de plotagem idêntica à da Seção 11.
    """
    if pesos is None:
        pesos = PESOS_PERFIS['A_cobertura_prioritaria']
 
    area_w, area_h = cenario_params['area']
    n_drones       = cenario_params['n_drones']
    radius         = cenario_params['radius']
    comm_radius    = COMM_RADIUS_FACTOR * radius
    res            = GRID_RES[area_w]
    grid           = build_grid(area_w, area_h, res)
    bounds         = np.tile([[0, area_w], [0, area_h]], (n_drones, 1))
 
    max_ov = (n_drones * (n_drones - 1) / 2) * (2 * radius)
 
    def make_obj(nd=n_drones, r=radius, g=grid, cr=comm_radius, mo=max_ov, w=pesos):
        return lambda x: objective_fast(x, nd, r, g, cr, mo,
                                         w['cobertura'], w['overlap'], w['conectividade'])
    obj = make_obj()
 
    cores = {
        'Aleatório': '#808080', 'PSO': '#1f77b4', 'GWO': '#2ca02c',
        'WOA': '#d62728', 'SSA': '#ff7f0e', 'CS': '#9467bd',
    }
 
    plt.figure(figsize=(10, 5))
    np.random.seed(42)
 
    for algo_nome, algo_func in algoritmos.items():
        _, _, history = algo_func(obj, bounds, n_agents, n_iterations)
 
        if algo_nome == 'Aleatório':
            plt.axhline(y=history[0], color=cores['Aleatório'],
                        linewidth=1.5, linestyle='--',
                        label='Aleatório (baseline)')
        else:
            plt.plot(history, label=algo_nome,
                     color=cores.get(algo_nome), linewidth=2)
 
    plt.xlabel('Iteração', fontsize=12)
    plt.ylabel('Melhor custo encontrado', fontsize=12)
    plt.title(f'Convergência dos Algoritmos — {cenario_nome}{sufixo}', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
 
    os.makedirs('figuras_tcc/convergencia', exist_ok=True)
    caminho = f'figuras_tcc/convergencia/convergencia_{cenario_nome}{sufixo}.png'
    plt.savefig(caminho, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()   # libera a figura da memória — essencial ao gerar muitas em loop
    print(f"Salvo: {caminho}")
 
 
def plot_cobertura(cenario_nome, cenario_params, resultados, sufixo=''):
    """
    Versão com sufixo e plt.close(). Lógica de plotagem idêntica à
    da Seção 11 — `resultados` continua sendo o dicionário de UM
    perfil específico (ex. resultados_todos_perfis['A_cobertura_prioritaria']).
    """
    area_w, area_h = cenario_params['area']
    n_drones       = cenario_params['n_drones']
    radius         = cenario_params['radius']
    comm_radius    = COMM_RADIUS_FACTOR * radius
    res            = GRID_RES[area_w]
    grid           = build_grid(area_w, area_h, res)
    algos          = list(resultados[cenario_nome].keys())
 
    cores = {
        'Aleatório': '#808080', 'PSO': '#1f77b4', 'GWO': '#2ca02c',
        'WOA': '#d62728', 'SSA': '#ff7f0e', 'CS': '#9467bd',
    }
 
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
 
    for ax, algo in zip(axes, algos):
        best_pos    = resultados[cenario_nome][algo]['best_pos']
        positions   = best_pos.reshape((n_drones, 2))
        ratio, mask = compute_coverage_fast(positions, radius, grid)
        cor         = cores.get(algo, '#333333')
 
        pts_cobertos = grid[mask]
        ax.scatter(pts_cobertos[:, 0], pts_cobertos[:, 1],
                   c=cor, alpha=0.12, s=1)
 
        for pos in positions:
            circulo = patches.Circle(
                pos, radius, fill=False,
                edgecolor=cor, linewidth=1.5, alpha=0.85
            )
            ax.add_patch(circulo)
 
        for i in range(n_drones):
            for j in range(i + 1, n_drones):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist <= comm_radius:
                    ax.plot([positions[i, 0], positions[j, 0]],
                           [positions[i, 1], positions[j, 1]],
                           color=cor, linewidth=0.8, alpha=0.4, zorder=3)
 
        ax.scatter(positions[:, 0], positions[:, 1],
                   c=cor, s=60, zorder=5,
                   edgecolors='black', linewidth=0.8)
 
        ax.scatter(*START_POINT, c='black', s=100,
                   marker='*', zorder=6)
 
        borda = patches.Rectangle(
            (0, 0), area_w, area_h,
            fill=False, edgecolor='black', linewidth=1.5
        )
        ax.add_patch(borda)
 
        pares_i, pares_j = [], []
        for i in range(n_drones):
            for j in range(i + 1, n_drones):
                if np.linalg.norm(positions[i] - positions[j]) <= comm_radius:
                    pares_i.append(i)
                    pares_j.append(j)
        maior_grupo = (compute_largest_component(n_drones, pares_i, pares_j)
                      if n_drones > 1 else 1)
        frac = maior_grupo / n_drones
 
        ax.set_xlim(-5, area_w + 5)
        ax.set_ylim(-5, area_h + 5)
        ax.set_aspect('equal')
        ax.set_title(f'{algo}\nCobert: {ratio:.1%} | Conect: {frac:.1%}',
                     fontsize=10)
        ax.set_xlabel('x (m)', fontsize=10)
        ax.set_ylabel('y (m)', fontsize=10)
        ax.grid(True, alpha=0.2)
 
    plt.suptitle(f'Posicionamento Final dos Drones — {cenario_nome}{sufixo}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
 
    os.makedirs('figuras_tcc/cobertura', exist_ok=True)
    caminho = f'figuras_tcc/cobertura/cobertura_{cenario_nome}{sufixo}.png'
    plt.savefig(caminho, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)   # libera a figura da memória — essencial ao gerar muitas em loop
    print(f"Salvo: {caminho}")
 
 
# ── Loop: gera convergência para todos os 9 cenários ───────────
"""
Por padrão, gera 1 curva de convergência por cenário, ilustrando o
Perfil A (cobertura prioritária). Para gerar também sob os outros 2
perfis (27 imagens no total em vez de 9), mude
GERAR_CONVERGENCIA_TODOS_PERFIS para True — fica mais lento porque
cada convergência dispara uma execução real dos 6 algoritmos.
"""
GERAR_CONVERGENCIA_TODOS_PERFIS = False
 
print("Gerando curvas de convergência...")
if GERAR_CONVERGENCIA_TODOS_PERFIS:
    for nome_perfil, pesos in PESOS_PERFIS.items():
        for cenario in CENARIOS_OFICIAL:
            plot_convergencia(cenario['nome'], cenario, ALGORITMOS,
                              N_AGENTS, N_ITERATIONS,
                              pesos=pesos, sufixo=f'_{nome_perfil}')
else:
    for cenario in CENARIOS_OFICIAL:
        plot_convergencia(cenario['nome'], cenario, ALGORITMOS,
                          N_AGENTS, N_ITERATIONS)
 
# ── Loop: gera mapa de cobertura para todos os 9 cenários × 3 perfis ──
print("\nGerando mapas de posicionamento (9 cenários × 3 perfis = 27 imagens)...")
for nome_perfil, resultados in resultados_todos_perfis.items():
    for cenario in CENARIOS_OFICIAL:
        plot_cobertura(cenario['nome'], cenario, resultados,
                       sufixo=f'_{nome_perfil}')
 
print("\nTudo gerado. Arquivos em 'figuras_tcc/convergencia/' e 'figuras_tcc/cobertura/'.")

# %% [markdown]
# ## 11b. Figuras Selecionadas para o Capítulo 4
# 
# Gera apenas as 7 figuras recomendadas para o corpo do TCC: 3 convergências e 3 mapas de cobertura (um por densidade, escala 100x100, Perfil A) + 1 mapa comparativo (100x100_redundante sob Perfil C), ilustrando o efeito dos perfis de peso.
# 
# As 3 escalas de área (100x100/250x250/500x500) produzem resultados geometricamente equivalentes por densidade — a razão raio/área é praticamente constante nas 3 escalas — por isso só a escala 100x100 foi selecionada como representativa.

# %%
CENARIOS_DESTAQUE = ['100x100_escasso', '100x100_suficiente', '100x100_redundante']

# Localiza os dicionários de cenário correspondentes na matriz oficial
cenarios_destaque_dict = [c for c in CENARIOS_OFICIAL if c['nome'] in CENARIOS_DESTAQUE]

print("Gerando convergências (Perfil A, 3 densidades)...")
for cenario in cenarios_destaque_dict:
    plot_convergencia(cenario['nome'], cenario, ALGORITMOS, N_AGENTS, N_ITERATIONS,
                      pesos=PESOS_PERFIS['A_cobertura_prioritaria'],
                      sufixo='_A_cobertura_prioritaria')

print("\nGerando mapas de cobertura (Perfil A, 3 densidades)...")
for cenario in cenarios_destaque_dict:
    plot_cobertura(cenario['nome'], cenario,
                  resultados_todos_perfis['A_cobertura_prioritaria'],
                  sufixo='_A_cobertura_prioritaria')

print("\nGerando mapa comparativo (100x100_redundante, Perfil C)...")
cenario_redundante = next(c for c in CENARIOS_OFICIAL if c['nome'] == '100x100_redundante')
plot_cobertura(cenario_redundante['nome'], cenario_redundante,
              resultados_todos_perfis['C_rede_prioritaria'],
              sufixo='_C_rede_prioritaria')

print("\n7 figuras selecionadas geradas em 'figuras_tcc/'.")
