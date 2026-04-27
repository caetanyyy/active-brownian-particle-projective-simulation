import numpy as np
import pickle
import os
import time

class MazeEnvironment(object):
    """
    Ambiente de simulação onde uma partícula Browniana ativa (ABP) busca um alvo
    dentro de um labirinto definido por um arquivo.
    
    O espaço de movimento é contínuo, mas as colisões são ditadas por uma
    grade discreta (o labirinto).
    """

    def __init__(
        self,
        maze_file: str,             # MUDANÇA: Arquivo do labirinto é obrigatório
        L: float = 100,
        Pe: float = 100,
        l: float = 1,
        tao: float = 1e+4,
        dt: float = 1,
        target_reward: float = 1,
        # Removido 'allow_colision' pois com labirinto, a colisão é implícita
    ):
        """
        Inicializa o ambiente do labirinto.
        """
        # --- Configuração Inicial ---
        self.rng = np.random.RandomState(None)
        self.L = L
        self.dt = dt
        self.max_steps_per_trial = int(tao)
        
        # --- Carregamento do Labirinto (Lógica Central Nova) ---
        if not os.path.exists(maze_file):
            raise FileNotFoundError(f"Arquivo do labirinto não encontrado em: {maze_file}")
            
        self.maze_grid = self._load_maze_from_file(maze_file)
        self.grid_size_y, self.grid_size_x = self.maze_grid.shape
        self.tile_size_x = self.L / self.grid_size_x
        self.tile_size_y = self.L / self.grid_size_y

        # --- Parâmetros de Percepção (para o agente) ---
        self.num_states = 2  # 1=ativo, 0=passivo
        self.colision_state = 2 # 1=colidiu, 0=não
        self.num_actions = 2  # 1=trocar, 0=manter
        
        # A percepção sempre incluirá colisão agora
        self.num_percepts_list = [self.num_states, self.max_steps_per_trial, self.colision_state]
        
        # --- Observáveis do Agente ---
        self.state = 0
        self.prev_state = 0
        self.timer = 0
        self.colision = 0
        self.prev_colision = 0

        # --- Recompensa e Estado do Episódio ---
        self.reward = 0
        self.target_reward = target_reward
        self.trial_finished = False

        # --- Pré-cálculo de Posições Válidas ---
        walkable_indices = np.argwhere(self.maze_grid == 0)
        self.walkable_indices = walkable_indices # Guarda os índices para o spawn
        if len(self.walkable_indices) < 2:
            raise ValueError("O labirinto precisa de pelo menos 2 células de caminho (para agente e alvo).")

        # --- Estado Inicial do Agente e Alvo ---
        self.r = self._get_random_spawn_point()
        
        self.target_radius = 0.05 * L
        self.target_position = self._get_random_spawn_point()
        self.distance = np.linalg.norm(self.r - self.target_position)

        # --- Parâmetros Físicos ---
        self.v = Pe * L / tao
        self.D = (L * L) / (4 * tao)
        self.D_theta = self.v / (l * L)

        # --- Vetores de Movimento ---
        self.theta_t = 2 * np.pi * self.rng.rand()
        self.u_t = np.array([np.cos(self.theta_t), np.sin(self.theta_t)])
        self.dr_active = np.zeros(2)

    def _load_maze_from_file(self, filepath):
        """Lê um arquivo de texto e o converte em uma matriz binária do labirinto."""
        char_map = {'1': 1, '0': 0}
        with open(filepath, 'r') as f:
            lines = [line.replace(',','').rstrip('\n') for line in f]
        
        binary_maze = [[char_map.get(char, 0) for char in line] for line in lines]
        return np.array(binary_maze, dtype=int)

    def _get_random_spawn_point(self):
        """Sorteia uma posição contínua aleatória dentro de uma célula de caminho válida."""
        random_cell_index = self.rng.randint(0, len(self.walkable_indices))
        chosen_tile_row, chosen_tile_col = self.walkable_indices[random_cell_index]

        x_min = chosen_tile_col * self.tile_size_x
        y_min = chosen_tile_row * self.tile_size_y
        
        random_x = x_min + self.rng.rand() * self.tile_size_x
        random_y = y_min + self.rng.rand() * self.tile_size_y
        
        return np.array([random_x, random_y])

    def _get_tile_coords(self, position):
        """Converte uma posição contínua (x, y) em coordenadas de tile (col, row)."""
        col = int(position[0] / self.tile_size_x)
        row = int(position[1] / self.tile_size_y)
        col = np.clip(col, 0, self.grid_size_x - 1)
        row = np.clip(row, 0, self.grid_size_y - 1)
        return col, row
        
    def reset_target(self):
        """Reseta a posição do alvo para um novo local válido."""
        self.target_position = self._get_random_spawn_point()
        # Garante que o novo alvo não apareça no mesmo lugar que o agente
        while np.linalg.norm(self.r - self.target_position) < self.target_radius:
            self.target_position = self._get_random_spawn_point()

    def reset_agent_state(self, new_state):
        """Reseta o estado do agente (ativo/passivo)."""
        self.r = self._get_random_spawn_point()
        self.timer = 0
        self.state = new_state
        if new_state == 1:
            self.reset_agent_ABP()

    # ESTA É A NOVA VERSÃO DA SUA FUNÇÃO _handle_maze_collision
    def _handle_maze_collision(self, current_pos, displacement):
        """
        Usa a lógica avançada do usuário para determinar a colisão e, em seguida,
        calcula a POSIÇÃO FINAL correta, incluindo reposicionamento e deslize.
        """
        next_pos = current_pos + displacement
        did_collide = False

        current_tile_col, current_tile_row = self._get_tile_coords(current_pos)
        next_tile_col, next_tile_row = self._get_tile_coords(next_pos)

        d_col = next_tile_col - current_tile_col
        d_row = next_tile_row - current_tile_row

        # ==============================================================================
        #      MUDANÇA 1: Introduzir a função auxiliar que faz o trabalho pesado
        # ==============================================================================
        def _calculate_final_pos(start_pos, disp, normal):
            """Calcula a posição final absoluta após uma colisão."""
            normal = normal.astype(float)
            norm_val = np.linalg.norm(normal)
            if norm_val > 0: normal /= norm_val
            else: return start_pos # Segurança

            # Componente de deslize
            dot = np.dot(disp, normal)
            slide_disp = disp - dot * normal

            # Posição final inicializada com o deslize
            final_pos = start_pos + slide_disp

            # Reposicionamento da componente normal
            if normal[0] != 0:
                if normal[0] < 0:
                    wall_edge = (int(start_pos[0] / self.tile_size_x) + 1) * self.tile_size_x
                    final_pos[0] = wall_edge - 0.001
                else:
                    wall_edge = int(start_pos[0] / self.tile_size_x) * self.tile_size_x
                    final_pos[0] = wall_edge + 0.001
            if normal[1] != 0:
                if normal[1] < 0:
                    wall_edge = (int(start_pos[1] / self.tile_size_y) + 1) * self.tile_size_y
                    final_pos[1] = wall_edge - 0.001
                else:
                    wall_edge = int(start_pos[1] / self.tile_size_y) * self.tile_size_y
                    final_pos[1] = wall_edge + 0.001

            return final_pos

        # --- CASO 1: SEM MUDANÇA DE TILE ---
        if d_col == 0 and d_row == 0:
            if self.maze_grid[current_tile_row, current_tile_col] == 1:
                return current_pos - displacement * 0.1, True # Recuo simples
            else:
                # MUDANÇA: Retorna a posição final, não o deslocamento
                return next_pos, False

        # --- CASO 2: MOVIMENTO PARA UM TILE VIZINHO ---
        is_diagonal_move = (d_col != 0 and d_row != 0)
        if not is_diagonal_move:
            if self.maze_grid[next_tile_row, next_tile_col] == 1:
                normal_vector = np.array([-d_col, -d_row], dtype=float)
                # MUDANÇA: Chama a função auxiliar para obter a posição final
                final_pos = _calculate_final_pos(current_pos, displacement, normal_vector)
                return final_pos, True
            else:
                return next_pos, False # Caminho livre

        # --- CASO 3: MOVIMENTO DIAGONAL ---
        if is_diagonal_move:
            # A sua 'get_slide' não é mais necessária, pois a nova função faz tudo.
            tile_diagonal_is_wall = (self.maze_grid[next_tile_row, next_tile_col] == 1)
            tile_horizontal_is_wall = (self.maze_grid[current_tile_row, next_tile_col] == 1)
            tile_vertical_is_wall = (self.maze_grid[next_tile_row, current_tile_col] == 1)

            # 1) TILE DIAGONAL DISPONÍVEL (corte de canto)
            if not tile_diagonal_is_wall:
                pos = np.copy(current_pos)
                did_collide = False
                if tile_horizontal_is_wall:
                    pos = _calculate_final_pos(pos, np.array([displacement[0], 0.0]), np.array([-d_col, 0.0]))
                    did_collide = True
                else:
                    pos[0] += displacement[0]
                if tile_vertical_is_wall:
                    pos = _calculate_final_pos(pos, np.array([0.0, displacement[1]]), np.array([0.0, -d_row]))
                    did_collide = True
                else:
                    pos[1] += displacement[1]
                return pos, did_collide

            # 2) TILE DIAGONAL INDISPONÍVEL
            else:
                did_collide = True
                # 2.1) Apenas um vizinho disponível
                if tile_horizontal_is_wall and not tile_vertical_is_wall:
                    return _calculate_final_pos(current_pos, displacement, np.array([-d_col, 0.0])), True
                if not tile_horizontal_is_wall and tile_vertical_is_wall:
                    return _calculate_final_pos(current_pos, displacement, np.array([0.0, -d_row])), True
                # 2.2) Nenhum vizinho disponível (canto sólido)
                if tile_horizontal_is_wall and tile_vertical_is_wall:
                    return _calculate_final_pos(current_pos, displacement, np.array([-d_col, -d_row])), True
                # 2.3) Ambos os vizinhos disponíveis
                if not tile_horizontal_is_wall and not tile_vertical_is_wall:
                    pos_x_first = _calculate_final_pos(current_pos, displacement, np.array([0.0, -d_row]))
                    pos_y_first = _calculate_final_pos(current_pos, displacement, np.array([-d_col, 0.0]))
                    if np.linalg.norm(pos_x_first - current_pos) > np.linalg.norm(pos_y_first - current_pos):
                        return pos_x_first, True
                    else:
                        return pos_y_first, True

        # Se nenhuma das condições acima foi atendida, retorna a posição final não corrigida.
        return next_pos, False


    def update_agent_position(self):
        """
        Calcula o deslocamento do agente e atualiza sua posição final
        usando a lógica de colisão do labirinto.
        """
        # 1. Calcula o deslocamento proposto
        E_t = self.rng.randn(2)
        dr_brownian = np.sqrt(2 * self.D * self.dt) * E_t
        displacement = self.dr_active + dr_brownian

        # 2. A função de colisão agora retorna a POSIÇÃO FINAL absoluta
        final_position, did_collide = self._handle_maze_collision(self.r, displacement)

        self.colision = 1 if did_collide else 0

        # 3. ATRIBUI a posição final, não soma mais um deslocamento
        self.r = final_position

        # 4. Calcula a distância
        self.distance = np.linalg.norm(self.r - self.target_position)

    def update_environment(self, action):
        """Atualiza o ambiente conforme a ação do agente."""
        self.timer += 1
        self.dr_active = np.zeros(2)
        self.prev_colision = self.colision
        self.prev_state = self.state

        if action:
            self.state = 1 - self.state
            self.timer = 0

        if self.state == 1:
            if self.prev_state == 0: self.reset_agent_ABP()
            else: self.update_agent_ABP()

        self.update_agent_position()
        self.update_reward()
        
        if self.prev_colision != self.colision:
            self.timer = 0 
        
        if self.timer >= self.max_steps_per_trial - 1:
            self.timer = 0
            self.state = 1 - self.state # Troca forçada de estado
            if self.state == 1: self.reset_agent_ABP()

        return self.reward, self.trial_finished

    def update_agent_ABP(self):
        """Atualiza os parâmetros do estado ABP."""
        n_t = self.rng.normal()
        self.theta_t = self.theta_t + np.sqrt(2 * self.D_theta * self.dt) * n_t
        self.u_t = np.array([np.cos(self.theta_t), np.sin(self.theta_t)])
        self.dr_active = self.v * self.u_t * self.state * self.dt

    def state_observation(self):
        """Retorna o estado observável do sistema para o agente."""
        return [self.state, self.timer, self.colision]

    def update_reward(self):
        """Atualiza a recompensa do agente."""
        self.trial_finished = False
        self.reward = 0
        if (self.distance < self.target_radius) and (self.state == 0):
            self.reward = self.target_reward
            self.trial_finished = True
            
    def reset_agent_ABP(self):
        """Reseta o agente para o estado ABP (ativo)."""
        self.theta_t = 2 * np.pi * self.rng.rand()
        self.u_t = np.array([np.cos(self.theta_t), np.sin(self.theta_t)])
        self.dr_active = self.v * self.u_t * self.state * self.dt