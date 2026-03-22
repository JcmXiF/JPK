# PlayerAgent：双Q-table Q-Learning Agent，移动与射击独立学习。
# 移动状态 = (网格位置, 最近敌人方向/距离, 危险计数, 弹道对齐)；射击状态 = (敌人方向, 距离, 可射击)。
# 探索时偏向合理行为（远离敌人/射向敌人），奖励以击杀/命中等离散事件为主。

import math
import random
import pickle

from game.src.const import boundary_rect, agent_config


MOVE_ACTIONS = [
    (0, 0),     # 0: 静止
    (0, -1),    # 1: 上
    (0, 1),     # 2: 下
    (-1, 0),    # 3: 左
    (1, 0),     # 4: 右
    (-1, -1),   # 5: 左上
    (1, -1),    # 6: 右上
    (-1, 1),    # 7: 左下
    (1, 1),     # 8: 右下
]

SHOOT_ACTIONS = [
    (0, 0),     # 0: 不射击
    (0, -1),    # 1: 上
    (0, 1),     # 2: 下
    (-1, 0),    # 3: 左
    (1, 0),     # 4: 右
    (-1, -1),   # 5: 左上
    (1, -1),    # 6: 右上
    (-1, 1),    # 7: 左下
    (1, 1),     # 8: 右下
]

N_MOVE = len(MOVE_ACTIONS)
N_SHOOT = len(SHOOT_ACTIONS)

GRID_SIZE = agent_config["grid_size"]
PLAY_X = boundary_rect[0]
PLAY_Y = boundary_rect[1]
PLAY_W = boundary_rect[2]
PLAY_H = boundary_rect[3]
CELL_W = PLAY_W / GRID_SIZE
CELL_H = PLAY_H / GRID_SIZE

DIST_CLOSE    = agent_config["dist_close"]
DIST_MEDIUM   = agent_config["dist_medium"]
DANGER_RADIUS = agent_config["danger_radius"]
DANGER_MAX    = agent_config["danger_max"]

# 扇区1~8（1=右,2=右下,...,8=右上）→ 知情探索动作映射
_SECTOR_TO_SHOOT = {1: 4, 2: 8, 3: 2, 4: 7, 5: 3, 6: 5, 7: 1, 8: 6}
_SECTOR_TO_FLEE  = {1: 3, 2: 5, 3: 1, 4: 6, 5: 4, 6: 8, 7: 2, 8: 7}

# 8方向归一化向量，用于弹道对齐检测
_INV_SQRT2 = 0.7071067811865476
_SHOOT_DIRS_NORM = [
    (1, 0), (-1, 0), (0, 1), (0, -1),
    (_INV_SQRT2, _INV_SQRT2), (-_INV_SQRT2, -_INV_SQRT2),
    (_INV_SQRT2, -_INV_SQRT2), (-_INV_SQRT2, _INV_SQRT2),
]
ALIGN_TOLERANCE = agent_config["align_tolerance"]


class PlayerAgent:
    """双 Q-table Q-Learning Agent。

    Move Q-table:  state = (gx, gy, enemy_dir, enemy_dist, danger_count, aligned)
    Shoot Q-table: state = (enemy_dir, enemy_dist, can_shoot)
    """

    def __init__(self,
                 alpha=0.10,
                 gamma=0.95,
                 epsilon=1.0,
                 epsilon_min=0.05,
                 epsilon_decay=0.9995):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_move = {}
        self.q_shoot = {}

    @staticmethod
    def _distance(pos1, pos2):
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return math.sqrt(dx * dx + dy * dy)

    @staticmethod
    def _angle_to_sector(dx, dy):
        """方向向量 → 扇区 1-8，无方向返回 0。"""
        if dx == 0 and dy == 0:
            return 0
        angle = math.atan2(dy, dx)
        if angle < 0:
            angle += 2 * math.pi
        sector = int((angle + math.pi / 8) / (math.pi / 4)) % 8
        return sector + 1

    @staticmethod
    def _is_aligned(px, py, ex, ey):
        """检查敌人是否处于8方向弹道射击范围内（垂直距离 < ALIGN_TOLERANCE 且在前方）。"""
        dx = ex - px
        dy = ey - py
        for nx, ny in _SHOOT_DIRS_NORM:
            if dx * nx + dy * ny > 0:
                if abs(dx * ny - dy * nx) < ALIGN_TOLERANCE:
                    return True
        return False

    def _enemy_info(self, px, py, enemies):
        """计算最近敌人方向、距离等级、危险区内敌人数。"""
        if not enemies:
            return 0, 0, 0

        min_dist = float('inf')
        nearest = None
        danger_count = 0
        for e in enemies:
            d = self._distance([px, py], e)
            if d < min_dist:
                min_dist = d
                nearest = e
            if d < DANGER_RADIUS:
                danger_count += 1

        enemy_dir = self._angle_to_sector(nearest[0] - px, nearest[1] - py)

        if min_dist < DIST_CLOSE:
            enemy_dist = 1
        elif min_dist < DIST_MEDIUM:
            enemy_dist = 2
        else:
            enemy_dist = 3

        danger_count = min(danger_count, DANGER_MAX)
        return enemy_dir, enemy_dist, danger_count

    def discretize_move_state(self, state_dict):
        """移动状态离散化：(gx, gy, enemy_dir, enemy_dist, danger_count, aligned)"""
        px, py = state_dict['player_pos']
        gx = max(0, min(GRID_SIZE - 1, int((px - PLAY_X) / CELL_W)))
        gy = max(0, min(GRID_SIZE - 1, int((py - PLAY_Y) / CELL_H)))
        enemy_dir, enemy_dist, danger_count = self._enemy_info(
            px, py, state_dict['enemies'])

        aligned = 0
        if state_dict['enemies']:
            nearest = min(state_dict['enemies'],
                          key=lambda e: self._distance([px, py], e))
            if self._is_aligned(px, py, nearest[0], nearest[1]):
                aligned = 1

        return (gx, gy, enemy_dir, enemy_dist, danger_count, aligned)

    def discretize_shoot_state(self, state_dict):
        """射击状态离散化：(enemy_dir, enemy_dist, can_shoot)（位置无关，极速收敛）"""
        px, py = state_dict['player_pos']
        enemies = state_dict['enemies']
        can_shoot = 1 if state_dict.get('can_shoot', True) else 0

        if enemies:
            min_dist = float('inf')
            nearest = None
            for e in enemies:
                d = self._distance([px, py], e)
                if d < min_dist:
                    min_dist = d
                    nearest = e
            enemy_dir = self._angle_to_sector(nearest[0] - px, nearest[1] - py)
            if min_dist < DIST_CLOSE:
                enemy_dist = 1
            elif min_dist < DIST_MEDIUM:
                enemy_dist = 2
            else:
                enemy_dist = 3
        else:
            enemy_dir = 0
            enemy_dist = 0

        return (enemy_dir, enemy_dist, can_shoot)

    def _get_q_move(self, state, action):
        return self.q_move.get((state, action), 0.0)

    def _get_q_shoot(self, state, action):
        return self.q_shoot.get((state, action), 0.0)

    def choose_move_action(self, move_state):
        """Epsilon-greedy，探索时30%概率选择远离最近敌人的方向。"""
        if random.random() < self.epsilon:
            enemy_dir = move_state[2]
            if enemy_dir > 0 and random.random() < agent_config["explore_flee_prob"]:
                return _SECTOR_TO_FLEE[enemy_dir]
            return random.randint(0, N_MOVE - 1)

        best_a, best_q = 0, self._get_q_move(move_state, 0)
        for a in range(1, N_MOVE):
            q = self._get_q_move(move_state, a)
            if q > best_q:
                best_q = q
                best_a = a
        return best_a

    def choose_shoot_action(self, shoot_state):
        """Epsilon-greedy，探索时50%概率选择朝向最近敌人的射击方向。"""
        if random.random() < self.epsilon:
            enemy_dir = shoot_state[0]
            if enemy_dir > 0 and random.random() < agent_config["explore_aim_prob"]:
                return _SECTOR_TO_SHOOT[enemy_dir]
            return random.randint(0, N_SHOOT - 1)

        best_a, best_q = 0, self._get_q_shoot(shoot_state, 0)
        for a in range(1, N_SHOOT):
            q = self._get_q_shoot(shoot_state, a)
            if q > best_q:
                best_q = q
                best_a = a
        return best_a

    def calculate_move_reward(self, curr_state, enemies_killed, player_hit):
        """移动奖励：存活 + 弹道对齐 + 击杀 - 近距离危险 - 被击 - 死亡。"""
        reward = agent_config["move_r_survive"]

        px, py = curr_state['player_pos']
        enemies = curr_state['enemies']

        if enemies:
            nearest = min(enemies, key=lambda e: self._distance([px, py], e))
            min_dist = self._distance([px, py], nearest)

            if self._is_aligned(px, py, nearest[0], nearest[1]):
                reward += agent_config["move_r_aligned"]

            prox_dist = agent_config["move_r_prox_dist"]
            if min_dist < prox_dist:
                reward -= (prox_dist - min_dist) / prox_dist * agent_config["move_r_prox_max"]

        reward += enemies_killed * agent_config["move_r_kill"]

        if player_hit:
            reward -= agent_config["move_p_hit"]

        if curr_state['player_health'] <= 0:
            reward -= agent_config["move_p_death"]

        return reward

    def calculate_shoot_reward(self, prev_state, curr_state,
                               enemies_killed, bullets_hit,
                               player_hit, shoot_action):
        """射击奖励：击杀/命中 + 瞄准方向即时反馈。"""
        reward = 0.0
        sx, sy = shoot_action

        reward += enemies_killed * agent_config["shoot_r_kill"]
        reward += bullets_hit * agent_config["shoot_r_hit"]

        if sx or sy:
            enemies_prev = prev_state['enemies']
            if enemies_prev:
                px, py = prev_state['player_pos']
                nearest = min(enemies_prev,
                              key=lambda e: self._distance([px, py], e))
                dx = nearest[0] - px
                dy = nearest[1] - py
                dist = math.sqrt(dx * dx + dy * dy)
                shoot_len = math.sqrt(sx * sx + sy * sy)
                if dist > 1e-6 and shoot_len > 1e-6:
                    cos_angle = (sx * dx + sy * dy) / (shoot_len * dist)
                    if cos_angle > agent_config["aim_cos_thresh"]:
                        reward += agent_config["shoot_r_good_aim"]
                    elif cos_angle > 0:
                        reward -= agent_config["shoot_p_miss_aim"]
                    else:
                        reward -= agent_config["shoot_p_bad_aim"]

        return reward

    def learn_move(self, state, action, reward, next_state, done):
        cur = self._get_q_move(state, action)
        if done:
            target = reward
        else:
            target = reward + self.gamma * max(
                self._get_q_move(next_state, a) for a in range(N_MOVE))
        self.q_move[(state, action)] = cur + self.alpha * (target - cur)

    def learn_shoot(self, state, action, reward, next_state, done):
        cur = self._get_q_shoot(state, action)
        if done:
            target = reward
        else:
            target = reward + self.gamma * max(
                self._get_q_shoot(next_state, a) for a in range(N_SHOOT))
        self.q_shoot[(state, action)] = cur + self.alpha * (target - cur)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path='q_agent.pkl', **kwargs):
        data = {
            'q_move': self.q_move,
            'q_shoot': self.q_shoot,
            'epsilon': self.epsilon,
        }
        data.update(kwargs)
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path='q_agent.pkl'):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.q_move = data['q_move']
            self.q_shoot = data['q_shoot']
            self.epsilon = data['epsilon']
            return data
