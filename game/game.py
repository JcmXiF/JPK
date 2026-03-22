# Game类：管理完整游戏状态。提供两套接口：
# main_loop()用于键盘手动游玩；reset()/step()/render()用于RL Agent训练与推理。
# 无尽模式：刷怪间隔随时间线性缩短，难度持续上升。
import pygame
import sys
import random

from game.src.const import (
    screen_size, boundary_rect, endless_config,
    objectdata, enemy_entrance, enemy_entrance_keys, game_fps,
)
from game.src.player import Player
from game.src.enemy import Enemy


class Game:
    def __init__(self, headless=False):
        self.headless = headless
        if headless:
            self.sc = pygame.display.set_mode((1, 1))
        else:
            self.sc = pygame.display.set_mode(screen_size)

        self.clock = pygame.time.Clock()

        self.player = Player(dataname=objectdata, id="0", position=None)

        self.bullets = []
        self.enemies = []

        self.last_spawn_time = 0
        self._game_start_time = 0

        self.kill_count = 0

        self._boundary_rect = pygame.Rect(boundary_rect)

        if not headless:
            self._font_small = pygame.font.SysFont(None, 24)
            self._font_large = pygame.font.SysFont(None, 96)

        self.game_over = False
        self._sim_time = 0

    def _current_spawn_interval(self, elapsed_ms):
        """根据已过毫秒数计算当前刷怪间隔（难度随时间增加）。"""
        steps = elapsed_ms // endless_config["ramp_every"]
        interval = endless_config["spawn_interval_start"] - steps * endless_config["ramp_step"]
        return max(interval, endless_config["spawn_interval_min"])

    def _current_level(self, elapsed_ms):
        """返回当前难度等级（1-based，每ramp_every毫秒+1）。"""
        raw_level = int(elapsed_ms // endless_config["ramp_every"]) + 1
        max_level = 1 + (endless_config["spawn_interval_start"] - endless_config["spawn_interval_min"]) // endless_config["ramp_step"]
        return min(raw_level, max_level)

    def spawn_enemy(self, current_time):
        """无尽模式：按当前间隔随机在四侧入口生成敌人。"""
        elapsed = current_time - self._game_start_time
        if current_time - self.last_spawn_time > self._current_spawn_interval(elapsed):
            entrance_rect = enemy_entrance[random.choice(enemy_entrance_keys)]
            spawn_x = random.randint(entrance_rect[0], entrance_rect[0] + entrance_rect[2])
            spawn_y = random.randint(entrance_rect[1], entrance_rect[1] + entrance_rect[3])
            self.enemies.append(Enemy(dataname=objectdata, id="2", position=[spawn_x, spawn_y]))
            self.last_spawn_time = current_time

    def hit_detection(self):
        """检测子弹-敌人碰撞与玩家-敌人碰撞。"""
        hit_bullets = set()
        dead_enemies = set()

        for i, bullet in enumerate(self.bullets):
            for j, enemy in enumerate(self.enemies):
                if bullet.rect.colliderect(enemy.rect):
                    hit_bullets.add(i)
                    enemy.hurt(10)
                    if enemy.health <= 0:
                        dead_enemies.add(j)
                        self.kill_count += 1

        if not self.player.invincible:
            for j, enemy in enumerate(self.enemies):
                if j not in dead_enemies and enemy.rect.colliderect(self.player.rect):
                    self.player.hurt(10)
                    dead_enemies.add(j)
                    break

        if hit_bullets:
            self.bullets = [b for i, b in enumerate(self.bullets) if i not in hit_bullets]
        if dead_enemies:
            self.enemies = [e for j, e in enumerate(self.enemies) if j not in dead_enemies]

    def display_game_over(self):
        """绘制 Game Over 画面（显示最终击杀数与存活时间）。"""
        self.sc.fill((255, 255, 255))
        pygame.draw.rect(self.sc, (0, 0, 0), self._boundary_rect, 2)

        text = self._font_large.render("GAME OVER", True, (0, 0, 0))
        rect = text.get_rect(center=(screen_size[0] // 2, screen_size[1] // 2 - 40))
        self.sc.blit(text, rect)

        elapsed_s = self._sim_time // 1000
        stats = self._font_small.render(
            f"Kills: {self.kill_count}   Time: {elapsed_s}s", True, (0, 0, 0))
        stats_rect = stats.get_rect(center=(screen_size[0] // 2, screen_size[1] // 2 + 40))
        self.sc.blit(stats, stats_rect)

        pygame.display.update()

    def main_loop(self):
        """游戏主循环：处理输入 → 更新状态 → 绘制画面。"""
        dt = self.clock.tick(game_fps)
        self._sim_time += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if self.game_over:
            self.display_game_over()
            return

        self.spawn_enemy(self._sim_time)

        new_bullet = self.player.update(dt)
        if new_bullet:
            self.bullets.append(new_bullet)

        self.bullets = [b for b in self.bullets if not b.update(dt)]

        player_pos = self.player.position
        for enemy in self.enemies:
            enemy.update(player_pos, dt)

        self.hit_detection()

        if self.player.health <= 0:
            self.game_over = True
            return

        self.sc.fill((255, 255, 255))
        pygame.draw.rect(self.sc, (0, 0, 0), self._boundary_rect, 2)

        elapsed_s = self._sim_time // 1000
        level = self._current_level(self._sim_time)
        hud = self._font_small.render(
            f'HP: {self.player.health}   Kills: {self.kill_count}   '
            f'Time: {elapsed_s}s   Lv.{level}', True, (0, 0, 0))
        self.sc.blit(hud, (10, 10))

        self.player.draw(self.sc)
        for bullet in self.bullets:
            bullet.spawn(self.sc)
        for enemy in self.enemies:
            enemy.draw(self.sc)

        pygame.display.update()

    def reset(self):
        """重置游戏状态，开始新一局。返回初始 state_dict。"""
        self.player = Player(dataname=objectdata, id="0", position=None)
        self.bullets = []
        self.enemies = []
        self._sim_time = 0
        self.last_spawn_time = 0
        self._game_start_time = 0
        self.kill_count = 0
        self.game_over = False
        return self.get_state()

    def get_state(self):
        """提取当前游戏状态，供 agent 观测。"""
        return {
            'player_pos': list(self.player.position),
            'player_health': self.player.health,
            'invincible': self.player.invincible,
            'enemies': [list(e.position) for e in self.enemies],
            'bullets': [list(b.position) for b in self.bullets],
            'can_shoot': (self._sim_time - self.player.last_shot_time
                          >= self.player.shot_cooldown),
        }

    def step(self, move_action, shoot_action, dt=16):
        """执行一帧游戏逻辑（agent 模式）。返回 (state_dict, done, info)。"""
        self._sim_time += dt

        self.spawn_enemy(self._sim_time)

        new_bullet = self.player.update_by_agent(
            move_action, shoot_action, dt, self._sim_time)
        if new_bullet:
            self.bullets.append(new_bullet)

        self.bullets = [b for b in self.bullets if not b.update(dt)]

        player_pos = self.player.position
        for enemy in self.enemies:
            enemy.update(player_pos, dt)

        enemies_killed, player_hit, bullets_hit = self._hit_detection_sim(self._sim_time)

        done = self.player.health <= 0
        if done:
            self.game_over = True

        state_dict = self.get_state()
        info = {
            'enemies_killed': enemies_killed,
            'player_hit': player_hit,
            'bullets_hit': bullets_hit,
            'sim_time': self._sim_time,
        }
        return state_dict, done, info

    def _hit_detection_sim(self, current_time):
        """碰撞检测（模拟模式），玩家受击使用外部时间戳。返回 (enemies_killed, player_hit, bullets_hit)。"""
        hit_bullets = set()
        dead_enemies = set()
        enemies_killed = 0

        for i, bullet in enumerate(self.bullets):
            for j, enemy in enumerate(self.enemies):
                if bullet.rect.colliderect(enemy.rect):
                    hit_bullets.add(i)
                    enemy.hurt(10)
                    if enemy.health <= 0:
                        dead_enemies.add(j)
                        enemies_killed += 1
                        self.kill_count += 1

        player_hit = False
        if not self.player.invincible:
            for j, enemy in enumerate(self.enemies):
                if j not in dead_enemies and enemy.rect.colliderect(self.player.rect):
                    self.player.hurt_at_time(10, current_time)
                    dead_enemies.add(j)
                    player_hit = True
                    break

        if hit_bullets:
            self.bullets = [b for i, b in enumerate(self.bullets) if i not in hit_bullets]
        if dead_enemies:
            self.enemies = [e for j, e in enumerate(self.enemies) if j not in dead_enemies]

        return enemies_killed, player_hit, len(hit_bullets)

    def render(self):
        """渲染当前帧画面（仅非 headless 模式有效）。"""
        if self.headless:
            return

        self.sc.fill((255, 255, 255))
        pygame.draw.rect(self.sc, (0, 0, 0), self._boundary_rect, 2)

        elapsed_s = self._sim_time // 1000
        level = self._current_level(self._sim_time)
        hud = self._font_small.render(
            f'HP: {self.player.health}   Kills: {self.kill_count}   '
            f'Time: {elapsed_s}s   Lv.{level}', True, (0, 0, 0))
        self.sc.blit(hud, (10, 10))

        self.player.draw(self.sc)
        for bullet in self.bullets:
            bullet.spawn(self.sc)
        for enemy in self.enemies:
            enemy.draw(self.sc)

        pygame.display.update()
