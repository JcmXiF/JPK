# Player类：继承Role，支持键盘控制（WASD移动/方向键射击）与Agent控制两种模式，
# 内置射击冷却和受击无敌期（无敌期间画面闪烁）。
import pygame

from game.src.const import boundary_rect
from game.src.role import Role
from game.src.bullet import Bullet


class Player(Role):
    def __init__(self, dataname, id, position):
        pos = dataname[id]["respawn_position"] if position is None else position
        super().__init__(dataname, id, pos)

        self.last_shot_time = 0
        self.shot_cooldown = 300
        self.health = self.data["health"]
        self.invincible = False
        self.invincible_start = 0
        self.invincible_duration = 1000

        self.rect = self.calculate_rect(self.position, self.size)

    def hurt(self, damage):
        if self.invincible:
            return
        self.health -= damage
        self.invincible = True
        self.invincible_start = pygame.time.get_ticks()

    def movebyPlayer(self, keys, dt):
        role_x_dir = 0
        role_y_dir = 0
        if keys[pygame.K_w]:
            role_y_dir -= 1
        if keys[pygame.K_s]:
            role_y_dir += 1
        if keys[pygame.K_a]:
            role_x_dir -= 1
        if keys[pygame.K_d]:
            role_x_dir += 1

        if role_x_dir or role_y_dir:
            self.move(role_x_dir, role_y_dir, dt)
            self.collision()

    def shootbyPlayer(self, keys):
        bullet_x_dir = 0
        bullet_y_dir = 0
        if keys[pygame.K_UP]:
            bullet_y_dir -= 1
        if keys[pygame.K_DOWN]:
            bullet_y_dir += 1
        if keys[pygame.K_LEFT]:
            bullet_x_dir -= 1
        if keys[pygame.K_RIGHT]:
            bullet_x_dir += 1

        if not (bullet_x_dir or bullet_y_dir):
            return None

        current_time = pygame.time.get_ticks()
        if current_time - self.last_shot_time < self.shot_cooldown:
            return None

        self.last_shot_time = current_time
        return Bullet(dataname=self.dataname,
                      id="1",
                      position=list(self.position),
                      direction=[bullet_x_dir, bullet_y_dir])

    def update(self, dt):
        if self.invincible:
            if pygame.time.get_ticks() - self.invincible_start >= self.invincible_duration:
                self.invincible = False

        keys = pygame.key.get_pressed()
        self.movebyPlayer(keys, dt)
        return self.shootbyPlayer(keys)

    def update_by_agent(self, move_dir, shoot_dir, dt, current_time):
        """Agent驱动的更新，绕过键盘输入。返回 Bullet 或 None。"""
        if self.invincible:
            if current_time - self.invincible_start >= self.invincible_duration:
                self.invincible = False

        mx, my = move_dir
        if mx or my:
            self.move(mx, my, dt)
            self.collision()

        sx, sy = shoot_dir
        if not (sx or sy):
            return None
        if current_time - self.last_shot_time < self.shot_cooldown:
            return None
        self.last_shot_time = current_time
        return Bullet(dataname=self.dataname,
                      id="1",
                      position=list(self.position),
                      direction=[sx, sy])

    def hurt_at_time(self, damage, current_time):
        """模拟模式下用外部时间戳触发受击与无敌。"""
        if self.invincible:
            return
        self.health -= damage
        self.invincible = True
        self.invincible_start = current_time

    def draw(self, sc):
        if self.invincible:
            elapsed = pygame.time.get_ticks() - self.invincible_start
            if (elapsed // 100) % 2:
                return
        super().draw(sc)
