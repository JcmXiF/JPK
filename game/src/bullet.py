# Bullet类：继承Object，沿方向向量匀速运动，每帧更新位置并检测是否越界。
import pygame

from .const import *
from .object import Object

class Bullet(Object):
    def __init__(self, dataname, id, position: list, direction):
        super().__init__(dataname, id)

        self.position = position
        self.direction = direction

        self.rect = self.calculate_rect(self.position, self.size)

    def spawn(self, sc):
        self.draw(sc)

    def self_move(self, dt):
        self.position[0] += self.direction[0] * self.speed * dt
        self.position[1] += self.direction[1] * self.speed * dt
        self.rect.center = self.position

    def collision(self):
        return (self.position[0] < boundary_rect[0] or
                self.position[0] > boundary_rect[0] + boundary_rect[2] or
                self.position[1] < boundary_rect[1] or
                self.position[1] > boundary_rect[1] + boundary_rect[3])

    def update(self, dt):
        self.self_move(dt)
        return self.collision()
