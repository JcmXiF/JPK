# Role类：角色基类，继承Object，封装移动（带delta time）与边界碰撞限位逻辑。
import pygame

from .const import *
from .object import Object

class Role(Object):
    def __init__(self, dataname, id, position):
        super().__init__(dataname, id)

        self.position = position

        self._left_bound   = boundary_rect[0] + self.size[0] / 2
        self._right_bound  = boundary_rect[0] + boundary_rect[2] - self.size[0] / 2
        self._top_bound    = boundary_rect[1] + self.size[1] / 2
        self._bottom_bound = boundary_rect[1] + boundary_rect[3] - self.size[1] / 2

    def move(self, x_dir, y_dir, dt):
        self.position[0] += x_dir * self.speed * dt
        self.position[1] += y_dir * self.speed * dt

    def collision(self):
        if self.position[0] < self._left_bound:
            self.position[0] = self._left_bound
        elif self.position[0] > self._right_bound:
            self.position[0] = self._right_bound
        if self.position[1] < self._top_bound:
            self.position[1] = self._top_bound
        elif self.position[1] > self._bottom_bound:
            self.position[1] = self._bottom_bound

        self.rect.center = self.position
