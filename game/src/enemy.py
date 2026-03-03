# Enemy类：继承Role，每帧将位置向玩家方向归一化移动（追逐AI），受击扣血。
import pygame

from .const import *
from .role import Role

class Enemy(Role):
    def __init__(self, dataname, id, position):
        super().__init__(dataname, id, position)

        self.data = self.obtaindata()
        self.health = self.data["health"]
        self.rect = self.calculate_rect(self.position, self.size)

    def chase(self, target_position, dt):
        dx = target_position[0] - self.position[0]
        dy = target_position[1] - self.position[1]
        length = (dx * dx + dy * dy) ** 0.5
        if length > 0:
            self.position[0] += (dx / length) * self.speed * dt
            self.position[1] += (dy / length) * self.speed * dt
            self.collision()

    def hurt(self, damage):
        self.health -= damage

    def update(self, target_position, dt):
        self.chase(target_position, dt)
