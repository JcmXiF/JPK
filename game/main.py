# 程序入口：通过 flag 变量切换游戏模式（game手动游玩 / train训练Agent / watch观看Agent）。
import pygame
import sys
import os

# 设置工作目录为项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

from game import Game

# "game"  : 手动游玩（键盘操控）
# "train" : 训练 agent（headless，纯计算）
# "watch" : 观看训练好的 agent 游玩
FLAG = "watch"

pygame.init()

if __name__ == "__main__":
    if FLAG == "game":
        game = Game()
        while True:
            game.main_loop()

    elif FLAG == "train":
        from game.train import train
        train(render=False)

    elif FLAG == "watch":
        from game.train import watch
        watch()
