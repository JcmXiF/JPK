# 程序入口：通过 flag 变量切换游戏模式（game手动游玩 / train训练Agent / watch观看Agent）。
import pygame
import sys
import os

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
os.chdir(_project_root)

from game import Game

# "game"  : 手动游玩（键盘操控）
# "train" : 训练 agent（headless，纯计算）
# "watch" : 观看训练好的 agent 游玩
flag = "watch"

pygame.init()

if __name__ == "__main__":
    if flag == "game":
        game = Game()
        while True:
            game.main_loop()

    elif flag == "train":
        from train import train
        train(render=False)

    elif flag == "watch":
        from train import watch
        watch()
