# 游戏全局常量与配置：屏幕/边界参数、角色数据、敌人入口、无尽模式配置、Q-Learning超参数与Agent行为参数。

game_fps = 60
screen_size = (800, 800)
boundary_rect = (50, 50, screen_size[0] - 100, screen_size[1] - 100)

# 速度单位：像素/毫秒，配合 delta time 使用
directionVector = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
    "upleft": (-1, -1),
    "upright": (1, -1),
    "downleft": (-1, 1),
    "downright": (1, 1),
}

objectdata = {
    "0": {
        "name": "player",
        "path": "assets/role/role-%d.png",
        "frames": 2,
        "size": (48, 48),
        "speed": 0.2,
        "health": 30,
        "respawn_position": [boundary_rect[0] + boundary_rect[2] // 2,
                             boundary_rect[1] + boundary_rect[3] // 2],
        "colorkey": None,
    },
    "1": {
        "name": "bullet",
        "path": "assets/bullet/bullet-%d.png",
        "frames": 1,
        "size": (10, 10),
        "speed": 0.5,
        "colorkey": None,
    },
    "2": {
        "name": "enemy-1",
        "path": "assets/enemy/enemy-%d.png",
        "frames": 1,
        "size": (48, 48),
        "speed": 0.1,
        "health": 30,
        "colorkey": None,
    }
}

enemy_entrance = {
    "left_entrance":   (boundary_rect[0], boundary_rect[1] + boundary_rect[3] // 2 - 75, 100, 150),
    "right_entrance":  (boundary_rect[0] + boundary_rect[2] - 100, boundary_rect[1] + boundary_rect[3] // 2 - 75, 100, 150),
    "top_entrance":    (boundary_rect[0] + boundary_rect[2] // 2 - 75, boundary_rect[1], 150, 100),
    "bottom_entrance": (boundary_rect[0] + boundary_rect[2] // 2 - 75, boundary_rect[1] + boundary_rect[3] - 100, 150, 100),
}

enemy_entrance_keys = list(enemy_entrance.keys())

endless_config = {
    "spawn_interval_start": 2000,   # 初始刷怪间隔 (ms)
    "spawn_interval_min":   1000,   # 最快刷怪间隔 (ms)
    "ramp_every":          10000,   # 每 10s 提升一次难度
    "ramp_step":             100,   # 每次减少的间隔 (ms)
}

train_config = {
    "num_episodes":    100000,      # 训练总回合数
    "max_steps":        15000,      # 每回合最大步数
    "fixed_dt":            16,      # 模拟帧间隔 (ms)

    "save_interval":      200,      # 每 N 回合保存检查点
    "print_interval":      20,      # 每 N 回合打印统计

    "alpha":             0.10,      # 学习率（固定，不衰减）
    "gamma":             0.95,      # 折扣因子
    "epsilon_start":      1.0,      # 初始探索率
    "epsilon_min":       0.05,      # 最低探索率
    "epsilon_decay":   0.9995,      # 每回合衰减（~6000回合达到最低，平衡探索与收敛）

    "agent_path":      "models/q_agent.pkl",       # 训练检查点
    "agent_best_path": "models/q_agent_best.pkl",  # 滑动平均最优模型
}

agent_config = {
    # 状态空间（8×8网格 → 状态总数~13,824，收敛快）
    "grid_size":        8,      # 地图网格划分数
    "dist_close":     120,      # 近距离阈值 (px)
    "dist_medium":    250,      # 中距离阈值 (px)
    "danger_radius":  150,      # 危险范围半径 (px)
    "danger_max":       2,      # 危险计数上限（0/1/2+，3级足够）
    "align_tolerance": 30,      # 弹道对齐垂直距离容差 (px)

    # 知情探索概率
    "explore_flee_prob":  0.3,  # 移动探索时选择逃离方向的概率
    "explore_aim_prob":   0.5,  # 射击探索时选择朝向敌人方向的概率

    # 移动奖励（精简：存活 + 对齐 + 击杀 - 近距离 - 受伤 - 死亡）
    "move_r_survive":    0.05,  # 每步存活奖励
    "move_r_aligned":     0.3,  # 与敌人弹道对齐奖励
    "move_r_kill":       15.0,  # 击杀奖励
    "move_r_prox_dist":  55.0,  # 近距离惩罚阈值 (px)
    "move_r_prox_max":    1.5,  # 近距离惩罚最大值
    "move_p_hit":        30.0,  # 被击中惩罚
    "move_p_death":      60.0,  # 死亡惩罚

    # 射击奖励
    "shoot_r_kill":      20.0,  # 击杀奖励
    "shoot_r_hit":        8.0,  # 子弹命中奖励
    "shoot_r_good_aim":   2.0,  # 朝向敌人射击奖励（cos > aim_cos_thresh）
    "shoot_p_miss_aim":   1.5,  # 未精确瞄准最近敌人惩罚（0 < cos < aim_cos_thresh）
    "shoot_p_bad_aim":    1.0,  # 背向敌人射击惩罚（cos < 0）
    "aim_cos_thresh":    0.85,  # 判定"朝向敌人"的cos阈值（cos(32°)≈0.85，仅偏差<32°算精确瞄准）
}
