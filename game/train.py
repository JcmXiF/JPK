# Q-Learning训练脚本：双Q-table（移动/射击独立学习），epsilon-greedy探索，
# 滑动平均奖励创新高时保存最优模型，定期保存检查点。
# 用法: python train.py [watch | train_render]（默认为训练模式）

import pygame
import sys
import os
from collections import deque

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
os.chdir(_project_root)

from game import Game
from src.player_agent import PlayerAgent, MOVE_ACTIONS, SHOOT_ACTIONS
from src.const import game_fps, train_config

NUM_EPISODES  = train_config["num_episodes"]
MAX_STEPS     = train_config["max_steps"]
FIXED_DT      = train_config["fixed_dt"]
SAVE_INTERVAL = train_config["save_interval"]
PRINT_INTERVAL= train_config["print_interval"]
ALPHA         = train_config["alpha"]
GAMMA         = train_config["gamma"]
EPSILON_START = train_config["epsilon_start"]
EPSILON_MIN   = train_config["epsilon_min"]
EPSILON_DECAY = train_config["epsilon_decay"]

AGENT_PATH      = train_config["agent_path"]
AGENT_BEST_PATH = train_config["agent_best_path"]


def train(render=False):
    pygame.init()
    os.makedirs(os.path.dirname(AGENT_PATH), exist_ok=True)

    game = Game(headless=not render)
    agent = PlayerAgent(
        alpha=ALPHA, gamma=GAMMA,
        epsilon=EPSILON_START, epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
    )

    reward_records = []
    kills_records = []
    steps_records = []
    epsilon_records = []
    start_episode = 0

    if os.path.exists(AGENT_PATH):
        data = agent.load(AGENT_PATH)
        print(f"[INFO] Resumed from checkpoint: {AGENT_PATH}")
        print(f"       move={len(agent.q_move)} shoot={len(agent.q_shoot)} "
              f"eps={agent.epsilon:.4f}")

        reward_records = data.get('reward_records', [])
        kills_records = data.get('kills_records', [])
        steps_records = data.get('steps_records', [])
        epsilon_records = data.get('epsilon_records', [])
        start_episode = len(reward_records)

    best_avg_reward = float('-inf')

    reward_history = deque(maxlen=100)
    kills_history  = deque(maxlen=100)
    steps_history  = deque(maxlen=100)

    print(f"[INFO] Training {NUM_EPISODES - start_episode} episodes (Resumed from {start_episode}) "
          f"(max {MAX_STEPS} steps, dt={FIXED_DT}ms)")
    print("-" * 75)

    for episode in range(start_episode, NUM_EPISODES):
        state_dict = game.reset()
        move_state  = agent.discretize_move_state(state_dict)
        shoot_state = agent.discretize_shoot_state(state_dict)

        ep_reward = 0.0
        ep_kills  = 0
        steps     = 0

        for step in range(MAX_STEPS):
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        agent.save(AGENT_PATH)
                        pygame.quit()
                        sys.exit()
            elif step % 200 == 0:
                pygame.event.pump()

            move_idx  = agent.choose_move_action(move_state)
            shoot_idx = agent.choose_shoot_action(shoot_state)
            move_dir  = MOVE_ACTIONS[move_idx]
            shoot_dir = SHOOT_ACTIONS[shoot_idx]

            prev_state_dict = state_dict
            state_dict, done, info = game.step(move_dir, shoot_dir, dt=FIXED_DT)

            if render:
                game.render()

            move_reward  = agent.calculate_move_reward(
                state_dict, info['enemies_killed'], info['player_hit'])
            shoot_reward = agent.calculate_shoot_reward(
                prev_state_dict, state_dict,
                info['enemies_killed'], info['bullets_hit'],
                info['player_hit'], shoot_dir)

            next_move_state  = agent.discretize_move_state(state_dict)
            next_shoot_state = agent.discretize_shoot_state(state_dict)

            agent.learn_move(move_state,  move_idx,  move_reward,
                             next_move_state,  done)
            agent.learn_shoot(shoot_state, shoot_idx, shoot_reward,
                              next_shoot_state, done)

            move_state  = next_move_state
            shoot_state = next_shoot_state
            ep_reward  += move_reward + shoot_reward
            ep_kills   += info['enemies_killed']
            steps      += 1

            if done:
                break

        agent.decay_epsilon()

        reward_history.append(ep_reward)
        kills_history.append(ep_kills)
        steps_history.append(steps)
        
        reward_records.append(ep_reward)
        kills_records.append(ep_kills)
        steps_records.append(steps)
        epsilon_records.append(agent.epsilon)

        if len(reward_history) >= 20:
            avg_r = sum(reward_history) / len(reward_history)
            if avg_r > best_avg_reward:
                best_avg_reward = avg_r
                agent.save(
                    AGENT_BEST_PATH,
                    reward_records=reward_records,
                    kills_records=kills_records,
                    steps_records=steps_records,
                    epsilon_records=epsilon_records
                )

        if (episode + 1) % PRINT_INTERVAL == 0:
            avg_r = sum(reward_history) / len(reward_history)
            avg_k = sum(kills_history)  / len(kills_history)
            avg_s = sum(steps_history)  / len(steps_history)
            print(f"Ep {episode+1:>5} | "
                  f"Steps {avg_s:>6.0f} | "
                  f"Reward {avg_r:>8.1f} | "
                  f"Kills {avg_k:>5.1f} | "
                  f"HP {state_dict['player_health']:>3} | "
                  f"Eps {agent.epsilon:.4f} | "
                  f"Qm {len(agent.q_move):>6} Qs {len(agent.q_shoot):>4}")

        if (episode + 1) % SAVE_INTERVAL == 0:
            agent.save(
                AGENT_PATH,
                reward_records=reward_records,
                kills_records=kills_records,
                steps_records=steps_records,
                epsilon_records=epsilon_records
            )
            print(f"  [SAVE] checkpoint saved → {AGENT_PATH}")

    agent.save(
        AGENT_PATH,
        reward_records=reward_records,
        kills_records=kills_records,
        steps_records=steps_records,
        epsilon_records=epsilon_records
    )
    print("-" * 75)
    print(f"[DONE] Best avg reward: {best_avg_reward:.1f}")
    print(f"       Q-move: {len(agent.q_move)}, Q-shoot: {len(agent.q_shoot)}")
    pygame.quit()


def watch():
    pygame.init()

    game  = Game(headless=False)
    agent = PlayerAgent(epsilon=0.0)

    if os.path.exists(AGENT_BEST_PATH):
        path = AGENT_BEST_PATH
    elif os.path.exists(AGENT_PATH):
        path = AGENT_PATH
    else:
        path = None

    if path:
        agent.load(path)
        print(f"[INFO] Loaded model: {path}")
        print(f"       move={len(agent.q_move)} shoot={len(agent.q_shoot)}")
    else:
        print("[WARN] No trained model found! Acting randomly.")

    state_dict  = game.reset()
    total_kills = 0

    while True:
        game.clock.tick(game_fps)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        move_state  = agent.discretize_move_state(state_dict)
        shoot_state = agent.discretize_shoot_state(state_dict)
        move_idx    = agent.choose_move_action(move_state)
        shoot_idx   = agent.choose_shoot_action(shoot_state)
        move_dir    = MOVE_ACTIONS[move_idx]
        shoot_dir   = SHOOT_ACTIONS[shoot_idx]

        state_dict, done, info = game.step(move_dir, shoot_dir, dt=FIXED_DT)
        total_kills += info['enemies_killed']

        game.render()

        if done:
            game.display_game_over()
            print(f"[Game Over] Kills: {total_kills}")
            pygame.time.wait(3000)
            state_dict  = game.reset()
            total_kills = 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"

    if mode == "watch":
        watch()
    elif mode == "train_render":
        train(render=True)
    else:
        train(render=False)
