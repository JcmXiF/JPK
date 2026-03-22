import pickle
import matplotlib.pyplot as plt
import os
import sys

# 设置工作目录为项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

from game.src.const import train_config

AGENT_PATH = train_config["agent_path"]


def moving_average(data, window_size=50):
    if len(data) < window_size:
        return []
    result = []
    current_sum = sum(data[:window_size])
    result.append(current_sum / window_size)
    for i in range(window_size, len(data)):
        current_sum += data[i] - data[i - window_size]
        result.append(current_sum / window_size)
    return result


def plot_results():
    # 构建绝对路径
    agent_path = AGENT_PATH
    if not os.path.isabs(agent_path):
        agent_path = os.path.join(PROJECT_ROOT, agent_path)

    if not os.path.exists(agent_path):
        print(f"File not found: {agent_path}")
        fallback = os.path.join(PROJECT_ROOT, 'models', 'q_agent.pkl')
        if os.path.exists(fallback):
            print(f"Trying fallback: {fallback}")
            agent_path = fallback
        else:
            return

    try:
        with open(agent_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    reward_records = data.get('reward_records', [])
    kills_records = data.get('kills_records', [])
    steps_records = data.get('steps_records', [])
    epsilon_records = data.get('epsilon_records', [])

    if not reward_records:
        print("No training records found in the model file.")
        return

    episodes = range(1, len(reward_records) + 1)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Training Results ({len(reward_records)} Episodes)')

    # Reward
    axs[0, 0].plot(episodes, reward_records, label='Reward', alpha=0.3, color='blue', linewidth=1)
    if len(reward_records) >= 50:
        ma_reward = moving_average(reward_records, 50)
        axs[0, 0].plot(range(50, len(reward_records) + 1), ma_reward, label='Avg (50)', color='blue', linewidth=2)
    axs[0, 0].set_title('Reward per Episode')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Reward')
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    # Kills
    axs[0, 1].plot(episodes, kills_records, label='Kills', alpha=0.3, color='red', linewidth=1)
    if len(kills_records) >= 50:
        ma_kills = moving_average(kills_records, 50)
        axs[0, 1].plot(range(50, len(kills_records) + 1), ma_kills, label='Avg (50)', color='red', linewidth=2)
    axs[0, 1].set_title('Kills per Episode')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Kills')
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    # Steps
    axs[1, 0].plot(episodes, steps_records, label='Steps', alpha=0.3, color='green', linewidth=1)
    if len(steps_records) >= 50:
        ma_steps = moving_average(steps_records, 50)
        axs[1, 0].plot(range(50, len(steps_records) + 1), ma_steps, label='Avg (50)', color='green', linewidth=2)
    axs[1, 0].set_title('Steps per Episode')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Steps')
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)

    # Epsilon
    axs[1, 1].plot(episodes, epsilon_records, label='Epsilon', color='orange', linewidth=2)
    axs[1, 1].set_title('Epsilon Decay')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('Epsilon')
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存结果到result文件夹
    result_dir = os.path.join(PROJECT_ROOT, 'result')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    save_path = os.path.join(result_dir, 'training_result.png')
    plt.savefig(save_path)
    print(f"Result plot saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    plot_results()
