import sys
import torch
import warnings
import numpy as np
import gymnasium as gym
from Agent import Agent
from gymnasium.wrappers import FrameStack
from torch.utils.tensorboard import SummaryWriter



warnings.filterwarnings("ignore", category=UserWarning)

batch_size = 8
update_rate = 10
epsilon_min = 0.05
epsilon_decay = 0.995
gamma = 0.95
epsilon_init = 1.0
episode_rewards = list()


def main():
    training_mode = True if len(sys.argv) >= 2 and sys.argv[1] == 'train' else False
    episodes = int(sys.argv[2]) if training_mode else 5
    env = FrameStack(gym.make('ALE/Pong-v5', render_mode='rgb_array' if training_mode else 'human'), num_stack=4)
    writer = SummaryWriter()

    agent = Agent(env, epsilon_init, batch_size, gamma, training_mode)
    if training_mode:
        for episode in range(episodes):
            terminate = False
            while not terminate:
                agent.epsilon = agent.epsilon*epsilon_decay if agent.epsilon >= epsilon_min else epsilon_min
                terminate, reward = agent.train()
                if terminate:
                    writer.add_scalar('reward/episode', reward, episode)
                    episode_rewards.append(reward)
                    mean_reward = round(np.mean(episode_rewards[-100:]),3)
                    print(f"episode {episode}, mean reward: {mean_reward}\n")
            if episode % update_rate == 0:
                agent.update_weights()

        writer.flush()
        torch.save(agent.model.state_dict(), f'model/pong_batch{batch_size}_rate{update_rate}.pth')
    else:
        agent.demo()

    env.close()


if __name__ == '__main__':
    sys.exit(main())
