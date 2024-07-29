import os
import sys
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
ave_rewards = list()


def main():
    training_mode = True if len(sys.argv) >= 2 and sys.argv[1] == 'train' else False
    episodes = int(sys.argv[2]) if training_mode else 5
    env = FrameStack(gym.make('ALE/Pong-v5', render_mode='rgb_array' if training_mode else 'human'), num_stack=4)
    if not training_mode:
        env.metadata['render_fps'] = 60

    model_path = f'model/pong_batch{batch_size}_rate{update_rate}_ep{episodes}.pth'
    rewards_path = f'model/episode_rewards_batch{batch_size}_rate{update_rate}.npy'
    ave_rewards_path = f'model/ave_rewards_batch{batch_size}_rate{update_rate}.npy'
    writer = SummaryWriter()

    agent = Agent(env, epsilon_init, batch_size, gamma, training_mode)
    if training_mode and len(sys.argv) >= 4 and os.path.exists(sys.argv[3]):
        agent.load_checkpoint(sys.argv[3])
        try:
            episode_rewards = np.load(rewards_path).to_list()
            ave_rewards = np.load(ave_rewards_path).to_list()
        except:
            episode_rewards = list()
            ave_rewards = list()

    if training_mode:
        for episode in range(1, episodes):
            terminate = False
            while not terminate:
                agent.epsilon = agent.epsilon*epsilon_decay if agent.epsilon >= epsilon_min else epsilon_min
                terminate, reward = agent.train()
                if terminate:
                    episode_rewards.append(reward)
                    mean_reward = round(np.mean(episode_rewards[-5:]), 3)
                    ave_rewards.append(mean_reward)
                    writer.add_scalar('Reward/Episode', reward, episode)
                    writer.add_scalar('Average_Cumulative_Reward/Last_5_Episodes', mean_reward, episode)

                    print(f"episode {episode}, mean reward: {mean_reward}\n")
                if len(agent.replay_memory) >= update_rate:
                    agent.update_weights()

            if episode % 10 == 0:
                agent.save_checkpoint(episode)
                np.save(rewards_path, np.array(episode_rewards))
                np.save(ave_rewards_path, np.array(ave_rewards))

        writer.flush()
        agent.save(model_path)
    else:
        if len(sys.argv) >= 2 and os.path.exists(sys.argv[1]):
            agent.demo(sys.argv[1])
        else:
            sys.exit('Please specify the model file.')

    env.close()


if __name__ == '__main__':
    sys.exit(main())
