import os
import sys
import warnings
import numpy as np
import gymnasium as gym
from Agent import Agent
from gymnasium.wrappers import FrameStack
from torch.utils.tensorboard import SummaryWriter



warnings.filterwarnings("ignore", category=UserWarning)

batch_size = 32
update_rate = 40
epsilon_min = 0.05
epsilon_decay = 0.995
gamma = 0.95
epsilon_init = 1.0


def main():
    training_mode = True if len(sys.argv) >= 2 and sys.argv[1] == 'train' else False
    episodes = int(sys.argv[2]) if training_mode else 1
    env = FrameStack(gym.make('ALE/Pong-v5', render_mode='rgb_array' if training_mode else 'human'), num_stack=4)
    if not training_mode:
        env.metadata['render_fps'] = 60

    rewards_path = f'model/episode_rewards_batch{batch_size}_rate{update_rate}.npy'
    ave_rewards_path = f'model/ave_rewards_batch{batch_size}_rate{update_rate}.npy'
    writer = SummaryWriter()

    agent = Agent(env, epsilon_init, batch_size, gamma, training_mode)
    if training_mode and len(sys.argv) >= 4 and os.path.exists(sys.argv[3]):
        agent.load_checkpoint(sys.argv[3])

    if training_mode and len(sys.argv) >= 4 and os.path.exists(rewards_path):
        episode_rewards = np.load(rewards_path).tolist()
    else:
        episode_rewards = list()

    if training_mode and len(sys.argv) >= 4 and os.path.exists(ave_rewards_path):
        ave_rewards = np.load(ave_rewards_path).tolist()
    else:
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

            if episode % 50 == 0:
                np.save(rewards_path, np.array(episode_rewards))
                np.save(ave_rewards_path, np.array(ave_rewards))
                agent.save_checkpoint(episode)

        writer.flush()
        model_path = f'model/pong_batch{batch_size}_rate{update_rate}_ep{len(episode_rewards)}.'
        agent.save(model_path+'pth')
        agent.export_onnx(model_path+'onnx')
    else:
        if len(sys.argv) >= 2 and os.path.exists(sys.argv[1]):
            agent.demo(sys.argv[1])
        else:
            sys.exit('Please specify the model file.')

    writer.close()
    env.close()


if __name__ == '__main__':
    sys.exit(main())
