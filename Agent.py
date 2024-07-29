import torch
import torch.optim as optim
import numpy as np
from DQN import DQN
from collections import deque, namedtuple 
from assignment3_utils import process_frame


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

Replay = namedtuple('Replay',field_names=['state', 'action', 'reward', 'done', 'next_state'])


class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)


    def __len__(self):
        return len(self.buffer)


    def append(self, experience):
        self.buffer.append(experience)


    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions, dtype=np.int64), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(next_states, dtype=np.float32)


class Agent:
    def __init__(self, env, start_epsilon, batch_size, gamma, mode=False):
        self.env = env
        self.gamma = gamma
        self.epsilon = start_epsilon
        self.batch_size = batch_size
        self.replay_memory = ReplayMemory(10000)
        self.model = DQN((4, 84, 80), env.action_space.n).to(device)
        self.target_model = DQN((4, 84, 80), env.action_space.n).to(device)
        self.train_mode = mode
        self.episode = 0
        self.learns = 0
        self.reset()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)


    def reset(self):
        self.state = process_frame(self.env.reset(seed=0)[0])
        self.steps = 0
        self.total_reward = 0


    def act(self):
        if self.train_mode:
            if np.random.random() < self.epsilon:
                action = self.env.action_space.sample()
            else:
                state = torch.tensor(np.array(self.state), dtype=torch.float32).to(device)
                action = np.argmax(self.model(state).cpu().detach().numpy())
        else:
            state = torch.tensor(np.array(self.state), dtype=torch.float32).to(device)
            action = np.argmax(self.model(state).cpu().detach().numpy())
        return action


    def train(self):
        episode_reward = None
        action = self.act()
        next_state, reward, terminate, _, _ = self.env.step(action)
        next_state = process_frame(next_state)
        self.replay_memory.append(Replay(np.squeeze(self.state, axis=0), action, reward, terminate, np.squeeze(next_state, axis=0)))
        self.state = next_state
        self.steps += 1
        self.total_reward += reward

        if terminate:
            episode_reward = self.total_reward
            print(f"steps {self.steps} Score: {episode_reward}")
            self.episode += 1
            self.reset()
            return True, episode_reward

        return False, episode_reward


    def update_weights(self):
        states, actions, rewards, dones, next_states = self.replay_memory.sample(self.batch_size)

        states_t = torch.tensor(states, dtype=torch.float32).to(device)
        next_states_t = torch.tensor(next_states).to(device)
        actions_t = torch.tensor(actions).to(device)
        rewards_t = torch.tensor(rewards).to(device)
        done_mask = torch.BoolTensor(dones).to(device)
        action_values = self.model(states_t).gather(1,actions_t.unsqueeze(-1)).squeeze(-1)

        next_action_values = self.target_model(next_states_t).max(1)[0]
        next_action_values[done_mask] = 0.0
        next_action_values = next_action_values.detach()

        expected_action_values = rewards_t + next_action_values*self.gamma
        loss_t = torch.nn.MSELoss()(action_values, expected_action_values)
        
        self.optimizer.zero_grad()
        loss_t.backward()
        self.optimizer.step()
        self.learns += 1

        if self.learns % 1000 == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            print(f"episode {self.episode}: target model weights updated")


    def demo(self, model_path):
        self.load(model_path)
        self.model.eval()

        self.reset()
        done = False
        stop = False
        steps = 0
        episode_reward = 0
        reward = 0
        while not done and not stop:
            action = self.act()
            next_state, reward, done, stop, _ = self.env.step(action)
            self.state = process_frame(next_state)
            episode_reward += reward
            steps += 1

        print(f"Steps: {steps:}, "f"Reward: {episode_reward:.2f}, \n")


    def save(self, path):
        return torch.save(self.model.state_dict(), path)


    def load(self, path):
        self.model.load_state_dict(torch.load(path))


    def save_checkpoint(self, episode):
        checkpoint = {
            'episode': episode,
            'model_state_dict': self.model.state_dict(),
            'target_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'replay_memory': self.replay_memory,
            'epsilon': self.epsilon
        }
        torch.save(checkpoint, f'model/checkpoint.pth')
        print(f'Checkpoint saved at episode {episode}')


    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.replay_memory = checkpoint['replay_memory']
        self.epsilon = checkpoint['epsilon']
        return checkpoint['episode']


    def export_onnx(self, model_path):
        onnx_model = torch.onnx.dynamo_export(self.model, torch.randn(1, 4, 84, 80))
        onnx_model.save(model_path)
