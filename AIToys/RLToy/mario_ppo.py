# python3
# Create date: 2023-11-29
# Func: test mario
# ===============================================================
from tqdm.auto import tqdm
import os
import gym 
from argparse import Namespace
from torchvision import transforms
from gym.spaces import Box
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT
import numpy as np
import torch
from gym.wrappers import FrameStack 
from nes_py.wrappers import JoypadSpace
from torch import nn
from torch.distributions import Categorical
import matplotlib.pyplot as plt
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
try:
    CURR_FILE = os.path.dirname(__file__)
except Exception as e:
    CURR_FILE = os.path.dirname('__file__')

if len(CURR_FILE) == 0:
    CURR_FILE = "/home/scc/sccWork/myGitHub/My_Learn/AIToys/RLToy"



class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip
    
    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info, _ = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info, _


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(
            low=0, high=255, shape=self.observation_space.shape[:2], dtype=np.uint8
        )
    
    def observation(self, observation):
        tf = transforms.Grayscale()
        # channel first
        return tf(torch.tensor(np.transpose(observation, (2, 0, 1)).copy(), dtype=torch.float))


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = (shape, shape)
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transformations = transforms.Compose([transforms.Resize(self.shape), transforms.Normalize(0, 255)])
        return transformations(observation).squeeze(0)


env = gym_super_mario_bros.make('SuperMarioBros-1-2-v0')
env = JoypadSpace(env, [["right"], ["right", "A"]])
# env = JoypadSpace(env, RIGHT_ONLY)
env = FrameStack(
    ResizeObservation(
        GrayScaleObservation(SkipFrame(env, skip=4)), 
        shape=84), 
    num_stack=4)


env.seed(42)
env.action_space.seed(42)
# torch.manual_seed(42)
# torch.random.manual_seed(42)
np.random.seed(42)



class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.action_space.n)
        )
        self.critic = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, obs):
        return Categorical(logits=self.actor(obs)), self.critic(obs).reshape(-1)


DEFAULT_CONFIG = Namespace(
    gamma=0.95,
    lamda=0.95,
    worker_steps=4096,
    n_mini_batch=4, # one batch 4 samples
    epochs=30,
    save_dir=f"{CURR_FILE}/weights/ppo",
    env=env,
    actor_lr=0.00025,
    critic_lr=0.001
)

class PPOSolver:
    def __init__(self, cfg=None):
        if cfg is None:
            cfg = DEFAULT_CONFIG
        self.rewards = []
        self.gamma = cfg.gamma
        self.lamda = cfg.lamda
        self.worker_steps = cfg.worker_steps
        self.n_mini_batch = cfg.n_mini_batch
        self.epochs = cfg.epochs
        self.save_dir = cfg.save_dir
        self.env_name = str(cfg.env).split('<')[-1].replace(">", "")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.batch_size = self.worker_steps
        self.mini_batch_size = self.batch_size // self.n_mini_batch
        res = cfg.env.reset()
        self.obs = res[0].__array__()
        self.policy = Model().to(device)
        self.mse_loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': cfg.actor_lr},
            {'params': self.policy.critic.parameters(), 'lr': cfg.critic_lr},
        ], eps=1e-4)
        self.policy_old = Model().to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.all_episode_rewards = []
        self.all_mean_rewards = []
        self.episode = 0
        self.best_rewards = -float('inf')
    
    def save_checkpoint(self, best_flag=False):
        filename = os.path.join(self.save_dir, '{}-checkpoint_{}.pth'.format(self.env_name, 'best' if best_flag else self.episode))
        torch.save(self.policy_old.state_dict(), f=filename)
        print('Checkpoint saved to \'{}\''.format(filename))

    def load_checkpoint(self, filename):
        self.policy.load_state_dict(torch.load(os.path.join(self.save_dir, filename), map_location=device))
        self.policy_old.load_state_dict(torch.load(os.path.join(self.save_dir, filename), map_location=device))
        print('Resuming training from checkpoint \'{}\'.'.format(filename))

    def sample(self, render=False, test_flag=False):
        rewards = np.zeros(self.worker_steps, dtype=np.float32)
        actions = np.zeros(self.worker_steps, dtype=np.int32)
        done = np.zeros(self.worker_steps, dtype=bool)
        obs = np.zeros((self.worker_steps, 4, 84, 84), dtype=np.float32)
        log_pis = np.zeros(self.worker_steps, dtype=np.float32)
        values = np.zeros(self.worker_steps, dtype=np.float32)
        for t in range(self.worker_steps):
            with torch.no_grad():
                obs[t] = self.obs
                pi, v = self.policy_old(torch.tensor(self.obs, dtype=torch.float32, device=device).unsqueeze(0))
                values[t] = v.cpu().numpy()
                a = pi.sample()
                actions[t] = a.cpu().numpy()
                log_pis[t] = pi.log_prob(a).cpu().numpy()
        
            self.obs, rewards[t], done[t], info, _ = env.step(actions[t])
            self.obs = self.obs.__array__()
            if test_flag or (render and self.episode % 100 == 0):
                env.render()
            self.rewards.append(rewards[t])
            if done[t]:
                self.episode += 1
                self.all_episode_rewards.append(np.sum(self.rewards))
                self.rewards = []
                env.reset()
                if self.episode % 10 == 0:
                    recent_reward = np.mean(self.all_episode_rewards[-10:])
                    print('Episode: {}, average reward: {}'.format(self.episode, recent_reward))
                    self.all_mean_rewards.append(recent_reward)
                    if self.best_rewards < recent_reward:
                        self.best_rewards = recent_reward
                        self.save_checkpoint(best_flag=True)
                    if self.episode % 500 == 0:
                        plt.plot(self.all_mean_rewards)
                        plt.savefig("{}/{}-mean_reward_{}.png".format(self.save_dir, self.env_name, self.episode))
                        plt.clf()
                        self.save_checkpoint()

        returns, advantages = self.calculate_advantages(done, rewards, values)
        return {
            'obs': torch.tensor(obs.reshape(obs.shape[0], *obs.shape[1:]), dtype=torch.float32, device=device),
            'actions': torch.tensor(actions, device=device),
            'values': torch.tensor(values, device=device),
            'log_pis': torch.tensor(log_pis, device=device),
            'advantages': torch.tensor(advantages, device=device, dtype=torch.float32),
            'returns': torch.tensor(returns, device=device, dtype=torch.float32)
        }

    def calculate_advantages(self, done, rewards, values):
        _, last_value = self.policy_old(torch.tensor(self.obs, dtype=torch.float32, device=device).unsqueeze(0))
        last_value = last_value.cpu().data.numpy()
        values = np.append(values, last_value)
        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            mask = 1.0 - done[i]
            delta = rewards[i] + self.gamma * values[i + 1] * mask - values[i]
            gae = delta + self.gamma * self.lamda * mask * gae
            returns.insert(0, gae + values[i])
        adv = np.array(returns) - values[:-1]
        return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-8)

    def train(self, samples, clip_range):
        indexes = torch.randperm(self.batch_size)
        for start in range(0, self.batch_size, self.mini_batch_size):
            end = start + self.mini_batch_size
            mini_batch_indexes = indexes[start: end]
            mini_batch = {}
            for k, v in samples.items():
                mini_batch[k] = v[mini_batch_indexes]
            for _ in range(self.epochs):
                loss = self.calculate_loss(clip_range=clip_range, samples=mini_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.policy_old.load_state_dict(self.policy.state_dict())

    def calculate_loss(self, samples, clip_range):
        sampled_returns = samples['returns']
        sampled_advantages = samples['advantages']
        pi, value = self.policy(samples['obs'])
        ratio = torch.exp(pi.log_prob(samples['actions']) - samples['log_pis'])
        clipped_ratio = ratio.clamp(min=1.0 - clip_range, max=1.0 + clip_range)
        policy_reward = torch.min(ratio * sampled_advantages, clipped_ratio * sampled_advantages)
        entropy_bonus = pi.entropy()
        vf_loss = self.mse_loss(value, sampled_returns)
        loss = -policy_reward + 0.5 * vf_loss - 0.01 * entropy_bonus
        return loss.mean()


def sp_load_checkpoint(solver, actor_file, critic_file, device, test_flag=False):
    solver.policy.actor.load_state_dict(
        torch.load(actor_file, map_location=device)
    )
    solver.policy.critic.load_state_dict(
        torch.load(critic_file, map_location=device)
    )
    print(f"Resuming training from checkpoint:\nactor: {actor_file}\ncritic: {critic_file}")
    if test_flag:
        print('Start Test')
        solver.sample(render=True, test_flag=True)
        env.close()


def test(checkpoint_file="checkpoint_best.pth"):
    # checkpoint_period = 10
    solver = PPOSolver()
    if checkpoint_file is not None:
        solver.load_checkpoint(checkpoint_file)

    solver.sample(render=True, test_flag=True)
    env.close()

# test-only
test(None)

solver = PPOSolver()
for i in tqdm(range(300)):
    solver.train(solver.sample(render=True, test_flag=False), 0.2)


test("SuperMarioBros-1-1-v0-checkpoint_best.pth")
test("SuperMarioBros-1-2-v0-checkpoint_best.pth")

checkpoint_file = "SuperMarioBros-1-2-v0-checkpoint_best.pth"
solver = PPOSolver()
if checkpoint_file is not None:
    solver.load_checkpoint(checkpoint_file)




# sp load
torch.save(solver.policy.actor.state_dict(), f=f"{DEFAULT_CONFIG.save_dir}/1-2-v0-actor.tar")
torch.save(solver.policy.critic.state_dict(), f=f"{DEFAULT_CONFIG.save_dir}/1-2-v0-critic.tar")

solver = PPOSolver()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

sp_load_checkpoint(
    solver, 
    f"{DEFAULT_CONFIG.save_dir}/1-2-v0-actor.tar", 
    f"{DEFAULT_CONFIG.save_dir}/1-2-v0-critic.tar", 
    device,
    True
)