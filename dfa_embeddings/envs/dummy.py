import gymnasium as gym
from gymnasium import spaces

class DummyEnv(gym.Env):
    def __init__(self, n_tokens:int, timeout:int):
        self.n_tokens = n_tokens
        self.timeout = timeout
        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Discrete(self.n_tokens)
        self.num_episodes = 0
        self.time = 0
        self.proposition = None

    def reset(self, seed=None):
        self.time = 0
        self.num_episodes += 1
        done = self.time > self.timeout
        return done

    def step(self, action):
        self.time += 1
        self.proposition = action
        done = self.time > self.timeout
        return done

    def get_events(self):
        return [self.proposition]

