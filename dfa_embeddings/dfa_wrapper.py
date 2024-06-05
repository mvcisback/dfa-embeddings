import numpy as np
import gymnasium as gym
from functools import reduce
import operator as OP
from dfa import dfa2dict

class DFAEnv(gym.Wrapper):
    def __init__(self, env, sampler, compositional=False):
        super().__init__(env)
        self.env = env
        self.compositional = compositional
        self.sampler = sampler

    def reset(self, seed=None):
        self.env.reset()
        self.dfa_goal = next(self.sampler)
        return self._to_dict(self.dfa_goal), None

    def step(self, action):
        env_done = self.env.step(action)

        token = self.env.get_events()

        old_dfa_goal  = self.dfa_goal
        self.dfa_goal = self._advance(self.dfa_goal, token)

        dfa_reward, dfa_done = self.get_dfa_reward(old_dfa_goal, self.dfa_goal)

        assert dfa_reward >= -1 and dfa_reward <= 1
        assert dfa_reward !=  1 or dfa_done
        assert dfa_reward != -1 or dfa_done
        assert (dfa_reward <=  -1 or dfa_reward >= 1) or not dfa_done

        reward  = dfa_reward
        done    = env_done or dfa_done

        return self._to_dict(self.dfa_goal), reward, done, False, None

    def get_dfa_reward(self, old_dfa_goal, dfa_goal):
        if old_dfa_goal != dfa_goal:
            if self.compositional:
                mono_dfa = self._to_monolithic_dfa(dfa_goal)
            else:
                mono_dfa = dfa_goal
            if mono_dfa._label(mono_dfa.start):
                return 1.0, True
            if mono_dfa.find_word() is None:
                return -1.0, True
        return 0.0, False

    def _to_monolithic_dfa(self, dfa_goal):
        return reduce(OP.and_, map(lambda dfa_clause: reduce(OP.or_, dfa_clause), dfa_goal))

    def _advance(self, dfa_goal, truth_assignment):
        if self.compositional:
            return tuple(tuple(dfa.advance(truth_assignment).minimize() for dfa in dfa_clause) for dfa_clause in dfa_goal)
        return dfa_goal.advance(truth_assignment).minimize()

    def _to_dict(self, dfa_goal):
        if self.compositional:
            return tuple(tuple(dfa2dict(dfa) for dfa in dfa_clause) for dfa_clause in dfa_goal)
        return dfa2dict(dfa_goal)

