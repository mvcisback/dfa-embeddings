import numpy as np
import gymnasium as gym
from functools import reduce
import operator as OP
from dfa import dfa2dict, dict2dfa
import dfa_embeddings.utils as utils
from scipy.special import log_softmax

class DFAEnv(gym.Wrapper):
    def __init__(self, env, sampler, alphabet_type='deterministic'):
        super().__init__(env)
        self.env = env
        self.alphabet_type = alphabet_type
        self.sampler = sampler
        self.prev_state_belief = 0.0

    def reset(self, seed=None):
        self.env.reset()
        self.dfa_original = next(self.sampler)
        self.dfa = self.dfa_original
        if self.alphabet_type == 'deterministic':
            # self.state_belief = 0
            self.state_belief = np.zeros(len(self.dfa.states()))
            self.state_belief[self.dfa.start] = 1.0
        else:
            self.state_belief = np.zeros(len(self.dfa.states()))
            self.state_belief[self.dfa.start] = 1
        return self._to_dict(self.dfa, self.state_belief), None

    def step(self, action):
        if self.alphabet_type == 'probabilistic':
            action = np.exp(log_softmax(action))
        return self._step(action)

    def _step(self, action):
        env_done = self.env.step(action)

        token = self.env.get_events()

        self.dfa, self.state_belief = self._advance(self.dfa, self.state_belief, token)

        dfa_reward, dfa_done = self._get_dfa_reward(self.dfa, self.state_belief)

        assert dfa_reward >= -1 and dfa_reward <= 1
        assert dfa_reward !=  1 or dfa_done
        assert dfa_reward != -1 or dfa_done
        assert (dfa_reward <=  -1 or dfa_reward >= 1) or not dfa_done

        reward  = dfa_reward
        done    = env_done or dfa_done

        self.prev_reward = reward

        return self._to_dict(self.dfa_original, self.state_belief), reward, done, False, None

    def _advance(self, dfa, state_belief, truth_assignment):
        self.prev_state_belief = self.state_belief
        if self.alphabet_type == 'probabilistic':
            # In the probabilistic case, dfa stays the same and state_belief gets updated.
            new_state_belief = utils.get_state_belief(dfa, state_belief, truth_assignment)
            return dfa, new_state_belief
        else:
            # In the deterministic case, dfa gets updated and state_belief stays the same.
            new_dfa = dfa.advance([truth_assignment])
            state_belief[dfa.start] = 0.0
            state_belief[new_dfa.start] = 1.0
            return new_dfa, state_belief
            # return dfa.advance([truth_assignment]).minimize(), state_belief

    def _get_dfa_reward(self, dfa, state_belief):
        if self.alphabet_type == 'probabilistic':
            reward = 0.0
            for s in dfa.states():
                # if state_belief[s] > 0.0 and not dfa._label(s) and sum(s != dfa._transition(s, a) for a in range(self.env.n_tokens)) == 0:
                #     return -1.0, True
                if dfa._label(s):
                    if state_belief[s] < 1.0:
                        delta = (state_belief[s] - self.prev_state_belief[s])
                        delta = delta/1_000 if delta > 0 else delta
                        reward += delta
                    else:
                        return 1.0, True
                elif sum(s != dfa._transition(s, a) for a in range(self.env.n_tokens)) == 0:
                    if state_belief[s] < 1.0:
                        delta = -(state_belief[s] - self.prev_state_belief[s])
                        delta = delta/1_000 if delta > 0 else delta
                        reward += delta
                    else:
                        return -1.0, True
            return reward, False
            # for s in dfa.states():
            #     if state_belief[s] > 0.9 and dfa._label(s):
            #         return 1.0, True
            #     elif state_belief[s] > 0.0 and not dfa._label(s) and sum(s != dfa._transition(s, a) for a in range(self.env.n_tokens)) == 0:
            #         return -1.0, True
            # s = np.where(state_belief > 0.9)[0]
            # if s.size > 0 and s[0] != dfa.start:
            #     dfa_dict, _ = dfa2dict(dfa)
            #     _dfa = dict2dfa(dfa_dict, s[0])
            #     if _dfa._label(_dfa.start):
            #         return 1.0, True
            #     elif _dfa.find_word() is None:
            #         return -1.0, True
        else:
            if dfa._label(dfa.start):
                return 1.0, True
            if dfa.find_word() is None:
                return -1.0, True
        return 0.0, False

    def _to_dict(self, dfa, state_belief):
        dfa_dict, _ = dfa2dict(dfa)
        return (dfa_dict, state_belief)
