import dgl
import torch
import numpy as np
import gymnasium as gym
from functools import reduce
import operator as OP
from dfa import dfa2dict, dict2dfa
import dfa_embeddings.utils as utils
from scipy.special import log_softmax

class cDFAEnv(gym.Wrapper):
    def __init__(self, env, sampler, alphabet_type='deterministic'):
        super().__init__(env)
        self.env = env
        self.alphabet_type = alphabet_type
        self.sampler = sampler
        # self.dfa2vec = dfa2vec

    def reset(self, seed=None):
        self.env.reset()
        self.dfa_goal = next(self.sampler)
        self.dfa_goal_state_belief = []
        for dfa_clause in self.dfa_goal:
            dfa_clause_state_belief = []
            for dfa in dfa_clause:
                if self.alphabet_type == 'deterministic':
                    # state_belief = 0
                    state_belief = np.zeros(len(dfa.states()))
                    state_belief[dfa.start] = 1.0
                else:
                    state_belief = np.zeros(len(dfa.states()))
                    state_belief[dfa.start] = 1.0
                dfa_clause_state_belief.append(state_belief)
            self.dfa_goal_state_belief.append(tuple(dfa_clause_state_belief))
        self.dfa_goal_state_belief = tuple(self.dfa_goal_state_belief)
        return self._to_dict(self.dfa_goal, self.dfa_goal_state_belief), None

    # def reset(self, seed=None):
    #     self.env.reset()
    #     self.dfa_goal = next(self.sampler)
    #     self.dfa_goal_state_belief = []
    #     self.dfa_goal_dgl = []
    #     for dfa_clause in self.dfa_goal:
    #         dfa_clause_state_belief = []
    #         dfa_clause_dgl = []
    #         for dfa in dfa_clause:
    #             if self.alphabet_type == 'deterministic':
    #                 state_belief = 0
    #                 dfa_dgl = None
    #             else:
    #                 state_belief = np.zeros(len(dfa.states()))
    #                 state_belief[dfa.start] = 1.0
    #                 dfa_dgl = self.dfa2vec((dfa2dict(dfa)[0], state_belief))
    #             dfa_clause_state_belief.append(state_belief)
    #             dfa_clause_dgl.append(dfa_dgl)
    #         self.dfa_goal_state_belief.append(tuple(dfa_clause_state_belief))
    #         self.dfa_goal_dgl.append(tuple(dfa_clause_dgl))
    #     self.dfa_goal_state_belief = tuple(self.dfa_goal_state_belief)
    #     self.dfa_goal_dgl = tuple(self.dfa_goal_dgl)
    #     return self._to_dict(self.dfa_goal_dgl, self.dfa_goal_state_belief), None

    def step(self, action):
        if self.alphabet_type == 'probabilistic':
            action = np.exp(log_softmax(action))
        return self._step(action)

    def _step(self, action):
        env_done = self.env.step(action)

        token = self.env.get_events()

        old_dfa_goal  = self.dfa_goal
        self.dfa_goal, self.dfa_goal_state_belief = self._advance(self.dfa_goal, self.dfa_goal_state_belief, token)

        dfa_reward, dfa_done = self._get_dfa_reward(self.dfa_goal, self.dfa_goal_state_belief)

        assert dfa_reward >= -1 and dfa_reward <= 1
        assert dfa_reward !=  1 or dfa_done
        assert dfa_reward != -1 or dfa_done
        assert (dfa_reward <=  -1 or dfa_reward >= 1) or not dfa_done

        reward  = dfa_reward
        done    = env_done or dfa_done

        return self._to_dict(self.dfa_goal, self.dfa_goal_state_belief), reward, done, False, None

    def _advance(self, dfa_goal, dfa_goal_state_belief, truth_assignment):
        if self.alphabet_type == 'probabilistic':
            # In the probabilistic case, dfa_goal stays the same and dfa_goal_state_belief gets updated.
            new_dfa_goal_state_belief = []
            for dfa_clause, dfa_clause_state_belief in zip(dfa_goal, dfa_goal_state_belief):
                new_dfa_clause_state_belief = []
                for dfa, state_belief in zip(dfa_clause, dfa_clause_state_belief):
                    new_state_belief = utils.get_state_belief(dfa, state_belief, truth_assignment)
                    new_dfa_clause_state_belief.append(new_state_belief)
                new_dfa_goal_state_belief.append(tuple(new_dfa_clause_state_belief))
            new_dfa_goal_state_belief = tuple(new_dfa_goal_state_belief)
            return dfa_goal, new_dfa_goal_state_belief
        else:
            # In the deterministic case, dfa_goal gets updated and dfa_goal_state_belief stays the same.
            # return tuple(tuple(dfa.advance([truth_assignment]).minimize() for dfa in dfa_clause) for dfa_clause in dfa_goal), dfa_goal_state_belief
            dfa_goal = tuple(tuple(dfa.advance([truth_assignment]).minimize() for dfa in dfa_clause) for dfa_clause in dfa_goal)
            new_dfa_goal_state_belief = []
            for dfa_clause, dfa_clause_state_belief in zip(dfa_goal, dfa_goal_state_belief):
                new_dfa_clause_state_belief = []
                for dfa, state_belief in zip(dfa_clause, dfa_clause_state_belief):
                    new_dfa = dfa.advance([truth_assignment])
                    new_state_belief = np.zeros(len(dfa.states()))
                    new_state_belief[new_dfa.start] = 1.0
                    new_dfa_clause_state_belief.append(new_state_belief)
                new_dfa_goal_state_belief.append(tuple(new_dfa_clause_state_belief))
            new_dfa_goal_state_belief = tuple(new_dfa_goal_state_belief)
            return dfa_goal, new_dfa_goal_state_belief

    def _get_dfa_reward(self, dfa_goal, dfa_goal_state_belief):
        if self.alphabet_type == 'probabilistic':
            dfa_goal_rewards = []
            for dfa_clause, dfa_clause_state_belief in zip(dfa_goal, dfa_goal_state_belief):
                dfa_clause_rewards = []
                for dfa, state_belief in zip(dfa_clause, dfa_clause_state_belief):
                    s = np.where(state_belief > 0.9)[0]
                    if s.size > 0 and s[0] != dfa.start:
                        dfa_dict, _ = dfa2dict(dfa)
                        _dfa = dict2dfa(dfa_dict, s[0])
                        if _dfa._label(_dfa.start):
                            dfa_clause_rewards.append(1.0)
                            continue
                        elif _dfa.find_word() is None:
                            dfa_clause_rewards.append(-1.0)
                            continue
                    dfa_clause_rewards.append(0.0)
                dfa_goal_rewards.append(dfa_clause_rewards)
            reward = reduce(min, map(lambda dfa_clause_rewards: reduce(max, dfa_clause_rewards), dfa_goal_rewards))
            done = reward != 0.0
            return reward, done
        else:
            mono_dfa = reduce(OP.and_, map(lambda dfa_clause: reduce(OP.or_, dfa_clause), dfa_goal))
            if mono_dfa._label(mono_dfa.start):
                return 1.0, True
            if mono_dfa.find_word() is None:
                return -1.0, True
            return 0.0, False

    def _to_dict(self, dfa_goal, dfa_goal_state_belief):
        dfa_dict_goal = []
        for dfa_clause, dfa_clause_state_belief in zip(dfa_goal, dfa_goal_state_belief):
            dfa_dict_clause = []
            for dfa, state_belief in zip(dfa_clause, dfa_clause_state_belief):
                dfa_dict_clause.append((dfa2dict(dfa)[0], state_belief))
            dfa_dict_goal.append(tuple(dfa_dict_clause))
        dfa_dict_goal = tuple(dfa_dict_goal)
        return dfa_dict_goal

    # def _to_dict(self, dfa_goal_dgl, dfa_goal_state_belief):
    #     dfa_dict_goal = []
    #     for dfa_clause_dgl, dfa_clause_state_belief in zip(dfa_goal_dgl, dfa_goal_state_belief):
    #         dfa_dict_clause = []
    #         for dfa_dgl, state_belief in zip(dfa_clause_dgl, dfa_clause_state_belief):
    #             # dfa_dict_clause.append((dfa2dict(dfa)[0], state_belief))
    #             idx = torch.argwhere(dfa_dgl.ndata['id']>=0)
    #             dfa_dgl.ndata['is_root'][idx] = torch.from_numpy(state_belief).float().reshape(dfa_dgl.ndata['is_root'][idx].shape)
    #             hg = dgl.sum_nodes(dfa_dgl, 'h', weight='is_root')
    #             dfa_dict_clause.append(hg)
    #         dfa_dict_goal.append(tuple(dfa_dict_clause))
    #     dfa_dict_goal = tuple(dfa_dict_goal)
    #     return dfa_dict_goal
