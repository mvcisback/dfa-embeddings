import random
from dfa import dict2dfa, DFA
from dfa_mutate import change_transition


def reach_avoid_sampler(n_tokens, max_size=6, prob_stutter=0.9):
    assert n_tokens > 1

    n = random.randint(3, max_size)
    success, fail = n - 2, n - 1

    tokens = list(range(n_tokens))
    while True:
        transitions = {
          success: (True,  {t: success for t in range(n_tokens)}),
          fail:    (False, {t: fail    for t in range(n_tokens)}),
        }
        for state in range(n - 2):
            noop, good, bad = partition = (set(), set(), set())
            random.shuffle(tokens)
            good.add(tokens[0])
            bad.add(tokens[1])
            for token in tokens[2:]:
                if random.random() <= prob_stutter:
                    noop.add(token)
                else:
                    partition[random.randint(1, 2)].add(token)

            _transitions = dict()
            for token in good:
                _transitions[token] = state + 1
            for token in bad:
                _transitions[token] = fail
            for token in noop:
                _transitions[token] = state

            transitions[state] = (False, _transitions)

        yield dict2dfa(transitions, start=0).minimize()


def accepting_is_sink(d: DFA):
    def transition(s, c):
        if d._label(s) is True:
            return s
        return d._transition(s, c)
    return DFA(start=d.start,
               inputs=d.inputs,
               label=d._label,
               transition=transition)


def dfa_sampler(n_tokens=12, max_mutations=5):
    dfas = reach_avoid_sampler(n_tokens)
    while True:
        candidate = next(dfas)
        for _ in range(random.randint(0, max_mutations)):
            tmp =  accepting_is_sink(change_transition(candidate))
            if tmp is None: continue
            tmp = tmp.minimize()
            if len(tmp.states()) == 1: continue
            candidate = tmp.minimize()
        yield candidate
