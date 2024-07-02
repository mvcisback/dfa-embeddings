import sys
from dfa_embeddings.dfa2vec import DFA2Vec
from dfa_embeddings.cdfa2vec import cDFA2Vec

if __name__ == "__main__":
    n_tokens = int(sys.argv[1])
    alphabet_type = sys.argv[2]
    architecture_type = sys.argv[3]
    seed = int(sys.argv[4])
    if architecture_type == 'monolithic':
        dfa2vec = DFA2Vec(pretrained=False, seed=seed, n_tokens=n_tokens, alphabet_type=alphabet_type)
        dfa2vec.train()
        print(dfa2vec.encoder)
        print(dfa2vec.decoder)
    else:
        cdfa2vec = cDFA2Vec(pretrained=False, seed=seed, n_tokens=n_tokens, alphabet_type=alphabet_type)
        cdfa2vec.train()
        print(cdfa2vec.encoder)
        print(cdfa2vec.decoder)
