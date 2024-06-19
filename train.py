from dfa_embeddings.dfa2vec import DFA2Vec

if __name__ == "__main__":
    dfa2vec = DFA2Vec(pretrained=False)
    dfa2vec.train()
    print(dfa2vec.encoder)
    print(dfa2vec.decoder)
