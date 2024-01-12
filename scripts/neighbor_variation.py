import numpy as np
import click

def var_score(w2v, vec, n=20):
    w2v_vec = list(set(w2v) & set(vec))
    print(w2v_vec)
    var = 1 - len(w2v_vec) / n
    return var

def get_neighbors(neigh):
    return sorted([i.split()[1] for i in neigh.split("\n")[1:]])

def main():
    with open("data/letter_to_santa_claus.txt", "r") as f:
        words = f.read().strip().split("\n")
    with open("test_w2v.txt", "r") as f:
        w2v = f.read().strip().split("\n\n")
    with open("test_ppmi.txt", "r") as f:
        ppmi = f.read().strip().split("\n\n")
    with open("test_ppmi_pca.txt", "r") as f:
        ppmi_pca = f.read().strip().split("\n\n")
    #print(len(w2v))
    #print(len(ppmi))
    #print(len(ppmi_pca))
    for i in range(len(ppmi)):
        #print(w2v[i])
        #print(ppmi[i])
        #print(ppmi_pca[i])
        w_w2v = get_neighbors(w2v[i])
        w_ppmi = get_neighbors(ppmi[i])
        w_pca = get_neighbors(ppmi_pca[i])
        print(w_w2v, w_ppmi)
        w2v_ppmi_score = var_score(w_w2v, w_ppmi)
        w2v_pca_score = var_score(w_w2v, w_pca)
        ppmi_pca_score = var_score(w_ppmi, w_pca)
        print(w2v_ppmi_score, w2v_pca_score)

if __name__ == "__main__":
    main()
