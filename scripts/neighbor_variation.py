import numpy as np
import click

def var_score(w2v, vec, n=10):
    # calcul du score de variation
    w2v_vec = list(set(w2v) & set(vec))
    #print(w2v_vec)
    var = 1 - len(w2v_vec) / n
    return var

def get_neighbors(neigh):
    # extrait les voisins du fichier texte pour un mot donné
    return sorted([i.split()[1] for i in neigh.split("\n")[1:]])

def main():
    # ouverture des listes de voisins et de la liste de mots cible
    with open("data/new_letter_to_santa_claus.txt", "r") as f:
        words = f.read().strip().split("\n")
    with open("voisin/liste2/neighbors_w2v_10.txt", "r") as f:
        w2v = f.read().strip().split("\n\n")
    with open("voisin/liste2/neighbors_ppmi_10.txt", "r") as f:
        ppmi = f.read().strip().split("\n\n")
    with open("voisin/liste2/neighbors_ppmi_pca_10.txt", "r") as f:
        ppmi_pca = f.read().strip().split("\n\n")

    for i in range(len(words)):
        w_w2v = get_neighbors(w2v[i])
        w_ppmi = get_neighbors(ppmi[i])
        w_pca = get_neighbors(ppmi_pca[i])
        # print(w_w2v, w_ppmi)
        w2v_ppmi_score = var_score(w_w2v, w_ppmi)
        w2v_pca_score = var_score(w_w2v, w_pca)
        ppmi_pca_score = var_score(w_ppmi, w_pca)
        print(words[i], w2v_ppmi_score, w2v_pca_score, ppmi_pca_score)

if __name__ == "__main__":
    main()
