import numpy as np
from sklearn.neighbors import NearestNeighbors
def find_knn(query_vector, vectors, k=5):
    knn = NearestNeighbors(n_neighbors=k, metric='cosine')
    knn.fit(vectors)
    distances, indices = knn.kneighbors([query_vector])
    return distances, indices

def read_vectors_file(file_path):
    vectors = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.split()
            word = parts[0]
            vector = [float(value) for value in parts[1:]]
            vectors[word] = vector
    return vectors

def find_knn_from_file(query_word, vectors, feature_names, k=5):
    if query_word not in vectors:
        print(f"Le mot '{query_word}' n'est pas présent dans le fichier.")
        return
    
    query_vector = vectors[query_word]
    distances, indices = find_knn(query_vector, list(vectors.values()), k+1)
    filtered_distances = []
    filtered_indices = []

    for dist, ind in zip(distances[0], indices[0]):
        if feature_names[ind] != query_word:
            filtered_distances.append(dist)
            filtered_indices.append(ind)

    print(f"Les {k} mots les plus similaires à '{query_word}':")
    for i, index in enumerate(filtered_indices[:k]):
        print(f"{i + 1}. {feature_names[index]} (distance: {filtered_distances[i]})")


file_path = 'vectors_word2vec.txt'
vectors_word2vec = read_vectors_file(file_path)
feature_names_word2vec = list(vectors_word2vec.keys())

query_word = "fox"
find_knn_from_file(query_word, vectors_word2vec, feature_names_word2vec)