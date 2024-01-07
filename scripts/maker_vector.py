import os
import spacy
import gensim
from sklearn.feature_extraction.text import CountVectorizer
from maker_corpus_clean import lire_contenu_corpus, pretraiter_donnees
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
def vectoriser_word2vec(documents_pretraites):
    model = gensim.models.Word2Vec(sentences=documents_pretraites, vector_size=100, window=5, min_count=1, workers=4)
    
    vectors = []
    entity_names = []
    for doc in documents_pretraites:
        for lemma in doc:
            if lemma in model.wv:
                vectors.append(model.wv[lemma])
                entity_names.append(lemma)
    return vectors, entity_names

#test avec ppmi vu en cours

def vectoriser_countvectorizer(documents_pretraites):
    vectorizer = CountVectorizer()
    X_count = vectorizer.fit_transform([' '.join(doc) for doc in documents_pretraites])
    cooccurrence_matrix = X_count.T.dot(X_count).toarray()
    total_sum = cooccurrence_matrix.sum()
    word_probabilities = cooccurrence_matrix.sum(axis=1) / total_sum
    context_probs = cooccurrence_matrix / total_sum
    ppmi_matrix = np.maximum(np.log2(context_probs / np.outer(word_probabilities, word_probabilities)), 0)
    feature_names = vectorizer.get_feature_names_out()
    df_ppmi = pd.DataFrame(ppmi_matrix, index=feature_names, columns=feature_names)

    vectors = []
    entity_names = []
    for word in df_ppmi.index:
        vector = df_ppmi.loc[word].values
        vectors.append(vector)
        entity_names.append(word)

    return vectors, entity_names
  
def vectoriser_countvectorizer_with_pca(documents_pretraites, n_components=10, cumulative_variance_threshold=0.95):
    vectorizer = CountVectorizer()
    X_count = vectorizer.fit_transform([' '.join(doc) for doc in documents_pretraites])
    cooccurrence_matrix = X_count.T.dot(X_count).toarray()
    total_sum = cooccurrence_matrix.sum()
    word_probabilities = cooccurrence_matrix.sum(axis=1) / total_sum
    context_probs = cooccurrence_matrix / total_sum
    ppmi_matrix = np.maximum(np.log2(context_probs / np.outer(word_probabilities, word_probabilities)), 0)
    feature_names = vectorizer.get_feature_names_out()
    df_ppmi = pd.DataFrame(ppmi_matrix, index=feature_names, columns=feature_names)

    pca = PCA(n_components=n_components)
    vectors_pca = pca.fit_transform(df_ppmi.values)

    if n_components is None:
        cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance_ratio >= cumulative_variance_threshold) + 1

        pca = PCA(n_components=n_components)
        vectors_pca = pca.fit_transform(df_ppmi.values)

    entity_names = df_ppmi.index

    # Plot cumulative explained variance
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance vs. Number of Components')
    plt.show()

    return vectors_pca, entity_names

def sauvegarder_vectors(vectors, feature_names, nom_fichier):
    with open(nom_fichier, 'w', encoding='utf-8') as file:
        saved_vectors = set()  
        sorted_indices = np.argsort(feature_names)
        sorted_feature_names = np.array(feature_names)[sorted_indices]
        sorted_vectors = np.array(vectors)[sorted_indices]

        for word, vector in zip(sorted_feature_names, sorted_vectors):
            key = f"{word}_{tuple(vector)}"
            if key not in saved_vectors:
                vector_w = ' '.join(map(str, vector))
                file.write(f"{word} {vector_w}\n")
                saved_vectors.add(key)

def main():
    path_corpus = 'corpus'
    stop_words_spacy = spacy.lang.en.stop_words.STOP_WORDS
    contenu_corpus = lire_contenu_corpus(path_corpus)
    documents_pretraites = [pretraiter_donnees(doc, stop_words_spacy) for doc in contenu_corpus]
    
    vectors_word2vec,feature_names_word2vec = vectoriser_word2vec(documents_pretraites)
    sauvegarder_vectors(vectors_word2vec, feature_names_word2vec, 'vectors_word2vec.txt')
    vectors_countvectorizer, feature_names_countvectorizer = vectoriser_countvectorizer(documents_pretraites)
    sauvegarder_vectors(vectors_countvectorizer, feature_names_countvectorizer, 'vectors_countvectorizer_ppmi.txt')
    vectoriser_countvectorizerdim,featurenames=vectoriser_countvectorizer_with_pca(documents_pretraites)
    sauvegarder_vectors(vectoriser_countvectorizerdim, featurenames, 'vectors_countvectorizer_ppmi_pca.txt')
if __name__ == "__main__":
    main()
    
