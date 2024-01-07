import os
import spacy
import gensim
from sklearn.feature_extraction.text import CountVectorizer
from maker_corpus_clean import lire_contenu_corpus, pretraiter_donnees

import os
import spacy
import gensim
from sklearn.feature_extraction.text import CountVectorizer
from maker_corpus_clean import lire_contenu_corpus, pretraiter_donnees
from sklearn.decomposition import PCA
def vectoriser_word2vec(documents_pretraites):
    model = gensim.models.Word2Vec(sentences=documents_pretraites, vector_size=100, window=5, min_count=1, workers=4)
    
    vectors = []
    for doc in documents_pretraites:
        for lemma in doc:
            if lemma in model.wv:
                vectors.append(model.wv[lemma])
    return vectors

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
  
def vectoriser_countvectorizer_reduit(documents_pretraites, dimensions_reduites=50):
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(documents_pretraites).toarray()
    svd = TruncatedSVD(n_components=dimensions_reduites)
    vectors_reduits = svd.fit_transform(vectors)

    return vectors_reduits


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
    
    vectors_word2vec = vectoriser_word2vec(documents_pretraites)
    feature_names_word2vec = [word for doc in documents_pretraites for word in doc]
    sauvegarder_vectors(vectors_word2vec, feature_names_word2vec, 'vectors_word2vec.txt')
    vectors_countvectorizer, feature_names_countvectorizer = vectoriser_countvectorizer(documents_pretraites)
    sauvegarder_vectors(vectors_countvectorizer, feature_names_countvectorizer, 'vectors_countvectorizer_ppmi.txt')
    


if __name__ == "__main__":
    main()
    

