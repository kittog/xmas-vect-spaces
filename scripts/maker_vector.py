import os
import spacy
import gensim
from sklearn.feature_extraction.text import CountVectorizer
from maker_corpus_clean import lire_contenu_corpus, pretraiter_donnees
def vectoriser_word2vec(documents_pretraites):
    model = gensim.models.Word2Vec(sentences=[doc.split() for doc in documents_pretraites], vector_size=100, window=5, min_count=1, workers=4)
    
    vectors = []
    for doc in documents_pretraites:
        for lemma in doc.split():
            if lemma in model.wv:
                vectors.append(model.wv[lemma])
            pass
    return vectors

def vectoriser_countvectorizer(documents_pretraites):
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(documents_pretraites).toarray()
    return vectors
  
def vectoriser_countvectorizer_reduit(documents_pretraites, dimensions_reduites=50):
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(documents_pretraites).toarray()
    svd = TruncatedSVD(n_components=dimensions_reduites)
    vectors_reduits = svd.fit_transform(vectors)

    return vectors_reduits
def sauvegarder_vectors(vectors, feature_names, nom_fichier):
    with open(nom_fichier, 'w', encoding='utf-8') as file:
        for word, vector in zip(feature_names, vectors):
            vector_str = ' '.join(map(str, vector))
            file.write(f"{word} {vector_str}\n")

def main():
    path_corpus = 'bla'
    stop_words_spacy = spacy.lang.en.stop_words.STOP_WORDS
    contenu_corpus = lire_contenu_corpus(path_corpus)
    documents_pretraites = [pretraiter_donnees(doc, stop_words_spacy) for doc in contenu_corpus]
    vectors_word2vec = vectoriser_word2vec(documents_pretraites)
    feature_names_word2vec = [word for doc in documents_pretraites for word in doc.split()]
    sauvegarder_vectors(vectors_word2vec, feature_names_word2vec, 'vectors_word2vec.txt')
    vectors_countvectorizer = vectoriser_countvectorizer(documents_pretraites)
    feature_names_countvectorizer = vectorizer.get_feature_names_out()
    sauvegarder_vectors(vectors_countvectorizer, feature_names_countvectorizer, 'vectors_countvectorizer.txt')

if __name__ == "__main__":
    main()

