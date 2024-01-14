import os
import re
import spacy
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer, TfidfVectorizer
import numpy as np
import pandas as pd

def lire_contenu_corpus(path):
    contenu_corpus = []
    for fichier in os.listdir(path):
        if fichier.endswith(".txt"):
            chemin_fichier = os.path.join(path, fichier)
            with open(chemin_fichier, 'r', encoding='utf-8') as file:
                contenu = file.read()
                contenu_corpus.append(contenu)
    return contenu_corpus


def pretraiter_donnees(texte, stop_words_spacy):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(texte)
    tokens = [token.lemma_.lower() for token in doc if token.text.isalpha() and token.text.lower() not in stop_words_spacy]
    cleaned_tokens = [re.sub('\s+', ' ', token).strip() for token in tokens]
    return cleaned_tokens

def main():
    path_corpus = 'corpus'
    nlp = spacy.load('en_core_web_sm')
    stop_words_spacy = nlp.Defaults.stop_words
    contenu_corpus = lire_contenu_corpus(path_corpus)
    documents_pretraites = [pretraiter_donnees(doc, stop_words_spacy) for doc in contenu_corpus]
    clean_texts = [" ".join(documents_pretraites[i]) for i in range(len(documents_pretraites))]

    n_toks = np.sum([len(i) for i in documents_pretraites])
    print(n_toks)

    # matrice terme-document
    # dictionnaire document
    dic = {}
    for i in range(len(clean_texts)):
        dic[i] = clean_texts[i]
    df1 = pd.DataFrame(dic, index=[0])
    # tfidf
    vectorizer = TfidfVectorizer()
    doc_vec = vectorizer.fit_transform(df1.iloc[0])
    df2 = pd.DataFrame(doc_vec.toarray().transpose(), index=vectorizer.get_feature_names_out())
    df2.to_csv("term_document_matrix.csv")
    #for i, doc in enumerate(documents_pretraites):
        #print(f"Document {i + 1} après prétraitement:\n{doc}\n")

if __name__ == "__main__":
    main()

from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

def main():
    path_corpus = '../corpus'
    nlp = spacy.load('en_core_web_sm')
    stop_words_spacy = nlp.Defaults.stop_words
    contenu_corpus = lire_contenu_corpus(path_corpus)
    documents_pretraites = [' '.join(pretraiter_donnees(doc, stop_words_spacy)) for doc in contenu_corpus]

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(documents_pretraites)
    feature_names = vectorizer.get_feature_names_out()

    word_counts = np.sum(X.toarray(), axis=0)
    word_freq_df = pd.DataFrame({'Word': feature_names, 'Frequency': word_counts})
    word_freq_df = word_freq_df.sort_values(by='Frequency', ascending=False)
    word_freq_df.to_csv("word_frequency.csv")
    top_25_words = word_freq_df.head(25)
    print(top_25_words)

if __name__ == "__main__":
    main()
