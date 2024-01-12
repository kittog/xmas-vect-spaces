import os
import re
import spacy
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
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
    #for i, doc in enumerate(documents_pretraites):
        #print(f"Document {i + 1} après prétraitement:\n{doc}\n")
        
if __name__ == "__main__":
    main()
