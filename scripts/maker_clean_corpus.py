import os
import spacy

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
    nlp = spacy.load('en_core_news_sm')
    doc = nlp(texte)
    tokens = [token.lemma_ for token in doc if token.text.lower() not in stop_words_spacy and not token.is_punct]

    # Retourner le texte prétraité sous forme de chaîne
    return ' '.join(tokens)

def main():
    path_corpus = 'bla'
    stop_words_spacy = spacy.lang.en.stop_words.STOP_WORDS
    contenu_corpus = lire_contenu_corpus(path_corpus)
    documents_pretraites = [pretraiter_donnees(doc, stop_words_spacy) for doc in contenu_corpus]
    for i, doc in enumerate(documents_pretraites):
        print(f"Document {i + 1} après prétraitement:\n{doc}\n")

if __name__ == "__main__":
    main()
