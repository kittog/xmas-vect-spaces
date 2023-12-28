import spacy
from gensim.models import Word2Vec
import re

nlp = spacy.load("en_core_web_sm")

with open("christmas-carol.txt", "r", encoding="utf-8") as file:
    text = file.read()

doc = nlp(text)

tokens = [token.text.lower() for token in doc if not token.is_stop and token.is_alpha]

model = Word2Vec([tokens], vector_size=100, window=5, min_count=1, workers=4)

model.save("word2vec_model")

vector = model.wv["christmas"]
print("Vecteur pour 'Christmas':", vector)

similar_words = model.wv.most_similar("christmas", topn=6)  
similar_words = [(word, score) for word, score in similar_words if word.isalpha()]

print("Les 5 voisins de 'Christmas' sont :")
for word, score in similar_words[1:]:  
    print(f"{word}: {score}")
