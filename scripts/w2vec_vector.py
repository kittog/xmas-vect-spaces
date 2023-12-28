import spacy
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
import re

nlp = spacy.load("en_core_web_sm")

with open("christmas-carol.txt", "r", encoding="utf-8") as file:
    text = file.read()

text = re.sub(r"[^\w\s]", "", text)

doc = nlp(text)

tokens = [token.text for token in doc]

model = Word2Vec([tokens], vector_size=100, window=5, min_count=1, workers=4)
print(model)
model.save("word2vec_model")

vector = model.wv["Christmas"]
print("christmas",vector)
