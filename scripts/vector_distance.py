nlp = spacy.load("en_core_web_sm")

with open("christmas-carol.txt", "r", encoding="utf-8") as file:
    text = file.read()

doc = nlp(text)

tokens = [token.text.lower() for token in doc if not token.is_stop and token.is_alpha]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(tokens)
vectors_array = X.toarray()
df = pd.DataFrame(data=vectors_array, columns = vectorizer.get_feature_names_out())
