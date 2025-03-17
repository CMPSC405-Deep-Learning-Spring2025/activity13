#!pip install gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import nltk
nltk.download('punkt_tab')

# Sample text corpus
corpus = [
    "Deep learning is transforming artificial intelligence.",
    "Neural networks power modern natural language processing.",
    "Recurrent neural networks are useful for sequential data.",
    "Convolutional networks are effective for images."
]

# Tokenize sentences
tokenized_corpus = [word_tokenize(sent.lower()) for sent in corpus]

# Train word2vec model
model = Word2Vec(sentences=tokenized_corpus, vector_size=50, window=5, min_count=1, workers=4)

print(model.wv.most_similar("deep"))
print(model.wv.similarity("learning", "networks"))

words = list(model.wv.key_to_index)
X = model.wv[words]

# Reduce dimensions with PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot embeddings
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1])

for i, word in enumerate(words):
    plt.annotate(word, (X_pca[i, 0], X_pca[i, 1]))

plt.show()

