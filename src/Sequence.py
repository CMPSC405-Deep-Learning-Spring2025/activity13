from collections import defaultdict

# Sample text corpus
text = "I love deep learning. I love AI. I love PyTorch. AI is the future."

def build_ngram_model(text, n):
    tokens = text.split()
    model = defaultdict(lambda: defaultdict(int))

    for i in range(len(tokens) - n + 1):
        context = tuple(tokens[i:i + n - 1])
        next_word = tokens[i + n - 1]
        model[context][next_word] += 1

    return model

# Build unigram, bigram, and trigram models
unigram_model = build_ngram_model(text, 1)
bigram_model = build_ngram_model(text, 2)
trigram_model = build_ngram_model(text, 3)

def predict_next_word(model, context):
    next_word_probs = model.get(context, {})
    return max(next_word_probs, key=next_word_probs.get) if next_word_probs else None

# Test predictions
print("Unigram prediction: ", predict_next_word(unigram_model, ()))
print("Bigram prediction ('I', ?): ", predict_next_word(bigram_model, ("I",)))
print("Trigram prediction ('I', 'love', ?): ", predict_next_word(trigram_model, ("I", "love")))
