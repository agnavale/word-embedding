import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# Example continuous text
text = '''Machine learning is the study of computer algorithms that \
improve automatically through experience. It is seen as a \
subset of artificial intelligence. Machine learning algorithms \
build a mathematical model based on sample data, known as \
training data, in order to make predictions or decisions without \
being explicitly programmed to do so. Machine learning algorithms \
are used in a wide variety of applications, such as email filtering \
and computer vision, where it is difficult or infeasible to develop \
conventional algorithms to perform the needed tasks.'''

# Define delimiters
delimiters = [".", ",", ";", ":", "?", "!", "-", "'", '"', "(", ")", "{", "}", "[", "]"]

# Remove delimiters
for delimiter in delimiters:
    text = text.replace(delimiter, " ")

# Tokenize the text into words
corpus = text.lower().split()

# Define stopwords
stopwords = set([
    'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there',
    'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an',
    'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself',
    'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the',
    'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 
    'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 
    'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 
    'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 
    'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why',
    'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where',
    'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if',
    'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'
])

# Remove stopwords from the corpus
corpus = [word for word in corpus if word not in stopwords]

# Create word vocabulary
word_vocab = set(corpus)
word2idx = {word: idx for idx, word in enumerate(word_vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}

# Generate training data
window_size = 5
train_data = []
for idx, word in enumerate(corpus):
    for neighbor in corpus[max(0, idx - window_size): min(len(corpus), idx + window_size + 1)]:
        if neighbor != word:
            train_data.append((word, neighbor))

# Convert training data into one-hot encoded vectors
X_train = []
y_train = []
for data in train_data:
    X_train.append(to_categorical(word2idx[data[0]], num_classes=len(word_vocab)))
    y_train.append(to_categorical(word2idx[data[1]], num_classes=len(word_vocab)))

X_train = np.array(X_train)
y_train = np.array(y_train)

# Define Word2Vec model
embedding_dim = 64
model = Sequential([
    Dense(units=embedding_dim, input_shape=(len(word_vocab),),use_bias=False),
    Dense(units=len(word_vocab), activation='softmax',use_bias=False)
])

# Compile and train the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100)


# def plot_word_vectors(word_vocab, word_embeddings):
#     pca = PCA(n_components=2)
#     word_embeddings_pca = pca.fit_transform(word_embeddings)

#     plt.figure(figsize=(10, 10))
#     for idx, word in enumerate(word_vocab):
#         plt.scatter(word_embeddings_pca[idx, 0], word_embeddings_pca[idx, 1])
#         plt.annotate(word, (word_embeddings_pca[idx, 0], word_embeddings_pca[idx, 1]))

#     plt.xlabel('Principal Component 1')
#     plt.ylabel('Principal Component 2')
#     plt.title('Word Vectors Visualization')
#     plt.show()

# # Get word embeddings
# word_embeddings = model.layers[0].get_weights()[0]

# # Test the function
# plot_word_vectors(word_vocab, word_embeddings)

def predict_closest_words(input_word, top_n=5):
    input_word = input_word.lower()
    if input_word not in word2idx:
        print("Input word not found in the vocabulary.")
        return
    input_idx = word2idx[input_word]
    input_vector = np.zeros((1, len(word_vocab)))
    input_vector[0, input_idx] = 1  # One-hot encode the input word

    predicted_probabilities = model.predict(input_vector)[0]
    
    # Sort the predicted probabilities to find the closest words
    closest_indices = np.argsort(predicted_probabilities)[-top_n:]
    closest_words = [idx2word[idx] for idx in closest_indices]
    return closest_words[::-1]  # Reverse the list to get the highest probabilities first

# Test the function
input_word = 'machine'
top_n = 5
closest_words = predict_closest_words(input_word, top_n)
print(f"The top {top_n} closest words to '{input_word}' are: {closest_words}")
