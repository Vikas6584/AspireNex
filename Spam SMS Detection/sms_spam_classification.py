import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from nltk.corpus import stopwords  # Import stopwords for preprocessing

# Load data (replace with your file path)
data = pd.read_csv('spam.csv')

# Separate messages and labels
messages = data['message'].tolist()
labels = data['label'].tolist()  # Assuming 'label' column has spam/legitimate labels

# Preprocessing (replace with your custom cleaning steps)
def preprocess_text(text):
  text = text.lower()  # Convert to lowercase
  text = ''.join([char for char in text if char.isalnum() or ' ' in char])  # Remove special characters
  stop_words = stopwords.words('english')  # Download stopwords if needed
  text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
  return text

messages_cleaned = [preprocess_text(text) for text in messages]

# Feature Extraction (choose either TF-IDF or Word Embeddings)

# Option A: TF-IDF
def extract_features_tfidf(messages):
  vectorizer = TfidfVectorizer(max_features=2000)  # Adjust max_features as needed
  features = vectorizer.fit_transform(messages)
  return features

tfidf_features = extract_features_tfidf(messages_cleaned)

# Option B: Word Embeddings (replace with your implementation)
# You'll need to load pre-trained word embeddings (e.g., Word2Vec, GloVe)
# This example assumes a function `get_word_embedding(word)` that returns the embedding vector for a word

def extract_features_word2vec(messages, embedding_dim):
  embeddings = []
  for message in messages:
    message_vec = np.zeros(embedding_dim)  # Replace with actual word embedding vectors
    for word in message.split():
      word_vec = get_word_embedding(word)  # Replace with your function call
      if word_vec is not None:
        message_vec += word_vec
    if len(message.split()) > 0:
      message_vec /= len(message.split())  # Average word vectors
    embeddings.append(message_vec)
  return np.array(embeddings)

# word2vec_features = extract_features_word2vec(messages_cleaned, embedding_dim=300)  # Adjust embedding dimension

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(tfidf_features, labels, test_size=0.2)

# Function to evaluate and print model performance
def evaluate_model(model, X_test, y_test):
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
  accuracy = accuracy_score(y_test, y_pred)
  precision = precision_score(y_test, y_pred)
  recall = recall_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred)
  print(f"\n{model.__class__.__name__} Results:")
  print(f"Accuracy: {accuracy:.4f}")
  print(f"Precision: {precision:.4f}")
  print(f"Recall: {recall:.4f}")
  print(f"F1 Score: {f1:.4f}")

# Evaluate different models (replace word2vec_features with your implementation if using)
evaluate_model(MultinomialNB(), X_train, y_train)
evaluate_model(LogisticRegression(solver='lbfgs'), X_train, y_train)  # Adjust solver parameter as needed
evaluate_model(SVC(), X_train, y_train)
# evaluate_model(SVC(), X_train, y_train, word2vec_features)  # Uncomment if using word embeddings