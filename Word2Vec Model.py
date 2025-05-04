#Second Code

import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from gensim.models import Word2Vec
from scipy.sparse import hstack, csr_matrix
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pyplot as plt

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
data = pd.read_csv(r"C:\Users\Alisher Amer\Downloads\email.csv")

# Clean the data
def clean_message(message):
    message = re.sub(r'<.*?>', '', message)
    message = re.sub(r'[^a-zA-Z\s]', '', message, re.I|re.A)
    message = re.sub(r'\s+', ' ', message).strip()
    return message

data['cleaned_message'] = data['Message'].apply(clean_message)

# Normalize the data
def normalize_message(message):
    tokens = word_tokenize(message.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    stemmer = SnowballStemmer('english')
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    return ' '.join(stemmed_tokens)

data['normalized_message'] = data['cleaned_message'].apply(normalize_message)

# TF-IDF feature extraction
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_tfidf = tfidf_vectorizer.fit_transform(data['normalized_message'])

# Prepare data for Word2Vec
data['tokenized_message'] = data['normalized_message'].apply(word_tokenize)

# Train Word2Vec model
word2vec_model = Word2Vec(sentences=data['tokenized_message'], vector_size=100, window=5, min_count=1, workers=4)
word_vectors = word2vec_model.wv

# Function to create feature vector for each email
def vectorize_text(text, model):
    vector = np.zeros(100)
    count = 0
    for word in text:
        if word in model:
            vector += model[word]
            count += 1
    if count != 0:
        vector /= count
    return vector

# Create feature vectors for the dataset
data['word2vec_vector'] = data['tokenized_message'].apply(lambda x: vectorize_text(x, word_vectors))
X_word2vec = np.vstack(data['word2vec_vector'])

# Standardize the Word2Vec features
scaler = StandardScaler()
X_word2vec_scaled = scaler.fit_transform(X_word2vec)

# Combine TF-IDF and Word2Vec features
X_combined = hstack([X_tfidf, csr_matrix(X_word2vec_scaled)])

# Encode labels
label_encoder = LabelEncoder()
data['labels'] = label_encoder.fit_transform(data['Category'])
y = data['labels']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Upsample the minority class in the training data
def upsample_data(X_train, y_train):
    X_train_np = X_train.toarray()
    train_data = pd.DataFrame(X_train_np)
    train_data['label'] = y_train.values
    minority_class = train_data['label'].value_counts().idxmin()
    majority_class = train_data['label'].value_counts().idxmax()
    
    minority_data = train_data[train_data['label'] == minority_class]
    majority_data = train_data[train_data['label'] == majority_class]
    
    minority_upsampled = resample(minority_data,
                                  replace=True,
                                  n_samples=len(majority_data),
                                  random_state=42)
    
    upsampled_train_data = pd.concat([majority_data, minority_upsampled])
    
    X_train_upsampled = csr_matrix(upsampled_train_data.drop('label', axis=1))
    y_train_upsampled = upsampled_train_data['label']
    
    return X_train_upsampled, y_train_upsampled

# Upsample the training data
X_train_upsampled, y_train_upsampled = upsample_data(X_train, y_train)

# Initialize and tune SVM model
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['linear', 'rbf']
}

svm = SVC()
grid = GridSearchCV(svm, param_grid, refit=True, verbose=2, cv=5)
grid.fit(X_train_upsampled, y_train_upsampled)

# Best parameters
best_svm = grid.best_estimator_

# Predictions and evaluation
y_pred = best_svm.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlGnBu', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

