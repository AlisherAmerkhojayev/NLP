#First code
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

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

# BoW feature extraction
def extract_bow_features(data):
    count_vectorizer = CountVectorizer()
    bow_features = count_vectorizer.fit_transform(data['normalized_message'])
    return bow_features, count_vectorizer

bow_features, count_vectorizer = extract_bow_features(data)

# TF-IDF feature extraction
def extract_tfidf_features(data):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_features = tfidf_vectorizer.fit_transform(data['normalized_message'])
    return tfidf_features, tfidf_vectorizer

tfidf_features, tfidf_vectorizer = extract_tfidf_features(data)

# Encode labels: 'ham' as 0 and 'spam' as 1
label_encoder = LabelEncoder()
data['Category_encoded'] = label_encoder.fit_transform(data['Category'])

# Split data into training and testing sets for each feature type
X_train_bow, X_test_bow, y_train_bow, y_test_bow = train_test_split(bow_features, data['Category_encoded'], test_size=0.2, random_state=42)
X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(tfidf_features, data['Category_encoded'], test_size=0.2, random_state=42)

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
    
    X_train_upsampled = csr_matrix(upsampled_train_data.drop('label', axis=1))  # Convert back to sparse matrix
    y_train_upsampled = upsampled_train_data['label']
    
    return X_train_upsampled, y_train_upsampled

# Upsample training data for BoW features
X_train_bow_upsampled, y_train_bow_upsampled = upsample_data(X_train_bow, y_train_bow)

# Upsample training data for TF-IDF features
X_train_tfidf_upsampled, y_train_tfidf_upsampled = upsample_data(X_train_tfidf, y_train_tfidf)

# Helper function to train and evaluate models
def evaluate_model(model, X_train, X_test, y_train, y_test, dense=False): 
    if dense:  # Convert to dense format for SVM
        X_train = X_train.toarray()
        X_test = X_test.toarray()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    return accuracy, precision, recall, f1, predictions

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "SVM": SVC(kernel='linear')
}

# Features setup
features = {
    "BoW": (X_train_bow_upsampled, X_test_bow, y_train_bow_upsampled, y_test_bow),
    "TF-IDF": (X_train_tfidf_upsampled, X_test_tfidf, y_train_tfidf_upsampled, y_test_tfidf)
}

# Evaluation results
results = []

# Evaluate each model on each feature set
for feature_name, feature_data in features.items():
    X_train, X_test, y_train, y_test = feature_data
    for model_name, model in models.items():
        dense = (model_name == "SVM")  # SVM requires dense input
        accuracy, precision, recall, f1, predictions = evaluate_model(model, X_train, X_test, y_train, y_test, dense)
        results.append({
            "Model": model_name,
            "Features": feature_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Predictions": predictions,
            "True Labels": y_test
        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Display the results table
print(results_df[['Model', 'Features', 'Accuracy', 'Precision', 'Recall', 'F1 Score']])

# Create confusion matrix heatmap for each model and feature set
for index, result in results_df.iterrows():
    model_name = result['Model']
    feature_name = result['Features']
    predictions = result['Predictions']
    y_test = result['True Labels']
    
    conf_matrix = confusion_matrix(y_test, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlGnBu', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.title(f'Confusion Matrix for {model_name} + {feature_name}')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

# Identify misclassified messages for SVM with BoW features
misclassified_indices = np.where(predictions != y_test_bow)[0]
misclassified_test_indices = y_test_bow.index[misclassified_indices]
misclassified_messages = data.iloc[misclassified_test_indices]

misclassified_messages['Predicted Labels'] = predictions[misclassified_indices]

print("Misclassified Messages for SVM + BoW:")
print(misclassified_messages[['Message', 'Category', 'Predicted Labels']])

# Plotting the bar chart for ham and spam messages
label_counts = data['Category'].value_counts()
plt.figure(figsize=(8, 6))
sns.barplot(x=label_counts.index, y=label_counts.values, palette='viridis')
plt.title('Distribution of Ham and Spam Messages')
plt.xlabel('Message Category')
plt.ylabel('Count')
plt.show()
