#Third Code

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch

# Load dataset
data = pd.read_csv(r"C:\Users\Alisher Amer\Downloads\email.csv")

# Encode labels
label_encoder = LabelEncoder()
data['labels'] = label_encoder.fit_transform(data['Category'])

# Split data into training and testing
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Determine the minority class
class_counts = train_data['labels'].value_counts()
minority_class = class_counts.idxmin()
majority_class = class_counts.idxmax()

# Upsample the minority class in the training data
minority_data = train_data[train_data['labels'] == minority_class]
majority_data = train_data[train_data['labels'] == majority_class]
minority_upsampled = resample(minority_data,
                              replace=True,
                              n_samples=len(majority_data),
                              random_state=42)

# Combine majority data with upsampled minority data
upsampled_train_data = pd.concat([majority_data, minority_upsampled])

# Initialize tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Function to encode the texts
def tokenize_function(texts):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=512, return_tensors="pt")

# Prepare data for tokenization
train_texts = upsampled_train_data['Message'].tolist()
train_labels = upsampled_train_data['labels'].tolist()
test_texts = test_data['Message'].tolist()
test_labels = test_data['labels'].tolist()

# Apply tokenization
train_encodings = tokenize_function(train_texts)
test_encodings = tokenize_function(test_texts)

# Define a dataset class for PyTorch (adapted from Hugging Face)
class SpamDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create datasets
train_dataset = SpamDataset(train_encodings, train_labels)
test_dataset = SpamDataset(test_encodings, test_labels)

# Metrics computation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='binary')
    recall = recall_score(labels, predictions, average='binary')
    f1 = f1_score(labels, predictions, average='binary')
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"
)

# Initialize the Trainer
trainer = Trainer(
    model=DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(label_encoder.classes_)),
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model after training
evaluation_results = trainer.evaluate()
print("Post-Training Evaluation Results:")
print(evaluation_results)

# Make predictions on the test dataset
predictions = trainer.predict(test_dataset)

# Extract the logits and calculate the predicted labels
logits = predictions.predictions
predicted_labels = np.argmax(logits, axis=-1)

# Compare predicted labels with true labels to find misclassifications
misclassified_indices = np.where(predicted_labels != test_labels)[0]
misclassified_emails = test_data.iloc[misclassified_indices]

# Display the misclassified emails
print("Misclassified Emails:")
print(misclassified_emails[['Message', 'Category', 'labels']])

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Predict on the test set
predictions, labels, _ = trainer.predict(test_dataset)
predictions = np.argmax(predictions, axis=-1)

# Generate confusion matrix
conf_matrix = confusion_matrix(labels, predictions)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlGnBu', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()