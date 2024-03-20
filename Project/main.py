# import nltk
# nltk.download('stopwords')


# import pandas as pd
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.pipeline import Pipeline
# import hashlib  # Import hashlib library for hashing

# # Load data from CSV files
# train_data = pd.read_csv("train.csv")
# train_labels = pd.read_csv("trainLabels.csv")
# test_data = pd.read_csv("test.csv")

# # Print the shapes of the datasets
# print("Shape of train_data:", train_data.shape)
# print("Shape of train_labels:", train_labels.shape)

# # Hashing the Text data
# def hash_text(text):
#     hashed_text = hashlib.sha256(text.encode('utf-8')).hexdigest()
#     return hashed_text

# train_data["hashed_text"] = train_data["text"].apply(hash_text)
# test_data["hashed_text"] = test_data["text"].apply(hash_text)

# # Text Preprocessing
# # Converting text data to lowercase
# train_data["text"] = train_data["text"].str.lower()
# test_data["text"] = test_data["text"].str.lower()

# # Removing stop words (Optional)
# from nltk.corpus import stopwords
# stop_words = stopwords.words("english")

# def remove_stop_words(text):
#     filtered_text = [word for word in text.split() if word not in stop_words]
#     return " ".join(filtered_text)

# train_data["text"] = train_data["text"].apply(remove_stop_words)
# test_data["text"] = test_data["text"].apply(remove_stop_words)

# # Feature Engineering using Bag of Words
# vectorizer = CountVectorizer(max_features=10000)
# train_features = vectorizer.fit_transform(train_data["text"])
# test_features = vectorizer.transform(test_data["text"])

# # Feature Engineering using TF-IDF
# vectorizer = TfidfVectorizer(max_features=10000)
# train_features = vectorizer.fit_transform(train_data["text"])
# test_features = vectorizer.transform(test_data["text"])

# # Train Test Split
# X_train, X_test, y_train, y_test = train_test_split(train_features, train_labels, test_size=0.2, random_state=42)

# # Machine Learning Models
# # Logistic Regression
# text_classifier = Pipeline([("tfidf", TfidfVectorizer(max_features=10000)), ("clf", LogisticRegression())])
# text_classifier.fit(X_train, y_train)

# # Random Forest Classifier
# text_classifier = Pipeline([("tfidf", TfidfVectorizer(max_features=10000)), ("clf", RandomForestClassifier())])
# text_classifier.fit(X_train, y_train)

# # Model Evaluation
# # You can use metrics like accuracy, precision, recall, F1 score for evaluation

# # Prediction on Test Data
# y_pred = text_classifier.predict(X_test)

# # Saving the Model
# import joblib
# joblib.dump(text_classifier, "text_classifier.pkl")

# # Loading the Model
# loaded_model = joblib.load("text_classifier.pkl")
# predictions = loaded_model.predict(test_features)

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import hashlib

# Download NLTK stopwords
nltk.download('stopwords')

# Load data from CSV files
train_data = pd.read_csv("train.csv")
train_labels = pd.read_csv("trainLabels.csv")
test_data = pd.read_csv("test.csv")

# Print the shapes of the datasets
print("Shape of train_data:", train_data.shape)
print("Shape of train_labels:", train_labels.shape)

# Hashing the Text data
def hash_text(text):
    hashed_text = hashlib.sha256(text.encode('utf-8')).hexdigest()
    return hashed_text

train_data["hashed_text"] = train_data["text"].apply(hash_text)
test_data["hashed_text"] = test_data["text"].apply(hash_text)

# Text Preprocessing
# Convert text to lowercase
train_data["text"] = train_data["text"].str.lower()
test_data["text"] = test_data["text"].str.lower()

# Remove stopwords
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    filtered_text = [word for word in text.split() if word not in stop_words]
    return " ".join(filtered_text)

train_data["text"] = train_data["text"].apply(remove_stopwords)
test_data["text"] = test_data["text"].apply(remove_stopwords)

# Feature Engineering using Bag of Words
vectorizer = CountVectorizer(max_features=10000)
train_features = vectorizer.fit_transform(train_data["text"])
test_features = vectorizer.transform(test_data["text"])

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(train_features, train_labels, test_size=0.2, random_state=42)

# Machine Learning Models
# Logistic Regression
text_classifier_lr = Pipeline([("clf", LogisticRegression())])
text_classifier_lr.fit(X_train, y_train)

# Random Forest Classifier
text_classifier_rf = Pipeline([("clf", RandomForestClassifier())])
text_classifier_rf.fit(X_train, y_train)

# Model Evaluation (Accuracy)
lr_accuracy = text_classifier_lr.score(X_test, y_test)
rf_accuracy = text_classifier_rf.score(X_test, y_test)

print("Logistic Regression Accuracy:", lr_accuracy)
print("Random Forest Accuracy:", rf_accuracy)
