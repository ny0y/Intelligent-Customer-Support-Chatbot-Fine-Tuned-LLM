# To train a model to predict the intent based on user input.
# Using TF-IDF and a Random Forest
# intents_classifier.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier


"""Trains a model to predict user intent based on input patterns using TF-IDF vectorization and a Random Forest classifier.
Provides functions to train the classifier (train_intent_classifier) and classify new messages (classify_intent)."""


def train_intent_classifier(intents_data):
    patterns = []
    tags = []

    for intent in intents_data['intents']:
        for pattern in intent['patterns']:
            patterns.append(pattern)
            tags.append(intent['tag'])

    # Convert patterns into TF-IDF vectors
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(patterns)

    # Train a classifier
    classifier = RandomForestClassifier()
    classifier.fit(X, tags)

    return vectorizer, classifier


def classify_intent(message, vectorizer, classifier):
    X_test = vectorizer.transform([message])
    prediction = classifier.predict(X_test)[0]
    return prediction