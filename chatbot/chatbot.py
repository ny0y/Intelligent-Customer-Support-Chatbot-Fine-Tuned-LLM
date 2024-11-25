import json
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string


def load_intents():
    with open('src/data/intents.json') as file:
        data = json.load(file)
    return data


nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


# Initialize the lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    # Tokenize the sentence
    tokens = nltk.word_tokenize(text)

    # Lemmatize and remove punctuation & stopwords
    tokens = [lemmatizer.lemmatize(token.lower())
              for token in tokens
              if token not in string.punctuation and token.lower()
              not in stop_words]

    return tokens


def chatbot_response(message, intents_data):
    processed_message = preprocess_text(message)  # Preprocess the user input

    # Loop through the intents to find a match
    for intent in intents_data['intents']:
        for pattern in intent['patterns']:
            processed_pattern = preprocess_text(pattern)  # Preprocess the pattern

            # Check if all tokens from the user message are in the pattern
            if all(token in processed_pattern for token in processed_message):
                return intent['tag']  # Return the intent tag as the prediction

    return "unknown"  # If no intent matches, return "unknown"
