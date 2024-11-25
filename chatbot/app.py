from flask import Flask, request, jsonify
from chatbot import chatbot_response, load_intents
import nltk

# Set up the Flask web server.
app = Flask(__name__)

# Path to the dataset CSV file for the BERT service
dataset_path = ('src/data/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv')


# Load the intents data for the non-BERT chatbot once when the app starts
intents_data = load_intents()


# Define an endpoint for the traditional chatbot
@app.route('/chat/chatbot', methods=['POST'])
def chat():
    data = request.json
    if 'message' not in data:
        return jsonify({'error': 'Message field is required'}), 400
    message = data['message']

    response = chatbot_response(message, intents_data)
    return jsonify({'response': response})


# Setup NLTK function to download necessary NLTK resources
def setup_nltk():
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')


# Run the Flask web server if executed as a script
if __name__ == "__main__":
    setup_nltk()  # Ensure NLTK resources are downloaded
    app.run(debug=True)  # Set debug=False for production
