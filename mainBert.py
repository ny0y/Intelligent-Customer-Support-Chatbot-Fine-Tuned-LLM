from flask import Flask, request, jsonify
from bert.bert_chatbot_service import BERTChatbotService
import logging
import nltk

app = Flask(__name__)

# Use absolute path for the dataset
dataset_path = ('src/data/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv')

fine_tuned_model_path = 'src/bert/bert_fine_tuned_model/best_model'

bert_service = BERTChatbotService(dataset_path, fine_tuned_model_path)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.route('/chat/bert', methods=['POST'])
def bert():
    try:
        data = request.json
        if 'message' not in data:
            return jsonify({'error': 'Message field is required'}), 400

        user_input = data['message']
        logger.info(f"Received message: {user_input}")

        response = bert_service.get_response(user_input)
        logger.info(f"Generated response: {response}")

        return jsonify({'response': response})

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        return jsonify({'error': str(e)}), 500


def setup_nltk():
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')


if __name__ == "__main__":
    setup_nltk()
    app.run(debug=True)
