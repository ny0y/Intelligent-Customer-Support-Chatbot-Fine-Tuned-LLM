import pandas as pd
import logging
from distailbert.distailbert_classifier import DistilBertClassifier
import random
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DistailBERTChatbotService:
    def __init__(self, dataset: str, model_name: str = 'bert-base-uncased', fine_tuned_model_path: str = None):
        self.dataset = self.load_dataset(dataset)
        self.classifier = DistilBertClassifier(model_name, self.dataset, fine_tuned_model_path)
        logger.info("Dataset loaded successfully.")
        logger.info("Dataset head:\n%s", self.dataset.head())
        logger.debug(f"Dataset structure: {self.dataset.columns.tolist()}")

    def load_dataset(self, path: str) -> pd.DataFrame:
        try:
            dataset = pd.read_csv(path)
            logger.info("Dataset loaded from %s", path)
            return dataset
        except Exception as e:
            logger.error("Failed to load dataset from %s: %s", path, str(e))
            raise

    def get_response(self, user_input: str) -> str:
        try:
            intent = self.classifier.classify_intent(user_input)
            response = self._generate_response_based_on_intent(intent)
            logger.info("Generated response: %s", response)
            return response
        except Exception as e:
            logger.error("Error generating response: %s", e)
            return "Sorry, something went wrong."

    def _generate_response_based_on_intent(self, intent: str) -> str:
        """
        Generate a response based on the predicted intent.
        First, attempt to generate from the dataset. If no match, use predefined responses.
        Args:
           intent (str): The predicted intent.
        Returns:
            str: The response from the dataset or a predefined response.
        """
        logger.debug(f"Generating response for intent: {intent}")  # Debug print statement

        # Step 1: Try to generate response from the dataset
        matching_rows = self.dataset[self.dataset['intent'] == intent]

        if not matching_rows.empty:
            # Randomly select a response from matching rows if multiple exist
            return random.choice(matching_rows['response'].tolist())

        # Step 2: Fallback to predefined responses if dataset doesn't provide a match
        predefined_responses = {
            "cancel_order": "I can help you with your order cancellation.",
            "change_address": "I can assist you with updating your address.",
            "termination_charges": "I can help you understand the early termination charges.",
        }
        
        # Return the predefined response if available, else a default error message
        return predefined_responses.get(intent, "I'm sorry, I didn't understand your request.")
