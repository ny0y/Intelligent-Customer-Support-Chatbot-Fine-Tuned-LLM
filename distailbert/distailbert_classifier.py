import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DistilBertClassifier:
    def __init__(self, model_name: str, dataset, fine_tuned_model_path: str = None):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)

        # Load fine-tuned model if available
        if fine_tuned_model_path:
            self.model = DistilBertForSequenceClassification.from_pretrained(fine_tuned_model_path)
            logger.info("Loaded fine-tuned model from %s", fine_tuned_model_path)
        else:
            self.model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=len(dataset['intent'].unique()))
            logger.info("Loaded pre-trained BERT model: %s", model_name)

        self.intents = dataset['intent'].unique().tolist()
        self.confidence_threshold = 0.5  # Minimum confidence threshold for predictions

    def tokenize_input(self, text: str) -> dict:
        return self.tokenizer(text, padding='max_length', truncation=True, return_tensors='pt')

    def classify_intent(self, text: str) -> str:
        try:
            inputs = self.tokenize_input(text)
            self.model.eval()

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            probabilities = F.softmax(logits, dim=-1)
            max_prob = torch.max(probabilities).item()
            predicted_class = torch.argmax(probabilities, dim=-1).item()

            if max_prob < self.confidence_threshold:
                logger.warning("Low confidence (%f) for intent classification, returning 'unknown_intent'", max_prob)
                return "unknown_intent"

            return self.intents[predicted_class] if predicted_class < len(self.intents) else "unknown_intent"

        except Exception as e:
            logger.error("Error classifying intent: %s", e)
            return "error"
