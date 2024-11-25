import os
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

intent_classes = [
    "cancel_order", "change_order", "change_shipping_address", "check_cancellation_fee",
    "check_invoice", "check_payment_methods", "check_refund_policy", "complaint",
    "contact_customer_service", "contact_human_agent", "create_account", "delete_account",
    "delivery_options", "delivery_period", "edit_account", "get_invoice", "get_refund",
    "newsletter_subscription", "payment_issue", "place_order", "recover_password",
    "registration_problems", "review", "set_up_shipping_address", "switch_account",
    "track_order", "track_refund"
]


class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def train(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_train_loss = 0
    correct_train_predictions = 0
    total_train_examples = 0

    for batch in tqdm(dataloader, desc="Training", unit="batch"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        _, predicted = torch.max(outputs.logits, dim=1)
        correct_train_predictions += (predicted == labels).sum().item()
        total_train_examples += labels.size(0)

    train_accuracy = correct_train_predictions / total_train_examples
    avg_train_loss = total_train_loss / len(dataloader)

    return avg_train_loss, train_accuracy


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # confusion matrix to range from 0 to 100 as percentages
    conf_matrix = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

    return accuracy, conf_matrix


def plot_confusion_matrix(conf_matrix, intent_classes):
    plt.figure(figsize=(12, 10))  # Adjust size for better readability
    sns.set(font_scale=1.4)  # Adjust font size

    # Create a heatmap
    ax = sns.heatmap(conf_matrix, annot=True, fmt=".0f", cmap='Blues',
                     square=True, cbar_kws={'shrink': 0.7},
                     linewidths=0.5, linecolor='gray')

    # Set axis labels and title
    plt.xlabel('Predicted Labels', fontsize=16)
    plt.ylabel('True Labels', fontsize=16)
    plt.title('Confusion Matrix', fontsize=20)

    # Set tick labels
    ax.set_xticklabels(intent_classes, rotation=45, ha='right', fontsize=12)
    ax.set_yticklabels(intent_classes, rotation=0, fontsize=12)

    plt.tight_layout()
    plt.show()


def main(dataset_path, model_name='distilbert-base-uncased', batch_size=32,
         epochs=10, save_dir='src/Distailbert/distilbert_fine_tuned_model/best_model'):

    # Load and prepare dataset
    data = pd.read_csv(dataset_path)
    texts = data['instruction'].values
    labels = data['intent'].astype('category').cat.codes.values

    # Initialize tokenizer and dataset
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    dataset = IntentDataset(texts, labels, tokenizer)

    # Split data into train and test sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                [train_size, test_size])

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True)

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights).float().to('cuda')

    # Initialize model, optimizer, and loss function
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=len(intent_classes))
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Training loop
    for epoch in range(epochs):
        avg_train_loss, train_accuracy = train(model, train_dataloader, optimizer, loss_fn, device)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

    # Evaluate the model
    accuracy, conf_matrix = evaluate(model, test_dataloader, device)
    print(f"Test Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")

    # Plot confusion matrix with the intent classes
    plot_confusion_matrix(conf_matrix, intent_classes)  # Pass intent_classes here

    # Save the model
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it does not exist
    model.save_pretrained(save_dir)  # Save the model
    tokenizer.save_pretrained(save_dir)  # Save the tokenizer as well


if __name__ == "__main__":
    dataset_path = 'src/data/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv'
    main(dataset_path)
