import os
import pandas as pd
import torch
from transformers import BertForSequenceClassification, BertTokenizer, AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tqdm import tqdm
import time
from torch.cuda.amp import GradScaler, autocast
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


def save_model(model, tokenizer, epoch, path):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(os.path.join(path, f'epoch-{epoch}'))
    tokenizer.save_pretrained(os.path.join(path, f'epoch-{epoch}'))


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

    # Normalize the confusion matrix to range from 0 to 100 as percentages
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


def train_model(dataset_path, model_name='bert-base-uncased', batch_size=16, epochs=10, save_dir='bert_fine_tuned_model'):
    # Load dataset
    data = pd.read_csv(dataset_path)
    texts = data['instruction'].values
    labels = data['intent'].astype('category').cat.codes.values

    # Split data into train and test sets
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, stratify=labels)

    tokenizer = BertTokenizer.from_pretrained(model_name)
    train_dataset = IntentDataset(train_texts, train_labels, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights).float().to('cuda')

    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(np.unique(labels)))
    model = model.to('cuda')
    optimizer = AdamW(model.parameters(), lr=2e-5)

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    scaler = GradScaler()

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        correct_train_predictions = 0
        total_train_examples = 0

        start_time = time.time()

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs} - Training", unit="batch"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            labels = batch['labels'].to('cuda')

            with autocast():
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            _, predicted = torch.max(outputs.logits, dim=1)
            correct_train_predictions += (predicted == labels).sum().item()
            total_train_examples += labels.size(0)

        train_accuracy = correct_train_predictions / total_train_examples
        avg_train_loss = total_train_loss / len(train_dataloader)
        epoch_duration = time.time() - start_time
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Time: {epoch_duration:.2f} seconds")

        save_model(model, tokenizer, epoch + 1, save_dir)

    # After training, evaluate the model on the test dataset
    test_dataset = IntentDataset(test_texts, test_labels, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True)
    
    accuracy, conf_matrix = evaluate(model, test_dataloader, 'cuda')
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Plot confusion matrix with the intent classes
    plot_confusion_matrix(conf_matrix, intent_classes)


if __name__ == "__main__":
    dataset_path = 'src/data/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv'
    train_model(dataset_path)
