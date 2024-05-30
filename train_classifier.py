import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
import numpy as np
import os
from tqdm import tqdm

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

def load_data(filepath):
    """Loads the CSV file containing the classed rows."""
    df = pd.read_csv(filepath)
    assert 'extracted_content' in df.columns and 'class' in df.columns, "CSV must contain 'extracted_content' and 'class' columns."
    return df

def preprocess_data(df):
    """Preprocess the data by encoding labels and tokenizing text."""
    # Encode the class labels
    label_encoder = LabelEncoder()
    df['class_label'] = label_encoder.fit_transform(df['class'])
    return df, label_encoder

def tokenize_data(texts, tokenizer):
    """Tokenize the text data using BERT tokenizer."""
    return tokenizer(texts.tolist(), truncation=True, padding=True, max_length=512)

def train(model, train_loader, val_loader, epochs=3):
    """Train the BERT model."""
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        total_train_accuracy = 0

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
            optimizer.zero_grad()

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            total_train_loss += loss.item()
            _, preds = torch.max(logits, dim=1)
            total_train_accuracy += (preds == batch['labels']).sum().item()

            loss.backward()
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_accuracy = total_train_accuracy / len(train_loader.dataset)

        print(f"Epoch {epoch + 1}: Train loss = {avg_train_loss:.3f}, Train accuracy = {avg_train_accuracy:.3f}")

        model.eval()
        total_val_loss = 0
        total_val_accuracy = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                logits = outputs.logits

                total_val_loss += loss.item()
                _, preds = torch.max(logits, dim=1)
                total_val_accuracy += (preds == batch['labels']).sum().item()

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_accuracy = total_val_accuracy / len(val_loader.dataset)

        print(f"Epoch {epoch + 1}: Val loss = {avg_val_loss:.3f}, Val accuracy = {avg_val_accuracy:.3f}")

def evaluate(model, test_loader):
    """Evaluate the BERT model."""
    model.eval()
    total_test_accuracy = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits

            _, preds = torch.max(logits, dim=1)
            total_test_accuracy += (preds == batch['labels']).sum().item()

    avg_test_accuracy = total_test_accuracy / len(test_loader.dataset)
    print(f"Test accuracy = {avg_test_accuracy:.3f}")

def main(input_csv, output_dir, epochs=3):
    # Load the data
    df = load_data(input_csv)
    
    # Preprocess the data
    df, label_encoder = preprocess_data(df)
    
    # Split the data into training, validation, and test sets
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # Initialize the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize the data
    train_encodings = tokenize_data(train_df['extracted_content'], tokenizer)
    val_encodings = tokenize_data(val_df['extracted_content'], tokenizer)
    test_encodings = tokenize_data(test_df['extracted_content'], tokenizer)

    # Create datasets
    train_dataset = CustomDataset(train_encodings, train_df['class_label'].values)
    val_dataset = CustomDataset(val_encodings, val_df['class_label'].values)
    test_dataset = CustomDataset(test_encodings, test_df['class_label'].values)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Initialize the model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))
    model.to(device)

    # Train the model
    train(model, train_loader, val_loader, epochs=epochs)

    # Evaluate the model
    evaluate(model, test_loader)

    # Save the model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model and tokenizer saved to {output_dir}")

if __name__ == "__main__":
    input_csv = 'dataset_small_only_classed.csv'  # Path to the CSV file with classed rows
    output_dir = 'bert_model'  # Directory to save the trained model and tokenizer
    main(input_csv, output_dir, epochs=3)