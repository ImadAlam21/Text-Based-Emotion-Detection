import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
from tqdm import tqdm
import os

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total

def main():
    print("Loading dataset...")
    df = pd.read_csv('emotion_dataset_raw.csv')
    
    # Convert labels to numerical values
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['Emotion'])
    
    # Save label mapping
    label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
    os.makedirs('emotion_model', exist_ok=True)
    with open('emotion_model/label_mapping.json', 'w') as f:
        json.dump(label_mapping, f)
    
    # Split dataset
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['Text'],
        df['label'],
        test_size=0.1,
        random_state=42,
        stratify=df['label']
    )
    
    print("Initializing tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=len(label_mapping)
    )
    
    # Save tokenizer
    tokenizer.save_pretrained('emotion_model')
    
    print("Preparing datasets...")
    train_dataset = EmotionDataset(train_texts.values, train_labels.values, tokenizer)
    val_dataset = EmotionDataset(val_texts.values, val_labels.values, tokenizer)
    
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16)
    
    # Set device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Training settings
    optimizer = AdamW(model.parameters(), lr=2e-5)
    num_epochs = 5
    best_accuracy = 0
    
    print("Training model...")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}:")
        train_loss = train_epoch(model, train_dataloader, optimizer, device)
        print(f"Average training loss: {train_loss:.4f}")
        
        print("Validating...")
        accuracy = evaluate(model, val_dataloader, device)
        print(f"Validation Accuracy: {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print(f"New best model saved with accuracy: {accuracy:.4f}")
            model.save_pretrained('emotion_model')

if __name__ == "__main__":
    main() 