#!/usr/bin/env python3
"""
BERT fine-tuning for Chinese couplet parallelism detection.
Fine-tunes SikuBERT on parallel/non-parallel couplet classification.
"""

import argparse
import json
import random
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm.auto import tqdm

class CoupletsDataset(Dataset):
    """Dataset class for Chinese couplets."""
    
    def __init__(self, positive, negative, tokenizer, line_length=5):
        self.tokenizer = tokenizer
        self.couplets = positive + negative
        self.labels = [1] * len(positive) + [0] * len(negative)
        self.line_length = line_length
        self.max_length = 2 * line_length + 3  # [CLS] line1 [comma] line2 [SEP]

        combined = list(zip(self.couplets, self.labels))
        random.shuffle(combined)
        self.couplets, self.labels = zip(*combined)

    def __len__(self):
        return len(self.couplets)

    def __getitem__(self, idx):
        line = self.couplets[idx][0] + "，" + self.couplets[idx][1]
        
        encoding = self.tokenizer(
            line, 
            add_special_tokens=True,
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def load_dataset(dataset_file):
    """Load training dataset from JSON file."""
    with open(dataset_file, "r", encoding='utf-8') as f:
        data = json.load(f)
    
    parallel_couplets = []
    nonparallel_couplets = []
    
    for item in data:
        couplet = (item["line1"], item["line2"])
        if item["label"] == 1:
            parallel_couplets.append(couplet)
        else:
            nonparallel_couplets.append(couplet)
    
    return parallel_couplets, nonparallel_couplets


def create_data_loaders(parallel_couplets, nonparallel_couplets, tokenizer, 
                       batch_size=16, val_split=0.05, seed=42):
    """Create training and validation data loaders."""
    random.seed(seed)
    torch.manual_seed(seed)
    
    dataset = CoupletsDataset(parallel_couplets, nonparallel_couplets, tokenizer)
    print(f"Total dataset size: {len(dataset)}")
    
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Train size: {train_size}, Validation size: {val_size}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def validate_model(model, val_loader, loss_function, device):
    """Validate the model and return average loss."""
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()
    
    return val_loss / len(val_loader)


def train_model(model, train_loader, val_loader, optimizer, loss_function, 
               device, tokenizer, epochs=1, patience=2, save_dir="models", 
               val_check_interval=1000, max_batches=None):
    """Train the model with early stopping."""
    os.makedirs(save_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    batch_num = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        train_iterator = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}")
        
        for batch in train_iterator:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            train_iterator.set_postfix(loss=loss.item())
            
            if batch_num % val_check_interval == 0 and batch_num > 0:
                avg_val_loss = validate_model(model, val_loader, loss_function, device)
                print(f"\nBatch {batch_num} - Validation Loss: {avg_val_loss:.4f}")
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    
                    model_path = Path(save_dir) / f"sikubert-parallelism-best"
                    model.save_pretrained(model_path)
                    tokenizer.save_pretrained(model_path)
                    print(f"New best model saved to {model_path}")
                    
                else:
                    patience_counter += 1
                    print(f"No improvement. Patience: {patience_counter}/{patience}")
                    
                    if patience_counter >= patience:
                        print("Early stopping triggered!")
                        return model
                
                model.train()
            
            batch_num += 1
            
            if max_batches is not None and batch_num >= max_batches:
                print(f"\nReached maximum batch limit ({max_batches}). Stopping training.")
                return model
        
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Average Training Loss: {avg_train_loss:.4f}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Fine-tune BERT for couplet parallelism detection")
    parser.add_argument("-i", "--input", required=True,
                       help="Training dataset JSON file")
    parser.add_argument("-m", "--model", required=True,
                       help="Pre-trained model name or path")
    parser.add_argument("-o", "--output", required=True,
                       help="Directory to save trained models")
    parser.add_argument("-b", "--batch-size", type=int, default=16,
                       help="Training batch size")
    parser.add_argument("-l", "--lr", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("-e", "--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("-v", "--val-split", type=float, default=0.05,
                       help="Validation split ratio")
    parser.add_argument("-p", "--patience", type=int, default=0,
                       help="Early stopping patience")
    parser.add_argument("--val-interval", type=int, default=1000,
                       help="Validation check interval (batches)")
    parser.add_argument("--max-batches", type=int, default=None,
                       help="Maximum number of batches to train (for early stopping)")
    parser.add_argument("-s", "--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading training dataset...")
    parallel_couplets, nonparallel_couplets = load_dataset(args.input)
    
    print(f"Loading tokenizer and model: {args.model}")
    tokenizer = BertTokenizer.from_pretrained(args.model)
    model = BertForSequenceClassification.from_pretrained(
        args.model, 
        num_labels=2,
        attn_implementation="eager"
    ).to(device)
    
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        parallel_couplets, nonparallel_couplets, tokenizer,
        batch_size=args.batch_size, val_split=args.val_split, seed=args.seed
    )
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_function = nn.CrossEntropyLoss()
    
    print("Starting training...")
    trained_model = train_model(
        model, train_loader, val_loader, optimizer, loss_function,
        device, tokenizer, epochs=args.epochs, patience=args.patience,
        save_dir=args.output, val_check_interval=args.val_interval,
        max_batches=args.max_batches
    )
    
    final_model_path = Path(args.output) / "sikubert-parallelism-final"
    trained_model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    print("Training complete!")


if __name__ == "__main__":
    main()
