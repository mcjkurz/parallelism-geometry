#!/usr/bin/env python3
"""
Model evaluation for Chinese couplet parallelism detection.
Evaluates trained BERT models on test data.
"""

import argparse
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tqdm.auto import tqdm


class InferenceDataset(Dataset):
    """Dataset class for inference on couplets."""
    
    def __init__(self, couplets, tokenizer, max_length=13):
        self.tokenizer = tokenizer
        self.couplets = couplets
        self.max_length = max_length

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
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }


def load_test_data(test_file):
    """Load test data from JSON file."""
    print(f"Loading test data from {test_file}...")
    
    with open(test_file, "r", encoding='utf-8') as f:
        data = json.load(f)
    
    test_couplets = []
    test_labels = []
    
    for item in data:
        couplet = (item["line1"], item["line2"])
        test_couplets.append(couplet)
        test_labels.append(int(item["label"]))
    
    print(f"Loaded {len(test_couplets)} test examples")
    return test_couplets, test_labels


def inference(couplets, model, tokenizer, batch_size=16, device='cuda', return_probs=False):
    """Run inference on couplets and return predictions."""
    model.eval()
    model.to(device)

    dataset = InferenceDataset(couplets, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    predictions = []
    probabilities = []

    print("Running inference...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            
            batch_predictions = torch.argmax(probs, dim=1).cpu().numpy()
            batch_probs = probs[:, 1].cpu().numpy()

            predictions.extend(batch_predictions)
            probabilities.extend(batch_probs)

    if return_probs:
        return predictions, probabilities
    else:
        return predictions


def evaluate_performance(true_labels, predicted_labels, class_names=None):
    """
    Evaluate model performance with standard classification metrics.
    
    Args:
        true_labels: Ground truth labels
        predicted_labels: Model predictions
        class_names: Optional class names for display
    
    Returns:
        Dictionary containing computed metrics
    """
    if class_names is None:
        class_names = ["Non-parallel", "Parallel"]
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    
    report = classification_report(
        true_labels, predicted_labels, 
        target_names=class_names,
        digits=4
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': report
    }
    
    return metrics


def print_results(results, model_name="Model"):
    """Print evaluation results in a formatted way."""
    print(f"\n{'='*50}")
    print(f"Results for {model_name}")
    print(f"{'='*50}")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1_score']:.4f}")
    print(f"\nDetailed Classification Report:")
    print(results['classification_report'])


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on test data")
    parser.add_argument("-m", "--model", required=True,
                       help="Path to trained model directory")
    parser.add_argument("-i", "--input", required=True,
                       help="Test data JSON file")
    parser.add_argument("-b", "--batch-size", type=int, default=16,
                       help="Inference batch size")
    parser.add_argument("-d", "--device", default="auto",
                       help="Device to use (auto, cuda, cpu)")
    parser.add_argument("-p", "--probs", action="store_true",
                       help="Return prediction probabilities")
    parser.add_argument("-o", "--output", 
                       help="Save predictions to file (optional)")
    
    args = parser.parse_args()
    
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    test_couplets, test_labels = load_test_data(args.input)
    
    print(f"Loading model from {args.model}...")
    try:
        tokenizer = BertTokenizer.from_pretrained(args.model)
        model = BertForSequenceClassification.from_pretrained(args.model, attn_implementation="eager")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure the model path is correct (local path or HuggingFace model name)")
        return
    
    if args.probs:
        predictions, probabilities = inference(
            test_couplets, model, tokenizer, 
            batch_size=args.batch_size, device=device, return_probs=True
        )
    else:
        predictions = inference(
            test_couplets, model, tokenizer,
            batch_size=args.batch_size, device=device, return_probs=False
        )
    
    results = evaluate_performance(test_labels, predictions)
    
    model_name = f"SikuBERT ({args.model})"
    print_results(results, model_name)
    
    if args.output:
        output_data = {
            "predictions": predictions.tolist() if hasattr(predictions, 'tolist') else predictions,
            "true_labels": test_labels
        }
        
        if args.probs:
            output_data["probabilities"] = probabilities.tolist() if hasattr(probabilities, 'tolist') else probabilities
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"Predictions saved to {args.output}")
    
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
