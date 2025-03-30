import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import json

def compute_metrics(pred):
    """
    Compute metrics for model evaluation.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_model():
    print("Starting model training...")
    
    # Create model directory
    os.makedirs("models/saved", exist_ok=True)
    
    # Load processed data
    train_df = pd.read_csv("data/processed/imdb_train_processed.csv")
    test_df = pd.read_csv("data/processed/imdb_test_processed.csv")
    
    # For quick testing, limit the dataset size
    # Remove these lines for full training
    train_df = train_df.sample(1000, random_state=42)
    test_df = test_df.sample(200, random_state=42)
    
    # Convert sentiment to numeric labels
    label_map = {'positive': 1, 'negative': 0}
    train_df['labels'] = train_df['sentiment'].map(label_map).astype('int64')
    test_df['labels'] = test_df['sentiment'].map(label_map).astype('int64')
    
    # Keep only necessary columns
    train_df = train_df[['text', 'labels']]
    test_df = test_df[['text', 'labels']]
    
    # Convert to Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # Load tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=2,
        problem_type="single_label_classification"
    )
    
    # Tokenize data
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=512
        )
    
    # Tokenize and format datasets
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )
    test_dataset = test_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )
    
    # Set format for pytorch
    train_dataset.set_format('torch')
    test_dataset.set_format('torch')
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir='models/saved/results',
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='models/saved/logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train model
    print("Training model...")
    trainer.train()
    
    # Evaluate model
    print("Evaluating model...")
    eval_result = trainer.evaluate()
    
    # Save model
    print("Saving model...")
    model.save_pretrained("models/saved/sentiment_model")
    tokenizer.save_pretrained("models/saved/sentiment_tokenizer")
    
    # Save evaluation results
    with open("models/saved/eval_results.json", "w") as f:
        json.dump(eval_result, f, indent=2)
    
    print(f"Training complete. Model saved to models/saved/")
    print(f"Evaluation results: {eval_result}")

if __name__ == "__main__":
    train_model() 