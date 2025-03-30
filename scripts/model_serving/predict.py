import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os

class SentimentPredictor:
    def __init__(self, model_path=None, tokenizer_path=None):
        """
        Initialize the sentiment predictor with a trained model and tokenizer.
        """
        # Get the project root directory (2 levels up from this script)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        
        # Set default paths if not provided
        if model_path is None:
            model_path = os.path.join(project_root, 'models', 'saved', 'sentiment_model')
        if tokenizer_path is None:
            tokenizer_path = os.path.join(project_root, 'models', 'saved', 'sentiment_tokenizer')
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        print(f"Loading tokenizer from: {tokenizer_path}")
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        
        print(f"Loading model from: {model_path}")
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()  # Set model to evaluation mode
        
        print("Model and tokenizer loaded successfully")
    
    def predict(self, text):
        """
        Predict sentiment for a given text.
        Returns a dict with prediction details.
        """
        # Tokenize the input text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Move inputs to the same device as model
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        # Map prediction to label
        label = "positive" if prediction == 1 else "negative"
        
        return {
            "text": text,
            "sentiment": label,
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                "negative": round(probabilities[0][0].item() * 100, 2),
                "positive": round(probabilities[0][1].item() * 100, 2)
            }
        }

# Example usage
if __name__ == "__main__":
    # Create predictor
    predictor = SentimentPredictor()
    
    # Test examples
    examples = [
        "This movie was fantastic! I really enjoyed every moment of it.",
        "The service was terrible and the food was cold.",
        "It was okay, not great but not terrible either."
    ]
    
    for example in examples:
        result = predictor.predict(example)
        print(f"\nText: {result['text']}")
        print(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']}%)")
        print(f"Probabilities: Positive: {result['probabilities']['positive']}%, Negative: {result['probabilities']['negative']}%") 