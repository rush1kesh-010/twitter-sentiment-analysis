import joblib
import re

def load_model(model_path):
    """
    Load the pre-trained Logistic Regression model and vectorizer.
    
    Args:
        model_path (str): Path to the saved model file (.pkl).
    
    Returns:
        tuple: (model, vectorizer)
    """
    model = joblib.load(model_path)
    vectorizer = joblib.load(model_path.replace('logistic_regression_model.pkl', 'vectorizer.pkl'))
    return model, vectorizer

def preprocess_text(text):
    """
    Preprocess input text (e.g., remove URLs, special characters, convert to lowercase).
    
    Args:
        text (str): Input text to preprocess.
    
    Returns:
        str: Cleaned text.
    """
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Convert to lowercase
    text = text.strip().lower()
    return text
