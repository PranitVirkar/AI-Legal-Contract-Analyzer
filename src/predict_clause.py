import torch
import joblib
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

# CONFIGURATION

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLASSIFIER_PATH = PROJECT_ROOT / "models" / "clause_classifier.pkl"
LABEL_ENCODER_PATH = PROJECT_ROOT / "models" / "label_encoder.pkl"


# LOAD MODELS


print("Loading Legal-BERT...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

bert_model = AutoModel.from_pretrained(MODEL_NAME)

bert_model.eval()

print("Loading trained classifier...")

classifier = joblib.load(CLASSIFIER_PATH)

label_encoder = joblib.load(LABEL_ENCODER_PATH)

print("Models loaded successfully!")


# CREATE EMBEDDING


def create_embedding(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

    with torch.no_grad():

        outputs = bert_model(
            **inputs
        )

    # CLS token embedding
    embedding = outputs.last_hidden_state[:, 0, :]

    return embedding.squeeze().numpy()


# PREDICT CLAUSE


def predict_clause(text):

    embedding = create_embedding(text)

    # Convert single embedding into 2D array
    embedding = embedding.reshape(1, -1)

    prediction = classifier.predict(
        embedding
    )

    predicted_label = label_encoder.inverse_transform(
        prediction
    )[0]

    # Get confidence
    probabilities = classifier.predict_proba(
        embedding
    )[0]

    confidence = probabilities.max()

    return {
        "label": predicted_label,
        "confidence": float(confidence)
    }


# TEST


if __name__ == "__main__":

    test_clause = """
    The subscription shall automatically renew for another
    twelve-month period unless the customer provides written
    notice before the renewal date.
    """

    result = predict_clause(
        test_clause
    )

    print("\n==============================")
    print("CLAUSE PREDICTION")
    print("==============================")

    print(
        f"Clause: {test_clause.strip()}"
    )

    print(
        f"Predicted Label: {result['label']}"
    )

    print(
        f"Confidence: {result['confidence']:.4f}"
    )