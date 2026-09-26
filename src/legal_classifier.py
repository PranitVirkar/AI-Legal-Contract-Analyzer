import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()


def create_embedding(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    token_embeddings = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"]

    # Expand attention mask to match embedding dimensions
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    # Mean pooling over actual tokens only
    summed = torch.sum(token_embeddings * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)

    embedding = summed / counts

    return embedding