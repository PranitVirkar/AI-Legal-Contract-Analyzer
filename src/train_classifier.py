import pandas as pd
import torch
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from transformers import AutoTokenizer, AutoModel


# CONFIGURATION

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "data" / "clause_dataset.csv"

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_OUTPUT = MODEL_DIR / "clause_classifier.pkl"

LABEL_ENCODER_OUTPUT = MODEL_DIR / "label_encoder.pkl"


# CREATE MODEL DIRECTORY


MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# LOAD DATASET


print("Loading dataset...")

df = pd.read_csv(DATASET_PATH)

print(
    f"Loaded {len(df)} examples."
)

print("\nClass distribution:")

print(
    df["label"].value_counts()
)


# BASIC DATA VALIDATION


required_columns = {"text", "label"}

if not required_columns.issubset(df.columns):

    raise ValueError(
        "Dataset must contain 'text' and 'label' columns."
    )


if df["text"].isnull().any():

    raise ValueError(
        "Dataset contains empty text values."
    )


if df["label"].isnull().any():

    raise ValueError(
        "Dataset contains empty labels."
    )


# LABEL ENCODING


label_encoder = LabelEncoder()

y = label_encoder.fit_transform(
    df["label"]
)


# LOAD LEGAL-BERT


print("\nLoading Legal-BERT...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

legal_bert = AutoModel.from_pretrained(
    MODEL_NAME
)

legal_bert.eval()


# CREATE LEGAL-BERT EMBEDDINGS

def create_embeddings(texts):

    embeddings = []

    for i, text in enumerate(texts):

        print(
            f"Processing {i + 1}/{len(texts)}"
        )

        inputs = tokenizer(
            str(text),
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )

        with torch.no_grad():

            outputs = legal_bert(
                **inputs
            )

        
        # MEAN POOLING
        token_embeddings = outputs.last_hidden_state

        attention_mask = inputs["attention_mask"]

        mask = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        ).float()

        masked_embeddings = (
            token_embeddings * mask
        )

        summed_embeddings = torch.sum(
            masked_embeddings,
            dim=1
        )

        token_count = torch.clamp(
            mask.sum(dim=1),
            min=1e-9
        )

        mean_embedding = (
            summed_embeddings / token_count
        )

        embeddings.append(
            mean_embedding.squeeze().cpu().numpy()
        )

    return embeddings

# GENERATE EMBEDDINGS


print(
    "\nCreating Legal-BERT embeddings..."
)

X = create_embeddings(
    df["text"].tolist()
)


# TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.30,

    random_state=42,

    stratify=y
)


print(
    f"\nTraining examples: {len(X_train)}"
)

print(
    f"Testing examples: {len(X_test)}"
)


# LOGISTIC REGRESSION CONFIGURATIONS


configs = [

    {
        "C": 0.1,
        "class_weight": None
    },

    {
        "C": 0.5,
        "class_weight": None
    },

    {
        "C": 1.0,
        "class_weight": None
    },

    {
        "C": 2.0,
        "class_weight": None
    },

    {
        "C": 5.0,
        "class_weight": None
    },

    {
        "C": 1.0,
        "class_weight": "balanced"
    },

    {
        "C": 2.0,
        "class_weight": "balanced"
    },

    {
        "C": 5.0,
        "class_weight": "balanced"
    }
]


# TRAIN AND COMPARE MODELS

best_model = None

best_config = None

best_macro_f1 = -1

best_accuracy = 0


print(
    "\nTesting Logistic Regression configurations..."
)

print(
    "=" * 60
)


for config in configs:

    print(
        f"\nTraining:"
        f" C={config['C']}"
        f" | class_weight={config['class_weight']}"
    )

    classifier = Pipeline(

        [

            (
                "scaler",
                StandardScaler()
            ),

            (

                "classifier",

                LogisticRegression(

                    C=config["C"],

                    class_weight=config[
                        "class_weight"
                    ],

                    max_iter=3000,

                    random_state=42

                )

            )

        ]

    )


    # TRAIN


    classifier.fit(
        X_train,
        y_train
    )


    # PREDICT


    predictions = classifier.predict(
        X_test
    )


    # EVALUATE


    report = classification_report(

        y_test,

        predictions,

        output_dict=True,

        zero_division=0

    )


    accuracy = accuracy_score(

        y_test,

        predictions

    )


    macro_f1 = report[
        "macro avg"
    ][
        "f1-score"
    ]


    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        f"Macro F1: {macro_f1:.4f}"
    )



    if (

        macro_f1 > best_macro_f1

        or

        (
            macro_f1 == best_macro_f1
            and accuracy > best_accuracy
        )

    ):

        best_macro_f1 = macro_f1

        best_accuracy = accuracy

        best_model = classifier

        best_config = config


# BEST MODEL


print(
    "\n"
    + "=" * 60
)

print(
    "BEST MODEL"
)

print(
    "=" * 60
)


print(
    f"Best C: {best_config['C']}"
)

print(
    f"Best class_weight: "
    f"{best_config['class_weight']}"
)

print(
    f"Best Accuracy: "
    f"{best_accuracy:.4f}"
)

print(
    f"Best Macro F1: "
    f"{best_macro_f1:.4f}"
)


# FINAL EVALUATION


final_predictions = best_model.predict(
    X_test
)


print(
    "\nClassification Report:"
)

print(

    classification_report(

        y_test,

        final_predictions,

        target_names=label_encoder.classes_,

        zero_division=0

    )

)


# SAVE BEST MODEL


joblib.dump(

    best_model,

    MODEL_OUTPUT

)


joblib.dump(

    label_encoder,

    LABEL_ENCODER_OUTPUT

)


# SAVE CONFIRMATION

print(
    "\nBest model saved successfully!"
)

print(
    f"Classifier: {MODEL_OUTPUT}"
)

print(
    f"Label encoder: {LABEL_ENCODER_OUTPUT}"
)

print(
    "\nTraining completed successfully."
)

