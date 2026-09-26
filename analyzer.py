import re
import joblib
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel


# CLAUSE KEYWORDS


CLAUSE_KEYWORDS = {

    "payment": [
        "payment",
        "fee",
        "salary",
        "compensation",
        "invoice",
        "rent",
        "subscription fee",
        "paid",
        "pay"
    ],

    "termination": [
        "termination",
        "terminate",
        "terminated",
        "notice period"
    ],

    "confidentiality": [
        "confidentiality",
        "confidential information",
        "non-disclosure",
        "nda"
    ],

    "intellectual_property": [
        "intellectual property",
        "copyright",
        "source code",
        "ownership",
        "proprietary"
    ],

    "liability": [
        "liability",
        "liable",
        "limitation of liability",
        "damages"
    ],

    "indemnity": [
        "indemnity",
        "indemnify",
        "hold harmless"
    ],

    "non_compete": [
        "non-compete",
        "non compete",
        "competing business",
        "competitor"
    ],

    "arbitration": [
        "arbitration",
        "arbitrator",
        "binding arbitration"
    ],

    "renewal": [
        "renewal",
        "renew",
        "automatically renew",
        "automatic renewal"
    ],

    "penalty": [
        "penalty",
        "penalties",
        "fine"
    ],

    "obligations": [
        "shall",
        "must",
        "responsible",
        "obligation",
        "required"
    ]
}

# SENTENCE SPLITTER


def split_sentences(text):

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

# CLAUSE DETECTION


def detect_clauses(text):

    sentences = split_sentences(text)

    clauses = {}

    for clause_type, keywords in CLAUSE_KEYWORDS.items():

        matches = []

        for sentence in sentences:

            sentence_lower = sentence.lower()

            if any(
                keyword in sentence_lower
                for keyword in keywords
            ):

                matches.append(sentence)

        clauses[clause_type] = list(
            dict.fromkeys(matches)
        )

    return clauses

# PARTY EXTRACTION


def extract_parties(text):

    parties = []

    patterns = [

        r'between\s+(.*?)\s+\(".*?"\)\s+and\s+(.*?)\s+\(".*?"\)',

        r'between\s+(.*?)\s+and\s+(.*?)(?:,|\.|\n)'
    ]

    for pattern in patterns:

        matches = re.findall(
            pattern,
            text,
            re.IGNORECASE
        )

        if matches:

            for match in matches:

                if isinstance(match, tuple):

                    parties.extend(match)

                else:

                    parties.append(match)

            break

    return list(
        dict.fromkeys(
            party.strip()
            for party in parties
            if party.strip()
        )
    )

# DATE EXTRACTION

def extract_dates(text):

    date_patterns = [

        r'\b(?:January|February|March|April|May|June|July|August|'
        r'September|October|November|December)\s+'
        r'\d{1,2},?\s+\d{4}\b',

        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
    ]

    dates = []

    for pattern in date_patterns:

        dates.extend(
            re.findall(
                pattern,
                text,
                re.IGNORECASE
            )
        )

    return list(
        dict.fromkeys(dates)
    )

# MONEY EXTRACTION


def extract_amounts(text):

    patterns = [

        r'INR\s?[\d,]+(?:\.\d+)?',

        r'₹\s?[\d,]+(?:\.\d+)?',

        r'\$\s?[\d,]+(?:\.\d+)?',

        r'\bUSD\s?[\d,]+(?:\.\d+)?'
    ]

    amounts = []

    for pattern in patterns:

        amounts.extend(
            re.findall(
                pattern,
                text,
                re.IGNORECASE
            )
        )

    return list(
        dict.fromkeys(amounts)
    )

# CLAUSE-SPECIFIC RISK DETECTION


def detect_risks(clauses):

    risks = []

    # Penalty


    if clauses["penalty"]:

        risks.append({

            "type": "Penalty",

            "severity": "High",

            "reason":
                "The contract contains a financial penalty "
                "or fine clause.",

            "clauses":
                clauses["penalty"]
        })

    # Non-compete
    if clauses["non_compete"]:

        risks.append({

            "type": "Non-Compete",

            "severity": "High",

            "reason":
                "The contract restricts one party from "
                "working with competitors or competing businesses.",

            "clauses":
                clauses["non_compete"]
        })


    # Indemnity


    if clauses["indemnity"]:

        risks.append({

            "type": "Indemnity",

            "severity": "Medium",

            "reason":
                "The contract contains an indemnification "
                "obligation that may transfer financial risk "
                "to one of the parties.",

            "clauses":
                clauses["indemnity"]
        })


    # Arbitration


    if clauses["arbitration"]:

        risks.append({

            "type": "Arbitration",

            "severity": "Medium",

            "reason":
                "Disputes under the agreement may be resolved "
                "through arbitration.",

            "clauses":
                clauses["arbitration"]
        })


    # Automatic Renewal

    if clauses["renewal"]:

        risks.append({

            "type": "Automatic Renewal",

            "severity": "Medium",

            "reason":
                "The agreement contains a renewal provision. "
                "The notice period should be reviewed carefully.",

            "clauses":
                clauses["renewal"]
        })

    # Liability
 

    if clauses["liability"]:

        risks.append({

            "type": "Liability",

            "severity": "Medium",

            "reason":
                "The agreement contains liability or "
                "damages provisions.",

            "clauses":
                clauses["liability"]
        })

    # Termination


    if clauses["termination"]:

        termination_text = " ".join(
            clauses["termination"]
        ).lower()

        notice_match = re.search(
            r'(\d+)\s*(day|days|month|months)',
            termination_text
        )

        if notice_match:

            value = int(
                notice_match.group(1)
            )

            unit = notice_match.group(2)

            if "day" in unit and value < 15:

                severity = "Medium"

                reason = (
                    f"The termination notice period "
                    f"is only {value} days."
                )

                risks.append({

                    "type": "Short Termination Notice",

                    "severity": severity,

                    "reason": reason,

                    "clauses":
                        clauses["termination"]
                })

    # Confidentiality


    if clauses["confidentiality"]:

        risks.append({

            "type": "Confidentiality",

            "severity": "Low",

            "reason":
                "The contract contains confidentiality "
                "obligations.",

            "clauses":
                clauses["confidentiality"]
        })

    # Intellectual Property
  

    if clauses["intellectual_property"]:

        risks.append({

            "type": "Intellectual Property",

            "severity": "Medium",

            "reason":
                "The agreement contains intellectual property "
                "or ownership provisions.",

            "clauses":
                clauses["intellectual_property"]
        })


    return risks


# MISSING CLAUSE DETECTION


def detect_missing_clauses(clauses):

    important_clauses = [

        "payment",

        "termination",

        "confidentiality",

        "liability",

        "intellectual_property"
    ]

    missing = []

    for clause in important_clauses:

        if not clauses.get(clause):

            missing.append(clause)

    return missing


# CONTRACT SUMMARY


def generate_summary(
    parties,
    dates,
    amounts,
    clauses,
    risks
):

    summary = []


    if parties:

        summary.append(
            f"The contract identifies "
            f"{len(parties)} party/parties."
        )


    if dates:

        summary.append(
            f"{len(dates)} date(s) were detected."
        )


    if amounts:

        summary.append(
            f"{len(amounts)} monetary value(s) were detected."
        )


    detected_clause_count = sum(
        bool(value)
        for value in clauses.values()
    )


    summary.append(
        f"{detected_clause_count} contract clause "
        f"categories were detected."
    )


    if risks:

        high = sum(
            risk["severity"] == "High"
            for risk in risks
        )

        medium = sum(
            risk["severity"] == "Medium"
            for risk in risks
        )


        summary.append(
            f"{len(risks)} potential risk(s) were identified "
            f"({high} high, {medium} medium)."
        )

    else:

        summary.append(
            "No predefined contractual risks were detected."
        )


    return " ".join(summary)

# RISK SCORE

RISK_POINTS = {
    "Penalty": 20,
    "Non-Compete": 20,
    "Indemnity": 12,
    "Automatic Renewal": 10,
    "Liability": 10,
    "Arbitration": 5,
    "Short Termination Notice": 8,
    "Intellectual Property": 5,
    "Confidentiality": 3
}


def calculate_risk_score(risks, missing_clauses):

    score = 0

    breakdown = []

    # Score detected risks
 

    for risk in risks:

        risk_type = risk["type"]

        points = RISK_POINTS.get(
            risk_type,
            0
        )

        score += points

        breakdown.append({
            "type": risk_type,
            "points": points,
            "severity": risk["severity"]
        })

    # Missing important clauses

    missing_points = {
        "payment": 5,
        "termination": 5,
        "confidentiality": 3,
        "liability": 5,
        "intellectual_property": 5
    }


    for clause in missing_clauses:

        points = missing_points.get(
            clause,
            0
        )

        score += points

        breakdown.append({
            "type": f"Missing {clause.replace('_', ' ').title()}",
            "points": points,
            "severity": "Medium"
        })


    # Maximum score
    score = min(
        score,
        100
    )

    # Overall category

    if score >= 60:

        category = "High"

    elif score >= 30:

        category = "Medium"

    else:

        category = "Low"


    return {
        "score": score,
        "category": category,
        "breakdown": breakdown
    }

# LEGAL-BERT CLAUSE CLASSIFICATION

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"

PROJECT_ROOT = Path(__file__).resolve().parent

CLASSIFIER_PATH = PROJECT_ROOT / "models" / "clause_classifier.pkl"
LABEL_ENCODER_PATH = PROJECT_ROOT / "models" / "label_encoder.pkl"

_ml_tokenizer = None
_ml_bert_model = None
_ml_classifier = None
_ml_label_encoder = None


def _load_ml_models():
    """Load Legal-BERT and the trained classifier once."""
    global _ml_tokenizer
    global _ml_bert_model
    global _ml_classifier
    global _ml_label_encoder

    if (
        _ml_tokenizer is not None
        and _ml_bert_model is not None
        and _ml_classifier is not None
        and _ml_label_encoder is not None
    ):
        return

    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(
            f"Classifier not found: {CLASSIFIER_PATH}"
        )

    if not LABEL_ENCODER_PATH.exists():
        raise FileNotFoundError(
            f"Label encoder not found: {LABEL_ENCODER_PATH}"
        )

    _ml_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _ml_bert_model = AutoModel.from_pretrained(MODEL_NAME)
    _ml_bert_model.eval()

    _ml_classifier = joblib.load(CLASSIFIER_PATH)
    _ml_label_encoder = joblib.load(LABEL_ENCODER_PATH)


def _create_ml_embedding(sentence):
    """Create one Legal-BERT mean-pooled embedding.

    This must match the embedding method used during training.
    """
    inputs = _ml_tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

    with torch.no_grad():
        outputs = _ml_bert_model(**inputs)

    token_embeddings = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"]

    mask = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()
    ).float()

    masked_embeddings = token_embeddings * mask

    summed_embeddings = torch.sum(
        masked_embeddings,
        dim=1
    )

    token_count = torch.clamp(
        mask.sum(dim=1),
        min=1e-9
    )

    embedding = summed_embeddings / token_count

    return embedding.squeeze().cpu().numpy()

def classify_clauses_with_legal_bert(text):
    """
    Classify meaningful contract clauses using the trained
    Legal-BERT + Logistic Regression model.

    The classifier uses the same mean-pooling method as the
    training pipeline. Obvious document titles and party/date
    lines are filtered, while legitimate legal clauses are
    preserved.
    """

    _load_ml_models()

    if (
        _ml_tokenizer is None
        or _ml_bert_model is None
        or _ml_classifier is None
        or _ml_label_encoder is None
    ):
        return []

    # Build candidate clauses

    raw_lines = text.splitlines()
    candidates = []

    for raw_line in raw_lines:

        line = raw_line.strip()

        if not line:
            continue

        line = " ".join(line.split())

        lower_line = line.lower()
        upper_line = line.upper()

        # Ignore obvious document titles/headings.
        heading_only_patterns = [
            "AGREEMENT",
            "CONTRACT",
            "TERMS AND CONDITIONS",
            "STATEMENT OF WORK",
            "FREELANCE DEVELOPMENT AGREEMENT",
            "SERVICE AGREEMENT",
            "EMPLOYMENT AGREEMENT",
            "NON-DISCLOSURE AGREEMENT",
            "CONFIDENTIALITY AGREEMENT"
        ]

        if (
            upper_line in heading_only_patterns
            or (
                len(line.split()) <= 8
                and (
                    upper_line.endswith(" AGREEMENT")
                    or upper_line.endswith(" CONTRACT")
                )
            )
        ):
            continue

        # Ignore obvious party/date-only lines.
        party_date_patterns = [
            "this agreement is between",
            "entered into by",
            "entered into between"
        ]

        if any(
            pattern in lower_line
            for pattern in party_date_patterns
        ):
            continue

        if (
            "effective " in lower_line
            and len(line.split()) < 18
        ):
            continue

        # Ignore very short fragments.
        if len(line.split()) < 6:
            continue

        candidates.append(line)

    # If line-based extraction produced nothing, fall back to sentence splitting.
    if not candidates:

        candidates = [
            sentence
            for sentence in split_sentences(text)
            if len(sentence.split()) >= 6
        ]

    # Remove duplicates while preserving order.
    candidates = list(
        dict.fromkeys(candidates)
    )

    predictions = []

    # Classify candidate clauses

    for sentence in candidates:

        try:

            embedding = _create_ml_embedding(
                sentence
            )

            # The saved classifier is a Pipeline containing
            # StandardScaler + Logistic Regression.
            embedding_2d = embedding.reshape(1, -1)

            prediction = _ml_classifier.predict(
                embedding_2d
            )[0]

            probabilities = (
                _ml_classifier.predict_proba(
                    embedding_2d
                )[0]
            )

            confidence = float(
                max(probabilities)
            )

            clause_label = (
                _ml_label_encoder.inverse_transform(
                    [prediction]
                )[0]
            )

            predictions.append(
                {
                    "sentence": sentence,
                    "clause": clause_label,
                    "confidence": confidence
                }
            )

        except Exception as error:

            print(
                f"Legal-BERT prediction failed: {error}"
            )

            continue

    return predictions

# MAIN ANALYZER

def analyze_contract(text):

    if not text or not text.strip():

        return {

            "summary":
                "No contract text was provided.",

            "parties": [],

            "dates": [],

            "amounts": [],

            "clauses": {},

            "risks": [],

            "missing_clauses": []
        }


    parties = extract_parties(text)

    dates = extract_dates(text)

    amounts = extract_amounts(text)

    clauses = detect_clauses(text)

    # Legal-BERT semantic classification is an additional layer.
    # The existing rule-based engine remains responsible for
    # deterministic risk detection and risk scoring.
    try:
        ml_clause_predictions = classify_clauses_with_legal_bert(
            text
        )
    except Exception as error:
        print(
            f"Legal-BERT classification unavailable: {error}"
        )
        ml_clause_predictions = []

    risks = detect_risks(
        clauses
    )

    missing_clauses = detect_missing_clauses(
        clauses
    )

    risk_score = calculate_risk_score(
    risks,
    missing_clauses
    )

    summary = generate_summary(
        parties,
        dates,
        amounts,
        clauses,
        risks
    )


    return {
    "summary": summary,
    "parties": parties,
    "dates": dates,
    "amounts": amounts,
    "clauses": clauses,
    "risks": risks,
    "missing_clauses": missing_clauses,
    "risk_score": risk_score,
    "ml_clause_predictions": ml_clause_predictions
    }