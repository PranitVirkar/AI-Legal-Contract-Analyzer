# AI-Legal-Contract-Analyzer
An AI-powered contract screening application that analyzes legal contracts to identify important clauses, contractual risks, missing clauses, key contract information, and semantic clause classifications using Legal-BERT and machine learning.

> **Disclaimer:** This project is intended for automated contract screening and educational purposes. It does not provide legal advice. Contract analysis and identified risks should be reviewed by a qualified legal professional.

---

## Overview

The AI Legal Contract Analyzer combines traditional rule-based contract analysis with a machine-learning-based clause classification system.

Users can upload a contract in PDF or TXT format. The application then:

* Extracts contract text
* Identifies parties, dates, and monetary values
* Detects important contractual clauses
* Identifies potential contractual risks
* Calculates an overall risk score
* Detects potentially missing clauses
* Classifies contract clauses using Legal-BERT embeddings
* Displays the results through an interactive Streamlit dashboard

---

## Features

### Contract Upload

Supports:

* PDF documents
* TXT documents

The application extracts readable text from the uploaded contract before analysis.

### Contract Information Extraction

The analyzer identifies:

* Contract parties
* Important dates
* Monetary values

### Clause Detection

The rule-based analyzer detects the following clause categories:

* Payment
* Termination
* Confidentiality
* Intellectual Property
* Liability
* Indemnity
* Non-Compete
* Arbitration
* Renewal
* Penalty
* Obligations

### Risk Analysis

The application identifies predefined contractual risk factors and calculates a score from 0 to 100.

|  Score | Category |
| -----: | -------- |
|   0–29 | Low      |
|  30–59 | Medium   |
| 60–100 | High     |

The risk engine considers factors such as:

* Penalties
* Non-compete clauses
* Indemnity
* Automatic renewal
* Liability
* Arbitration
* Short termination notice
* Intellectual property
* Confidentiality
* Missing important clauses

### Legal-BERT Clause Classification

The project uses:

`nlpaueb/legal-bert-base-uncased`

Legal-BERT generates semantic embeddings for contract sentences. These embeddings are passed through a trained machine-learning classifier to predict the clause category.

Architecture:

```text
Contract Sentence
       |
       v
Legal-BERT
       |
       v
Mean Pooling
       |
       v
StandardScaler
       |
       v
Logistic Regression
       |
       v
Clause Classification
```

Supported ML clause categories:

```text
arbitration
confidentiality
indemnity
intellectual_property
liability
non_compete
obligation
payment
penalty
renewal
termination
```

---

## Model Performance

The final classifier was trained using a labeled dataset containing:

* 242 examples
* 11 clause categories

Evaluation results:

| Metric        | Result |
| ------------- | -----: |
| Test Accuracy | 89.04% |
| Macro F1      | 89.28% |
| Weighted F1   | 89.00% |

The model uses mean pooling over Legal-BERT token embeddings followed by a scaled Logistic Regression classifier.

> These metrics are based on the project's held-out test set and should not be interpreted as production-level legal accuracy.

---

## Architecture

```text
                    +---------------------+
                    |    Streamlit UI     |
                    |       app.py        |
                    +----------+----------+
                               |
                               v
                    +---------------------+
                    | Contract Extraction |
                    |    extractor.py     |
                    +----------+----------+
                               |
                               v
                    +---------------------+
                    | Contract Analyzer   |
                    |    analyzer.py      |
                    +---------+-----------+
                              |
                 +------------+------------+
                 |                         |
                 v                         v
        +----------------+       +------------------+
        | Rule-Based     |       | Legal-BERT       |
        | Analysis       |       | Classification   |
        +-------+--------+       +--------+---------+
                |                         |
                v                         v
        +----------------+       +------------------+
        | Risk Detection |       | Logistic         |
        | & Scoring      |       | Regression       |
        +-------+--------+       +--------+---------+
                |                         |
                +------------+------------+
                             |
                             v
                    +---------------------+
                    | Analysis Dashboard  |
                    +---------------------+
```

---

## Project Structure

```text
AI-Legal-Contract-Analyzer/
|
├── app.py
├── analyzer.py
├── extractor.py
├── requirements.txt
├── README.md
├── .gitignore
|
├── data/
│   └── clause_dataset.csv
|
├── models/
│   ├── clause_classifier.pkl
│   └── label_encoder.pkl
|
├── outputs/
|
└── src/
    ├── legal_classifier.py
    ├── train_classifier.py
    └── predict_clauses.py
```

---

## Technologies Used

### Programming

* Python

### Machine Learning and NLP

* PyTorch
* Hugging Face Transformers
* Legal-BERT
* Scikit-learn
* Logistic Regression

### Data Processing

* Pandas
* NumPy

### Document Processing

* PyMuPDF

### Web Application

* Streamlit

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/PranitVirkar/AI-Legal-Contract-Analyzer.git
cd AI-Legal-Contract-Analyzer
```

### 2. Create a Virtual Environment

Windows:

```bash
python -m venv venv
```

Activate the environment:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## Example Workflow

### Step 1: Upload a Contract

Upload a PDF or TXT contract through the Streamlit interface.

### Step 2: Extract Text

The application extracts readable text from the uploaded document.

### Step 3: Analyze the Contract

The analyzer processes the contract using:

* Rule-based clause detection
* Risk analysis
* Contract information extraction
* Legal-BERT semantic classification

### Step 4: Review Results

The dashboard presents:

```text
Contract Overview
       |
       v
Risk Assessment
       |
       v
Contract Information
       |
       v
Detected Clauses
       |
       v
Potential Risks
       |
       v
Legal-BERT Classification
       |
       v
Potentially Missing Clauses
```

---

## Model Training

The classifier can be retrained using the provided dataset.

Training pipeline:

```text
clause_dataset.csv
        |
        v
Label Encoding
        |
        v
Legal-BERT
        |
        v
Mean Pooling
        |
        v
Train/Test Split
        |
        v
StandardScaler
        |
        v
Logistic Regression
        |
        v
Model Evaluation
        |
        v
clause_classifier.pkl
label_encoder.pkl
```

The trained model artifacts are stored in:

```text
models/
```

---

## Example Output

For a clause such as:

```text
The Client shall pay a total fee of INR 2,00,000.
```

the semantic classifier can identify:

```text
Clause: Payment
Confidence: 78.06%
```

For a complete contract, the application combines these semantic predictions with the rule-based analyzer to provide a broader contract-screening report.

---

## Privacy and Security

This project is designed as a local application.

Uploaded documents are processed locally by the application for analysis and are not inherently stored in a third-party contract database by the application itself.

Users should still avoid uploading confidential or sensitive legal documents when running the project in an environment where document privacy cannot be guaranteed.


## Limitations

This project is a contract screening tool and is not a replacement for professional legal review.

Known limitations include:

* The training dataset is relatively small.
* The dataset contains 242 labeled examples.
* Some semantically ambiguous legal sentences may be misclassified.
* Rule-based risk detection depends on predefined patterns.
* Model confidence does not represent legal certainty.
* The risk score is a screening heuristic rather than a legal risk assessment.
* Real-world contracts can contain language significantly different from the training examples.

The reported model metrics should therefore be interpreted within the context of the project's dataset and evaluation setup.

---

## Project Goals

The project demonstrates the practical application of:

* Natural Language Processing
* Transformer-based language models
* Legal-domain NLP
* Machine Learning classification
* Rule-based reasoning
* Risk scoring
* Document processing
* Streamlit application development

The primary goal is to demonstrate how traditional rule-based techniques can be combined with transformer-based semantic classification for automated legal document screening.

---

## Author

**Pranit Virkar**

Computer Science Graduate
Artificial Intelligence & Analytics

Areas of Interest:

* Artificial Intelligence
* Machine Learning
* Data Science
* Data Engineering
* Natural Language Processing
* Generative AI

---

## License

This project is intended for educational and portfolio purposes.

If you plan to distribute or modify the project, add an appropriate open-source license according to your intended usage.
