# Consumer Complaint Classification

An end-to-end NLP project that classifies customer complaints into the correct product category using deep learning, comparing classic RNN-based architectures against a fine-tuned Transformer.

## Objective

Build a deep learning text classification pipeline that automatically predicts a complaint's category from its written narrative — removing the need to manually read and route every complaint.

The project:
- Builds a complete NLP pipeline: EDA → preprocessing → tokenization → modeling → evaluation → deployment
- Compares SimpleRNN, LSTM, and GRU models built from scratch against a fine-tuned Transformer (DistilBERT)
- Evaluates every model on Accuracy, Precision, Recall, F1-score, and Confusion Matrix
- Deploys the best-performing model as an interactive Gradio web app

## Dataset

[Consumer Complaint dataset](https://www.kaggle.com/datasets/shashwatwork/consume-complaints-dataset-fo-nlp) — customer complaint narratives labeled with one of 5 product categories:
`credit_reporting`, `debt_collection`, `mortgages_and_loans`, `credit_card`, `retail_banking`

The dataset is not included in this repo. Download it from the link above and place `complaints_processed.csv` in your Kaggle working directory (or update `DATA_PATH` in the notebook) before running.

## Project Pipeline

| Stage | Description |
|---|---|
| 1. Data Exploration | Load dataset, check missing values/duplicates, analyze class distribution |
| 2. Text Preprocessing | Lowercase, remove punctuation/numbers/special characters, remove stopwords, lemmatize |
| 3. Text Tokenization | Tokenizer, vocabulary, sequence padding |
| 4. Word Embedding | Trainable embedding layer |
| 5. Model Building | SimpleRNN, LSTM, GRU, fine-tuned DistilBERT |
| 6. Model Evaluation | Accuracy, Precision, Recall, F1-score, Confusion Matrix |
| 7. Performance Comparison | Side-by-side comparison table and chart |
| 8. Deployment | Best model deployed with Gradio |

## Results

| Model | Accuracy | Precision | Recall | F1-score | Training Time (s) |
| Transformer (DistilBERT) | 0.856381 | 0.856186 | 0.856381 | 0.855056  |1693.604893 |
| LSTM | 0.815143 | 0.820339 | 0.815143 | 0.815967 | 37.990365 |
| SimpleRNN | 0.462190 | 0.360177  | 0.462190 | 0.336662 | 25.330165 |
| GRU | 0.238952 | 0.452029 | 0.238952 | 0.180422 | 10.957480 |


## Repository Structure

```
consumer-complaint-classification/
├── README.md
├── requirements.txt
├── notebooks/
│   └── complaint_classification.ipynb
├── app/
│   └── gradio_app.py
└── results/
    └── comparison_results.csv
```

Note: the trained model weights (`best_model/`) are not included in this repo due to file size. Run the notebook to regenerate them before running the app.

## How to Run

### 1. Train the models (Kaggle, GPU recommended)

- Upload the dataset to Kaggle, or add it via the Kaggle dataset search
- Open `notebooks/complaint_classification.ipynb` in Kaggle
- Update `DATA_PATH` to point to your dataset
- Enable GPU (Settings → Accelerator)
- Run all cells — this trains all 4 models, evaluates them, and saves the best one to `/kaggle/working/best_model`
- Download the `best_model` folder (zip it first for convenience)

### 2. Run the app locally

Requires Python 3.11 (TensorFlow does not yet support newer Python versions).

```bash
# create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# install dependencies
pip install -r requirements.txt

# place the downloaded best_model folder in the project root, then run:
python app/gradio_app.py
```

Open the printed local URL (typically `http://127.0.0.1:7860`) in your browser. Enter a complaint narrative and the app will predict its category with a confidence score.

## Technologies Used

| Task | Tool |
|---|---|
| Programming Language | Python |
| Data Analysis | Pandas |
| Visualization | Matplotlib, Seaborn |
| NLP | NLTK |
| Deep Learning | TensorFlow / Keras |
| Transformer | HuggingFace Transformers |
| Deployment | Gradio |
