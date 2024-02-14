This project leverages the IMDB dataset to fine-tune a Hugging Face DistilBERT model for sentiment analysis purposes. 
After fine-tuning, the model is further evaluated on a dataset of Amazon reviews to assess its performance in real-world inference scenarios.

## Dependencies
- Python 3.8.3

## Installation
To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

To fine-tune the BERT model on IMDB dataset:

```bash
Python bert.py
```

To test on Amazon reviews dataset:

```bash
Python bert_inference.py
```
