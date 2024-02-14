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

### Fine-tuning the BERT Model
To fine-tune the DistilBERT model on the IMDB dataset, execute the following command:

```bash
Python bert.py
```

### Testing on a differnt dataset
To test on Amazon reviews dataset, execute the following command:

```bash
Python bert_inference.py
```
