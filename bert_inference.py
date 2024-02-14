from transformers import pipeline
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# Load the fine-tuned sentiment analysis model
sentiment_model = pipeline(
    "sentiment-analysis",
    model="Kimia124/finetuning-sentiment-model-3000-samples",
    truncation=True  # Ensure sequences longer than model's max length are truncated
)


# Analyze the sentiments of two sample texts
results = sentiment_model(["I love this movie", "This movie sucks!"])

# Print the results
for result in results:
    print(result)

# Load Amazon dataset
dataset = pd.read_csv("./amazon.csv")
# print(dataset.columns)

# Pre-processing text: Convert non-string values to strings and filter out None values
reviews = [str(text) for text in dataset["reviewText"] if text is not None]
reviews = [text if text.strip() != "" else "No review text." for text in reviews]
# print(dataset.columns)
results = sentiment_model(reviews)

test_reviews = reviews[:10]  
test_results = sentiment_model(test_reviews)

for result in test_results:
    print(result)
