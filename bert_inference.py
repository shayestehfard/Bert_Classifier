from transformers import pipeline
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# Load the sentiment analysis model
sentiment_model = pipeline("sentiment-analysis", model="Kimia124/finetuning-sentiment-model-3000-samples")

# Analyze the sentiments of two sample texts
results = sentiment_model(["I love this movie", "This movie sucks!"])

# Print the results
for result in results:
    print(result)


dataset = pd.read_csv("/mnt/c/Users/shkim/Desktop/LLM Projects/Bert_Classifier/amazon.csv")
print(dataset.columns)
# Load the sentiment analysis model
# sentiment_model = pipeline("sentiment-analysis", model="Kimia124/finetuning-sentiment-model-3000-samples")

# # For a pandas DataFrame
# results = sentiment_model(list(dataset["reviewText"]))  # Replace "review_column" with the actual column name

# # For the Hugging Face 'datasets' library
# # Convert the dataset to a list of texts if necessary
# results = sentiment_model([example['text'] for example in dataset])
# # Assuming you have ground truth labels
# true_labels = dataset["label"].tolist()  # Or any method to get your true labels
# predicted_labels = [result['label'].upper() for result in results]  # Adjust as needed

# # Calculate accuracy
# accuracy = accuracy_score(true_labels, predicted_labels)
# print(f"Accuracy: {accuracy}")

# # Detailed performance report
# print(classification_report(true_labels, predicted_labels))

sentiment_model = pipeline(
    "sentiment-analysis",
    model="Kimia124/finetuning-sentiment-model-3000-samples",
    truncation=True  # Ensure sequences longer than model's max length are truncated
)

# Load your dataset
dataset = pd.read_csv("/mnt/c/Users/shkim/Desktop/LLM Projects/Bert_Classifier/amazon.csv")
# Convert non-string values to strings and filter out None values
reviews = [str(text) for text in dataset["reviewText"] if text is not None]
reviews = [text if text.strip() != "" else "No review text." for text in reviews]
print(dataset.columns)
results = sentiment_model(reviews)

test_reviews = reviews[:10]  # Adjust the number as needed
test_results = sentiment_model(test_reviews)

for result in test_results:
    print(result)
# # Ensure 'reviewText' column exists in your DataFrame
# if "reviewText" in dataset.columns:
#     # Apply the sentiment model to the review texts
#     # Depending on your dataset size, consider processing in smaller batches to manage memory usage
#     results = sentiment_model(dataset["reviewText"].tolist())

#     # Example of processing the results
#     for result in results[:5]:  # Just showing the first 5 results for brevity
#         print(result)
# else:
#     print("Column 'reviewText' not found in the dataset.")