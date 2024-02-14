import transformers
from transformers import AutoTokenizer, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
import torch

import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from datasets import load_dataset
import evaluate
from evaluate import evaluator


RANDOM_SEED = 123
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device
imdb = load_dataset("imdb")
small_train_dataset = imdb["train"].shuffle(RANDOM_SEED).select([i for i in list(range(3000))])
small_test_dataset = imdb["test"].shuffle(RANDOM_SEED).select([i for i in list(range(500))])
small_val_dataset = imdb["test"].shuffle(RANDOM_SEED).select([i for i in list(range(500, 1000))])

# Checking the label names and corresponding classes
print("Classes in the IMDb dataset:", small_train_dataset.features['label'].names)


#print(small_train_dataset[0])
class_names = ['negative',  'positive']

PRE_TRAINED_MODEL_NAME = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

def preprocess_function(examples):
   return tokenizer(examples["text"], truncation=True)
 
tokenized_train = small_train_dataset.map(preprocess_function, batched=True)
tokenized_test = small_test_dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)


def compute_metrics(eval_pred):
   
   load_accuracy = evaluate.load("accuracy")
   load_f1 = evaluate.load("f1")
  
   logits, labels = eval_pred
   predictions = np.argmax(logits, axis=-1)
   accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
   f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
   print(f"accuracy: {accuracy}, f1: {f1}")
   return {"accuracy": accuracy, "f1": f1}


training_args = TrainingArguments(
    output_dir="before-finetuning",
    do_train=False,
    do_eval=True,
    per_device_eval_batch_size=16,
    
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_test,   
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

eval_results_before = trainer.evaluate()

print("Evaluation results before fine-tuning:", eval_results_before)
print("*"*50)
repo_name = "finetuning-sentiment-model-3000-samples"
 
training_args = TrainingArguments(
   output_dir=repo_name,
   learning_rate=2e-5,
   per_device_train_batch_size=16,
   per_device_eval_batch_size=16,
   num_train_epochs=2,
   weight_decay=0.01,
   save_strategy="epoch",
   push_to_hub=True,
)
 
trainer = Trainer(
   model=model,
   args=training_args,
   train_dataset=tokenized_train,
   eval_dataset=tokenized_test,
   tokenizer=tokenizer,
   data_collator=data_collator,
   compute_metrics=compute_metrics,
)
trainer.train()
trainer.evaluate()

trainer.push_to_hub()
