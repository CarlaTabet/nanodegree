import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, EvalPrediction
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# TODO: In this cell, describe your choices for each of the following
# PEFT technique: I chose the LoRa technique.
# This technique allows large pre-trained models to be fine tuned by reducing the number of parameters.
# Model: I chose the distilbert-base-uncased model, as it is smaller than gpt-2
# Evaluation approach: The evaluation approach included calculating the accuracy metric using the Hugging Face Trainer.
# The accuracy metric compared the correctness of the model's predictions with the true labels.
# Fine-tuning dataset: I selected the ag_news dataset.
# It is a relatively small dataset, and was therefore a good choice for demonstrating the
# effectiveness of fine-tuning techniques

# load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForSequenceClassification.from_pretrained("gpt2", num_labels=4)

model.config.pad_token_id = tokenizer.pad_token_id

dataset = load_dataset("ag_news")

# From class example
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)


tokenized_datasets = dataset.map(tokenize_function, batched=True)


def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='macro')
    acc = accuracy_score(p.label_ids, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    #fp16=True,
)

# made smaller subset for sake of running in time
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"].shuffle().select([i for i in range(100)]),
    eval_dataset=tokenized_datasets["test"].shuffle().select([i for i in range(100)]),
    compute_metrics=compute_metrics
)

initial_results = trainer.evaluate()
print(f"Initial results: {initial_results}")

# LoRA configuration
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["attn.c_attn", "mlp.c_fc", "mlp.c_proj"],
    lora_dropout=0.1,
    bias="none"
)

# create a PEFT model
lora_model = get_peft_model(model, config)
lora_model.print_trainable_parameters()

trainer.model = lora_model
trainer.train()
lora_model.save_pretrained("peft-lora")

lora_model = PeftModelForSequenceClassification.from_pretrained(model, "peft-lora", from_transformers=True)
# now train with fine-tuned results
trainer.model = lora_model
fine_tuned_result = trainer.evaluate()
print(f"Fine-tuned results: {fine_tuned_result}")

# use accuracy to evaluate improvement
initial_accuracy = initial_results["eval_accuracy"]
fine_tuned_accuracy = fine_tuned_result["eval_accuracy"]
change = fine_tuned_accuracy - initial_accuracy
print(f"Accuracy Improvement: {change}")

