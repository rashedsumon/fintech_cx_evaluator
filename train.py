import os
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
from data_loader import load_or_download_dataset

MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
OUTPUT_DIR = "./trained_model"

def train_cx_classifier():
    """Fine-tunes a Spanish Transformer model to identify CX friction points and compliance issues."""
    print("Initializing dataset and model tokenizer...")
    dataset = load_or_download_dataset()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    label_list = list(set(dataset["train"]["category"]))
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}

    def preprocess_function(examples):
        tokens = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
        tokens["labels"] = [label2id[cat] for cat in examples["category"]]
        return tokens

    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir="./checkpoints",
        eval_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        use_cpu=not torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        tokenizer=tokenizer
    )

    print("Starting Hugging Face Fine-Tuning execution...")
    trainer.train()
    
    print(f"Saving fine-tuned artifact to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    train_cx_classifier()