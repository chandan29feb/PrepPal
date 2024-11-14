from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
import torch
from torch.utils.data import Dataset
import config
import os

class QuestionDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_NAME)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def train_model(label_type):
    # print(os.path.isfile("../data/questions.json"))
    df = pd.read_json("./data/questions.json")
    label_map = {label: idx for idx, label in enumerate(df[label_type].unique())}
    df[f'{label_type}_label'] = df[label_type].map(label_map)

    dataset = QuestionDataset(df['question'].tolist(), df[f'{label_type}_label'].tolist())
    
    model = BertForSequenceClassification.from_pretrained(config.BERT_MODEL_NAME, num_labels=len(label_map))

    training_args = TrainingArguments(
        output_dir=f"{config.MODEL_SAVE_PATH}/bert_{label_type}",
        num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )
    
    trainer.train()
    model.save_pretrained(f"{config.MODEL_SAVE_PATH}/bert_{label_type}")
    BertTokenizer.from_pretrained(config.BERT_MODEL_NAME).save_pretrained(f"{config.MODEL_SAVE_PATH}/bert_{label_type}")

if __name__ == "__main__":
    for label in config.LABEL_TYPES:
        train_model(label)
