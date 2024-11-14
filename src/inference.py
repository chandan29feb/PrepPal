from transformers import BertTokenizer, BertForSequenceClassification
import torch
import config

class ModelInference:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.label_maps = {}

        for label_type in config.LABEL_TYPES:
            model_path = f"{config.MODEL_SAVE_PATH}/bert_{label_type}"
            self.models[label_type] = BertForSequenceClassification.from_pretrained(model_path)
            self.tokenizers[label_type] = BertTokenizer.from_pretrained(model_path)
            self.label_maps[label_type] = self._load_label_map(label_type)

    def _load_label_map(self, label_type):
        # Logic to load label mapping for decoding predictions
        return {idx: label for label, idx in enumerate(config.LABEL_TYPES)}

    def predict(self, text):
        predictions = {}
        for label_type in config.LABEL_TYPES:
            inputs = self.tokenizers[label_type](text, return_tensors="pt", truncation=True)
            outputs = self.models[label_type](**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
            predictions[label_type] = self.label_maps[label_type][pred]
        return predictions
