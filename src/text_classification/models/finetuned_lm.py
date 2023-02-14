from typing import List, Optional
import os
import math

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)
from datasets import Dataset, DatasetDict, NamedSplit
import evaluate
import numpy as np
from tqdm.auto import tqdm

from text_classification.data import Samples
from text_classification.models.base import Model
from text_classification.transforms import concat_fields


class FinetunedLM(Model):
    """
    A model that fine-tunes a language model on the target dataset.
    """
    def __init__(
        self,
        version: str,
        fields: List[str],
        labels: List[str],
        language_model: str,
        learning_rate: float = 2e-5,
        batch_size: int = 32,
        num_train_epochs: float = 5,
        weight_decay: float = 0.01,
        logging_steps: int = 500,
        validation_samples: int = 2500  # saves time
    ):
        self.version = version
        self.fields = fields
        self.labels = labels
        self.language_model = language_model
        self.validation_samples = validation_samples
        self.model_name = self.__class__.__name__
        self.batch_size = batch_size
        self.num_train_epochs = num_train_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.logging_steps = logging_steps
        self._tokenizer = AutoTokenizer.from_pretrained(language_model)
        self._data_collator = DataCollatorWithPadding(tokenizer=self._tokenizer)
        self._model: Optional[torch.nn.Module] = None

    def fit(
        self,
        X_train: Samples,
        y_train: np.ndarray,
        X_valid: Samples,
        y_valid: np.ndarray,
    ) -> 'FinetunedLM':
        X_train = self._add_label(X_train, y_train)
        X_train = self._preprocess_samples(X_train)
        X_valid = self._add_label(X_valid, y_valid)
        X_valid = self._preprocess_samples(X_valid)
        # sampled to save time, already shuffled during data split
        X_valid = X_valid[:self.validation_samples] 
        train_dataset = Dataset.from_list(X_train, split=NamedSplit("train"))
        validation_dataset = Dataset.from_list(X_valid, split=NamedSplit("validation"))
        datasets = DatasetDict({"train": train_dataset, "validation": validation_dataset})
        tokenized_datasets = datasets.map(self._tokenize_batch, batched=True)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.language_model,
            num_labels=len(self.labels),
            id2label={i: l for i, l in enumerate(self.labels)},
            label2id={l: i for i, l in enumerate(self.labels)}
        )
        training_args = TrainingArguments(
            output_dir=f"{os.environ['MODELS_ROOT']}/{self.model_name}/{self.version}",
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.num_train_epochs,
            weight_decay=self.weight_decay,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=self.logging_steps,
            logging_steps=self.logging_steps,
            load_best_model_at_end=True
        )
        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],  # type: ignore
            eval_dataset=tokenized_datasets["validation"],  # type: ignore
            tokenizer=self._tokenizer,
            data_collator=self._data_collator,
            compute_metrics=self._compute_metrics  # type: ignore
        )
        trainer.train()
        return self
    
    def _add_label(self, X: Samples, y: np.ndarray) -> Samples:
        return [
            {**sample, "label": label}
            for sample, label in zip(X, y)
        ]

    def _preprocess_samples(self, X: Samples) -> Samples:
        output = [
            {
                **sample,
                "text": concat_fields(sample, self.fields)
            }
            for sample in X
        ]
        return output

    def _tokenize_batch(self, batch):
        return self._tokenizer(batch["text"], truncation=True)
    
    def _compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        metric = evaluate.load("f1")
        return metric.compute(
            predictions=predictions,
            references=labels,
            labels=labels,
            average="macro"
        )
    
    def predict(self, X: Samples) -> np.ndarray:
        # would have used pipeline, but there was multithreading issues.
        # would debug with more time, but this works for now.
        samples = self._preprocess_samples(X)
        num_batches = math.ceil(len(samples) / self.batch_size)
        preds_batches = []
        for batch_idx in tqdm(range(num_batches)):
            batch = samples[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
            text = [sample["text"] for sample in batch]
            inputs = self._tokenizer(text, truncation=True, padding=True, return_tensors="pt")  # type: ignore
            with torch.no_grad():
                outputs = self._model(**inputs)  # type: ignore
            preds = np.argmax(outputs.logits.numpy(), axis=1)
            preds_batches.append(preds)
        preds = np.concatenate(preds_batches)
        return preds
