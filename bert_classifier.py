from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader
import torch

from text_dataset import TextDataset


class BERTClassifier:
    """
    Classifier using BERT for sequence classification.
    """

    def __init__(self, model_name, num_labels, device):
        """
        Initialize the classifier with the specified BERT model and the device (GPU/CPU).

        :param model_name: str, name of the BERT model to use
        :param num_labels: int, number of labels in the classification task
        :param device: torch.device, device to run the model on
        """
        self.device = device
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def prepare_data(self, texts, labels, batch_size=16):
        """
        Prepare data for training or evaluation.

        :param texts: list of str, the texts to process
        :param labels: list of int, corresponding labels for the texts
        :param batch_size: int, size of the batch for DataLoader
        :return: DataLoader, prepared data loader for training or evaluation
        """
        dataset = TextDataset(texts, labels, self.tokenizer)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train(self, train_loader, val_loader, epochs=3, learning_rate=5e-5):
        """
        Train the BERT model.

        :param train_loader: DataLoader, data loader for the training set
        :param val_loader: DataLoader, data loader for the validation set
        :param epochs: int, number of epochs to train
        :param learning_rate: float, learning rate for the optimizer
        """
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        self.model.train()

        for epoch in range(epochs):
            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
            self.evaluate(val_loader)

    def evaluate(self, data_loader):
        """
        Evaluate the model on a given data loader.

        :param data_loader: DataLoader, data loader for evaluation
        """
        self.model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for batch in data_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == batch['labels']).sum().item()
                total += batch['labels'].size(0)

        accuracy = correct / total
        print(f'Validation Accuracy: {accuracy}')