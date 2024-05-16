import datetime
import os
import time
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, random_split
from transformers import (AdamW, BertForSequenceClassification, BertTokenizer,
                          get_linear_schedule_with_warmup)

from logger import Logger

warnings.filterwarnings('ignore')


class BERTClassifier:
    """
    Классификатор, использующий BERT для классификации фрагментов текста.
    """

    def __init__(self, model_name, num_labels, enable_logging=True, output_dir='./model_save/'):
        """
        Инициализация классификатора с указанным BERT моделью и устройством (GPU/CPU).

        :param model_name: str, название BERT модели для использования
        :param num_labels: int, количество меток в задаче классификации
        :param enable_logging: bool, активация логирования
        :param output_dir: str, путь для сохранения модели
        """
        self.output_dir = output_dir
        self.log = Logger(enable_logging)
        self.device = self._get_device()
        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            output_attentions=False,
            output_hidden_states=False
        )
        self.train_dataset = None
        self.test_dataset = None

    def _get_device(self):
        """
        Определение и установка устройства для вычислений.
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
            self.log.info(f'There are {torch.cuda.device_count()} GPU(s) available.')
            self.log.important('We will use the GPU:' + torch.cuda.get_device_name(0))
        else:
            self.log.warning('No GPU available, using the CPU instead.')
            device = torch.device("cpu")
        return device

    def _get_max_sentence_len(self, sentences):
        """
        Определение максимальной длины предложения после токенизации.

        :param sentences: list, список предложений для обработки
        :return: int, максимальная длина предложения
        """
        max_len = 0
        for sent in sentences:
            input_ids = self.tokenizer.encode(sent, add_special_tokens=True)
            max_len = max(max_len, len(input_ids))

        self.log.info('Max sentence length: ' + str(max_len))
        if max_len > 0 and (max_len & (max_len - 1)) == 0:
            return max_len
        return 1 << (max_len.bit_length())

    def _print_model_params(self):
        """
        Вывод параметров модели.
        """
        params = list(self.model.named_parameters())
        self.log.important('\n\nThe BERT model has {:} different named parameters.\n'.format(len(params)))
        self.log.important('==== Embedding Layer ====\n')
        for p in params[0:5]:
            self.log.info("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

        self.log.important('\n==== First Transformer ====\n')
        for p in params[5:21]:
            self.log.info("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

        self.log.important('\n==== Output Layer ====\n')
        for p in params[-4:]:
            self.log.info("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    @staticmethod
    def _flat_accuracy(preds, labels):
        """
        Расчёт точности классификации.

        :param preds: numpy.ndarray, предсказания модели
        :param labels: numpy.ndarray, истинные метки
        :return: float, точность классификации
        """
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    @staticmethod
    def _format_time(elapsed):
        """
        Форматирование времени в читаемый формат.

        :param elapsed: float, время в секундах
        :return: str, время в формате hh:mm:ss
        """
        elapsed_rounded = int(round(elapsed))
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def prepare_data(self, sentences, labels):
        """
        Подготовка данных для обучения или оценки.

        :param sentences: list of str, тексты для обработки
        :param labels: list of int, соответствующие метки для текстов
        :return: tuple, содержащий тренировочный и валидационный DataLoader
        """
        input_ids, attention_masks = [], []
        max_len = self._get_max_sentence_len(sentences)
        for sent in sentences:
            encoded_dict = self.tokenizer.encode_plus(
                sent,
                add_special_tokens=True,
                max_length=max_len,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels)

        self.log.info('Original: ' + sentences[0])
        self.log.info('Token IDs: ' + str(input_ids[0]))
        self.log.info('Attention masks: ' + str(attention_masks[0]))
        self.log.info('Labels: ' + str(labels[0]))

        dataset = TensorDataset(input_ids, attention_masks, labels)
        train_size = int(0.9 * len(dataset))
        test_size = len(dataset) - train_size
        self.train_dataset, self.test_dataset = random_split(dataset, [train_size, test_size])

        self.log.important('\n\n{:>5,} training samples'.format(train_size))
        self.log.important('{:>5,} validation samples\n\n'.format(test_size))

        return self.train_dataset, self.test_dataset

    def _create_dataloaders(self, batch_size):
        """
        Создание DataLoader'ов для тренировочных и валидационных данных.

        :param batch_size: int, размер батча
        :return: tuple, содержащий тренировочный и валидационный DataLoader
        """
        train_dataloader = DataLoader(
            self.train_dataset,
            sampler=RandomSampler(self.train_dataset),
            batch_size=batch_size
        )
        test_dataloader = DataLoader(
            self.test_dataset,
            sampler=SequentialSampler(self.test_dataset),
            batch_size=batch_size
        )
        return train_dataloader, test_dataloader

    def train(self, epochs=3, batch_size=32):
        """
        Обучение модели BERT.

        :param epochs: int, количество эпох для тренировки (рек. от 2 до 4)
        :param batch_size: int, размер батча для тренировки
        """
        if self.train_dataset is None or self.test_dataset is None:
            raise ValueError("Data must be prepared before training. Call prepare_data first.")

        train_dataloader, test_dataloader = self._create_dataloaders(batch_size)
        if torch.cuda.is_available():
            self.model.cuda()

        self._print_model_params()
        optimizer = AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        training_stats = []
        total_t0 = time.time()

        for epoch_i in range(epochs):
            avg_train_loss, training_time = self._train_step(train_dataloader, optimizer, scheduler, epoch_i, epochs)
            avg_val_loss, avg_val_accuracy, test_time = self._test_step(test_dataloader)

            training_stats.append({
                'Epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Validation Loss': avg_val_loss,
                'Validation Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Test Time': test_time
            })

        self.log.important("\n\n\nTraining complete! Total training took " +
                           "{:} (hh:mm:ss)".format(self._format_time(time.time() - total_t0)))
        self._save_model()

    def _train_step(self, train_dataloader, optimizer, scheduler, epoch_i, epochs):
        """
        Один шаг обучения (эпоха).

        :param train_dataloader: DataLoader, тренировочные данные
        :param optimizer: оптимизатор
        :param scheduler: планировщик изменения скорости обучения
        :param epoch_i: int, текущий номер эпохи
        :param epochs: int, общее количество эпох
        :return: tuple, среднее значение потерь и время обучения
        """
        self.log.important('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        t0 = time.time()
        total_train_loss = 0
        self.model.train()

        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and not step == 0:
                elapsed = self._format_time(time.time() - t0)
                self.log.info('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            b_input_ids, b_input_mask, b_labels = (batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device))
            self.model.zero_grad()
            outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        training_time = self._format_time(time.time() - t0)
        self.log.important("  Average training loss: {0:.2f}".format(avg_train_loss))
        self.log.important("  Training epoch took: {:}".format(training_time))

        return avg_train_loss, training_time

    def _test_step(self, test_dataloader):
        """
        Валидация модели на тестовых данных.

        :param test_dataloader: DataLoader, валидационные данные
        :return: tuple, средние значения потерь, точности и время тестирования
        """
        self.log.important("Running Validation...")
        t0 = time.time()
        self.model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0

        for batch in test_dataloader:
            b_input_ids, b_input_mask, b_labels = (batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device))
            with torch.no_grad():
                outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs.loss
                logits = outputs.logits

            total_eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_eval_accuracy += self._flat_accuracy(logits, label_ids)

        avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
        avg_val_loss = total_eval_loss / len(test_dataloader)
        validation_time = self._format_time(time.time() - t0)
        self.log.info("  Validation Loss: {0:.2f}".format(avg_val_loss))
        self.log.info("  Validation took: {:}".format(validation_time))

        return avg_val_loss, avg_val_accuracy, validation_time

    def _save_model(self):
        """
        Сохранение обученной модели и токенизатора.
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.log.info("Saving model to %s" % self.output_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

    def load_model(self):
        """
        Загрузка модели и токенизатора для дальнейшего использования.
        """
        self.model = BertForSequenceClassification.from_pretrained(self.output_dir)
        self.tokenizer = BertTokenizer.from_pretrained(self.output_dir)
        if torch.cuda.is_available():
            self.model.to(self.device)
