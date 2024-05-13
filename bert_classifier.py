import time
import datetime
import random
import torch
import warnings as w
import numpy as np
import os

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler

from logger import Logger

w.filterwarnings('ignore')


class BERTClassifier:
    """
    Classifier using BERT for sequence classification.
    """

    def __init__(self, model_name, num_labels, enable_logging=True, output_dir='./model_save/'):
        """
        Initialize the classifier with the specified BERT model and the device (GPU/CPU).

        :param model_name: str, name of the BERT model to use
        :param num_labels: int, number of labels in the classification task
        """
        self.output_dir = output_dir
        self.log = Logger(enable_logging)
        self.device = self._get_device()
        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels,
                                                                   output_attentions=False,
                                                                   output_hidden_states=False)
        self.train_dataset = None
        self.test_dataset = None

    def _get_device(self):
        # Если в системе есть GPU ...
        if torch.cuda.is_available():
            # Тогда говорим PyTorch использовать GPU.
            device = torch.device("cuda")
            self.log.info('There are %d GPU(s) available.' % torch.cuda.device_count())
            self.log.important('We will use the GPU:' + torch.cuda.get_device_name(0))
        # Если нет GPU, то считаем на обычном процессоре ...
        else:
            self.log.warning('No GPU available, using the CPU instead.')
            device = torch.device("cpu")
        return device

    def _get_max_sentence_len(self, sentences):
        max_len = 0
        # Считаем какой максимальный размер имеет предложение разбитое на токены и разбавленное спец. токенами.
        for sent in sentences:
            # Токенизируем текст и добавляем `[CLS]` и `[SEP]` токены.
            input_ids = self.tokenizer.encode(sent, add_special_tokens=True)
            # Обновляем максимум.
            max_len = max(max_len, len(input_ids))

        self.log.info('Max sentence length: ' + str(max_len))
        # Проверяем, является ли число уже степенью двойки
        if max_len > 0 and (max_len & (max_len - 1)) == 0:
            return max_len
        # Находим следующую степень двойки
        return 1 << (max_len.bit_length())

    def _print_model_params(self):
        """
        Получаем все параметры модели как список кортежей и выводим сводную информацию по модели.
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
        Функция для расчёта точности. Сравниваются предсказания и реальная разметка к данным
        :param preds:
        :param labels:
        :return:
        """
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    @staticmethod
    def _format_time(elapsed):
        """
        На вход время в секундах и возвращается строка в формате hh:mm:ss
        :param elapsed:
        :return:
        """
        # Округляем до ближайшей секунды.
        elapsed_rounded = int(round(elapsed))

        # Форматируем как hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def prepare_data(self, sentences, labels):
        """
        Prepare data for training or evaluation.

        :param sentences: list of str, the texts to process
        :param labels: list of int, corresponding labels for the texts
        :return: DataLoader, prepared data loader for training or evaluation
        """
        input_ids, attention_masks = [], []

        max_len = self._get_max_sentence_len(sentences)
        # Для всех предложений...
        for sent in sentences:
            encoded_dict = self.tokenizer.encode_plus(
                sent,  # Текст для токенизации.
                add_special_tokens=True,  # Добавляем '[CLS]' и '[SEP]'
                max_length=max_len,  # Дополняем [PAD] или обрезаем текст до 64 токенов.
                pad_to_max_length=True,
                return_attention_mask=True,  # Возвращаем также attn. masks.
                return_tensors='pt',  # Возвращаем в виде тензоров pytorch.
            )

            # Добавляем токенизированное предложение в список
            input_ids.append(encoded_dict['input_ids'])
            # И добавляем attention mask в список
            attention_masks.append(encoded_dict['attention_mask'])

        # Конвертируем списки в полноценные тензоры Pytorch.
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels)

        # Печатаем предложение с номером 0, его токены (теперь в виде номеров в словаре) и.т.д.
        self.log.info('Original: ' + sentences[0])
        self.log.info('Token IDs: ' + str(input_ids[0]))
        self.log.info('Attention masks: ' + str(attention_masks[0]))
        self.log.info('Labels: ' + str(labels[0]))

        # Объединяем все тренировочные данные в один TensorDataset.
        dataset = TensorDataset(input_ids, attention_masks, labels)

        # Делаем разделение случайное разбиение 90% - тренировка 10% - валидация.

        # Считаем число данных для тренировки и для валидации.
        train_size = int(0.9 * len(dataset))
        test_size = len(dataset) - train_size

        # Разбиваем датасет с учетом посчитанного количества.
        self.train_dataset, self.test_dataset = random_split(dataset, [train_size, test_size])

        self.log.important('\n\n{:>5,} training samples'.format(train_size))
        self.log.important('{:>5,} validation samples\n\n'.format(test_size))

        return self.train_dataset, self.test_dataset

    def _create_dataloaders(self, batch_size):
        # DataLoader должен знать размер батча для тренировки мы задаем его здесь.
        # Размер батча – это сколько текстов будет подаваться на сеть для вычисления градиентов
        # Авторы BERT предлагают ставить его 16 или 32.

        # Создаем отдельные DataLoaders для наших тренировочного и валидационного наборов

        # Для тренировки мы берем тексты в случайном порядке.
        train_dataloader = DataLoader(
            self.train_dataset,  # Тренировочный набор данных.
            sampler=RandomSampler(self.train_dataset),  # Выбираем батчи случайно
            batch_size=batch_size  # Тренируем с таким размером батча.
        )

        # Для валидации порядок не важен, поэтому зачитываем их последовательно.
        test_dataloader = DataLoader(
            self.test_dataset,  # Валидационный набор данных.
            sampler=SequentialSampler(self.test_dataset),  # Выбираем батчи последовательно.
            batch_size=batch_size  # Считаем качество модели с таким размером батча.
        )
        return train_dataloader, test_dataloader

    def train(self, batch_size=32):
        """
        Train the BERT model.

        :param train_loader: DataLoader, data loader for the training set
        :param val_loader: DataLoader, data loader for the validation set
        :param epochs: int, number of epochs to train
        :param learning_rate: float, learning rate for the optimizer
        """

        if self.train_dataset is None or self.test_dataset is None:
            raise ValueError("Прежде чем обучать модель необходимо подготовить данные. Функция prepare_data")

        train_dataloader, test_dataloader = self._create_dataloaders(batch_size=batch_size)
        if torch.cuda.is_available():
            self.model.cuda()

        self._print_model_params()

        optimizer = AdamW(self.model.parameters(),
                          lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                          eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                          )

        # Количество эпох для тренировки. Авторы BERT рекомендуют от 2 до 4.
        # Мы выбираем 3
        epochs = 3

        # Общее число шагов тренировки равно [количество батчей] x [число эпох].
        total_steps = len(train_dataloader) * epochs

        # Создаем планировщик learning rate (LR). LR будет плавно уменьшаться в процессе тренировки
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,  # Default value in run_glue.py
                                                    num_training_steps=total_steps)

        # В этой переменной сохраним всякую статистику по тренировке: точность, функцию потерь и время выполнения.
        training_stats = []
        # Переменная что бы измерить время всей тренировки.
        total_t0 = time.time()

        # Для каждой эпохи...
        for epoch_i in range(0, epochs):
            # Запустить одну эпоху тренировки (следующий слайд)
            avg_train_loss, training_time = self._train_step(train_dataloader, optimizer, scheduler, epoch_i, epochs)
            # Запустить валидацию что бы проверить качество модели на данном этапе (следующий слайд)
            avg_val_loss, avg_val_accuracy, test_time = self._test_step(test_dataloader)

            # Сохраняем статистику тренировки на данной эпохе.
            training_stats.append(
                {
                    'Epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Validation Loss': avg_val_loss,
                    'Validation Accur.': avg_val_accuracy,
                    'Training Time': training_time,
                    'Test Time': test_time
                }
            )

        self.log.important("\n\n\nTraining complete! Total training took " +
                           "{:} (hh:mm:ss)".format(self._format_time(time.time() - total_t0)))
        self._save_model()

    def _train_step(self, train_dataloader, optimizer, scheduler, epoch_i, epochs):
        self.log.important('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        t0 = time.time()
        total_train_loss = 0
        # Переводим модель в режим тренировки.
        self.model.train()

        # Для каждого батча из тренировочных данных...
        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and not step == 0:
                elapsed = self._format_time(time.time() - t0)
                self.log.info('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader),
                                                                                    elapsed))

            # Извлекаем все компоненты из полученного батча
            b_input_ids, b_input_mask, b_labels = (batch[0].to(self.device),
                                                   batch[1].to(self.device),
                                                   batch[2].to(self.device))
            # Очищаем все ранее посчитанные градиенты (это важно)
            self.model.zero_grad()
            # Выполняем прямой проход по данным
            outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            # Накапливаем тренировочную функцию потерь по всем батчам
            total_train_loss += loss.item()
            # Выполняем обратное распространение ошибки что бы посчитать градиенты.
            loss.backward()
            # Ограничиваем максимальный размер градиента до 1.0. Это позволяет избежать проблемы "exploding gradients".
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            # Обновляем параметры модели используя рассчитанные градиенты с помощью выбранного оптимизатора и текущего learning rate.
            optimizer.step()
            # Обновляем learning rate.
            scheduler.step()

        # Считаем среднее значение функции потерь по всем батчам.
        avg_train_loss = total_train_loss / len(train_dataloader)
        # Сохраняем время тренировки одной эпохи.
        training_time = self._format_time(time.time() - t0)
        self.log.important("  Average training loss: {0:.2f}".format(avg_train_loss))
        self.log.important("  Training epcoh took: {:}".format(training_time))
        return avg_train_loss, training_time

    def _test_step(self, test_dataloader):
        self.log.important("Running Validation...")
        t0 = time.time()
        # Переводим модель в режим evaluation – некоторые слои, например dropout ведут себя по другому.
        self.model.eval()

        # Переменные для подсчёта функции потерь и точности
        total_eval_accuracy = 0
        total_eval_loss = 0
        # Прогоняем все данные из валидации
        for batch in test_dataloader:
            # Извлекаем все компоненты из полученного батча.
            b_input_ids, b_input_mask, b_labels = (batch[0].to(self.device),
                                                   batch[1].to(self.device),
                                                   batch[2].to(self.device))

            # Говорим pytorch что нам не нужен вычислительный граф для подсчёта градиентов (всё будет работать намного быстрее)
            with torch.no_grad():
                # Прямой проход по нейронной сети и получение выходных значений.
                outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask,
                                            labels=b_labels)
                loss = outputs.loss
                logits = outputs.logits

            # Накапливаем значение функции потерь для валидации.
            total_eval_loss += loss.item()

            # Переносим значения с GPU на CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Считаем точность для отдельного батча с текстами и накапливаем значения.
            total_eval_accuracy += self._flat_accuracy(logits, label_ids)

        # Выводим точность для всех валидационных данных.
        avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
        self.log.important("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # Считаем среднюю функцию потерь для всех батчей.
        avg_val_loss = total_eval_loss / len(test_dataloader)
        # Измеряем как долго считалась валидация.
        validation_time = self._format_time(time.time() - t0)
        self.log.info("  Validation Loss: {0:.2f}".format(avg_val_loss))
        self.log.info("  Validation took: {:}".format(validation_time))
        return avg_val_loss, avg_val_accuracy, validation_time

    def _save_model(self):
        # Если она не существует создаем её
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.log.info("Saving model to %s" % self.output_dir)

        # Сохраняем натренированную модель и её токенайзер используя `save_pretrained()`.
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

    def load_model(self):
        # Загружаем натренированную модель и её словарь
        self.model = BertForSequenceClassification.from_pretrained(self.output_dir)
        self.tokenizer = BertTokenizer.from_pretrained(self.output_dir)

        # Отправляем модель на GPU.
        if torch.cuda.is_available():
            self.model.to(self.device)
