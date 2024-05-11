from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd


class CustomDataset(Dataset):
    """Класс для создания PyTorch-совместимого датасета."""
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten().to(device),
            'attention_mask': encoding['attention_mask'].flatten().to(device),
            'labels': torch.tensor(label, dtype=torch.long).to(device)
        }


# Определение устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))


# Загрузка токенизатора и модели
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Подготовка данных
data = pd.read_csv('customer_support_tickets.csv')
texts = data['Ticket Description'].tolist()   # Примеры текстов
categories = data['Ticket Type'].tolist()  # Категории в текстовом формате

# Создание и обучение LabelEncoder
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(categories)  # Преобразование текстовых меток в числовые

# Разделение данных
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Создание тренировочного и тестового датасетов
train_dataset = CustomDataset(train_texts, train_labels, tokenizer)
test_dataset = CustomDataset(test_texts, test_labels, tokenizer)

# Настройка модели
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(set(labels))).to(device)

# Параметры обучения
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Обучение
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

print(torch.cuda.memory_allocated())
print(torch.cuda.memory_reserved())

trainer.train()

# Оценка модели на тестовом наборе
results = trainer.evaluate()
print(results)


# Функция для преобразования числовых меток обратно в текстовые
def decode_labels(encoded_labels):
    return label_encoder.inverse_transform(encoded_labels)


# Пример использования функции
# predicted_labels_numeric = [0, 2, 1]  # предполагаемые числовые метки от модели
# predicted_labels_text = decode_labels(predicted_labels_numeric)
# print(predicted_labels_text)
