from data_processor import DataProcessor
from bert_classifier import BERTClassifier
from sklearn.model_selection import train_test_split
import torch


def main():

    # Получение датасета через DataProcessor
    data_processor = DataProcessor('customer_support_tickets.csv')
    data = data_processor.load_and_filter_data()

    # Разделение на признаки и метки
    texts = data['Ticket Type'].tolist()  # Замените 'text_column_name' на имя столбца с текстами
    labels = data['Ticket Description'].tolist()  # Замените 'label_column_name' на имя столбца с метками

    # Разделение на тренировочный и тестовый датасеты
    texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # Создание устройства для обучения на GPU, если доступно
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classifier = BERTClassifier("bert-base-multilingual-cased", num_labels=5, device=device)
    train_loader = classifier.prepare_data(texts_train, labels_train)
    val_loader = classifier.prepare_data(texts_test, labels_test)
    classifier.train(train_loader, val_loader)


if __name__ == '__main__':
    main()