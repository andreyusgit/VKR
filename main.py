from data_processor import DataProcessor
from bert_classifier import BERTClassifier
from svm_classifier import SVMClassifier


def collect_data(file_name: str, data_columns: list, label_column: str):
    # Подготовка данных
    data = DataProcessor(file_name)
    categories, texts = data.load_and_filter_data(data_columns=data_columns, label_column=label_column)
    labels = data.code_labels(categories)  # Преобразование текстовых меток в числовые
    return labels, texts, categories


def run_bert(labels: list, texts: list, texts2, labels2, epochs: int = 3, batch_size: int = 32, enable_logging: bool = True):
    BERT_model = BERTClassifier('bert-base-uncased', num_labels=len(set(labels)),
                                enable_logging=enable_logging)
    BERT_model.prepare_data(texts, labels, texts2, labels2)
    BERT_model.train(epochs=epochs, batch_size=batch_size)


def run_svm(texts: list, labels: list, texts2, labels2, enable_logging: bool = True):
    classifier = SVMClassifier(enable_logging=enable_logging)
    classifier.prepare_data(texts, labels, texts2, labels2)
    classifier.train()
    results = classifier.evaluate()
    print(results)


if __name__ == '__main__':
    labels, texts, categories = collect_data('news.csv',
                                             data_columns=['title', 'text'], label_column='label')
    labels2, texts2, categories2 = collect_data('fake_and_real_news.csv',
                                             data_columns=['Text'], label_column='label')
    run_svm(texts, labels, texts2, labels2)
    run_bert(labels, texts, texts2, labels2, epochs=4)
