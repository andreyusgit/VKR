from data_processor import DataProcessor
from bert_classifier import BERTClassifier
from svm_classifier import SVMClassifier


def collect_data(file_name: str, data_columns: list, label_column: str):
    # Подготовка данных
    data = DataProcessor(file_name)
    categories, texts = data.load_and_filter_data(data_columns=data_columns, label_column=label_column)
    labels = data.code_labels(categories)  # Преобразование текстовых меток в числовые
    return labels, texts, categories


def run_bert(labels: list, texts: list, epochs: int = 3, batch_size: int = 32, enable_logging: bool = True):
    BERT_model = BERTClassifier('bert-base-uncased', num_labels=len(set(labels)),
                                enable_logging=enable_logging)
    BERT_model.prepare_data(texts, labels)
    BERT_model.train(epochs=epochs, batch_size=batch_size)


def run_svm(labels: list, texts: list, enable_logging: bool = True):
    classifier = SVMClassifier(enable_logging=enable_logging)
    classifier.prepare_data(texts, labels)
    classifier.train()
    results = classifier.evaluate()
    print(results)


if __name__ == '__main__':
    labels, texts, categories = collect_data('news.csv',
                                             data_columns=['title', 'text'], label_column='label')
    run_svm(labels, texts)
    # run_bert(labels, texts, epochs=4)
