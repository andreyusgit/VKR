from data_processor import DataProcessor
from bert_classifier import BERTClassifier
from svm_classifier import SVMClassifier


# Подготовка данных
data = DataProcessor('spam_email.csv')
categories, texts = data.load_and_filter_data(data_columns=['MESSAGE'], label_column='CATEGORY')
labels = data.code_labels(categories)  # Преобразование текстовых меток в числовые

# BERT_model = BERTClassifier('bert-base-uncased', num_labels=len(set(labels)))
# train_dataset, test_dataset = BERT_model.prepare_data(texts, labels)
# BERT_model.train()

classifier = SVMClassifier(enable_logging=True)
classifier.prepare_data(texts, labels)
classifier.train()
results = classifier.evaluate()
print(results)

# Пример использования функции
# predicted_labels_numeric = [0, 2, 1]  # предполагаемые числовые метки от модели
# predicted_labels_text = decode_labels(predicted_labels_numeric)
# print(predicted_labels_text)
