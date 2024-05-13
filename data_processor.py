import pandas as pd
import random

from sklearn.preprocessing import LabelEncoder
from logger import Logger


class DataProcessor:
    """
    Data processor for loading and filtering dataset to specific columns.
    """

    def __init__(self, file_path, enable_logging=True):
        """
        Initialize the data processor with the file path.

        :param file_path: str, path to the CSV file containing the dataset
        """
        self.file_path = file_path
        self.log = Logger(enable_logging)
        self.label_encoder = LabelEncoder()

    def load_and_filter_data(self, label_column, data_columns):
        """
        Load data from CSV and filter to keep only the 'x' and 'y' columns.
        :param required_columns: Какие колонки оставить в датасете
        :return: pd.DataFrame, filtered data with columns 'x' and 'y'
        """
        # Load the dataset
        data = pd.read_csv(self.file_path)

        # Проверяем, что все указанные столбцы существуют в DataFrame
        if label_column not in data.columns:
            raise ValueError(f"Label column '{label_column}' not found in DataFrame.")
        if not all(col in data.columns for col in data_columns):
            missing_cols = [col for col in data_columns if col not in data.columns]
            raise ValueError(f"Data columns '{missing_cols}' not found in DataFrame.")

        # Извлечение столбца лейблов и преобразование его в список
        labels = data[label_column].tolist()

        # Объединение данных из указанных столбцов, если их несколько
        if len(data_columns) == 1:
            texts = data[data_columns[0]].tolist()
        else:
            # Создание одного столбца, объединив данные из всех указанных столбцов
            data['combined_text'] = data[data_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
            texts = data['combined_text'].tolist()

        self.log.important('Number of training sentences: {:,}\n'.format(data.shape[0]))

        return labels, texts

    def code_labels(self, categories):
        # Создание и обучение LabelEncoder
        return self.label_encoder.fit_transform(categories)  # Преобразование текстовых меток в числовые

    def decode_labels(self, encoded_labels):
        return self.label_encoder.inverse_transform(encoded_labels)
