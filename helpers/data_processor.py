import pandas as pd

from sklearn.preprocessing import LabelEncoder
from helpers.logger import Logger


class DataProcessor:
    """
    Класс для обработки данных, загрузки и фильтрации набора данных по специфическим колонкам.
    """

    def __init__(self, file_path, enable_logging=True):
        """
        Инициализация обработчика данных с указанием пути к файлу.

        :param file_path: str, путь к CSV файлу, содержащему набор данных
        :param enable_logging: bool, флаг для включения логирования
        """
        self.file_path = file_path
        self._log = Logger(enable_logging)  # Создание экземпляра логгера
        self._label_encoder = LabelEncoder()  # Инициализация кодировщика меток

    def load_and_filter_data(self, label_column, data_columns):
        """
        Загрузка данных из CSV и фильтрация для сохранения только необходимых колонок.

        :param label_column: str, название колонки с метками (целевая переменная)
        :param data_columns: list, список колонок с данными, которые нужно сохранить
        :return: tuple, содержащий список меток и список текстов (данные из указанных колонок)
        """
        try:
            data = pd.read_csv(self.file_path)  # Загрузка датасета
        except FileNotFoundError as e:
            self._log.error(f"File not found: {self.file_path}")
            raise e
        except pd.errors.EmptyDataError as e:
            self._log.error(f"Empty data error: {self.file_path}")
            raise e
        except Exception as e:
            self._log.error(f"Error loading data: {e}")
            raise e

        # Проверка наличия необходимых колонок в датасете
        if label_column not in data.columns:
            error_msg = f"Label column '{label_column}' not found in DataFrame."
            self._log.error(error_msg)
            raise ValueError(error_msg)
        if not all(col in data.columns for col in data_columns):
            missing_cols = [col for col in data_columns if col not in data.columns]
            error_msg = f"Data columns '{missing_cols}' not found in DataFrame."
            self._log.error(error_msg)
            raise ValueError(error_msg)

        # Удаление строк с пустыми значениями в указанных колонках
        data = data.dropna(subset=[label_column] + data_columns)
        if data.empty:
            error_msg = "No data left after dropping rows with missing values."
            self._log.error(error_msg)
            raise ValueError(error_msg)

        # Извлечение меток и данных из указанных колонок
        labels = data[label_column].tolist()
        if len(data_columns) == 1:
            texts = data[data_columns[0]].tolist()
        else:
            data['combined_text'] = data[data_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
            texts = data['combined_text'].tolist()

        # Логирование количества обработанных предложений
        self._log.important('Number of training sentences: {:,}\n'.format(len(texts)))

        return labels, texts

    def code_labels(self, categories):
        """
        Кодирование меток в числовые значения с помощью LabelEncoder.

        :param categories: list, список категорий (текстовые метки) для кодирования
        :return: array, массив закодированных числовых меток
        """
        return self._label_encoder.fit_transform(categories)

    def decode_labels(self, encoded_labels):
        """
        Декодирование числовых меток обратно в текстовые.

        :param encoded_labels: ndarray, массив числовых меток для декодирования
        :return: list, список декодированных текстовых меток
        """
        return self._label_encoder.inverse_transform(encoded_labels)