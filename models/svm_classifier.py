import nltk
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from helpers.logger import Logger

# Загрузка ресурсов nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


class SVMClassifier:
    """
    Классификатор, использующий SVM для классификации текстов.
    """

    def __init__(self, enable_logging=True):
        """
        Инициализация классификатора с опциональным логированием.

        :param enable_logging: bool, активация логирования
        """
        self._log = Logger(enable_logging)
        self._vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        self._model = svm.SVC()
        self._lemmatizer = WordNetLemmatizer()
        self._stop_words = set(stopwords.words('english'))
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def _preprocess_text(self, text):
        """
        Предварительная обработка текста: токенизация, удаление стоп-слов и лемматизация.

        :param text: str, исходный текст
        :return: str, обработанный текст
        """
        words = word_tokenize(text)
        words = [self._lemmatizer.lemmatize(word.lower()) for word in words if
                 word.isalpha() and word.lower() not in self._stop_words]
        return ' '.join(words)

    def prepare_data(self, texts, labels, test_size=0.25, random_state=42):
        """
        Подготовка данных для обучения и тестирования.

        :param texts: list of str, тексты для обработки
        :param labels: list of int, соответствующие метки для текстов
        :param test_size: float, доля тестовой выборки
        :param random_state: int, сид для случайного разбиения данных
        """
        preprocessed_texts = [self._preprocess_text(text) for text in texts]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            preprocessed_texts, labels, test_size=test_size, random_state=random_state
        )

        self.X_train = self._vectorizer.fit_transform(self.X_train)
        self.X_test = self._vectorizer.transform(self.X_test)

        self._log.info(
            f'Training data prepared: {self.X_train.shape[0]} training samples, {self.X_test.shape[0]} test samples')

    def train(self):
        """
        Обучение модели SVM с подбором гиперпараметров.
        """
        parameters = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly'], 'gamma': ['scale', 'auto']}

        def log_best_params(gs):
            self._log.info(f"Best parameters found: {gs.best_params_}")
            self._log.info(f"Best cross-validation score: {gs.best_score_:.2f}")

        grid_search = GridSearchCV(self._model, parameters, cv=3, scoring='accuracy', verbose=3)
        grid_search.fit(self.X_train, self.y_train)

        self._model = grid_search.best_estimator_

        log_best_params(grid_search)
        self._log.important('Model trained successfully with best parameters.')

    def evaluate(self):
        """
        Оценка модели на тестовых данных.

        :return: dict, содержащий метрики классификации
        """
        predictions = self._model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        report = classification_report(self.y_test, predictions, output_dict=True)
        self._log.important(f'Evaluation completed. Accuracy: {accuracy:.2f}')
        return {'accuracy': accuracy, 'report': report}
