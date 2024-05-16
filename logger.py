class Logger:
    def __init__(self, enable_logging=True):
        """
        Инициализация логгера с возможностью включения или отключения логирования.

        :param enable_logging: bool, указывает, следует ли выводить логи (True - выводить логи, False - не выводить)
        """
        self.enable_logging = enable_logging

    def info(self, text):
        """
        Вывод информационного сообщения.

        :param text: str, текст информационного сообщения для вывода
        """
        # Выводит текст с обычным стилем
        print("\033[0m{}".format(text)) if self.enable_logging else None

    def important(self, text):
        """
        Вывод важного сообщения.

        :param text: str, текст важного сообщения для вывода
        """
        # Выводит текст с синим цветом
        print("\033[36m{}".format(text)) if self.enable_logging else None

    def warning(self, text):
        """
        Вывод предупреждения.

        :param text: str, текст предупредительного сообщения для вывода
        """
        # Выводит текст с жёлтым цветом
        print("\033[33m{}".format(text)) if self.enable_logging else None

    def error(self, text):
        """
        Вывод сообщения об ошибке.

        :param text: str, текст сообщения об ошибке для вывода
        """
        # Выводит текст с красным цветом
        print("\033[31m{}".format(text)) if self.enable_logging else None
