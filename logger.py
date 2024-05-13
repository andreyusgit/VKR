from termcolor import colored


class Logger:
    def __init__(self, enable_logging=True):
        """
        Initialize the data processor with the file path.

        :param file_path: str, path to the CSV file containing the dataset
        """
        self.enable_logging = enable_logging

    def info(self, text):
        """
        """
        # Load the dataset
        print("\033[0m{}".format(text)) if self.enable_logging else None

    def important(self, text):
        """
        """
        print("\033[36m{}".format(text)) if self.enable_logging else None

    def warning(self, text):
        """
        """
        print("\033[33m{}".format(text)) if self.enable_logging else None

    def error(self, text):
        """
        """
        print("\033[31m{}".format(text)) if self.enable_logging else None
