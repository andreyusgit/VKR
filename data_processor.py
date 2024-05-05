import pandas as pd


class DataProcessor:
    """
    Data processor for loading and filtering dataset to specific columns.
    """

    def __init__(self, file_path):
        """
        Initialize the data processor with the file path.

        :param file_path: str, path to the CSV file containing the dataset
        """
        self.file_path = file_path

    def load_and_filter_data(self):
        """
        Load data from CSV and filter to keep only the 'x' and 'y' columns.

        :return: pd.DataFrame, filtered data with columns 'x' and 'y'
        """
        # Load the dataset
        data = pd.read_csv(self.file_path)

        # Check if the specified columns exist in the dataframe
        required_columns = ['Ticket Type', 'Ticket Description']
        if not all(column in data.columns for column in required_columns):
            raise ValueError(f"The dataset must contain the columns: {required_columns}")

        # Filter necessary columns
        filtered_data = data[required_columns]

        return filtered_data
