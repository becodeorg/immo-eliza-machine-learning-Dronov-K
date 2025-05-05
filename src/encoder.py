import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


class DataEncode:
    """
    Class for encoding categorical data using different strategies:
    one-hot encoding, label encoding, and target encoding.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the encoder with a DataFrame.

        :param df: Input pandas DataFrame to be encoded.
        """
        self.df = df

    def one_hot_encoding(self, columns: list[str] = None) -> None:
        """
        Apply one-hot encoding to the specified columns.
        Adds binary columns for each category level (excluding the first one).

        :param columns: List of column names to encode (default is None).
        :raises KeyError: If any of the specified columns are not in the DataFrame.
        """
        if columns:
            self.check_data(columns)
            self.df = pd.get_dummies(self.df, columns=columns, drop_first=True)

    def label_encoding(self, columns: list[str] = None) -> None:
        """
        Apply label encoding to the specified columns.
        Converts categorical values to integer labels.

        :param columns: List of column names to encode (default is None).
        :raises KeyError: If any of the specified columns are not in the DataFrame.
        """

        if columns:
            self.check_data(columns)

            for col in columns:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col])

    def target_encoding(self, columns: list[str] = None, target: str = None) -> None:
        """
        Apply target (mean) encoding to the specified columns.
        Replaces each category with the mean of the target variable.

        :param columns: List of categorical columns to encode (default is None).
        :param target: Name of the target variable column (default is None).
        :raises KeyError: If any of the specified columns or the target are not in the DataFrame.
        """

        if columns and target:
            self.check_data(columns + [target])

            for col in columns:
                means = self.df.groupby(col)[target].mean()
                self.df[col] = self.df[col].map(means)

    def check_data(self, columns: list[str]):
        """
        Check if all specified columns exist in the DataFrame.

        :param columns: List of column names to check.
        :raises KeyError: If any column is missing from the DataFrame.
        """
        missing = [col for col in columns if col not in self.df.columns]

        if missing:
            raise KeyError(f"Columns: {missing} not found in DataFrame")

    def get_encode_data(self) -> pd.DataFrame:
        """
        Return the encoded DataFrame.

        :return: Encoded pandas DataFrame.
        """
        return self.df
