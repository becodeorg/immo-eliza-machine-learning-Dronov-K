import random
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter


class DataVisualizer:
    """
    A class to visualize real estate data
    """

    def __init__(self, df):
        """
        Initializes the DataVisualizer object with the DataFrame and regions for Flanders, Wallonia, and Brussels.

        :param df: pandas DataFrame, the original dataset containing real estate data.
        """
        self.df = df.copy()
        self._flanders_provinces = ['West flanders', 'East flanders', 'Antwerp', 'Flemish brabant']
        self._wallonia_provinces = ['Hainaut', 'Liège', 'Walloon brabant', 'Namur', 'Luxembourg']
        self._brussels_province = 'Brussels'

    def _filter_by_region(self, region: str) -> pd.DataFrame:
        """
        Filters the data by region (Flanders, Wallonia, or Brussels).

        :param region: str, the name of the region ('Flanders', 'Wallonia', 'Brussels')
        :return: pandas DataFrame, the filtered DataFrame based on the region
        :raises ValueError: if an invalid region is provided.
        """
        if region == 'Flanders':
            return self.df[self.df['province'].isin(self._flanders_provinces)]
        elif region == 'Wallonia':
            return self.df[self.df['province'].isin(self._wallonia_provinces)]
        elif region == 'Brussels':
            return self.df[self.df['province'] == self._brussels_province]
        else:
            raise ValueError("Region must be 'Flanders', 'Wallonia', or 'Brussels'")

    @staticmethod
    def _format_large_numbers(x: int | float, pos) -> str:
        """
        Formats numbers with thousands separators for better readability.

        :param x: number to be formatted.
        :return: str, formatted number as a string.
        """
        return f'{x:,.0f}'

    def _collect_data(self, region: str) -> pd.DataFrame:
        """
        Collects data on localities for the specified region.

        :param region: str, the name of the region ('Flanders', 'Wallonia', 'Brussels').
        :return: pandas DataFrame, collected data by locality.
        """
        filtered_data = self._filter_by_region(region)
        filtered_data['price_per_m2'] = filtered_data['price'] / filtered_data['habitableSurface']
        collected_data = filtered_data.groupby('locality').agg(
            average_price=('price', 'mean'),
            median_price=('price', 'median'),
            price_per_m2=('price_per_m2', 'mean')
        ).reset_index()
        return collected_data

    def plot_most_least_expensive_locality(self, region: str, metric: str = 'average_price', type_of_sort: bool = False,
                                           top: int = 10) -> None:
        """
        Plots a graph for the most expensive or least expensive localities in the given region.

        :param region: str, the name of the region ('Flanders', 'Wallonia', 'Brussels').
        :param metric: str, the metric to sort by ('average_price', 'median_price', 'price_per_m2') (default is 'average_price').
        :param type_of_sort: bool, if True, sort in ascending order; if False, sort in descending order (default is False).
        :param top: int, the number of top localities to display (default is 10).
        :return: None
        """
        if metric not in ('average_price', 'median_price', 'price_per_m2'):
            raise ValueError("Metric must be one of: 'average_price', 'median_price', 'price_per_m2'")

        collect_data = self._collect_data(region)
        most_expensive = collect_data.sort_values(metric, ascending=type_of_sort).head(top)

        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x=metric, y='locality', data=most_expensive, palette='Blues_r' if type_of_sort else 'Blues_d')
        order = 'Least' if type_of_sort else 'Most'
        plt.title(f"{order} Expensive Locality in {region} ({metric.replace('_', ' ').title()})",
                  fontsize=18,
                  fontweight='bold')
        plt.xlabel(f"{metric.replace('_', ' ').title()} (€)", fontsize=14, fontweight='bold', )
        plt.ylabel('Locality', fontsize=14, fontweight='bold')
        ax.xaxis.set_major_formatter(FuncFormatter(self._format_large_numbers))

        #  Automatically adjusts the arrangement of elements on the chart
        plt.tight_layout()
        plt.show()

    def plot_property_types_by_province(self) -> None:
        """
        Plots a graph showing the number of houses and apartments in each province.

        :return: None
        """

        # Grouping data by province and property type
        property_amount = self.df.groupby(['province', 'type']).size().unstack(fill_value=0)

        # Reset index
        property_amount = property_amount.reset_index()

        # melt turns a "wide" table into a "long" one. We transform each column
        # (property type) into a separate row with a type label and a value.
        property_amounts_melted = property_amount.melt(id_vars="province", value_vars=property_amount.columns[1:],
                                                       var_name='Property Type', value_name='Count')

        plt.figure(figsize=(12, 8))
        sns.barplot(x='province', y='Count', hue='Property Type', data=property_amounts_melted,
                    palette=self._choose_random_palette_theme())

        plt.title('Amount of Houses and Apartments by Province', fontsize=18, fontweight='bold', fontstyle='italic',
                  family='fantasy')
        plt.xlabel('Province', fontsize=14, fontweight='bold', fontstyle='oblique', family='monospace')
        plt.ylabel('Amount of Properties', fontsize=14, fontweight='bold', fontstyle='oblique', family='monospace')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Property Type', loc='upper left')

        plt.tight_layout()
        plt.show()

    def plot_correlation_heatmap(self, target_column: str = None) -> None:
        """
        Plots a heatmap showing correlations between all features and optionally highlights correlations
        with a specified target column (e.g., 'price').

        :param target_column: str, optional name of the column to focus on (default is None).
        :return: None
        """
        corr_matrix = self._mixed_correlation_matrix()

        if target_column:
            if target_column not in corr_matrix.columns:
                raise ValueError(f"Column '{target_column}' not found in data.")

            target_corr = corr_matrix[[target_column]].dropna().sort_values(by=target_column, ascending=False)

            plt.figure(figsize=(6, len(target_corr) * 0.5))
            sns.heatmap(target_corr,
                        annot=True,
                        cmap='coolwarm',
                        fmt='.2f',
                        linewidths=0.5,
                        linecolor='gray',
                        cbar=True)
            plt.title(f"Correlation with '{target_column}'", fontsize=16, fontweight='bold')

        else:
            corr_matrix = corr_matrix.loc[corr_matrix.columns, corr_matrix.columns]

            plt.figure(figsize=(max(10, len(corr_matrix) * 0.6), max(8, len(corr_matrix) * 0.5)))
            sns.heatmap(corr_matrix,
                        annot=True,
                        cmap='coolwarm',
                        fmt='.2f',
                        linewidths=0.5,
                        linecolor='gray',
                        cbar=True)
            plt.title("Correlation Heatmap", fontsize=16, fontweight='bold')

        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _cramers_v(x: pd.Series, y: pd.Series) -> float:
        """
        Computes Cramér's V statistic for categorical-categorical association.

        :param x: pd.Series of categorical values
        :param y: pd.Series of categorical values
        :return: float, Cramér's V value
        """
        confusion_matrix = pd.crosstab(x, y)
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        r, k = confusion_matrix.shape
        return np.sqrt(chi2 / (n * (min(k - 1, r - 1)))) if n > 0 else 0

    @staticmethod
    def _correlation_ratio(categories: pd.Series, measurements: pd.Series) -> float:
        """
        Computes correlation ratio (eta squared) for categorical → numerical relationships.

        :param categories: pd.Series of categorical values
        :param measurements: pd.Series of numeric values
        :return: float, correlation ratio
        """
        fcat, _ = pd.factorize(categories)
        cat_num = np.max(fcat) + 1
        y_avg = np.mean(measurements)
        numerator = sum([
            np.sum((measurements[np.argwhere(fcat == i).flatten()] - np.mean(
                measurements[np.argwhere(fcat == i).flatten()])) ** 2)
            for i in range(cat_num)
        ])
        denominator = np.sum((measurements - y_avg) ** 2)
        return 1 - numerator / denominator if denominator != 0 else 0

    def _mixed_correlation_matrix(self) -> pd.DataFrame:
        """
        Computes a correlation matrix that supports both numeric and categorical features.

        :return: pd.DataFrame, mixed-type correlation matrix
        """
        for col in self.df.select_dtypes(include='object').columns:
            self.df[col] = self.df[col].astype('category')

        cols = self.df.columns
        n = len(cols)
        corr = pd.DataFrame(np.zeros((n, n)), columns=cols, index=cols)

        for i in range(n):
            for j in range(n):
                col1, col2 = self.df[cols[i]], self.df[cols[j]]
                if pd.api.types.is_numeric_dtype(col1) and pd.api.types.is_numeric_dtype(col2):
                    corr.iloc[i, j] = col1.corr(col2)
                elif pd.api.types.is_categorical_dtype(col1) or col1.dtype == object:
                    if pd.api.types.is_numeric_dtype(col2):
                        corr.iloc[i, j] = self._correlation_ratio(col1, col2)
                    else:
                        corr.iloc[i, j] = self._cramers_v(col1, col2)
                elif pd.api.types.is_numeric_dtype(col1) and (
                        pd.api.types.is_categorical_dtype(col2) or col2.dtype == object):
                    corr.iloc[i, j] = self._correlation_ratio(col2, col1)
        return corr

    @staticmethod
    def _choose_random_palette_theme() -> str:
        """
        Chooses a random categorical palette theme from Seaborn's predefined palettes.

        :return: str, a random palette theme name.
        """
        categorical_palettes = list(sns.palettes.SEABORN_PALETTES.keys())

        return random.choice(categorical_palettes)
