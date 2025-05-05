from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score


class ModelTrainer:
    """
    Class for building, training and tuning an ML model using a pipeline and GridSearchCV.
    The K-Nearest Neighbors (KNN) classifier is used.
    """

    def __init__(self, df, target_column: str):
        """
        Initializing a class.

        :param df: Processed DataFrame without gaps and with encoded features.
        :param target_column: Name of the target feature (what we are predicting).
        """
        self.df = df
        self.target_column = target_column
        self.model = None  # Best classifier found after GridSearchCV
        self.pipeline = None
        self.X_train = self.X_test = self.y_train = self.y_test = None

    def apply_sample(self, sample_size: int, random_state: int = 42) -> None:
        """
        Apply a sample to the working DataFrame for faster experimentation.
        The original df remains untouched.

        :param sample_size: Number of rows to include in the sample.
        :param random_state: Random state for reproducibility.
        :raises ValueError: if sample_size is greater than or equal to the dataset size.
        """
        if sample_size >= len(self.df):
            raise ValueError(f"Sample size ({sample_size}) must be less than dataset size ({len(self.df)}).")

        self.df = self.df.sample(n=sample_size, random_state=random_state)

    def split_data(self, test_size=0.25, random_state=42) -> None:
        """
        Splitting data into training and testing samples.

        :param test_size: Proportion of test sample (default is 0.25).
        :param random_state: Для воспроизводимости результата (default is 42).
        :return: None
        """
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    def build_pipeline(self, model) -> None:
        """
        Building a sklearn pipeline with scaling and classification.

        :param model: Regression model to use in the pipeline.
        :raises ValueError: If model is None.
        :return: None
        """
        if model is None:
            raise ValueError("You must provide a model to build the pipeline.")

        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])

    def find_best_hyperparameters(self, model=None, param_grid=None, cv: int = 5,
                                  scoring: str = "neg_mean_absolute_error") -> None:
        """
        Finding the best hyperparameters using GridSearchCV.

        :param model: Model to use in pipeline. Required if pipeline is not already built (default is None).
        :param param_grid: Dictionary of parameters to search through (default is None).
        :param cv: Number of folds for cross-validation (default is 2).
        :param scoring: Metric for optimization (default is "mae").
        :raises ValueError if cv param less than 2 and if model not.
        :return: None
        """
        if cv < 2:
            raise ValueError("cv must be at least 2 or higher")

        if self.pipeline is None:
            if model is None:
                raise ValueError("Need to choose model")
            self.build_pipeline(model)

        if param_grid is None:
            param_grid = {}

        grid_search = GridSearchCV(
            estimator=self.pipeline,
            param_grid=param_grid,
            cv=cv,  # Cross-validation
            scoring=scoring,
            n_jobs=-1,  # Parallelization (all available cores)
            verbose=3
        )

        grid_search.fit(self.X_train, self.y_train)
        self.model = grid_search.best_estimator_  # The best model is saved

    def train(self) -> None:
        """
        Train the model based on the best hyperparameters (if GridSearch was called).

        :raises ValueError: If no model has been selected via find_best_hyperparameters().
        """
        if self.model is None:
            raise ValueError("No model to train. Use find_best_hyperparameters() first.")
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        """
        Generate predictions on the test set using the trained model or pipeline.

        If a model (e.g., result of GridSearchCV) is available, it is used for prediction.
        Otherwise, the raw pipeline is used, assuming it was previously trained.

        :return: NumPy array of predicted values for X_test.
        :raises ValueError: If neither a trained model nor pipeline is available.
        """
        if self.model:
            return self.model.predict(self.X_test)
        elif self.pipeline:
            return self.pipeline.predict(self.X_test)
        raise ValueError("Model is not trained yet")

    def evaluate(self) -> float:
        """
        Evaluate the trained model on the test set using MAE.

        :return: mean absolute error
        """
        y_pred = self.predict()
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"R² Score: {r2:.2f}")
        return mae
