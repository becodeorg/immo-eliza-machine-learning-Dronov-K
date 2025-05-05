from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor, XGBRFRegressor
import pandas as pd


def compare_models(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """
    Trains and evaluates multiple regression models on provided data.

    Each model is fitted on the training set and evaluated on the test set using
    Mean Absolute Error (MAE) and R² score. Results are returned as a sorted DataFrame.

    :param X_train: Features for training (array-like or DataFrame)
    :param X_test: Features for testing (array-like or DataFrame)
    :param y_train: Target values for training (array-like)
    :param y_test: Target values for testing (array-like)
    :return: DataFrame with model names, MAE, and R² scores, sorted by MAE
    """
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "DecisionTree": DecisionTreeRegressor(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "KNeighbors": KNeighborsRegressor(),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
        "XGBRegressor": XGBRegressor(),
        "XGBRFRegressor": XGBRFRegressor()

    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results.append({
            "Model": name,
            "MAE": mae,
            "R²": r2
        })

    return pd.DataFrame(results).sort_values(by="MAE")
