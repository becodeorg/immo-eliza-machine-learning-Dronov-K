import pandas as pd
from src.cleaner import DataCleaner
from src.visualizer import DataVisualizer
from src.encoder import DataEncode
from src.model import ModelTrainer
from src.model_utils import compare_models
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

if __name__ == '__main__':
    # file_path = 'datasets/Kangaroo.csv'
    # write_path = 'datasets/cleaned_kangaroo.csv'
    # cleaner = DataCleaner(file_path)
    # cleaner.remove_duplicates('id')
    # cleaner.remove_columns_by_missing_percentage(
    #     percent=100,
    #     to_drop=['url', 'id'])
    # cleaner.remove_spaces()
    # cleaner.remove_by_column_values('type', ['Apartment_group', 'House_group'])
    # cleaner.replace_rare_values('epcScore')
    # cleaner.handle_errors()
    # cleaner.handle_missing_values()
    # cleaner.write_to_csv(write_path)
    # cleaned_data = pd.read_csv('datasets/cleaned_kangaroo.csv')

    # visualizer = DataVisualizer(cleaned_data)
    # visualizer.plot_correlation_heatmap(target_column='price')
    # cleaned_data = cleaned_data.drop(
    #     columns=['kitchenSurface', 'terraceOrientation',
    #              'streetFacadeWidth', 'gardenOrientation',
    #              'floorCount', 'roomCount', 'diningRoomSurface', 'hasDiningRoom', 'hasDressingRoom'])
    #
    # encoder = DataEncode(cleaned_data)
    # encoder.one_hot_encoding(
    #     ['type', 'province', 'locality', 'subtype', 'kitchenType', 'buildingCondition', 'floodZoneType', 'heatingType'])
    # encoder.label_encoding(['epcScore'])
    # encode_df = encoder.get_encode_data()
    # encode_df.to_csv('datasets/encoding_kangaroo.csv')

    encode_df = pd.read_csv('datasets/encoding_kangaroo.csv')
    model_trainer = ModelTrainer(encode_df, 'price')
    # model_trainer.apply_sample(sample_size=25000)
    model_trainer.split_data()
    # result = compare_models(model_trainer.X_train, model_trainer.X_test, model_trainer.y_train, model_trainer.y_test)
    # print(result)
    # model_trainer.find_best_hyperparameters(
    #     model=XGBRegressor(),
    #     param_grid={
    #         'model__n_estimators': [100, 200],
    #         'model__learning_rate': [0.05, 0.1],
    #         'model__max_depth': [3, 5]
    #     }
    # )
    model_trainer.find_best_hyperparameters(
        model=RandomForestRegressor(),
        param_grid={
            'model__n_estimators': [100, 200],
            'model__max_depth': [20, 30],
            'model__min_samples_leaf': [1, 2],
        }
    )
    model_trainer.train()
    model_trainer.evaluate()

    predictions = model_trainer.predict()
    print(predictions)
