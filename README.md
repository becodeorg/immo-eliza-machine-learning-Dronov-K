# 🏠 ImmoEliza Price Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue) ![License](https://img.shields.io/badge/License-MIT-green)

## 📝 Description
The **ImmoEliza Price Prediction** project is designed to help the real estate company _ImmoEliza_ predict property prices across Belgium using machine learning techniques. The project focuses on building and training a machine learning model that can accurately predict property prices based on various features from the dataset.

The dataset, referred to as the _Kangaroo_ dataset, contains a variety of information about properties, including location, type, and size. The project is structured into the following main steps:

1. **Data Cleaning**: The dataset is pre-processed by removing duplicates, handling missing values, and ensuring there are no errors or blank spaces.
    
2. **Feature Engineering**: Key features from the dataset are transformed and prepared for the model, including encoding categorical variables and scaling numerical ones.
    
3. **Model Training**: A machine learning model (e.g., Random Forest or XGBoost) is trained using the cleaned and processed data to predict property prices.
    
4. **Model Evaluation**: The performance of the trained model is evaluated to determine its accuracy and generalization ability.



## 🌳 Project Structure

```
hangman/
│
├── datasets/
|   └── Kangaroo.csv
├── src/
│   ├── __init__.py
|   ├── cleaner.py
|   ├── encoder.py
|   └── model.py
├── __init__.py
├── main.py
├── README.md  
└── requirements.txt   
```

## 🚀 Installation and Execution

1. **Clone the repository:**
```bash
git clone https://github.com/becodeorg/immo-eliza-machine-learning-Dronov-K.git
cd immo-eliza-machine-learning-Dronov-K
```

2. Create and activate a virtual environment (optional):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run:

```sh
python main.py
```



## ✅ TODO

-  **Enhance Data Cleaning Process**: Improve the handling of missing values, and outliers. Implement more advanced techniques such as interpolation, imputation for missing data. Ensure data consistency across all features and columns.
- **Feature Engineering:** Add additional transformations to prepare features for the model, such as scaling or handling outliers.


## ⚖️ License

This project is licensed under the MIT License.