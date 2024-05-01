"""
Este módulo se encarga de entrenar un modelo de regresión
utilizando Random Forest.
Carga los datos procesados, ajusta el modelo utilizando
búsqueda aleatoria de hiperparámetros y guarda el modelo entrenado.
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import joblib


def train_model(input_path):
    """
    Entrena un modelo de regresión y guarda el modelo entrenado.

    Carga datos procesados, separa en características y etiqueta,
    divide en conjuntos de entrenamiento y prueba, configura y
    ejecuta RandomizedSearchCV para encontrar los mejores
    hiperparámetros, y finalmente guarda el mejor modelo encontrado.

    Parameters:
    input_path (str): Ruta al archivo CSV de entrada con los datos procesados.

    Raises:
    FileNotFoundError: Si el archivo de entrada no se encuentra en la ruta
    especificada.
    Exception: Para errores generales durante el entrenamiento del modelo.
    """
    try:
        data = pd.read_csv(input_path)
        features = data.drop('SalePrice', axis=1)
        target = data['SalePrice']

        features_train, _, target_train, _ = train_test_split(
            features, target, test_size=0.2, random_state=42)

        model = RandomForestRegressor()
        param_dist = {
            'n_estimators': [50, 100, 200, 300, 500],
            'max_depth': [3, 5, 10, 15, 20, 30],
            'min_samples_split': [5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 6, 10],
            'max_features': ['sqrt', 'log2'],
            'bootstrap': [True, False]
        }
        random_search = RandomizedSearchCV(
            model,
            param_distributions=param_dist,
            n_iter=100,
            cv=5,
            scoring='neg_mean_squared_error',
            random_state=42,
            n_jobs=-1
        )
        random_search.fit(features_train, target_train)

        print("Mejores hiperparámetros:", random_search.best_params_)
        joblib.dump(random_search.best_estimator_, 'model_rf.joblib')

    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"No se pudo encontrar el archivo en ruta: {input_path}")from exc
    except Exception as exc:
        raise RuntimeError(
            f"Error durante el entrenamiento del modelo: {str(exc)}")from exc
