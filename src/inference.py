"""
Este módulo se encarga de realizar inferencias en batch utilizando un modelo
entrenado.
Carga un modelo de regresión desde un archivo .joblib, aplica el modelo a
nuevos datos y guarda las predicciones en un archivo CSV.
"""

import pandas as pd
import joblib


def make_predictions(data_path, model_path, output_path):
    """
    Carga datos, un modelo entrenado y realiza predicciones en batch.

    Carga los datos desde un archivo CSV, utiliza un modelo guardado para
    hacer predicciones y guarda las predicciones en otro archivo CSV.

    Parameters:
    data_path (str): Ruta al archivo CSV de los datos de inferencia.
    model_path (str): Ruta al archivo .joblib del modelo entrenado.
    output_path (str): Ruta al archivo CSV donde se guardarán las predicciones.

    Raises:
    FileNotFoundError: Si alguno de los archivos de entrada no se encuentra
    en la ruta especificada.
    RuntimeError: Para errores durante el proceso de inferencia que no
    sean de E/S.
    """
    try:
        # Carga de datos de inferencia
        data = pd.read_csv(data_path)

        # Carga del modelo entrenado
        model = joblib.load(model_path)

        # Realización de predicciones
        predictions = model.predict(data)

        # Guardado de predicciones
        results = pd.DataFrame(predictions, columns=['Predictions'])
        results.to_csv(output_path, index=False)

        print("Las predicciones han sido guardadas en:", output_path)

    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "El archivo especificado no fue encontrado en ruta dada.") from exc
    except RuntimeError as exc:
        raise RuntimeError(
            f"Ocurrió un error durante el proceso: {str(exc)}") from exc
