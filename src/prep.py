"""
Este módulo se encarga de preparar los datos para el entrenamiento del modelo.
Realiza la carga de los datos, el procesamiento inicial incluyendo la
imputación de valores faltantes, la codificación one-hot de variables
categóricas y la transformación de variables numéricas.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


def prepare_data(input_path, output_path):
    """
    Procesa y guarda los datos preparados.

    Carga los datos desde un archivo CSV, aplica one-hot encoding, imputa
    valores faltantes,
    realiza transformaciones en variables sesgadas, crea nuevas variables y
    guarda el conjunto de datos procesado.

    Parameters:
    input_path (str): Ruta al archivo CSV de entrada (datos crudos).
    output_path (str): Ruta al archivo CSV de salida (datos procesados).

    Raises:
    FileNotFoundError: Si el archivo de entrada no se encuentra en la ruta
    especificada.
    RuntimeError: Para errores generales de procesamiento de datos.
    """
    try:
        # Carga de datos
        data = pd.read_csv(input_path)

        # Codificación one-hot
        data_encoded = pd.get_dummies(data, drop_first=True)

        # Imputación de valores faltantes
        imputer = SimpleImputer(strategy='mean')
        numeric_vars = data_encoded.select_dtypes(
            include=['float64', 'int64']).columns
        data_encoded[numeric_vars] = imputer.fit_transform(
            data_encoded[numeric_vars])

        # Transformación de variables sesgadas
        skewed_vars = ['LotFrontage', 'LotArea', 'MasVnrArea']
        for var in skewed_vars:
            data_encoded[var] = np.log1p(data_encoded[var])

        # Creación de nuevas variables
        data_encoded['TotalArea'] = (data_encoded['GrLivArea'] +
                                     data_encoded['TotalBsmtSF'] +
                                     data['GarageArea'])
        data_encoded['HouseAge'] = (data_encoded['YrSold'] -
                                    data_encoded['YearBuilt'])
        data_encoded['TotalSF'] = (data_encoded['TotalBsmtSF'] +
                                   data_encoded['1stFlrSF'] +
                                   data['2ndFlrSF'])
        data_encoded['TotalBathrooms'] = (data_encoded['FullBath'] +
                                          0.5 * data_encoded['HalfBath'])

        # Guardado de los datos procesados
        data_encoded.to_csv(output_path, index=False)

    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"No se pudo encontrar el archivo en ruta: {input_path}") from exc
    except Exception as exc:
        raise RuntimeError(
            f"Error durante el procesamiento de datos: {str(exc)}") from exc
