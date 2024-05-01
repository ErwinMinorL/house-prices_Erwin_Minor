# Predicción de precios de casas

## Tabla de Contenidos
- [Descripción](#descripción)
- [Dependencias](#dependencias)
- [Inputs/Outputs](#inputsoutputs)
- [Estructura del Repositorio](#estructura-del-repositorio)
- [Instalación](#instalación)
- [Cómo Ejecutar](#cómo-ejecutar)
- [Data Model](#data-model)

## Descripción
Es un proyecto que utiliza la data de Kaggle para realizar un flujo de trabajo en el que se realiza la preparación de datos, entrenamiento de modelos e inferencia para predecir el precio de casas utilizando un modelo de RandomForest.

## Dependencias


## Inputs/Outputs
- Input: Son los datos con los que se van a entrenar el modelo, incluye todas las variables.
- Output: Son los precios de las casas estimados con el modelo ya ajustado con los parámetros calculados.

## Estructura del Repositorio
```
.
├── README.md
├── data
│   ├── data_description.txt
│   └── train.csv
├── inference.py
├── notebooks
│   └── Model.ipynb
├── prep.py
├── src
└── train.py
```

## Instalación
Instrucciones paso a paso para configurar el entorno de desarrollo o producción:
```bash
git clone [URL del repositorio]
cd house-prices_Erwin_Minor
pip install -r requirements.txt
```

## Cómo ejecutar
```bash
python src/prep.py
python src/train.py
python src/inference.py
```

## Data Model

