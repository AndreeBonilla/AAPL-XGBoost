# Predicción del Precio de Acciones con XGBoost

Este proyecto aplica técnicas de Machine Learning para predecir el precio de cierre de la acción de Apple (AAPL) utilizando el modelo XGBoost Regressor, basado en datos históricos descargados con `yfinance`.

## Resultados

- Error porcentual medio: **16.14%**
- MAE (Error absoluto medio): **2.71**
- MSE (Error cuadrático medio): **22.68**

## Tecnologías utilizadas

- Python 
- yfinance
- pandas
- numpy
- scikit-learn
- XGBoost
- matplotlib

## Metodología

1. Se descargaron datos históricos de la acción AAPL desde 2015 a 2020.
2. Se seleccionaron las características: `Open`, `High`, `Low`, `Volume`, `Close`.
3. Se escalaron los datos con `StandardScaler`.
4. Se entrenó un modelo `XGBRegressor` y se evaluó el rendimiento en el conjunto de prueba.

## Cómo ejecutar

```bash
pip install yfinance pandas numpy scikit-learn xgboost matplotlib
python XGBoost.py
