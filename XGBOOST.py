import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt

# Descargar los datos
ticker = 'AAPL'
df = yf.download(ticker, start='2015-01-01', end='2020-01-01', auto_adjust=True)

# Verificar si se descargaron datos
if df.empty:
    raise ValueError("No se pudieron descargar datos. Verifica tu conexión a internet o el ticker.")

# Crear variables desfasadas (lag 1)
df['Close_lag1'] = df['Close'].shift(1)
df['High_lag1'] = df['High'].shift(1)
df['Low_lag1'] = df['Low'].shift(1)
df['Open_lag1'] = df['Open'].shift(1)
df['Volume_lag1'] = df['Volume'].shift(1)

# Eliminar filas con NaNs por el shift
df = df.dropna()

# Definir características y objetivo
X = df[['Close_lag1', 'High_lag1', 'Low_lag1', 'Open_lag1', 'Volume_lag1']]
y = df['Close']

# Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos (80% entrenamiento, 20% prueba) sin mezclar (shuffle=False)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# Crear y entrenar el modelo
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
model.fit(X_train, y_train)

# Predecir
y_pred = model.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
error_porcentual = np.mean(np.abs((y_test.values - y_pred) / y_test.values)) * 100

print(f"\nMSE: {mse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"Error porcentual medio: {error_porcentual:.2f}%")

# Crear gráfico
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Precio real', linewidth=2)
plt.plot(y_pred, label='Predicción', linewidth=2)
plt.title(f'Predicción del precio de cierre de {ticker} con XGBoost')
plt.xlabel('Días')
plt.ylabel('Precio ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
