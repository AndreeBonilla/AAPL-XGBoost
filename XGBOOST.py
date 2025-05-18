import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Descargar los datos
ticker = 'AAPL'
df = yf.download(ticker, start='2015-01-01', end='2020-01-01', auto_adjust=True)

# Verificar si se descargaron datos
if df.empty:
    raise ValueError("No se pudieron descargar datos. Verifica tu conexión a internet o el ticker.")

# Mostrar las primeras filas
print("\nPrimeras filas de los datos:")
print(df.head())

# Asegurarse de que no haya índice multinivel (por si acaso)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

# Elegimos las características (features) y el objetivo (target)
X = df[['Close', 'High', 'Low', 'Open', 'Volume']]
y = df[['Close']]  # o 'Adj Close' si prefieres ajustado

# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# Crear el modelo XGBoost
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
model.fit(X_train, y_train)

# Predecir
y_pred = model.predict(X_test)

# Evaluar: error porcentual medio
y_test_np = y_test.values.flatten()
error_porcentual = np.mean(np.abs((y_test_np - y_pred) / y_test_np)) * 100

print(f"\nError porcentual medio: {error_porcentual:.2f}%")
