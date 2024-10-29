import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from typing import Tuple
import yfinance as yf
from TimeSeriesELM import TimeSeriesELM, ModelState  # Asegúrate de que este archivo esté en el mismo directorio

# Función auxiliar para crear datos supervisados utilizando ventanas deslizantes
def create_supervised_data(series: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transforma una serie temporal en datos supervisados utilizando ventanas deslizantes.

    Parámetros:
    -----------
    series : np.ndarray
        Serie temporal unidimensional.
    lookback : int
        Número de pasos hacia atrás para crear las características.

    Retorna:
    --------
    Tuple[np.ndarray, np.ndarray]
        Características (ventanas deslizantes) y valores objetivo.
    """
    X, y = [], []
    for i in range(len(series) - lookback):
        X.append(series[i:i + lookback])
        y.append(series[i + lookback])
    return np.array(X), np.array(y)

# Configuración básica del modelo (puedes ajustar estos parámetros según tus necesidades)
DEFAULT_N_HIDDEN = 100
DEFAULT_ACTIVATION = 'relu'  # Opciones: 'sigmoid', 'tanh', 'relu'
DEFAULT_LOOKBACK = 24  # Número de pasos hacia atrás
DEFAULT_SCALE_DATA = True
DEFAULT_SCALER = 'standard'  # Opciones: 'standard', 'minmax', o None
DEFAULT_RANDOM_STATE = 42  # Para reproducibilidad

# Crear un estado de modelo por defecto
default_model_state = ModelState(
    n_hidden=DEFAULT_N_HIDDEN,
    activation=DEFAULT_ACTIVATION,
    lookback=DEFAULT_LOOKBACK,
    scale_data=DEFAULT_SCALE_DATA,
    scaler=DEFAULT_SCALER,
    random_state=DEFAULT_RANDOM_STATE
)

# Ejemplo 1: Predicción de una Onda Seno
def sine_wave_example():
    print("\n--- Ejemplo 1: Predicción de una Onda Seno ---")
    # Generación de datos de onda seno
    x = np.linspace(0, 50, 500)
    y = np.sin(x)
    
    # Crear datos supervisados
    lookback = 10
    X, y_supervised = create_supervised_data(y, lookback)
    
    # División en entrenamiento y prueba
    split_index = int(len(y_supervised) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y_supervised[:split_index], y_supervised[split_index:]
    
    # Configuración del modelo
    model = TimeSeriesELM(
        n_hidden=default_model_state.n_hidden,
        activation=default_model_state.activation,
        lookback=lookback,
        scale_data=default_model_state.scale_data,
        scaler=default_model_state.scaler,
        random_state=default_model_state.random_state
    )
    
    # Ajustar el modelo
    model.fit(X_train, y_train)
    
    # Realizar predicciones
    predictions = model.predict(X_test)
    
    # Visualización de resultados
    plt.figure(figsize=(14, 7))
    plt.plot(range(len(y)), y, label="Onda Seno Real")
    plt.plot(range(split_index + lookback, len(y)), predictions, label="Predicción", linestyle='--')
    plt.legend()
    plt.title("Predicción de Onda Seno")
    plt.xlabel("Tiempo")
    plt.ylabel("Amplitud")
    plt.show()
    
    # Evaluación del rendimiento
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"Onda Seno - MSE: {mse:.4f}, R2: {r2:.4f}")

# Ejemplo 2: Predicción de Precios de Acciones (AAPL)
def stock_price_example():
    print("\n--- Ejemplo 2: Predicción de Precios de Acciones (AAPL) ---")
    # Descargar datos de Yahoo Finance
    stock_data = yf.download("AAPL", start="2020-01-01", end="2023-01-01")
    close_prices = stock_data['Close'].values
    
    # Crear datos supervisados
    lookback = 10
    X, y_supervised = create_supervised_data(close_prices, lookback)
    
    # División en entrenamiento y prueba
    split_index = int(len(y_supervised) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y_supervised[:split_index], y_supervised[split_index:]
    
    # Configuración del modelo
    model = TimeSeriesELM(
        n_hidden=default_model_state.n_hidden,
        activation=default_model_state.activation,
        lookback=lookback,
        scale_data=default_model_state.scale_data,
        scaler=default_model_state.scaler,
        random_state=default_model_state.random_state
    )
    
    # Ajustar el modelo
    model.fit(X_train, y_train)
    
    # Realizar predicciones
    predictions_scaled = model.predict(X_test)
    
    # Invertir la escala si es necesario (Removido)
    # Como y no está escalado, no es necesario invertir la escala
    # predictions_unscaled = model.scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
    # y_test_unscaled = model.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Asignar directamente las predicciones y los valores reales
    predictions_unscaled = predictions_scaled
    y_test_unscaled = y_test
    
    # Visualización de resultados
    plt.figure(figsize=(14, 7))
    plt.plot(range(len(close_prices)), close_prices, label="Precio Real")
    plt.plot(range(split_index + lookback, len(close_prices)), predictions_unscaled, label="Predicción", linestyle='--')
    plt.legend()
    plt.title("Predicción de Precios de Acciones (AAPL)")
    plt.xlabel("Tiempo")
    plt.ylabel("Precio de Cierre (USD)")
    plt.show()
    
    # Evaluación del rendimiento
    mse = mean_squared_error(y_test_unscaled, predictions_unscaled)
    r2 = r2_score(y_test_unscaled, predictions_unscaled)
    print(f"Precios de Acciones - MSE: {mse:.4f}, R2: {r2:.4f}")

# Ejemplo 3: Predicción Multi-Paso de una Onda Seno
def multi_step_forecasting_example():
    print("\n--- Ejemplo 3: Predicción Multi-Paso de una Onda Seno ---")
    # Generación de datos de onda seno con ruido
    x = np.linspace(0, 100, 1000)
    y = np.sin(x) + 0.1 * np.random.normal(size=x.shape)
    
    # Crear datos supervisados
    lookback = 10
    X, y_supervised = create_supervised_data(y, lookback)
    
    # División en entrenamiento y prueba
    split_index = int(len(y_supervised) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y_supervised[:split_index], y_supervised[split_index:]
    
    # Configuración del modelo
    model = TimeSeriesELM(
        n_hidden=default_model_state.n_hidden,
        activation=default_model_state.activation,
        lookback=lookback,
        scale_data=default_model_state.scale_data,
        scaler=default_model_state.scaler,
        random_state=default_model_state.random_state
    )
    
    # Ajustar el modelo
    model.fit(X_train, y_train)
    
    # Predicción multi-paso utilizando una estrategia iterativa
    steps_ahead = 50  # Número de pasos futuros a predecir
    predictions = []
    
    # Iniciar con la primera ventana de prueba
    current_input = X_test[0].reshape(1, -1)
    
    for _ in range(steps_ahead):
        # Realizar la predicción
        pred_array = model.predict(current_input)
        pred = pred_array[0, 0]  # Extraer el valor escalar
        
        predictions.append(pred)
        
        # Actualizar la entrada desplazando una ventana y agregando la nueva predicción
        current_input = np.append(current_input[:, 1:], [[pred]], axis=1)
    
    # Invertir la escala si es necesario (Removido)
    # Como y no está escalado, no es necesario invertir la escala
    # predictions_unscaled = model.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    # y_test_unscaled = model.scaler.inverse_transform(y_test[:steps_ahead].reshape(-1, 1)).flatten()
    
    # Asignar directamente las predicciones y los valores reales
    predictions_unscaled = np.array(predictions)
    y_test_unscaled = y_test[:steps_ahead]
    
    # Visualización de resultados
    plt.figure(figsize=(14, 7))
    plt.plot(range(len(y)), y, label="Datos Reales")
    plt.plot(range(split_index + lookback, split_index + lookback + steps_ahead), predictions_unscaled, label="Predicción Multi-Paso", linestyle='--')
    plt.legend()
    plt.title("Predicción Multi-Paso de Onda Seno")
    plt.xlabel("Tiempo")
    plt.ylabel("Amplitud")
    plt.show()
    
    # Evaluación del rendimiento
    mse = mean_squared_error(y_test_unscaled, predictions_unscaled)
    r2 = r2_score(y_test_unscaled, predictions_unscaled)
    print(f"Predicción Multi-Paso - MSE: {mse:.4f}, R2: {r2:.4f}")

# Ejemplo 4: Predicción de Temperaturas Mínimas Diarias
def daily_min_temperature_example():
    print("\n--- Ejemplo 4: Predicción de Temperaturas Mínimas Diarias ---")
    # Descargar el conjunto de datos
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv'
    data = pd.read_csv(url, parse_dates=['Date'])
    
    # Mostrar las primeras filas del conjunto de datos
    print(data.head())
    
    # Extraer la serie temporal de temperaturas
    temperatures = data['Temp'].values
    
    # Visualizar la serie temporal
    plt.figure(figsize=(14, 5))
    plt.plot(data['Date'], temperatures, label='Temperatura Mínima Diaria')
    plt.legend()
    plt.title('Temperaturas Mínimas Diarias en Melbourne (1981-1990)')
    plt.xlabel('Fecha')
    plt.ylabel('Temperatura (°C)')
    plt.show()
    
    # Crear datos supervisados
    lookback = 10
    X, y_supervised = create_supervised_data(temperatures, lookback)
    
    # División en entrenamiento y prueba
    split_index = int(len(y_supervised) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y_supervised[:split_index], y_supervised[split_index:]
    
    # Configuración del modelo
    model = TimeSeriesELM(
        n_hidden=default_model_state.n_hidden,
        activation='tanh',
        lookback=lookback,
        scale_data=default_model_state.scale_data,
        scaler=default_model_state.scaler,
        random_state=default_model_state.random_state
    )
    
    # Ajustar el modelo
    model.fit(X_train, y_train)
    
    # Realizar predicciones
    predictions = model.predict(X_test)
    
    # Invertir la escala si es necesario (Removido)
    # Como y no está escalado, no es necesario invertir la escala
    # predictions_unscaled = model.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    # y_test_unscaled = model.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Asignar directamente las predicciones y los valores reales
    predictions_unscaled = predictions
    y_test_unscaled = y_test
    
    # Visualización de resultados
    plt.figure(figsize=(14, 7))
    plt.plot(range(len(temperatures)), temperatures, label="Temperatura Real")
    plt.plot(range(split_index + lookback, len(temperatures)), predictions_unscaled, label="Predicción", linestyle='--')
    plt.legend()
    plt.title("Predicción de Temperaturas Mínimas Diarias")
    plt.xlabel("Tiempo")
    plt.ylabel("Temperatura (°C)")
    plt.show()
    
    # Evaluación del rendimiento
    mse = mean_squared_error(y_test_unscaled, predictions_unscaled)
    r2 = r2_score(y_test_unscaled, predictions_unscaled)
    print(f"Temperaturas Mínimas Diarias - MSE: {mse:.4f}, R2: {r2:.4f}")

# Ejemplo 5: Predicción de Precios de Bitcoin
def bitcoin_price_example():
    print("\n--- Ejemplo 5: Predicción de Precios de Bitcoin ---")
    # Descargar datos de Yahoo Finance
    btc_data = yf.download("BTC-USD", start="2015-01-01", end="2023-12-31")
    close_prices = btc_data['Close'].values
    
    # Mostrar las primeras filas del conjunto de datos
    print(btc_data.head())
    
    # Visualizar la serie temporal
    plt.figure(figsize=(14, 5))
    plt.plot(btc_data.index, close_prices, label='Precio de Cierre BTC-USD')
    plt.legend()
    plt.title('Precio de Cierre de Bitcoin (BTC-USD)')
    plt.xlabel('Fecha')
    plt.ylabel('Precio en USD')
    plt.show()
    
    # Crear datos supervisados
    lookback = 10
    X, y_supervised = create_supervised_data(close_prices, lookback)
    
    # División en entrenamiento y prueba
    split_index = int(len(y_supervised) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y_supervised[:split_index], y_supervised[split_index:]
    
    # Configuración del modelo
    model = TimeSeriesELM(
        n_hidden=100,
        activation='relu',
        lookback=lookback,
        scale_data=default_model_state.scale_data,
        scaler='minmax',
        random_state=default_model_state.random_state
    )
    
    # Ajustar el modelo
    model.fit(X_train, y_train)
    
    # Realizar predicciones
    predictions_scaled = model.predict(X_test)
    
    # Invertir la escala si es necesario (Removido)
    # Como y no está escalado, no es necesario invertir la escala
    # predictions_unscaled = model.scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
    # y_test_unscaled = model.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Asignar directamente las predicciones y los valores reales
    predictions_unscaled = predictions_scaled
    y_test_unscaled = y_test
    
    # Visualización de resultados
    plt.figure(figsize=(14, 7))
    plt.plot(range(len(close_prices)), close_prices, label="Precio Real BTC-USD")
    plt.plot(range(split_index + lookback, len(close_prices)), predictions_unscaled, label="Predicción", linestyle='--')
    plt.legend()
    plt.title("Predicción de Precios de Bitcoin (BTC-USD)")
    plt.xlabel("Tiempo")
    plt.ylabel("Precio en USD")
    plt.show()
    
    # Evaluación del rendimiento
    mse = mean_squared_error(y_test_unscaled, predictions_unscaled)
    r2 = r2_score(y_test_unscaled, predictions_unscaled)
    print(f"Precios de Bitcoin - MSE: {mse:.4f}, R2: {r2:.4f}")

# Ejemplo 6: Predicción de Consumo Eléctrico de un Hogar
def household_power_consumption_example():
    print("\n--- Ejemplo 6: Predicción de Consumo Eléctrico de un Hogar ---")
    # Generar datos sintéticos de consumo eléctrico
    np.random.seed(42)
    days = 365 * 2  # Dos años de datos
    time = np.arange(days * 24)  # Datos horarios
    # Crear patrones diarios y estacionales con ruido
    daily_pattern = 10 + 5 * np.sin(2 * np.pi * time / 24)
    seasonal_pattern = 5 * np.sin(2 * np.pi * time / (24 * 365))
    noise = np.random.normal(0, 2, size=time.shape)
    consumption = daily_pattern + seasonal_pattern + noise
    consumption = consumption.clip(min=0)  # El consumo no puede ser negativo

    # Crear un DataFrame para facilitar el manejo
    dates = pd.date_range(start='2020-01-01', periods=len(consumption), freq='H')
    df = pd.DataFrame({'DateTime': dates, 'Consumption': consumption})

    # Visualizar la serie temporal
    plt.figure(figsize=(14, 5))
    plt.plot(df['DateTime'], df['Consumption'], label='Consumo Eléctrico')
    plt.legend()
    plt.title('Consumo Eléctrico Sintético de un Hogar (2 Años)')
    plt.xlabel('Fecha y Hora')
    plt.ylabel('Consumo (kWh)')
    plt.show()

    # Extraer la serie temporal de consumo
    consumption_values = df['Consumption'].values

    # Crear datos supervisados
    lookback = 24  # Uso de una ventana de 24 horas
    X, y_supervised = create_supervised_data(consumption_values, lookback)

    # División en entrenamiento y prueba
    split_index = int(len(y_supervised) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y_supervised[:split_index], y_supervised[split_index:]

    # Configuración del modelo
    model = TimeSeriesELM(
        n_hidden=100,
        activation='relu',
        lookback=lookback,
        scale_data=True,
        scaler='standard',
        random_state=42
    )

    # Ajustar el modelo
    model.fit(X_train, y_train)

    # Realizar predicciones
    predictions = model.predict(X_test)

    # Invertir la escala si es necesario (Removido)
    # Como y no está escalado, no es necesario invertir la escala
    # predictions_unscaled = model.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    # y_test_unscaled = model.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Asignar directamente las predicciones y los valores reales
    predictions_unscaled = predictions
    y_test_unscaled = y_test

    # Visualización de resultados
    plt.figure(figsize=(14, 7))
    plt.plot(range(len(consumption_values)), consumption_values, label="Consumo Real")
    plt.plot(range(split_index + lookback, len(consumption_values)), predictions_unscaled, label="Predicción", linestyle='--')
    plt.legend()
    plt.title("Predicción de Consumo Eléctrico de un Hogar")
    plt.xlabel("Tiempo (Horas)")
    plt.ylabel("Consumo (kWh)")
    plt.show()

    # Evaluación del rendimiento
    mse = mean_squared_error(y_test_unscaled, predictions_unscaled)
    r2 = r2_score(y_test_unscaled, predictions_unscaled)
    print(f"Consumo Eléctrico - MSE: {mse:.4f}, R2: {r2:.4f}")

# Función principal para ejecutar todos los ejemplos
def main():
    sine_wave_example()
    stock_price_example()
    multi_step_forecasting_example()
    daily_min_temperature_example()
    bitcoin_price_example()
    household_power_consumption_example()

# Ejecutar todos los ejemplos si el script se ejecuta directamente
if __name__ == "__main__":
    main()
