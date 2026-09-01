# TimeSeriesELM

Una implementación de **Extreme Learning Machine (ELM)** para el modelado de series temporales en Python.

## Descripción

Este proyecto proporciona una clase `TimeSeriesELM` que permite realizar predicciones en series temporales utilizando ELM, un tipo de red neuronal de una sola capa oculta donde los pesos de entrada se asignan aleatoriamente y no se actualizan durante el entrenamiento. El repositorio también incluye múltiples ejemplos que demuestran cómo utilizar la clase para diferentes tipos de datos de series temporales.

## Contenido del Repositorio

- `TimeSeriesELM.py`: Implementación de la clase `TimeSeriesELM`.
- `ExamplesELM.py`: Seis ejemplos que muestran cómo utilizar la clase `TimeSeriesELM` en diferentes escenarios:
  1. Predicción de una onda seno.
  2. Predicción de precios de acciones (AAPL).
  3. Predicción multi-paso de una onda seno.
  4. Predicción de temperaturas mínimas diarias.
  5. Predicción de precios de Bitcoin.
  6. Predicción de consumo eléctrico de un hogar.

## Requisitos

Para ejecutar el código, necesitas instalar las siguientes bibliotecas:

- numpy
- pandas
- matplotlib
- scikit-learn
- yfinance
- pytest (para ejecutar pruebas)

Puedes instalarlas utilizando el archivo `requirements.txt` proporcionado:

```bash
pip install -r requirements.txt
```

## Uso rápido

```python
import numpy as np
from TimeSeriesELM import TimeSeriesELM

# Dataset ejemplo
X = np.random.rand(200, 10)
y = np.random.rand(200)

model = TimeSeriesELM(
    n_hidden=128,
    activation="tanh",
    lookback=10,
    scale_data=True,
    scaler="standard",
    random_state=42,
    ridge_alpha=1e-2,
)
model.fit(X, y)
pred = model.predict(X[:5])

# Persistencia segura del escalador + pesos
model.save_model("model.json")
restored = TimeSeriesELM()
restored.load_model("model.json")
```

## Calidad y confiabilidad (production-ready)

- Validación estricta de entradas (NaN/Inf y forma de matrices).
- Inferencia reproducible usando `random_state` con generador local.
- Persistencia completa del estado del escalador para evitar drift entre entrenamiento e inferencia.
- Pruebas automatizadas en `tests/test_timeserieselm.py`.

## Ejecutar pruebas

```bash
pytest -q
```
