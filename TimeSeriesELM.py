import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from typing import Union, Optional, Dict, Any, Tuple
import logging
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import pandas as pd

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TimeSeriesELM')


@dataclass
class ModelState:
    """Dataclass para almacenar y validar el estado del modelo."""
    n_hidden: int
    activation: str
    lookback: int
    scale_data: bool
    scaler: Optional[str] = None
    random_state: Optional[int] = None

    def __post_init__(self):
        # Verificación de tipo y valor de n_hidden
        if not isinstance(self.n_hidden, int) or self.n_hidden <= 0:
            raise ValueError("n_hidden must be a positive integer")
        
        # Validación de la función de activación
        if self.activation not in ['sigmoid', 'tanh', 'relu']:
            raise ValueError("activation must be 'sigmoid', 'tanh', or 'relu'")
        
        # Verificación de tipo y valor de lookback
        if not isinstance(self.lookback, int) or self.lookback <= 0:
            raise ValueError("lookback must be a positive integer")
        
        # Validación del tipo de escalador
        if self.scale_data and self.scaler not in ['standard', 'minmax', None]:
            raise ValueError("scaler must be 'standard', 'minmax', or None")


class TimeSeriesELM(BaseEstimator, RegressorMixin):
    """
    Clase TimeSeriesELM mejorada con manejo robusto de errores, logging y diagnósticos.
    
    Parámetros:
    -----------
    n_hidden : int
        Número de neuronas en la capa oculta.
    activation : str
        Función de activación ('sigmoid', 'tanh', 'relu').
    lookback : int
        Número de periodos de retroceso.
    scale_data : bool
        Indica si se debe escalar los datos de entrada.
    scaler : Optional[str]
        Tipo de escalador a utilizar ('standard', 'minmax', None).
    random_state : Optional[int]
        Semilla aleatoria para reproducibilidad.
    """

    def __init__(
        self,
        n_hidden: int = 100,
        activation: str = 'sigmoid',
        lookback: int = 10,
        scale_data: bool = True,
        scaler: Optional[str] = 'standard',
        random_state: Optional[int] = None
    ):
        try:
            self.state = ModelState(
                n_hidden=n_hidden,
                activation=activation,
                lookback=lookback,
                scale_data=scale_data,
                scaler=scaler if scale_data else None,
                random_state=random_state
            )
            
            # Establecer la semilla aleatoria si se proporciona
            if random_state is not None:
                np.random.seed(random_state)
            
            self._initialize_attributes()
            logger.info(f"Initialized TimeSeriesELM with config: {self._get_config()}")
            
        except Exception as e:
            logger.error(f"Failed to initialize TimeSeriesELM: {str(e)}")
            raise

    def _initialize_attributes(self):
        """Inicializa los atributos del modelo con tipado adecuado."""
        self.n_hidden = self.state.n_hidden
        self.activation = self.state.activation
        self.lookback = self.state.lookback
        self.scale_data = self.state.scale_data
        self.random_state = self.state.random_state
        
        # Componentes del modelo
        self.input_weights_ = None
        self.output_weights_ = None
        self.biases_ = None
        self.input_shape_ = None
        self.scaler = self._get_scaler(self.state.scaler) if self.scale_data else None
        
        # Almacenamiento de métricas
        self.training_metrics_: Dict[str, float] = {}
        self.validation_metrics_: Dict[str, float] = {}
        self.fitted_ = False

    def _get_scaler(self, scaler_type: Optional[str]) -> Optional[Any]:
        """Inicializa el escalador apropiado basado en el tipo."""
        if not self.scale_data or scaler_type is None:
            return None
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler()
        }
        if scaler_type not in scalers:
            raise ValueError(f"Invalid scaler type. Choose from {list(scalers.keys())}")
        return scalers[scaler_type]

    def _validate_input(
        self, 
        X: Union[np.ndarray, pd.DataFrame, pd.Series], 
        y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Valida y preprocesa los datos de entrada."""
        try:
            # Convertir a numpy array si es necesario
            if isinstance(X, (pd.DataFrame, pd.Series)):
                X = X.values
            if isinstance(y, (pd.DataFrame, pd.Series)):
                y = y.values
                
            # Validar dimensiones
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            
            if y is not None:
                if y.ndim == 1:
                    y = y.reshape(-1, 1)
                if len(X) != len(y):
                    raise ValueError(f"X and y must have the same length. Got X: {len(X)}, y: {len(y)}")
            
            # Verificar valores NaN
            if np.isnan(X).any():
                raise ValueError("Input X contains NaN values")
            if y is not None and np.isnan(y).any():
                raise ValueError("Input y contains NaN values")
                
            return X, y
                
        except Exception as e:
            logger.error(f"Input validation failed: {str(e)}")
            raise

    def _activate(self, X: np.ndarray) -> np.ndarray:
        """Aplica la función de activación con manejo de errores."""
        try:
            if self.activation == 'sigmoid':
                return 1.0 / (1.0 + np.exp(-X))
            elif self.activation == 'tanh':
                return np.tanh(X)
            elif self.activation == 'relu':
                return np.maximum(0, X)
            else:
                raise ValueError(f"Unknown activation function: {self.activation}")
        except Exception as e:
            logger.error(f"Activation function failed: {str(e)}")
            raise

    def fit(
        self, 
        X: Union[np.ndarray, pd.DataFrame, pd.Series], 
        y: np.ndarray
    ) -> 'TimeSeriesELM':
        """
        Ajusta el modelo con manejo robusto de errores y logging.
        
        Parámetros:
        -----------
        X : array-like
            Muestras de entrada de entrenamiento.
        y : array-like
            Valores objetivo.
            
        Retorna:
        --------
        self : TimeSeriesELM
            Retorna la instancia del modelo.
        """
        try:
            logger.info("Starting model fitting")
            start_time = datetime.now()
            
            # Validar entradas
            X, y = self._validate_input(X, y)
            self.input_shape_ = X.shape
            
            # Escalar si es necesario
            if self.scaler is not None:
                X = self.scaler.fit_transform(X)
            
            # Inicializar pesos y biases
            self.input_weights_ = np.random.normal(size=(X.shape[1], self.n_hidden))
            self.biases_ = np.random.normal(size=self.n_hidden)
            
            # Calcular la salida de la capa oculta
            H = self._activate(np.dot(X, self.input_weights_) + self.biases_)
            
            # Calcular pesos de salida usando Ridge Regression para regularización
            ridge = Ridge(alpha=1e-3, fit_intercept=False)
            ridge.fit(H, y)
            self.output_weights_ = ridge.coef_.T
            
            # Calcular predicciones en entrenamiento directamente
            y_pred = np.dot(H, self.output_weights_)
            self.training_metrics_ = self._calculate_metrics(y, y_pred)
            
            # Marcar el modelo como ajustado
            self.fitted_ = True
            
            # Log de finalización del ajuste y métricas
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Model fitting completed in {duration:.2f} seconds")
            logger.info(f"Training metrics: {json.dumps(self.training_metrics_, indent=2)}")
            
            return self
            
        except Exception as e:
            logger.error(f"Model fitting failed: {str(e)}")
            raise

    def predict(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """
        Realiza predicciones con manejo robusto de errores.
        
        Parámetros:
        -----------
        X : array-like
            Muestras de entrada.
            
        Retorna:
        --------
        np.ndarray
            Valores predichos.
        """
        if not self.fitted_:
            raise ValueError("Model must be fitted before making predictions")
            
        try:
            # Validar entrada
            X, _ = self._validate_input(X)
            
            # Escalar si es necesario
            if self.scaler is not None:
                X = self.scaler.transform(X)
            
            # Generar predicciones
            H = self._activate(np.dot(X, self.input_weights_) + self.biases_)
            predictions = np.dot(H, self.output_weights_)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calcula y retorna las métricas de rendimiento."""
        try:
            metrics = {
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred)
            }
            return metrics
        except Exception as e:
            logger.error(f"Metrics calculation failed: {str(e)}")
            raise

    def save_model(self, path: Union[str, Path]) -> None:
        """Guarda el estado del modelo y los pesos en un archivo JSON."""
        try:
            path = Path(path)
            state_dict = {
                'config': self._get_config(),
                'weights': {
                    'input_weights': self.input_weights_.tolist() if self.input_weights_ is not None else None,
                    'output_weights': self.output_weights_.tolist() if self.output_weights_ is not None else None,
                    'biases': self.biases_.tolist() if self.biases_ is not None else None
                },
                'metrics': {
                    'training': self.training_metrics_,
                    'validation': self.validation_metrics_
                }
            }
            
            with open(path, 'w') as f:
                json.dump(state_dict, f, indent=2)
            
            logger.info(f"Model saved successfully to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise

    def load_model(self, path: Union[str, Path]) -> None:
        """Carga el estado del modelo y los pesos desde un archivo JSON."""
        try:
            path = Path(path)
            with open(path, 'r') as f:
                state_dict = json.load(f)
            
            # Restaurar configuración
            self.__init__(
                n_hidden=state_dict['config']['n_hidden'],
                activation=state_dict['config']['activation'],
                lookback=state_dict['config']['lookback'],
                scale_data=state_dict['config']['scale_data'],
                scaler=state_dict['config']['scaler'],
                random_state=state_dict['config']['random_state']
            )
            
            # Restaurar pesos
            self.input_weights_ = np.array(state_dict['weights']['input_weights'])
            self.output_weights_ = np.array(state_dict['weights']['output_weights'])
            self.biases_ = np.array(state_dict['weights']['biases'])
            
            # Restaurar métricas
            self.training_metrics_ = state_dict['metrics']['training']
            self.validation_metrics_ = state_dict['metrics']['validation']
            
            self.fitted_ = True
            logger.info(f"Model loaded successfully from {path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def _get_config(self) -> Dict[str, Any]:
        """Obtiene la configuración del modelo."""
        return asdict(self.state)

    def get_debug_info(self) -> Dict[str, Any]:
        """Obtiene información completa de depuración sobre el modelo."""
        debug_info = {
            'model_config': self._get_config(),
            'model_state': {
                'is_fitted': self.fitted_,
                'input_shape': self.input_shape_,
                'has_scaler': self.scaler is not None,
                'weights_initialized': self.input_weights_ is not None
            },
            'performance_metrics': {
                'training': self.training_metrics_,
                'validation': self.validation_metrics_
            }
        }
        
        if self.fitted_:
            debug_info.update({
                'weights_info': {
                    'input_weights_shape': self.input_weights_.shape,
                    'output_weights_shape': self.output_weights_.shape,
                    'input_weights_stats': {
                        'mean': float(np.mean(self.input_weights_)),
                        'std': float(np.std(self.input_weights_)),
                        'min': float(np.min(self.input_weights_)),
                        'max': float(np.max(self.input_weights_))
                    }
                }
            })
        
        return debug_info
