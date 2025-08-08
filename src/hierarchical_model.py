# src/hierarchical_model.py

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input, Concatenate, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
import numpy as np

class HierarchicalTemporalModel:
    def __init__(self, input_shapes, num_features):
        """
        Modèle hiérarchique multi-échelle
        
        Args:
            input_shapes: Dict des shapes pour chaque timeframe {'1m': (60, n), '5m': (12, n), ...}
            num_features: Nombre de caractéristiques par pas de temps
        """
        self.input_shapes = input_shapes
        self.num_features = num_features
        self.model = self._build_model()
        
    def _build_model(self):
        # Inputs pour chaque timeframe
        inputs = {}
        for tf_name in self.input_shapes.keys():
            inputs[tf_name] = Input(shape=self.input_shapes[tf_name])
        
        # Encoders LSTM pour chaque timeframe
        encoded = {}
        for tf_name, input_layer in inputs.items():
            x = LSTM(64, return_sequences=True)(input_layer)
            x = LayerNormalization()(x)
            x = LSTM(32, return_sequences=False)(x)
            encoded[tf_name] = x
        
        # Fusion hiérarchique (de la plus longue à la plus courte timeframe)
        timeframes = ['4h', '1h', '15m', '5m', '1m']
        context = None
        
        for tf_name in timeframes:
            if tf_name in encoded:
                if context is None:
                    context = encoded[tf_name]
                else:
                    # Mécanisme d'attention simple
                    attention = Dense(1, activation='tanh')(Concatenate()([context, encoded[tf_name]]))
                    attention = tf.nn.softmax(attention)
                    context = attention * context + (1 - attention) * encoded[tf_name]
        
        # Couches de décision finale
        x = Dense(64, activation='relu')(context)
        x = Dense(32, activation='relu')(x)
        output = Dense(1, activation='linear')(x)  # Prédiction de prix
        
        model = Model(inputs=list(inputs.values()), outputs=output)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=Huber(),
            metrics=['mae']
        )
        return model
    
    def train(self, X_train, y_train, epochs=100, batch_size=32):
        """Entraîne le modèle"""
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        return history
    
    def predict(self, X):
        """Fait des prédictions"""
        return self.model.predict(X)
    
    def save(self, path):
        """Sauvegarde le modèle"""
        self.model.save(path)
    
    @classmethod
    def load(cls, path):
        """Charge un modèle sauvegardé"""
        model = tf.keras.models.load_model(path)
        # Crée une instance avec des paramètres fictifs
        instance = cls({'1m': (60, 10)}, 10)
        instance.model = model
        return instance

class DynamicThreshold:
    """Module de seuil adaptatif"""
    def __init__(self, base_threshold=0.6):
        self.base_threshold = base_threshold
        self.volatility_factor = 1.0
        self.confidence_factor = 1.0
        self.performance_factor = 1.0
    
    def update(self, volatility, model_confidence, recent_performance):
        """
        Met à jour les facteurs basés sur les conditions du marché
        
        Args:
            volatility: Volatilité récente (écart-type des rendements)
            model_confidence: Confiance moyenne du modèle dans ses prédictions
            recent_performance: Performance récente (précision) du modèle
        """
        # Ajustement basé sur la volatilité (plus de volatilité = seuil plus strict)
        self.volatility_factor = 1.5 if volatility > 0.05 else 1.0
        
        # Ajustement basé sur la confiance du modèle
        self.confidence_factor = 0.8 if model_confidence > 0.7 else 1.2
        
        # Ajustement basé sur la performance récente
        self.performance_factor = 0.9 if recent_performance > 0.8 else 1.1
    
    def get_threshold(self):
        """Retourne le seuil dynamique"""
        return self.base_threshold * self.volatility_factor * self.confidence_factor * self.performance_factor

class FeedbackUpdater:
    """Mise à jour adaptative basée sur les résultats des trades"""
    def __init__(self, model):
        self.model = model
        self.reward_memory = []
        self.max_memory = 1000
    
    def add_trade_result(self, trade_result):
        """Ajoute un résultat de trade à la mémoire"""
        self.reward_memory.append(trade_result)
        if len(self.reward_memory) > self.max_memory:
            self.reward_memory.pop(0)
    
    def update_model(self, X, y):
        """Met à jour le modèle avec les nouvelles données étiquetées"""
        if len(self.reward_memory) > 100:
            # Réentraînement partiel avec les nouvelles données
            self.model.fit(X, y, epochs=10, batch_size=32, verbose=0)
