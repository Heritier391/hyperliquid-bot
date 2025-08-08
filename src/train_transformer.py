# core/train_transformer.py

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.transformer_model import build_transformer_model
import logging

# Configuration du logging
logger = logging.getLogger('TransformerTraining')
logger.setLevel(logging.INFO)

# Configuration des répertoires
MODEL_DIR = "models/transformer"
REPORT_DIR = "reports/transformer_predictions"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

def train_transformer_models(dataset: dict, lookback: int = 60):
    """
    Entraîne les modèles Transformer pour tous les symboles du dataset
    
    Args:
        dataset: Dictionnaire {symbol: DataFrame} des données préparées
        lookback: Nombre de périodes historiques à utiliser
    """
    if not dataset:
        logger.error("❌ Aucune donnée disponible pour l'entraînement Transformer")
        return
    
    logger.info(f"🧠 Début de l'entraînement des modèles Transformer sur {len(dataset)} symboles")
    
    for symbol, df in dataset.items():
        try:
            logger.info(f"🔧 Préparation des données pour: {symbol}")
            
            # Vérifier les données minimales
            if len(df) < lookback * 2:
                logger.warning(f"⚠️ Données insuffisantes pour {symbol} ({len(df)} < {lookback*2})")
                continue
            
            # Préparer les caractéristiques et la cible
            features = df.drop(columns=["timestamp", "symbol", "target"], errors="ignore").values
            target = df["target"].values
            
            # Créer des séquences temporelles
            X, y = [], []
            for i in range(lookback, len(features)):
                X.append(features[i-lookback:i])
                y.append(target[i])
            
            X, y = np.array(X), np.array(y)
            
            # Vérifier les dimensions
            if len(X) == 0:
                logger.warning(f"⚠️ Aucune séquence créée pour {symbol}")
                continue
                
            logger.info(f"📦 Données préparées: {X.shape} séquences pour {symbol}")
            
            # Diviser en train/validation
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Construire le modèle
            input_shape = (lookback, X_train.shape[2])
            model = build_transformer_model(
                sequence_len=lookback,
                num_features=X_train.shape[2]
            )
            
            # Callbacks
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
            
            checkpoint = ModelCheckpoint(
                os.path.join(MODEL_DIR, f"{symbol.replace('/', '_')}_best.h5"),
                save_best_only=True,
                monitor='val_loss',
                mode='min'
            )
            
            # Entraînement
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=64,
                callbacks=[early_stop, checkpoint],
                verbose=1
            )
            
            # Sauvegarder le modèle final
            model_path = os.path.join(MODEL_DIR, f"{symbol.replace('/', '_')}_transformer.keras")
            save_model(model, model_path)
            logger.info(f"✅ Modèle sauvegardé: {model_path}")
            
            # Évaluation et visualisation
            evaluate_transformer_model(symbol, X_val, y_val)
            
        except Exception as e:
            logger.error(f"❌ Erreur entraînement {symbol}: {str(e)}")

def evaluate_transformer_model(symbol: str, X_val: np.ndarray, y_val: np.ndarray):
    """Évalue un modèle et génère un graphique de prédiction"""
    try:
        model_path = os.path.join(MODEL_DIR, f"{symbol.replace('/', '_')}_transformer.keras")
        if not os.path.exists(model_path):
            logger.warning(f"⚠️ Modèle non trouvé pour {symbol}: {model_path}")
            return
            
        model = load_model(model_path)
        
        # Prédiction
        y_pred = model.predict(X_val, verbose=0).flatten()
        
        # Tracer les résultats
        plt.figure(figsize=(12, 6))
        plt.plot(y_val, label='Valeurs réelles', alpha=0.7)
        plt.plot(y_pred, label='Prédictions', alpha=0.7)
        plt.title(f"Prédictions vs Réel - {symbol}")
        plt.xlabel('Période')
        plt.ylabel('Valeur')
        plt.legend()
        plt.grid(True)
        
        # Sauvegarder le graphique
        plot_path = os.path.join(REPORT_DIR, f"{symbol.replace('/', '_')}_predictions.png")
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"📊 Graphique sauvegardé: {plot_path}")
        
    except Exception as e:
        logger.error(f"❌ Erreur évaluation {symbol}: {str(e)}")

def load_transformer_model(symbol: str):
    """Charge un modèle Transformer sauvegardé"""
    try:
        model_path = os.path.join(MODEL_DIR, f"{symbol.replace('/', '_')}_transformer.keras")
        if not os.path.exists(model_path):
            logger.warning(f"⚠️ Modèle non trouvé pour {symbol}: {model_path}")
            return None
            
        return load_model(model_path)
    except Exception as e:
        logger.error(f"❌ Erreur chargement modèle {symbol}: {str(e)}")
        return None
