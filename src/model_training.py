# src/model_training.py

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.hierarchical_model import HierarchicalTemporalModel
import logging

# Configuration du logging
logger = logging.getLogger('ModelTraining')
logger.setLevel(logging.INFO)

MODEL_DIR = "models/hierarchical"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_hierarchical_model(symbol: str, multi_scale_data: dict):
    """Entraîne un modèle hiérarchique pour un symbole donné"""
    try:
        # Vérifier les données disponibles
        if not multi_scale_data:
            logger.error(f"❌ Aucune donnée multi-échelle pour {symbol}")
            return
        
        # Préparer les données d'entrée et cible
        X_train, y_train = {}, None
        input_shapes = {}
        
        for tf_name, (X, y, confidence) in multi_scale_data.items():
            if tf_name == "dynamic_weights":
                continue
                
            # Utiliser seulement les données avec suffisamment d'échantillons
            if len(X) > lookback * 2:  # Au moins 2 fenêtres complètes
                X_train[tf_name] = X
                input_shapes[tf_name] = X.shape[1:]
                
                # Prendre y de la timeframe la plus courte comme référence
                if y_train is None or tf_name == "1m":
                    y_train = y
        
        if not X_train or y_train is None:
            logger.warning(f"⚠️ Données insuffisantes pour {symbol}")
            return
        
        # Créer et entraîner le modèle hiérarchique
        model = HierarchicalTemporalModel(input_shapes, X_train[list(X_train.keys())[0]].shape[-1])
        
        early_stop = EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        )
        
        checkpoint = ModelCheckpoint(
            os.path.join(MODEL_DIR, f"{symbol}_best.h5"),
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        )
        
        # Déterminer les paramètres d'entraînement adaptatifs
        min_samples = min([len(X) for X in X_train.values()])
        
        # Paramètres adaptatifs
        epochs = min(100, max(20, min_samples // 100))
        batch_size = min(64, max(16, min_samples // 50))
        
        logger.info(f"Paramètres adaptatifs: {epochs} epochs, batch_size={batch_size}")
        
        history = model.model.fit(
            list(X_train.values()),
            y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, checkpoint],
            verbose=1
        )
        
        # Sauvegarder le modèle final
        model_path = os.path.join(MODEL_DIR, f"{symbol}.h5")
        save_model(model.model, model_path)
        logger.info(f"✅ Modèle hiérarchique sauvegardé: {model_path}")
        
        return model
    
    except Exception as e:
        logger.error(f"❌ Erreur entraînement hiérarchique {symbol}: {e}")
        return None

def train_all_hierarchical_models(multi_scale_dataset: dict):
    """Entraîne les modèles hiérarchiques pour tous les symboles"""
    if not multi_scale_dataset:
        logger.error("❌ Aucune donnée disponible pour l'entraînement hiérarchique")
        return
    
    logger.info(f"🚀 Début de l'entraînement des modèles hiérarchiques sur {len(multi_scale_dataset)} symboles")
    
    for symbol, multi_scale_data in multi_scale_dataset.items():
        try:
            logger.info(f"🧠 Entraînement hiérarchique pour {symbol}")
            train_hierarchical_model(symbol, multi_scale_data)
        except Exception as e:
            logger.error(f"❌ Erreur entraînement hiérarchique {symbol}: {e}")

def load_hierarchical_model(symbol: str):
    """Charge un modèle hiérarchique sauvegardé"""
    model_path = os.path.join(MODEL_DIR, f"{symbol}.h5")
    if os.path.exists(model_path):
        return HierarchicalTemporalModel.load(model_path)
    return None
