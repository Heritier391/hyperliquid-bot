# core/train_xgboost.py

import os
import joblib
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import logging

# Configuration du logging
logger = logging.getLogger('XGBoostTraining')
logger.setLevel(logging.INFO)

# Configuration des répertoires
MODELS_DIR = "models/xgboost"
REPORTS_DIR = "reports/feature_importance"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def train_xgboost_models(dataset: dict, lookback: int = 60):
    """
    Entraîne les modèles XGBoost pour tous les symboles du dataset
    
    Args:
        dataset: Dictionnaire {symbol: DataFrame} des données préparées
        lookback: Nombre de périodes historiques à utiliser
    """
    if not dataset:
        logger.error("❌ Aucune donnée disponible pour l'entraînement XGBoost")
        return
    
    logger.info(f"🚀 Début de l'entraînement des modèles XGBoost sur {len(dataset)} symboles")
    
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
            for i in range(lookback, len(features) - 1):
                # Caractéristiques: fenêtre historique
                X.append(features[i-lookback:i].flatten())
                # Cible: direction du prix futur (1 = hausse, 0 = baisse)
                y.append(1 if target[i+1] > target[i] else 0)
            
            X, y = np.array(X), np.array(y)
            
            # Vérifier les dimensions
            if len(X) == 0:
                logger.warning(f"⚠️ Aucune séquence créée pour {symbol}")
                continue
                
            logger.info(f"📦 Données préparées: {X.shape} séquences pour {symbol}")
            
            # Diviser en train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Configurer et entraîner le modèle
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric='logloss',
                early_stopping_rounds=20
            )
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=10
            )
            
            # Évaluation
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            logger.info(f"📊 Performance {symbol}: Accuracy={accuracy:.4f}, F1-score={f1:.4f}")
            
            # Sauvegarder le modèle
            model_path = os.path.join(MODELS_DIR, f"{symbol.replace('/', '_')}_xgb_model.pkl")
            joblib.dump(model, model_path)
            logger.info(f"💾 Modèle sauvegardé: {model_path}")
            
            # Tracer l'importance des caractéristiques
            plot_feature_importance(model, symbol, X_train.shape[1])
            
        except Exception as e:
            logger.error(f"❌ Erreur entraînement {symbol}: {str(e)}")

def plot_feature_importance(model, symbol: str, n_features: int, top_n: int = 20):
    """Génère un graphique d'importance des caractéristiques"""
    try:
        importance = model.feature_importances_
        sorted_idx = np.argsort(importance)[-top_n:]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(sorted_idx)), importance[sorted_idx])
        plt.yticks(range(len(sorted_idx)), [f"feature_{i}" for i in sorted_idx])
        plt.title(f"Top {top_n} - Importance des caractéristiques: {symbol}")
        plt.xlabel("Score d'importance")
        plt.tight_layout()
        
        plot_path = os.path.join(REPORTS_DIR, f"{symbol.replace('/', '_')}_importance.png")
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"📈 Importance des caractéristiques sauvegardée: {plot_path}")
        
    except Exception as e:
        logger.error(f"❌ Erreur création importance caractéristiques {symbol}: {str(e)}")

def load_xgboost_model(symbol: str):
    """Charge un modèle XGBoost sauvegardé"""
    try:
        model_path = os.path.join(MODELS_DIR, f"{symbol.replace('/', '_')}_xgb_model.pkl")
        if not os.path.exists(model_path):
            logger.warning(f"⚠️ Modèle non trouvé pour {symbol}: {model_path}")
            return None
            
        return joblib.load(model_path)
    except Exception as e:
        logger.error(f"❌ Erreur chargement modèle {symbol}: {str(e)}")
        return None
