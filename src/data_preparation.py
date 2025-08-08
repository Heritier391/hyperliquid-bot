# src/data_preparation.py

import pandas as pd
import numpy as np
import sqlite3
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from ta.trend import EMAIndicator, SMAIndicator, ADXIndicator
from ta.volume import OnBalanceVolumeIndicator
from ta.momentum import StochRSIIndicator
from scipy.stats import entropy
import logging
import math

# Configuration du logging
logger = logging.getLogger('DataPreparation')
logger.setLevel(logging.INFO)

DATABASE_PATH = "data/hyperliquid_v2.db"
TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h"]

def load_symbol_data(symbol: str) -> dict:
    """Charge les données pour un symbole donné sur toutes les timeframes"""
    conn = sqlite3.connect(DATABASE_PATH)
    dfs = {}
    
    for tf in TIMEFRAMES:
        table_name = f"ohlcv_{tf}"
        query = f"""
            SELECT timestamp, open, high, low, close, volume
            FROM {table_name}
            WHERE symbol = ?
            ORDER BY timestamp
        """
        try:
            df = pd.read_sql_query(
                query, conn, 
                params=(symbol,), 
                parse_dates=["timestamp"]
            )
            if df.empty:
                logger.warning(f"⚠️ Aucune donnée pour {symbol} sur {tf}")
            else:
                dfs[tf] = df
        except sqlite3.OperationalError as e:
            logger.error(f"❌ Erreur base de données pour {symbol}/{tf}: {str(e)}")
            dfs[tf] = pd.DataFrame()
    
    conn.close()
    return dfs

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule les indicateurs techniques pour un DataFrame"""
    if df.empty:
        return df
    
    try:
        # Indicateurs de base
        df["ema_14"] = EMAIndicator(close=df["close"], window=14).ema_indicator()
        df["sma_14"] = SMAIndicator(close=df["close"], window=14).sma_indicator()
        df["sma_50"] = SMAIndicator(close=df["close"], window=50).sma_indicator()
        df["sma_200"] = SMAIndicator(close=df["close"], window=200).sma_indicator()
        
        if "high" in df.columns and "low" in df.columns:
            df["adx_14"] = ADXIndicator(
                high=df["high"], 
                low=df["low"], 
                close=df["close"], 
                window=14
            ).adx()
        
        if "volume" in df.columns:
            df["obv"] = OnBalanceVolumeIndicator(
                close=df["close"], 
                volume=df["volume"]
            ).on_balance_volume()
        
        df["stochrsi"] = StochRSIIndicator(close=df["close"]).stochrsi()
        
        # Caractéristiques comportementales avancées
        df["bullish_ratio_20"] = df["close"].gt(df["open"]).rolling(20).mean()
        
        # Breakouts récents (dépassement de 2%)
        df["breakouts_20"] = df["high"].gt(df["close"].shift(1) * 1.02).rolling(20).sum()
        
        # Distance aux MAs
        df["dist_to_sma_200"] = (df["close"] - df["sma_200"]) / df["sma_200"]
        
        # Temps passé au-dessus de la SMA 200
        df["above_sma_200"] = (df["close"] > df["sma_200"]).astype(int)
        df["pct_above_sma_200"] = df["above_sma_200"].rolling(50).mean()
        
    except Exception as e:
        logger.error(f"⚠️ Erreur calcul indicateurs: {str(e)}")
    
    return df

def add_anomaly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute des caractéristiques d'anomalie"""
    if df.empty:
        return df
    
    try:
        # Entropie des rendements
        returns = df["close"].pct_change().dropna()
        if not returns.empty:
            df.loc[returns.index, "return_entropy"] = entropy(pd.cut(returns, 20).value_counts(normalize=True))
        
        # Z-score de volatilité
        volatility = (df["high"] - df["low"]).rolling(20)
        df["volatility_mean"] = volatility.mean()
        df["volatility_std"] = volatility.std()
        df["volatility_zscore"] = (df["high"] - df["low"] - df["volatility_mean"]) / df["volatility_std"]
        
        # Distance L2 (sera calculée après normalisation)
        
    except Exception as e:
        logger.error(f"⚠️ Erreur caractéristiques d'anomalie: {str(e)}")
    
    return df

def clean_and_scale(df: pd.DataFrame, exclude=["timestamp", "symbol"]) -> pd.DataFrame:
    """Nettoie et normalise les données"""
    if df.empty:
        return df
    
    # Supprimer les colonnes non numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = df[numeric_cols + ["timestamp", "symbol"]]
    
    # Remplacer les infinis et NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=numeric_cols, how="all", inplace=True)
    
    # Interpolation des valeurs manquantes
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].interpolate(method="linear")
    
    # Supprimer les lignes restantes avec NaN
    df.dropna(inplace=True)
    
    # Normalisation
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(df[numeric_cols])
        df[numeric_cols] = scaled_values
    
    return df

def compute_timeframe_confidence(df: pd.DataFrame) -> float:
    """Calcule un score de confiance pour une timeframe"""
    if df.empty:
        return 0.0
    
    # 1. Complétude des données
    completeness = 1.0 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
    
    # 2. Volatilité moyenne
    volatility = (df["high"] - df["low"]).mean()
    
    # 3. Volume moyen
    volume = df["volume"].mean()
    
    # 4. Taux de bougies manquantes
    expected_periods = pd.date_range(start=df["timestamp"].min(), end=df["timestamp"].max(), freq="1min")
    missing_rate = 1.0 - (len(df) / len(expected_periods))
    
    # Score composite (pondérations ajustables)
    confidence = (
        0.4 * completeness +
        0.3 * min(1.0, volatility / 0.05) +  # Normaliser la volatilité
        0.2 * min(1.0, volume / 10000) +     # Normaliser le volume
        0.1 * (1.0 - missing_rate)
    )
    
    return max(0.0, min(1.0, confidence))

def prepare_multi_scale_data(
    db_path: str = "data/hyperliquid_v2.db", 
    lookback: int = 60,
    min_rows: int = 100
) -> dict:
    """
    Prépare les données pour tous les symboles trainables
    Retourne un dictionnaire {symbole: dict} avec:
    {
        '1m': (X, y, confidence),
        '5m': (X, y, confidence),
        ...
        'dynamic_weights': [w1, w2, ...]
    }
    """
    logger.info("\n📊 Préparation des données multi-échelles")
    
    # 1. Se connecter à la base de données et récupérer les symboles trainables
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT symbol FROM symbols WHERE trainable = 1")
    symbols = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    if not symbols:
        logger.error("❌ Aucun symbole trainable trouvé")
        return {}
    
    dataset = {}
    symbol_count = len(symbols)
    logger.info(f"🔍 {symbol_count} symboles trainables identifiés")
    
    # 2. Traiter chaque symbole
    for symbol in tqdm(symbols, desc="Préparation multi-échelle"):
        try:
            # 2.1. Charger les données pour toutes les timeframes
            dfs = load_symbol_data(symbol)
            confidence_scores = {}
            symbol_data = {}
            
            # 2.2. Préparer chaque timeframe séparément
            for tf in TIMEFRAMES:
                if tf not in dfs or dfs[tf].empty:
                    continue
                    
                df = dfs[tf]
                
                # Seuils minimaux adaptés par timeframe
                tf_min_rows = {
                    "1m": 5000,   # ~3.5 jours
                    "5m": 2000,   # ~7 jours
                    "15m": 1000,  # ~10 jours
                    "1h": 500,    # ~3 semaines
                    "4h": 200     # ~1 mois
                }.get(tf, min_rows)
                
                if len(df) < tf_min_rows:
                    logger.warning(f"⛔ Données insuffisantes pour {symbol} sur {tf} "
                                f"({len(df)}/{tf_min_rows} lignes)")
                    continue
                
                # Calculer les indicateurs techniques
                df = compute_technical_indicators(df)
                
                # Ajouter les caractéristiques d'anomalie
                df = add_anomaly_features(df)
                
                # Nettoyer et normaliser
                df = clean_and_scale(df)
                
                # Calculer le score de confiance
                confidence = compute_timeframe_confidence(df)
                confidence_scores[tf] = confidence
                
                # Créer des séquences temporelles
                features = df.drop(columns=["timestamp", "symbol", "target"], errors="ignore").values
                targets = df["target"].values if "target" in df.columns else None
                
                # Créer des séquences de lookback
                X, y = [], []
                for i in range(lookback, len(features)):
                    X.append(features[i-lookback:i])
                    if targets is not None:
                        y.append(targets[i])
                
                symbol_data[tf] = (np.array(X), np.array(y), confidence)
            
            # 2.3. Calculer les poids dynamiques
            total_confidence = sum(confidence_scores.values())
            dynamic_weights = {
                tf: conf / total_confidence if total_confidence > 0 else 1.0/len(confidence_scores)
                for tf, conf in confidence_scores.items()
            }
            
            symbol_data["dynamic_weights"] = dynamic_weights
            dataset[symbol] = symbol_data
            
            logger.info(f"✅ {symbol} - Données multi-échelles préparées")

        except Exception as e:
            logger.error(f"❌ Erreur traitement {symbol}: {str(e)}")
    
    logger.info(f"\n🎉 Préparation terminée: {len(dataset)}/{symbol_count} symboles traités")
    return dataset
