#!/usr/bin/env python3
# main_v2.py - Orchestrateur principal du trading bot avec architecture hiérarchique multi-échelle

import asyncio
import logging
import os
import sys
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Désactiver les logs TensorFlow trop verbeux
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configuration des chemins avec 'src'
try:
    from src.hyper_ws_client import HyperLiquidWSClient
    from src.data_fetcher_hyperliquid import run_data_fetcher
    from src.hierarchical_model import HierarchicalTemporalModel, DynamicThreshold, FeedbackUpdater
    from src.data_preparation import prepare_multi_scale_data
    from src.model_training import train_all_hierarchical_models, load_hierarchical_model
    from src.telegram_integration import TelegramBot
except ImportError as e:
    print(f"Erreur d'importation: {e}")
    sys.exit(1)

from dotenv import load_dotenv
import numpy as np
import pandas as pd

# Initialisation
load_dotenv()  # Charger les variables d'environnement

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot_execution.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MainOrchestrator')

def escape_markdown(text: str) -> str:
    """Échappe les caractères spéciaux Markdown pour Telegram"""
    escape_chars = '_*[]()~`>#+-=|{}.!'
    return ''.join('\\' + char if char in escape_chars else char for char in text)

class TradingBot:
    def __init__(self):
        self.ws_client: Optional[HyperLiquidWSClient] = None
        self.telegram = TelegramBot(
            os.getenv("TELEGRAM_TOKEN"),
            os.getenv("TELEGRAM_CHAT_ID")
        )
        self.active_symbols: List[str] = []
        self.models: Dict[str, HierarchicalTemporalModel] = {}
        self.threshold_adapters: Dict[str, DynamicThreshold] = {}
        self.feedback_updaters: Dict[str, FeedbackUpdater] = {}
        self.last_data_fetch = datetime.now()
        self.last_model_train = datetime.now()
        self.performance_history = []
        self.running = True
        self.DB_PATH = "data/hyperliquid_v2.db"
        self.symbol_performance = {}

    async def initialize(self):
        """Initialisation du bot"""
        logger.info("Démarrage du Trading Bot")
        await self.telegram.send_message("🤖 *Démarrage du Trading Bot*")

        await self.fetch_market_data()
        self.active_symbols = await self.get_trainable_symbols()
        await self.train_models()
        await self.start_websocket()

    async def fetch_market_data(self):
        """Récupération des données de marché"""
        logger.info("Démarrage du data fetcher")
        await self.telegram.send_message("📊 *Mise à jour des données de marché*")

        try:
            await run_data_fetcher(db_path=self.DB_PATH)
            self.last_data_fetch = datetime.now()
            logger.info("Data fetching terminé")
        except Exception as e:
            logger.error(f"Erreur lors du data fetching: {str(e)}")
            await self.telegram.send_message(f"⚠️ *Erreur data fetching*: {escape_markdown(str(e))}")
            raise

    async def train_models(self):
        """Entraînement des modèles de ML avec approche hiérarchique"""
        if not self.active_symbols:
            logger.error("Aucun symbole trainable disponible pour l'entraînement")
            return

        logger.info("Démarrage de l'entraînement des modèles")
        await self.telegram.send_message("🧠 *Entraînement des modèles hiérarchiques*")

        try:
            # Préparation des données multi-échelles
            logger.info("Préparation des données multi-échelles")
            multi_scale_dataset = prepare_multi_scale_data(
                db_path=self.DB_PATH,
                min_rows=200
            )
            
            # Entraînement des modèles hiérarchiques
            logger.info("Entraînement des modèles hiérarchiques")
            await asyncio.to_thread(train_all_hierarchical_models, multi_scale_dataset)
            
            # Charger les modèles entraînés et initialiser les composants
            for symbol in self.active_symbols:
                model = load_hierarchical_model(symbol)
                if model:
                    self.models[symbol] = model
                    self.threshold_adapters[symbol] = DynamicThreshold()
                    self.feedback_updaters[symbol] = FeedbackUpdater(model)
                    self.symbol_performance[symbol] = {
                        'total_trades': 0,
                        'successful_trades': 0,
                        'last_updated': datetime.now()
                    }
            
            self.last_model_train = datetime.now()
            await self.telegram.send_message("✅ *Entraînement hiérarchique terminé avec succès*")
            logger.info("Tous les modèles entraînés")
        except Exception as e:
            logger.error(f"Erreur entraînement modèles: {str(e)}")
            await self.telegram.send_message(f"⚠️ *Erreur entraînement*: {escape_markdown(str(e))}")
            raise

    async def start_websocket(self):
        """Connexion au WebSocket HyperLiquid"""
        if not self.active_symbols:
            logger.error("Aucun symbole disponible pour le WebSocket")
            await self.telegram.send_message("🔴 Aucun symbole trainable disponible")
            return

        logger.info("Connexion au WebSocket")
        await self.telegram.send_message("🔌 *Connexion à HyperLiquid*")

        # Configurer les topics à suivre
        topics_symbols = [("ticker", sym) for sym in self.active_symbols]

        # Démarrer le client WS
        private_key = os.getenv("HYPERLIQUID_PRIVATE_KEY")
        if not private_key:
            logger.error("Clé privée HyperLiquid non configurée!")
            await self.telegram.send_message("🔴 *ERREUR*: Clé privée HyperLiquid non configurée!")
            sys.exit(1)

        self.ws_client = HyperLiquidWSClient(private_key)
        asyncio.create_task(self.ws_runner(topics_symbols))

    async def ws_runner(self, topics_symbols: List[Tuple[str, str]]):
        """Gestion de la connexion WebSocket avec reconnexion"""
        while self.running:
            try:
                await self.ws_client.run(topics_symbols)
            except Exception as e:
                logger.error(f"Erreur WebSocket: {str(e)}. Reconnexion dans 30s...")
                await self.telegram.send_message(f"⚠️ *WebSocket déconnecté*: Reconnexion dans 30s")
                await asyncio.sleep(30)

    async def get_trainable_symbols(self) -> List[str]:
        """Récupère les symboles avec score > 0.2 depuis la base de données"""
        try:
            conn = sqlite3.connect(self.DB_PATH)
            cursor = conn.cursor()

            # Récupérer les symboles avec trainable_score >= 0.2
            cursor.execute("SELECT symbol FROM symbols WHERE trainable = 1")
            symbols = [row[0] for row in cursor.fetchall()]

            conn.close()
            logger.info(f"{len(symbols)} symboles trainables trouvés")
            return symbols
        except Exception as e:
            logger.error(f"Erreur récupération symboles: {str(e)}")
            return ["BTC", "ETH", "SOL"]  # Fallback

    async def trading_decision(self, symbol: str, market_data: dict) -> Optional[Tuple[str, float, float]]:
        """Prend une décision de trading avec seuil adaptatif"""
        try:
            # 1. Vérifier si nous avons un modèle pour ce symbole
            if symbol not in self.models:
                return None
                
            # 2. Préparer les données d'entrée multi-échelles
            inputs = {}
            for tf in ['1m', '5m', '15m', '1h', '4h']:
                if tf in market_data:
                    # Convertir les données en DataFrame temporaire
                    df = pd.DataFrame(market_data[tf])
                    # Ajouter des timestamps simulés pour la compatibilité
                    df['timestamp'] = pd.date_range(end=datetime.now(), periods=len(df), freq='1min')
                    inputs[tf] = self.preprocess_data(df)
            
            # 3. Faire une prédiction avec le modèle hiérarchique
            model = self.models[symbol]
            prediction = model.predict(list(inputs.values()))[0][0]
            
            # 4. Calculer la confiance du modèle
            confidence = self.calculate_prediction_confidence(prediction)
            
            # 5. Mettre à jour le seuil dynamique
            volatility = self.calculate_recent_volatility(market_data['1m'])
            recent_perf = self.get_recent_performance(symbol)
            
            threshold_adapter = self.threshold_adapters[symbol]
            threshold_adapter.update(volatility, confidence, recent_perf)
            
            # 6. Prendre une décision basée sur le seuil dynamique
            threshold = threshold_adapter.get_threshold()
            
            if prediction > threshold:
                return 'BUY', prediction, confidence
            elif prediction < -threshold:
                return 'SELL', prediction, confidence
            else:
                return 'HOLD', prediction, confidence
                
        except Exception as e:
            logger.error(f"Erreur décision trading {symbol}: {str(e)}")
            return None

    def preprocess_data(self, df: pd.DataFrame) -> np.ndarray:
        """Prétraite les données pour l'entrée du modèle (simplifié)"""
        # Dans une implémentation réelle, appliquer le même prétraitement que pendant l'entraînement
        return df[['open', 'high', 'low', 'close', 'volume']].values

    def calculate_prediction_confidence(self, prediction: float) -> float:
        """Calcule la confiance d'une prédiction (méthode simplifiée)"""
        # Basée sur la magnitude de la prédiction
        return min(1.0, abs(prediction) * 1.5)

    def calculate_recent_volatility(self, data: list) -> float:
        """Calcule la volatilité récente"""
        if len(data) < 20:
            return 0.0
            
        # Extraire les prix
        prices = [d['close'] for d in data]
        
        # Calculer les rendements
        returns = []
        for i in range(1, len(prices)):
            returns.append((prices[i] - prices[i-1]) / prices[i-1])
        
        # Calculer la volatilité (écart-type des rendements)
        return np.std(returns) if returns else 0.0

    def get_recent_performance(self, symbol: str) -> float:
        """Calcule la performance récente du modèle pour un symbole"""
        if symbol not in self.symbol_performance:
            return 0.5
            
        perf = self.symbol_performance[symbol]
        if perf['total_trades'] == 0:
            return 0.5
            
        return perf['successful_trades'] / perf['total_trades']

    async def execute_trade(self, symbol: str, decision: tuple):
        """Exécute un trade et enregistre les résultats pour l'apprentissage futur"""
        action, prediction, confidence = decision
        
        try:
            # Simuler l'exécution du trade (à remplacer par l'API réelle)
            logger.info(f"Exécution {action} sur {symbol}")
            trade_result = {
                'success': True,
                'pnl': 0.015 if action in ['BUY', 'SELL'] else 0.0
            }
            
            # Enregistrer les résultats pour l'apprentissage futur
            if action in ['BUY', 'SELL']:
                # Stocker les métriques de performance
                if symbol not in self.symbol_performance:
                    self.symbol_performance[symbol] = {
                        'total_trades': 0,
                        'successful_trades': 0
                    }
                
                self.symbol_performance[symbol]['total_trades'] += 1
                if trade_result['pnl'] > 0:
                    self.symbol_performance[symbol]['successful_trades'] += 1
                
                # Enregistrer dans le feedback updater
                self.feedback_updaters[symbol].add_trade_result({
                    'symbol': symbol,
                    'action': action,
                    'prediction': prediction,
                    'confidence': confidence,
                    'result': trade_result['pnl'],
                    'timestamp': datetime.now()
                })
            
            # Mettre à jour périodiquement le modèle
            if symbol in self.feedback_updaters:
                if len(self.feedback_updaters[symbol].reward_memory) % 10 == 0:
                    self.update_model_with_feedback(symbol)
                
        except Exception as e:
            logger.error(f"Erreur exécution trade {symbol}: {str(e)}")

    def update_model_with_feedback(self, symbol: str):
        """Met à jour le modèle avec les résultats des trades (simulé)"""
        try:
            if symbol not in self.feedback_updaters:
                return
                
            feedback_data = self.feedback_updaters[symbol].reward_memory
            logger.info(f"Mise à jour modèle {symbol} avec {len(feedback_data)} feedbacks")
            
            # Dans une implémentation réelle, on utiliserait:
            # X_new, y_new = self.prepare_feedback_data(feedback_data)
            # self.feedback_updaters[symbol].update_model(X_new, y_new)
            
        except Exception as e:
            logger.error(f"Erreur mise à jour modèle {symbol}: {str(e)}")

    async def handle_ws_message(self, data: dict):
        """Traite les messages du WebSocket"""
        try:
            if 'channel' in data and 'data' in data:
                channel = data['channel']
                symbol = channel.split('.')[1]
                
                # Prendre une décision de trading
                decision = await self.trading_decision(symbol, {channel: data['data']})
                
                if decision and decision[0] != 'HOLD':
                    await self.execute_trade(symbol, decision)
                    
        except Exception as e:
            logger.error(f"Erreur traitement message WS: {str(e)}")

    async def monitor_performance(self):
        """Surveillance continue des performances"""
        while self.running:
            try:
                # Vérifier les données toutes les 6 heures
                if (datetime.now() - self.last_data_fetch) > timedelta(hours=6):
                    logger.info("Mise à jour périodique des données")
                    await self.fetch_market_data()
                    self.active_symbols = await self.get_trainable_symbols()

                # Vérifier l'entraînement toutes les 12 heures
                if (datetime.now() - self.last_model_train) > timedelta(hours=12):
                    logger.info("Mise à jour périodique des modèles")
                    await self.train_models()

                # Vérifier les performances toutes les 24h
                await self.check_performance()
                await asyncio.sleep(3600)  # Attendre 1 heure
            except Exception as e:
                logger.error(f"Erreur dans monitor_performance: {str(e)}")

    async def check_performance(self):
        """Vérifie si les performances atteignent 85%"""
        try:
            # Calculer la performance globale
            total_trades = 0
            successful_trades = 0
            
            for symbol, perf in self.symbol_performance.items():
                total_trades += perf['total_trades']
                successful_trades += perf['successful_trades']
            
            current_perf = successful_trades / total_trades if total_trades > 0 else 0.0
            
            if current_perf >= 0.85:
                msg = (f"🎯 *OBJECTIF ATTEINT*: 85% de performance!\\n"
                       f"Performance actuelle: {current_perf*100:.2f}%")
                await self.telegram.send_message(msg)
                logger.info(msg)
            else:
                logger.info(f"Performance actuelle: {current_perf*100:.2f}%")
        except Exception as e:
            logger.error(f"Erreur check_performance: {str(e)}")

    async def shutdown(self):
        """Arrêt propre du bot"""
        self.running = False
        if self.ws_client and self.ws_client.ws:
            await self.ws_client.ws.close()
        await self.telegram.send_message("🔴 *Arrêt du Trading Bot*")

async def main():
    bot = TradingBot()
    try:
        await bot.initialize()

        # Démarrer la surveillance en arrière-plan
        asyncio.create_task(bot.monitor_performance())

        # Maintenir le bot en vie
        while bot.running:
            await asyncio.sleep(3600)
    except Exception as e:
        logger.exception(f"Erreur critique: {str(e)}")
        await bot.telegram.send_message(f"🔴 *CRASH DU BOT*: {escape_markdown(str(e))}")
        await bot.shutdown()
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Arrêt du bot par l'utilisateur")
        sys.exit(0)
