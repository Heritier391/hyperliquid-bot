import asyncio
import aiosqlite
import pandas as pd
import time
from datetime import datetime, timedelta
from tqdm import tqdm
import os
import aiohttp
from src.perp_hyperliquid import PerpHyperliquid

# Chemin de la base de données
DB_PATH = "data/hyperliquid_v2.db"
TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h"]

# Seuils de durée minimale par timeframe (en jours)
MIN_DAYS_REQUIRED = {
    "1m": 60,    # 2 mois
    "5m": 90,     # 3 mois
    "15m": 120,   # 4 mois
    "1h": 180,    # 6 mois
    "4h": 365     # 1 an
}

# Poids pour le calcul du score global
TIMEFRAME_WEIGHTS = {
    "1m": 0.1,
    "5m": 0.15,
    "15m": 0.2,
    "1h": 0.25,
    "4h": 0.3
}

def timeframe_to_ms(timeframe: str) -> int:
    """Convertit un timeframe en millisecondes"""
    return {
        "1m": 60000,
        "5m": 300000,
        "15m": 900000,
        "1h": 3600000,
        "4h": 14400000
    }[timeframe]

async def fetch_symbols() -> list:
    """Récupère dynamiquement les symboles via POST"""
    url = "https://api.hyperliquid.xyz/info"
    payload = {"type": "meta"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                data = await resp.json()
                return [s["name"] for s in data["universe"]]
    except Exception as e:
        print(f"❌ Erreur récupération symboles: {e}")
        return []

async def init_db(db_path: str):
    """Création des tables si inexistantes"""
    async with aiosqlite.connect(db_path) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS symbols (
                symbol TEXT PRIMARY KEY,
                trainable INTEGER,
                trainable_score REAL,
                last_updated DATETIME
            );
        """)
        
        # Table pour stocker les métadonnées des timeframes
        await db.execute("""
            CREATE TABLE IF NOT EXISTS timeframe_metadata (
                symbol TEXT,
                timeframe TEXT,
                first_timestamp DATETIME,
                last_timestamp DATETIME,
                PRIMARY KEY (symbol, timeframe)
            );
        """)
        
        for tf in TIMEFRAMES:
            table_name = f"ohlcv_{tf}"
            await db.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    symbol TEXT,
                    timestamp DATETIME,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    PRIMARY KEY (symbol, timestamp)
                );
            """)
        await db.commit()

async def get_last_db_timestamp(db: aiosqlite.Connection, symbol: str, timeframe: str) -> int:
    """Récupère le dernier timestamp en base pour un symbol/timeframe"""
    table_name = f"ohlcv_{timeframe}"
    try:
        cursor = await db.execute(
            f"SELECT MAX(timestamp) FROM {table_name} WHERE symbol = ?",
            (symbol,)
        )
        row = await cursor.fetchone()
        await cursor.close()
        
        if row and row[0]:
            last_dt = pd.to_datetime(row[0])
            return int(last_dt.timestamp() * 1000)  # Convertir en ms
    except Exception as e:
        print(f"⚠️ Erreur récupération dernier timestamp: {symbol} {timeframe} - {e}")
    return None

async def get_first_db_timestamp(db: aiosqlite.Connection, symbol: str, timeframe: str) -> int:
    """Récupère le premier timestamp en base pour un symbol/timeframe"""
    table_name = f"ohlcv_{timeframe}"
    try:
        cursor = await db.execute(
            f"SELECT MIN(timestamp) FROM {table_name} WHERE symbol = ?",
            (symbol,)
        )
        row = await cursor.fetchone()
        await cursor.close()
        
        if row and row[0]:
            first_dt = pd.to_datetime(row[0])
            return int(first_dt.timestamp() * 1000)  # Convertir en ms
    except Exception as e:
        print(f"⚠️ Erreur récupération premier timestamp: {symbol} {timeframe} - {e}")
    return None

async def update_timeframe_metadata(db: aiosqlite.Connection, symbol: str, timeframe: str):
    """Met à jour les métadonnées pour un symbol/timeframe"""
    table_name = f"ohlcv_{timeframe}"
    try:
        # Récupérer le premier et dernier timestamp
        cursor = await db.execute(
            f"SELECT MIN(timestamp), MAX(timestamp) FROM {table_name} WHERE symbol = ?",
            (symbol,)
        )
        row = await cursor.fetchone()
        await cursor.close()
        
        if row and row[0] and row[1]:
            await db.execute("""
                INSERT OR REPLACE INTO timeframe_metadata 
                (symbol, timeframe, first_timestamp, last_timestamp)
                VALUES (?, ?, ?, ?)
            """, (symbol, timeframe, row[0], row[1]))
            await db.commit()
    except Exception as e:
        print(f"⚠️ Erreur mise à jour métadonnées: {symbol} {timeframe} - {e}")

def compute_trainable_score(df: pd.DataFrame, timeframe: str) -> float:
    """Calcul d'un score trainable basé sur les données disponibles"""
    if df.empty or len(df) < 100:
        return 0.0
    
    # Récupérer le seuil minimal pour ce timeframe
    min_days = MIN_DAYS_REQUIRED.get(timeframe, 180)
    
    # Calculer la durée réelle en jours
    min_date = df["timestamp"].min()
    max_date = df["timestamp"].max()
    duration_days = (max_date - min_date).days
    
    # Calcul des métriques
    completeness = min(1.0, duration_days / min_days)
    volatility = df["close"].pct_change().std() * (len(df) ** 0.5)  # Volatilité annualisée
    liquidity = df["volume"].mean()
    
    # Normalisation des métriques
    volatility_norm = min(1.0, volatility * 100)  # Supposant que 100% = volatilité très élevée
    liquidity_norm = min(1.0, liquidity / 10000)  # Supposant 10k USD = bonne liquidité
    
    # Score composé
    score = (0.5 * completeness + 
             0.3 * volatility_norm + 
             0.2 * liquidity_norm)
    
    # Application du poids du timeframe
    return score * TIMEFRAME_WEIGHTS.get(timeframe, 0.5)

async def fetch_incremental_ohlcv(symbol: str, timeframe: str, db: aiosqlite.Connection) -> pd.DataFrame:
    """Télécharge uniquement les nouvelles données depuis la dernière timestamp en base"""
    client = PerpHyperliquid()
    try:
        # Récupérer le dernier timestamp en base
        last_ts = await get_last_db_timestamp(db, symbol, timeframe)
        interval_ms = timeframe_to_ms(timeframe)
        end_ts = int(time.time() * 1000)
        all_data = []
        
        # Déterminer le point de départ
        if last_ts:
            start_ts = last_ts + interval_ms
            print(f"ℹ️ {symbol} {timeframe}: Mise à jour depuis {datetime.fromtimestamp(last_ts/1000)}")
        else:
            # Première synchronisation
            first_ts = await client.get_first_candle_timestamp(symbol, timeframe)
            if not first_ts:
                print(f"⚠️ Aucune première bougie trouvée pour {symbol} {timeframe}")
                return pd.DataFrame()
            start_ts = first_ts
            print(f"ℹ️ {symbol} {timeframe}: Premier téléchargement depuis {datetime.fromtimestamp(first_ts/1000)}")
        
        # Vérifier s'il y a de nouvelles données
        if start_ts >= end_ts:
            print(f"✅ {symbol} {timeframe}: Déjà à jour")
            return pd.DataFrame()
        
        current_ts = start_ts
        max_candles = 5000
        
        # Calcul du nombre total de bougies estimé
        total_candles = (end_ts - current_ts) // interval_ms
        print(f"📊 Téléchargement {symbol} {timeframe}: ~{total_candles} nouvelles bougies")
        
        # Créer une barre de progression
        progress_bar = tqdm(total=total_candles, 
                          desc=f"{symbol} {timeframe}", 
                          leave=False)
        
        retry_count = 0
        max_retries = 5
        
        while current_ts < end_ts:
            try:
                # Calculer le nombre de bougies dans ce segment
                segment_candles = min(max_candles, (end_ts - current_ts) // interval_ms + 1)
                if segment_candles <= 0:
                    break
                    
                segment_end_ts = current_ts + (segment_candles * interval_ms)
                
                df = await client.get_ohlcv(
                    f"{symbol}/USD", 
                    timeframe, 
                    start_time=current_ts,
                    end_time=segment_end_ts
                )
                
                if df.empty:
                    # Avancer d'un segment complet
                    current_ts = segment_end_ts
                    progress_bar.update(segment_candles)
                    continue
                
                # Convertir et ajouter les données
                df = df.reset_index()
                df = df.rename(columns={'date': 'timestamp'})
                all_data.append(df[['timestamp', 'open', 'high', 'low', 'close', 'volume']])
                
                # Mettre à jour la progression
                downloaded_candles = len(df)
                progress_bar.update(downloaded_candles)
                
                # Avancer au prochain segment
                if not df.empty:
                    last_ts = int(df["timestamp"].iloc[-1].timestamp() * 1000)
                    current_ts = last_ts + interval_ms
                else:
                    current_ts = segment_end_ts
                
                # Réinitialiser le compteur de réessais
                retry_count = 0
                
                # Pause pour éviter le rate limiting
                await asyncio.sleep(0.3)
                
            except Exception as e:
                retry_count += 1
                if retry_count > max_retries:
                    print(f"❌ Échec après {max_retries} tentatives pour {symbol} {timeframe}")
                    break
                    
                wait_time = 2 ** retry_count
                print(f"⚠️ Erreur segment {symbol} {timeframe}: {e}. Réessai dans {wait_time}s...")
                await asyncio.sleep(wait_time)
        
        progress_bar.close()
        
        if all_data:
            full_df = pd.concat(all_data, ignore_index=True)
            full_df = full_df.drop_duplicates('timestamp').sort_values('timestamp')
            full_df = full_df.reset_index(drop=True)
            
            # Vérifier la continuité
            if len(full_df) > 1:
                time_diffs = full_df['timestamp'].diff().dt.total_seconds()
                expected_diff = interval_ms / 1000
                irregular = ((abs(time_diffs - expected_diff) > 10) & (abs(time_diffs - expected_diff) < 300) & time_diffs.notna()).sum()
                if irregular > 0:
                    irregular_indices = full_df.index[(abs(time_diffs - expected_diff) > tolerance) & time_diffs.notna()]
                    print(f"⚠️ {symbol} {timeframe}: {irregular} intervalles irréguliers aux positions: {irregular_indices.tolist()}")
                    # Afficher les timestamps problématiques
                    for idx in irregular_indices:
                        print(f"  - {full_df.loc[idx, 'timestamp']} (diff: {time_diffs[idx]:.1f}s)")
            
            print(f"✅ {symbol} {timeframe}: {len(full_df)} nouvelles bougies téléchargées")
            return full_df
            
        return pd.DataFrame()
        
    except Exception as e:
        print(f"❌ Erreur pour {symbol} {timeframe}: {e}")
        return pd.DataFrame()
    finally:
        await client.close()

async def load_db_data(db: aiosqlite.Connection, symbol: str, timeframe: str) -> pd.DataFrame:
    """Charge toutes les données d'un symbol/timeframe depuis la base"""
    table_name = f"ohlcv_{timeframe}"
    try:
        cursor = await db.execute(
            f"SELECT * FROM {table_name} WHERE symbol = ? ORDER BY timestamp",
            (symbol,)
        )
        rows = await cursor.fetchall()
        await cursor.close()
        
        if not rows:
            return pd.DataFrame()
        
        df = pd.DataFrame(rows, columns=['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    except Exception as e:
        print(f"⚠️ Erreur chargement données: {symbol} {timeframe} - {e}")
        return pd.DataFrame()

async def process_symbol(symbol: str, db_path: str):
    """Traitement incrémental d'un symbole"""
    async with aiosqlite.connect(db_path) as db:
        dfs = {}
        timeframe_scores = {}
        updated = False
        
        for tf in TIMEFRAMES:
            # Télécharger les nouvelles données
            df_new = await fetch_incremental_ohlcv(symbol, tf, db)
            
            if df_new.empty:
                # Charger les données existantes
                df = await load_db_data(db, symbol, tf)
                dfs[tf] = df
                continue
            
            # Charger les données existantes
            df_existing = await load_db_data(db, symbol, tf)
            
            # Fusionner avec les nouvelles données
            if not df_existing.empty:
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                df_combined = df_combined.drop_duplicates('timestamp').sort_values('timestamp')
            else:
                df_combined = df_new
            
            # Mettre à jour la base de données
            if not df_new.empty:
                # Convertir les timestamps en format string pour SQLite
                df_new["timestamp_str"] = df_new["timestamp"].dt.strftime('%Y-%m-%d %H:%M:%S')
                data_tuples = [
                    (symbol, row["timestamp_str"], row["open"], row["high"], row["low"], row["close"], row["volume"])
                    for _, row in df_new.iterrows()
                ]
                
                # Insérer par batch
                for i in range(0, len(data_tuples), 1000):
                    batch = data_tuples[i:i+1000]
                    await db.executemany(
                        f"""
                        INSERT OR IGNORE INTO ohlcv_{tf} 
                        (symbol, timestamp, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        batch
                    )
                await db.commit()
                
                # Mettre à jour les métadonnées
                await update_timeframe_metadata(db, symbol, tf)
                updated = True
            
            dfs[tf] = df_combined
        
        # Calculer les scores seulement si mise à jour ou premier chargement
        if updated or all(df.empty for df in dfs.values()):
            for tf in TIMEFRAMES:
                df = dfs.get(tf, pd.DataFrame())
                if df.empty or len(df) < 100:
                    print(f"⚠️ Données insuffisantes pour {symbol} sur {tf}")
                    timeframe_scores[tf] = 0.0
                else:
                    timeframe_scores[tf] = compute_trainable_score(df, tf)
            
            # Calculer le score global
            total_weight = sum(TIMEFRAME_WEIGHTS.values())
            global_score = sum(
                timeframe_scores[tf] * TIMEFRAME_WEIGHTS.get(tf, 0) 
                for tf in TIMEFRAMES
            ) / total_weight
            
            trainable = int(global_score >= 0.3)
            
            # Mettre à jour la table des symboles
            now_str = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            await db.execute(
                "INSERT OR REPLACE INTO symbols (symbol, trainable, trainable_score, last_updated) VALUES (?, ?, ?, ?);",
                (symbol, trainable, global_score, now_str)
            )
            await db.commit()
            print(f"💾 {symbol} sauvegardé - Score: {global_score:.2f} {'(Trainable)' if trainable else ''}")
        else:
            print(f"✅ {symbol} déjà à jour - Pas de changement")

async def run_data_fetcher(db_path: str = DB_PATH):
    """Routine principale du data fetcher"""
    await init_db(db_path)
    symbols = await fetch_symbols()
    print(f"🔍 {len(symbols)} symboles trouvés.")
    
    if not symbols:
        print("❌ Aucun symbole trouvé, arrêt du data fetcher")
        return

    # Traiter chaque symbole séquentiellement
    for symbol in tqdm(symbols, desc="Mise à jour OHLCV"):
        await process_symbol(symbol, db_path)

    print("✅ Données mises à jour avec succès.")
