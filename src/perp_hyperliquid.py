# src/perp_hyperliquid.py

import os, time, math
import ccxt.async_support as ccxt
import pandas as pd
from pydantic import BaseModel
from decimal import Decimal, getcontext, ROUND_DOWN
import ta
from dotenv import load_dotenv
from datetime import datetime, timedelta
import asyncio

load_dotenv()

# --- Models Pydantic ---
class UsdtBalance(BaseModel):
    total: float
    free: float
    used: float

class Market(BaseModel):
    internal_pair: str
    base: str
    quote: str
    price_precision: float
    contract_precision: float
    min_contracts: float
    min_cost: float
    coin_index: int
    market_price: float

# --- Helper functions ---
def get_price_precision(price: float) -> float:
    if price <= 0:
        return 0.0001
    order = math.floor(math.log10(price))
    return 10 ** (order - 4)

def number_to_str(n: float) -> str:
    s = format(n, 'f').rstrip('0').rstrip('.')
    return s

# --- Client wrapper CCXT ---
class PerpHyperliquid:
    def __init__(self):
        public = os.getenv("PUBLIC_ADDRESS")
        private = os.getenv("PRIVATE_KEY")
        getcontext().prec = 10
        if private:
            self._session = ccxt.hyperliquid({
                "walletAddress": public,
                "privateKey": private,
            })
        else:
            self._session = ccxt.hyperliquid()
        self.market: dict[str,Market] = {}

    async def close(self):
        await self._session.close()

    async def load_markets(self) -> dict[str,Market]:
        data_meta = await self._session.publicPostInfo({"type":"metaAndAssetCtxs"})
        universe, assetCtxs = data_meta[0]["universe"], data_meta[1]
        resp = {}
        for i,obj in enumerate(universe):
            name = obj["name"]
            mark_price = float(assetCtxs[i]["markPx"])
            precision_sz = int(obj["szDecimals"])
            resp[f"{name}/USD"] = Market(
                internal_pair=name,
                base=name, quote="USD",
                price_precision=get_price_precision(mark_price),
                contract_precision=1/(10**precision_sz),
                min_contracts=1/(10**precision_sz),
                min_cost=10,
                coin_index=i,
                market_price=mark_price
            )
        self.market = resp
        return resp

    def _timeframe_to_ms(self, timeframe: str) -> int:
        """Convertit un timeframe en millisecondes"""
        return {
            "1m": 60000,
            "5m": 300000,
            "15m": 900000,
            "1h": 3600000,
            "4h": 14400000
        }[timeframe]

    async def get_first_candle_timestamp(self, symbol: str, timeframe: str) -> int:
        """Trouve le timestamp de la première bougie disponible avec recherche large"""
        try:
            # Date de lancement approximative d'Hyperliquid
            start_ts = int(datetime(2022, 10, 1).timestamp() * 1000)
            end_ts = int(time.time() * 1000)
            
            # Recherche large de la première bougie
            data = await self._session.publicPostInfo({
                "type": "candleSnapshot",
                "req": {
                    "coin": symbol,
                    "interval": timeframe,
                    "startTime": start_ts,
                    "endTime": end_ts,
                    "limit": 1
                }
            })
            
            if not data:
                return None
                
            first_ts = int(data[0]['T'])
            
            # Vérifier la continuité
            verify_ts = first_ts + self._timeframe_to_ms(timeframe)
            verify_data = await self._session.publicPostInfo({
                "type": "candleSnapshot",
                "req": {
                    "coin": symbol,
                    "interval": timeframe,
                    "startTime": verify_ts,
                    "endTime": verify_ts + self._timeframe_to_ms(timeframe),
                    "limit": 1
                }
            })
            
            if not verify_data:
                # Ajuster si point isolé
                return int((datetime.now() - timedelta(days=30)).timestamp() * 1000)
            
            return first_ts
        except Exception as e:
            print(f"⚠️ Erreur détection première bougie {symbol}/{timeframe}: {e}")
            return int((datetime.now() - timedelta(days=365)).timestamp() * 1000)

    async def get_ohlcv(self, ext_pair: str, timeframe: str, 
                       start_time: int = None, 
                       end_time: int = None) -> pd.DataFrame:
        """Récupère les données OHLCV dans une plage spécifique"""
        base = ext_pair.split("/")[0]
        
        # Déterminer la plage temporelle
        end_ts = end_time or int(time.time() * 1000)
        start_ts = start_time or (end_ts - (5000 * self._timeframe_to_ms(timeframe)))
        
        try:
            data = await self._session.publicPostInfo({
                "type": "candleSnapshot",
                "req": {
                    "coin": base,
                    "interval": timeframe,
                    "startTime": start_ts,
                    "endTime": end_ts
                }
            })
            
            if not data or not isinstance(data, list):
                return pd.DataFrame()
            
            # Création du DataFrame
            df = pd.DataFrame(data)
            if not df.empty:
                # Gestion des colonnes (majuscules/minuscules)
                time_col = 'T' if 'T' in df.columns else 't'
                open_col = 'O' if 'O' in df.columns else 'o'
                high_col = 'H' if 'H' in df.columns else 'h'
                low_col = 'L' if 'L' in df.columns else 'l'
                close_col = 'C' if 'C' in df.columns else 'c'
                volume_col = 'V' if 'V' in df.columns else 'v'
                
                df['date'] = pd.to_datetime(df[time_col].astype(float), unit='ms')
                df.set_index('date', inplace=True)
                df = df[[open_col, high_col, low_col, close_col, volume_col]].astype(float)
                df = df.rename(columns={
                    open_col: 'open',
                    high_col: 'high',
                    low_col: 'low',
                    close_col: 'close',
                    volume_col: 'volume'
                })
                df = df.sort_index(ascending=True)
            return df
        except Exception as e:
            print(f"API Error {ext_pair} {timeframe}: {e}")
            return pd.DataFrame()
        except asyncio.TimeoutError:
            print(f"⌛ Timeout pour {ext_pair} {timeframe}")
            return pd.DataFrame()

    async def get_balance(self) -> UsdtBalance:
        st = await self._session.publicPostInfo({"type":"clearinghouseState","user":os.getenv("PUBLIC_ADDRESS")})
        total = float(st["marginSummary"]["accountValue"])
        used  = float(st["marginSummary"]["totalMarginUsed"])
        return UsdtBalance(total=total, free=total-used, used=used)

    async def fetch_order_book(self, symbol: str, limit: int = 100) -> dict:
        """Récupère le carnet d'ordres"""
        try:
            return await self._session.fetch_order_book(f"{symbol}/USD", limit)
        except Exception as e:
            print(f"❌ Erreur carnet d'ordres {symbol}: {e}")
            return {'bids': [], 'asks': []}

    async def create_order(self, symbol: str, type: str, side: str, amount: float, price: float = None):
        """Crée un ordre"""
        try:
            return await self._session.create_order(
                f"{symbol}/USD", 
                type, 
                side, 
                amount, 
                price
            )
        except Exception as e:
            print(f"❌ Erreur création ordre {symbol}: {e}")
            return None

    async def cancel_order(self, order_id: str, symbol: str):
        """Annule un ordre"""
        try:
            return await self._session.cancel_order(order_id, f"{symbol}/USD")
        except Exception as e:
            print(f"❌ Erreur annulation ordre {order_id}: {e}")
            return False

    async def fetch_open_orders(self, symbol: str):
        """Récupère les ordres ouverts"""
        try:
            return await self._session.fetch_open_orders(f"{symbol}/USD")
        except Exception as e:
            print(f"❌ Erreur récupération ordres ouverts {symbol}: {e}")
            return []
