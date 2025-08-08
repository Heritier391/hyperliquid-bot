# core/data_fetcher_hyperliquid.py
# -*- coding: utf-8 -*-
"""
Data fetcher Hyperliquid avec:
- Fetch natif multi-timeframes (incl. 4h)
- Fallback agrégation 1h -> 4h si API 4h indisponible
- Mise à jour incrémentale par timeframe (pas de refetch massif)
- Calcul trainable_score + trainable adapté à la profondeur réelle
- Conservation des symboles récents même non-trainables

Dépendance facultative: core.perp_hyperliquid (déjà présent chez toi)
"""

from __future__ import annotations
import time
import math
import logging
import sqlite3
from typing import Dict, Iterable, List, Tuple, Optional

try:
    # Ton module interne validé le 2025-08-03
    from core.perp_hyperliquid import get_ohlcv  # type: ignore
    _HAS_NATIVE = True
except Exception:
    # Si la signature diffère, on gérera via fallback agrégation 1h->4h
    _HAS_NATIVE = False

logger = logging.getLogger("DataFetcherHL")

# --- Mapping TF -> durée ms
HOUR_MS = 60 * 60 * 1000
TF_MS = {
    "1m": 60 * 1000,
    "5m": 5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "1h": HOUR_MS,
    "4h": 4 * HOUR_MS,
}

# --- Seuils réalistes (adaptés à ta profondeur réelle HL)
MIN_CANDLES = {
    "1m": 3000,   # ~2-3 jours utiles
    "5m": 1000,
    "15m": 500,
    "1h": 200,
    "4h": 80,
}

# --- Poids pour le score (simples et interprétables)
SCORE_WEIGHTS = {
    "1m": 0.5,
    "5m": 0.8,
    "15m": 1.2,
    "1h": 2.0,
    "4h": 3.0,
}
SCORE_CAP = 20000.0  # on cap au même max que ce que tu as déjà en base

# ---------------------- Utilitaires SQL ---------------------- #

def _ensure_tables(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS symbols (
      symbol TEXT PRIMARY KEY,
      trainable INTEGER DEFAULT 0,
      trainable_score REAL DEFAULT 0
    )
    """)
    for tf in ("1m","5m","15m","1h","4h"):
        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS ohlcv_{tf} (
          symbol TEXT NOT NULL,
          timestamp INTEGER NOT NULL,
          open REAL NOT NULL,
          high REAL NOT NULL,
          low REAL NOT NULL,
          close REAL NOT NULL,
          volume REAL DEFAULT 0,
          PRIMARY KEY (symbol, timestamp)
        )
        """)
    conn.commit()

def _max_ts(conn: sqlite3.Connection, tf: str, symbol: str) -> Optional[int]:
    cur = conn.cursor()
    row = cur.execute(f"SELECT MAX(timestamp) FROM ohlcv_{tf} WHERE symbol=?", (symbol,)).fetchone()
    return row[0] if row and row[0] is not None else None

def _upsert_ohlcv(conn: sqlite3.Connection, tf: str, symbol: str, rows: List[Tuple[int,float,float,float,float,float]]) -> int:
    """rows: List[(ts, o, h, l, c, v)] en millisecondes, ts alignés TF."""
    if not rows:
        return 0
    cur = conn.cursor()
    cur.executemany(
        f"""INSERT OR REPLACE INTO ohlcv_{tf} (symbol, timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [(symbol, ts, o, h, l, c, v) for (ts, o, h, l, c, v) in rows]
    )
    conn.commit()
    return len(rows)

def _count_by_tf(conn: sqlite3.Connection, symbol: str) -> Dict[str, int]:
    cur = conn.cursor()
    out = {}
    for tf in ("1m","5m","15m","1h","4h"):
        row = cur.execute(f"SELECT COUNT(*) FROM ohlcv_{tf} WHERE symbol=?", (symbol,)).fetchone()
        out[tf] = int(row[0] or 0)
    return out

# ---------------------- Fetch natif ---------------------- #

def _fetch_native(symbol: str, tf: str, since_ms: int, until_ms: int) -> List[Tuple[int,float,float,float,float,float]]:
    """Appel l’API interne si dispo. Retour: [(ts,o,h,l,c,v), ...] triés par ts asc."""
    if not _HAS_NATIVE:
        return []
    # NOTE: adapte si ta fonction a une signature différente
    # Ex attendu: get_ohlcv(symbol, timeframe, start_ts, end_ts) -> List[dict/tuple]
    try:
        data = get_ohlcv(symbol, tf, since_ms, until_ms)  # type: ignore
    except Exception as e:
        logger.warning("Native fetch failed %s %s: %s", symbol, tf, e)
        return []
    out: List[Tuple[int,float,float,float,float,float]] = []
    for r in data or []:
        # Adapte si 'r' est dict: r["t"], r["o"], ...
        if isinstance(r, dict):
            ts = int(r.get("timestamp") or r.get("t"))
            o = float(r.get("open") or r.get("o"))
            h = float(r.get("high") or r.get("h"))
            l = float(r.get("low") or r.get("l"))
            c = float(r.get("close") or r.get("c"))
            v = float(r.get("volume") or r.get("v") or 0.0)
        else:
            # suppose tuple-like
            ts, o, h, l, c, v = r
        out.append((ts, o, h, l, c, v))
    out.sort(key=lambda x: x[0])
    return out

# ---------------------- Agrégation 1h -> 4h ---------------------- #

def _aggregate_1h_to_4h(conn: sqlite3.Connection, symbol: str, since_ms: Optional[int] = None) -> int:
    """Agrège les bougies 1h en 4h (fallback) à partir de since_ms."""
    cur = conn.cursor()
    if since_ms is None:
        since_clause = ""
        params: Tuple = (symbol,)
    else:
        since_clause = "AND timestamp >= ?"
        params = (symbol, since_ms)

    rows = cur.execute(f"""
    SELECT timestamp, open, high, low, close, COALESCE(volume,0)
    FROM ohlcv_1h
    WHERE symbol = ? {since_clause}
    ORDER BY timestamp ASC
    """, params).fetchall()

    if not rows:
        return 0

    inserted = 0
    bucket_ts = None
    o = h = l = c = v = None
    count = 0

    def flush(bkt_ts, o_, h_, l_, c_, v_):
        nonlocal inserted
        cur.execute("""
            INSERT OR REPLACE INTO ohlcv_4h (symbol, timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (symbol, bkt_ts, o_, h_, l_, c_, v_))
        inserted += 1

    for ts, _o, _h, _l, _c, _v in rows:
        bkt = (ts // TF_MS["4h"]) * TF_MS["4h"]
        if bucket_ts is None:
            bucket_ts = bkt
            o, h, l, c, v = _o, _h, _l, _c, _v
            count = 1
        elif bkt == bucket_ts:
            h = max(h, _h)
            l = min(l, _l)
            c = _c
            v = (v or 0) + (_v or 0)
            count += 1
        else:
            if count >= 1:
                flush(bucket_ts, o, h, l, c, v or 0)
            bucket_ts = bkt
            o, h, l, c, v = _o, _h, _l, _c, _v
            count = 1
    if bucket_ts is not None and count >= 1:
        flush(bucket_ts, o, h, l, c, v or 0)

    conn.commit()
    return inserted

# ---------------------- Incrémental ---------------------- #

def _now_ms() -> int:
    return int(time.time() * 1000)

def update_symbol_incremental(
    conn: sqlite3.Connection,
    symbol: str,
    timeframes: Iterable[str] = ("1m","5m","15m","1h","4h"),
    lookback_hours_if_empty: int = 72,
    native4h_first: bool = True,
) -> Dict[str, int]:
    """
    Met à jour un symbole incrémentalement pour les TF demandées.
    - Si 4h est demandé:
        * tente le fetch natif 4h
        * sinon agrège 1h->4h
    Retourne un dict tf -> nb_lignes_insérées
    """
    _ensure_tables(conn)
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO symbols(symbol) VALUES (?)", (symbol,))
    conn.commit()

    inserted_by_tf: Dict[str, int] = {}

    now = _now_ms()
    for tf in timeframes:
        if tf not in TF_MS:
            continue

        last_ts = _max_ts(conn, tf, symbol)
        if last_ts is None:
            since_ms = now - lookback_hours_if_empty * HOUR_MS
        else:
            since_ms = last_ts + TF_MS[tf]

        # bornage
        if since_ms > now:
            inserted_by_tf[tf] = 0
            continue

        # fetch par batch (utile si API pagine)
        batch_inserted = 0
        cursor = since_ms
        # fenêtre de batch ~ 7 jours par appel pour limiter la charge
        batch_span = 7 * 24 * HOUR_MS

        while cursor <= now:
            until = min(cursor + batch_span, now)

            rows: List[Tuple[int,float,float,float,float,float]] = []
            if tf == "4h" and native4h_first:
                rows = _fetch_native(symbol, "4h", cursor, until)

            if tf != "4h":
                rows = _fetch_native(symbol, tf, cursor, until)
            elif tf == "4h" and not rows:
                # fallback: aggréger 1h -> 4h à partir de "cursor"
                # d'abord, s'assurer d'avoir la 1h jusqu'à "until"
                _ = _fetch_and_store_native(conn, symbol, "1h", cursor, until)
                # puis agrégation
                _ = _aggregate_1h_to_4h(conn, symbol, since_ms=cursor)
                # on ne sait pas combien de lignes exactes via agrég. On recalcule après.
                # on ne met pas rows ici.

            if rows:
                batch_inserted += _upsert_ohlcv(conn, tf, symbol, rows)

            # avance
            cursor = until + TF_MS[tf]

        inserted_by_tf[tf] = batch_inserted if tf != "4h" else _count_ohlcv_rows_new_since(conn, "4h", symbol, last_ts)

    return inserted_by_tf

def _fetch_and_store_native(conn: sqlite3.Connection, symbol: str, tf: str, since_ms: int, until_ms: int) -> int:
    rows = _fetch_native(symbol, tf, since_ms, until_ms)
    if not rows:
        return 0
    return _upsert_ohlcv(conn, tf, symbol, rows)

def _count_ohlcv_rows_new_since(conn: sqlite3.Connection, tf: str, symbol: str, prev_max_ts: Optional[int]) -> int:
    cur = conn.cursor()
    if prev_max_ts is None:
        row = cur.execute(f"SELECT COUNT(*) FROM ohlcv_{tf} WHERE symbol=?", (symbol,)).fetchone()
    else:
        row = cur.execute(f"SELECT COUNT(*) FROM ohlcv_{tf} WHERE symbol=? AND timestamp>?",
                          (symbol, prev_max_ts)).fetchone()
    return int(row[0] or 0)

# ---------------------- Trainable score/flag ---------------------- #

def recompute_trainable_flags(conn: sqlite3.Connection, verbose: bool = True) -> None:
    """
    Fixe trainable_score et trainable selon la profondeur réelle.
    Règles (OR):
      - (1h >= 200 ET 15m >= 500)
      - OU (4h >= 80)
      - OU (5m >= 1000)
    Score = somme(w_tf * n_tf), cap à 20000.
    """
    cur = conn.cursor()
    syms = [r[0] for r in cur.execute("SELECT symbol FROM symbols").fetchall()]
    for s in syms:
        counts = _count_by_tf(conn, s)

        score = 0.0
        for tf, n in counts.items():
            score += SCORE_WEIGHTS.get(tf, 0.0) * float(n)
        score = min(score, SCORE_CAP)

        cond = (
            (counts.get("1h", 0) >= MIN_CANDLES["1h"] and counts.get("15m", 0) >= MIN_CANDLES["15m"])
            or (counts.get("4h", 0) >= MIN_CANDLES["4h"])
            or (counts.get("5m", 0) >= MIN_CANDLES["5m"])
        )
        trainable = 1 if cond else 0

        cur.execute("UPDATE symbols SET trainable=?, trainable_score=? WHERE symbol=?",
                    (trainable, float(score), s))
        if verbose:
            logger.info(
                "trainable[%s]=%s score=%.1f (1m=%d 5m=%d 15m=%d 1h=%d 4h=%d)",
                s, trainable, score,
                counts.get("1m",0), counts.get("5m",0), counts.get("15m",0),
                counts.get("1h",0), counts.get("4h",0),
            )
    conn.commit()

# ---------------------- Helpers haut niveau ---------------------- #

def update_all_symbols_incremental(
    conn: sqlite3.Connection,
    symbols: Iterable[str],
    timeframes: Iterable[str] = ("1m","5m","15m","1h","4h"),
    lookback_hours_if_empty: int = 72,
) -> Dict[str, Dict[str, int]]:
    """
    Met à jour tous les symboles fournis de manière incrémentale.
    Retour: {symbol: {tf: inserted}}
    """
    out: Dict[str, Dict[str, int]] = {}
    for s in symbols:
        try:
            out[s] = update_symbol_incremental(
                conn, s, timeframes=timeframes, lookback_hours_if_empty=lookback_hours_if_empty
            )
        except Exception as e:
            logger.exception("update_symbol_incremental failed for %s: %s", s, e)
    recompute_trainable_flags(conn, verbose=False)
    return out
