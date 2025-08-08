# core/hyper_ws_client.py

import asyncio
import json
import websockets
import time
from eth_account import Account
from eth_account.messages import encode_defunct

class HyperLiquidWSClient:
    def __init__(self, wallet_private_key: str, base_url="wss://api.hyperliquid.xyz/ws"):
        self.wallet_private_key = wallet_private_key
        self.wallet_address = Account.from_key(wallet_private_key).address
        self.base_url = base_url
        self.ws = None
        self.connected = False

    async def connect(self):
        self.ws = await websockets.connect(self.base_url)
        await self.authenticate()
        self.connected = True
        print(f"[WS] Connected and authenticated as {self.wallet_address}")

    async def authenticate(self):
        nonce = str(int(time.time()))
        msg = encode_defunct(text=nonce)
        signed_msg = Account.sign_message(msg, self.wallet_private_key)
        payload = {
            "method": "auth",
            "params": {
                "wallet": self.wallet_address,
                "timestamp": nonce,
                "signature": signed_msg.signature.hex()
            },
            "id": 1
        }
        await self.ws.send(json.dumps(payload))
        response = await self.ws.recv()
        print(f"[WS] Auth response: {response}")

    async def subscribe(self, topic: str, symbol: str):
        payload = {
            "method": "subscribe",
            "params": {
                "channels": [f"{topic}.{symbol}"]
            },
            "id": 2
        }
        await self.ws.send(json.dumps(payload))
        print(f"[WS] Subscribed to {topic}.{symbol}")

    async def receive_messages(self):
        try:
            while True:
                message = await self.ws.recv()
                data = json.loads(message)
                self.handle_message(data)
        except websockets.exceptions.ConnectionClosed:
            print("[WS] Disconnected. Reconnecting...")
            self.connected = False
            await self.reconnect()

    def handle_message(self, data):
        print("[WS] Incoming message:", data)

    async def reconnect(self):
        await asyncio.sleep(5)
        await self.connect()

    async def run(self, topics_symbols: list):
        await self.connect()
        for topic, symbol in topics_symbols:
            await self.subscribe(topic, symbol)
        await self.receive_messages()
