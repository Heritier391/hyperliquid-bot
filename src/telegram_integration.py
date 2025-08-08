# src/telegram_integration.py

import requests
import logging
import asyncio

logger = logging.getLogger('TelegramBot')

def escape_markdown(text: str) -> str:
    """Échappe les caractères spéciaux Markdown pour Telegram"""
    escape_chars = '_*[]()~`>#+-=|{}.!'
    return ''.join('\\' + char if char in escape_chars else char for char in text)

class TelegramBot:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        
        if not token or not chat_id or "YOUR_" in token or "YOUR_" in chat_id:
            logger.error("Token ou Chat ID Telegram non configuré ou invalide!")
            self.valid = False
        else:
            self.base_url = f"https://api.telegram.org/bot{token}/sendMessage"
            self.valid = True
            logger.info("TelegramBot initialisé avec succès")
    
    async def send_message(self, text: str) -> bool:
        """Envoi d'un message Telegram avec gestion d'erreur"""
        if not self.valid:
            logger.warning("Token Telegram non configuré - message non envoyé")
            return False
            
        try:
            # Échapper les caractères Markdown problématiques
            escaped_text = escape_markdown(text)
            
            payload = {
                "chat_id": self.chat_id,
                "text": escaped_text,
                "parse_mode": "MarkdownV2"
            }
            
            # Utiliser une session asynchrone pour les requêtes HTTP
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: requests.post(self.base_url, json=payload, timeout=10)
            )
            
            # Vérifier la réponse
            if response.status_code == 200:
                return True
            else:
                logger.error(f"Erreur Telegram: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur réseau Telegram: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Erreur inattendue Telegram: {str(e)}")
            return False
