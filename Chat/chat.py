from typing import TypedDict, Dict
from ContextStore.cs import ContextStore

class Chat:
    def __init__(self, chat_id: str):
        self.chat_id = chat_id
        self.messages = []
        self.ContextStore = ContextStore()
        self.total_tokens = 0
        self.cost = 0

class ChatManager:
    def __init__(self):
        self.chats = {}

    def create_chat(self, chat_id: str):
