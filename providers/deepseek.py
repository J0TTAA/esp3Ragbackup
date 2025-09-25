# providers/deepseek.py
import os
from typing import List, Dict, Any
from openai import OpenAI
from .base import Provider

class DeepSeekProvider(Provider):
    def __init__(self, model: str = "deepseek-chat"):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("Falta configurar DEEPSEEK_API_KEY en .env")
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self._model = model

    @property
    def name(self) -> str:
        return f"DeepSeek-{self._model}"

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        response = self.client.chat.completions.create(
            model=self._model,
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content.strip()
