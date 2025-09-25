# providers/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class Provider(ABC):
    """Interfaz base para un proveedor LLM."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Envía una conversación al modelo y retorna el texto de respuesta."""
        ...
