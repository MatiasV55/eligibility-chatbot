import os
from typing import Optional, Any


class OllamaLLMProvider:
    """Proveedor de LLM usando Ollama"""
    
    def __init__(self, model_name: str = "llama3.2", base_url: Optional[str] = None, **kwargs):
        """
        Inicializa el proveedor Ollama
        
        Args:
            model_name: Nombre del modelo a usar
            base_url: URL base de Ollama (opcional)
            **kwargs: Parámetros adicionales para ChatOllama
        """
        from langchain_ollama import ChatOllama
        
        ollama_base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        self._llm = ChatOllama(
            model=model_name,
            base_url=ollama_base_url,
            temperature=kwargs.get("temperature", 0.3),
            **{k: v for k, v in kwargs.items() if k != "temperature"}
        )
    
    def invoke(self, messages: list) -> Any:
        """
        Invoca el modelo Ollama
        
        Args:
            messages: Lista de mensajes (HumanMessage, etc.)
            
        Returns:
            Respuesta del LLM con atributo content
        """
        return self._llm.invoke(messages)
