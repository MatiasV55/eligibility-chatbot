import os
from typing import Optional
from app.chatbot import EligibilityChatbot
from app.repositories import SQLiteConversationRepository
from app.llm_providers import OllamaLLMProvider


class ChatbotFactory:
    """
    Factory centralizado para crear instancias de EligibilityChatbot
    con diferentes configuraciones.
    """
    
    @staticmethod
    def create_default(use_llm_responses: bool = False) -> EligibilityChatbot:
        """
        Crea una instancia del chatbot con la configuración por defecto:
        - Persistencia: SQLite
        - LLM: Ollama (modelo llama3.2)
        - Respuestas: Pregeneradas (o LLM si use_llm_responses=True)
        
        Args:
            use_llm_responses: Si True, usa LLM para generar respuestas (por defecto False)
        
        Returns:
            Instancia configurada de EligibilityChatbot
        """
        repository = SQLiteConversationRepository()
        llm_provider = OllamaLLMProvider()
        return EligibilityChatbot(repository=repository, llm_provider=llm_provider, use_llm_responses=use_llm_responses)
    
    @staticmethod
    def create_with_ollama(
        model_name: str = "llama3.2",
        base_url: Optional[str] = None,
        db_path: Optional[str] = None,
        use_llm_responses: bool = False
    ) -> EligibilityChatbot:
        """
        Crea un chatbot usando Ollama con configuración personalizada
        
        Args:
            model_name: Nombre del modelo Ollama (por defecto: llama3.2)
            base_url: URL base de Ollama (por defecto: http://localhost:11434)
            db_path: Ruta a la base de datos SQLite
            use_llm_responses: Si True, usa LLM para generar respuestas (por defecto False)
            
        Returns:
            Instancia configurada de EligibilityChatbot
        """
        repository = SQLiteConversationRepository(db_path=db_path)
        llm_provider = OllamaLLMProvider(model_name=model_name, base_url=base_url)
        return EligibilityChatbot(repository=repository, llm_provider=llm_provider, use_llm_responses=use_llm_responses)
    
    @staticmethod
    def create_from_env() -> EligibilityChatbot:
        """
        Crea un chatbot basado en variables de entorno.
        
        Variables de entorno soportadas:
        - OLLAMA_BASE_URL: URL base de Ollama (por defecto: http://localhost:11434)
        - OLLAMA_MODEL: Nombre del modelo Ollama (por defecto: llama3.2)
        - DB_PATH: Ruta a la base de datos SQLite
        - USE_LLM_RESPONSES: "true" para usar LLM en respuestas (por defecto: "false")
        
        Returns:
            Instancia configurada de EligibilityChatbot
        """
        use_llm_responses = os.getenv("USE_LLM_RESPONSES", "false").lower() == "true"
        db_path = os.getenv("DB_PATH", None)
        model_name = os.getenv("OLLAMA_MODEL", "llama3.2")
        base_url = os.getenv("OLLAMA_BASE_URL", None)
        
        repository = SQLiteConversationRepository(db_path=db_path)
        llm_provider = OllamaLLMProvider(model_name=model_name, base_url=base_url)
        
        return EligibilityChatbot(repository=repository, llm_provider=llm_provider, use_llm_responses=use_llm_responses)
