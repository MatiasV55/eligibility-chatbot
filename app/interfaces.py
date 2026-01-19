from abc import ABC, abstractmethod
from typing import Optional, Protocol
from app.models import ChatState


class ConversationRepository(ABC):
    """
    Interfaz abstracta para el repositorio de conversaciones.
    Define el contrato que debe cumplir cualquier implementación de persistencia.
    """
    
    @abstractmethod
    def save_conversation(self, state: ChatState) -> None:
        """
        Guarda o actualiza una conversación
        
        Args:
            state: Estado de la conversación a guardar
        """
        pass
    
    @abstractmethod
    def load_conversation(self, conversation_id: str) -> Optional[ChatState]:
        """
        Carga una conversación desde la persistencia
        
        Args:
            conversation_id: ID de la conversación
            
        Returns:
            Estado de la conversación o None si no existe
        """
        pass
    
    @abstractmethod
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Elimina una conversación
        
        Args:
            conversation_id: ID de la conversación
            
        Returns:
            True si se eliminó correctamente, False en caso contrario
        """
        pass


class LLMProvider(Protocol):
    """
    Protocolo para proveedores de modelos de lenguaje.
    Define el contrato mínimo que debe cumplir cualquier implementación de LLM.
    
    Este es un Protocol (structural typing) que permite que cualquier objeto
    con un método invoke() que acepte una lista de mensajes sea compatible.
    """
    
    def invoke(self, messages: list) -> 'LLMResponse':
        """
        Invoca el modelo de lenguaje con una lista de mensajes
        
        Args:
            messages: Lista de mensajes (formato depende de la implementación)
            
        Returns:
            Respuesta del LLM que debe tener un atributo 'content'
        """
        ...


class LLMResponse(Protocol):
    """Protocolo para respuestas de LLM"""
    content: str
