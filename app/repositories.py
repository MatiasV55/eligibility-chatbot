from typing import Optional
from app.interfaces import ConversationRepository
from app.models import ChatState


class SQLiteConversationRepository(ConversationRepository):
    """
    Implementación concreta del repositorio usando SQLite.
    Esta clase envuelve la clase Database existente para implementar la interfaz.
    """
    
    def __init__(self, db_path: str = None):
        """
        Inicializa el repositorio SQLite
        
        Args:
            db_path: Ruta a la base de datos SQLite (opcional)
        """
        # Importación diferida para evitar dependencias circulares
        from app.database import Database
        self._db = Database(db_path=db_path)
    
    def save_conversation(self, state: ChatState) -> None:
        """Guarda o actualiza una conversación"""
        self._db.save_conversation(state)
    
    def load_conversation(self, conversation_id: str) -> Optional[ChatState]:
        """Carga una conversación desde SQLite"""
        return self._db.load_conversation(conversation_id)
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Elimina una conversación delegando a Database"""
        return self._db.delete_conversation(conversation_id)
