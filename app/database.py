import sqlite3
import json
from datetime import datetime
from typing import Optional, Dict, Any
from cryptography.fernet import Fernet
import os
import base64
from pathlib import Path
from app.models import ChatState, ConversationState, PersonalData, CarData


class Database:
    """Base de datos con cifrado para datos PII"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            project_root = Path(__file__).parent.parent
            db_path = str(project_root / "data" / "conversations.db")
        self.db_path = db_path
        self._ensure_db_directory()
        self._init_database()
        self._init_encryption()
    
    def _ensure_db_directory(self):
        """Asegura que el directorio de la base de datos existe"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    def _init_encryption(self):
        """Inicializa el sistema de cifrado"""
        project_root = Path(__file__).parent.parent
        key_file = str(project_root / "data" / "encryption.key")
        if os.path.exists(key_file):
            with open(key_file, "rb") as f:
                key = f.read()
        else:
            key = Fernet.generate_key()
            os.makedirs(os.path.dirname(key_file), exist_ok=True)
            with open(key_file, "wb") as f:
                f.write(key)
        self.cipher = Fernet(key)
    
    def _init_database(self):
        """Inicializa las tablas de la base de datos"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabla de conversaciones
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                current_step TEXT NOT NULL,
                encrypted_personal_data TEXT,
                encrypted_car_data TEXT,
                personal_data_confirmed INTEGER DEFAULT 0,
                car_data_confirmed INTEGER DEFAULT 0,
                eligibility_result TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Tabla de mensajes con relación N:1 a conversaciones
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
            )
        """)
        
        # Índice para mejorar consultas por conversación
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_conversation_id 
            ON messages(conversation_id)
        """)
        
        conn.commit()
        conn.close()
    
    def _encrypt_pii(self, data: Dict[str, Any]) -> str:
        """Cifra datos PII"""
        if not data:
            return None
        json_data = json.dumps(data)
        encrypted = self.cipher.encrypt(json_data.encode())
        return base64.b64encode(encrypted).decode()
    
    def _decrypt_pii(self, encrypted_data: str) -> Dict[str, Any]:
        """Descifra datos PII"""
        if not encrypted_data:
            return {}
        encrypted_bytes = base64.b64decode(encrypted_data.encode())
        decrypted = self.cipher.decrypt(encrypted_bytes)
        return json.loads(decrypted.decode())
    
    def save_conversation(self, state: ChatState):
        """Guarda o actualiza una conversación"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Cifrar datos PII
        personal_data_dict = state.personal_data.model_dump(exclude_none=True)
        car_data_dict = state.car_data.model_dump(exclude_none=True)
        
        encrypted_personal = self._encrypt_pii(personal_data_dict) if personal_data_dict else None
        encrypted_car = self._encrypt_pii(car_data_dict) if car_data_dict else None
        
        # Serializar elegibilidad
        eligibility_json = json.dumps(state.eligibility_result) if state.eligibility_result else None
        
        # Guardar o actualizar conversación
        cursor.execute("""
            INSERT OR REPLACE INTO conversations 
            (conversation_id, current_step, encrypted_personal_data, encrypted_car_data,
             personal_data_confirmed, car_data_confirmed, eligibility_result,
             created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            state.conversation_id,
            state.current_step.value,
            encrypted_personal,
            encrypted_car,
            1 if state.personal_data_confirmed else 0,
            1 if state.car_data_confirmed else 0,
            eligibility_json,
            state.created_at.isoformat(),
            datetime.now().isoformat()
        ))
        
        # Guardar mensajes en tabla separada
        # Primero, obtener los mensajes existentes para evitar duplicados
        cursor.execute("""
            SELECT COUNT(*) FROM messages WHERE conversation_id = ?
        """, (state.conversation_id,))
        existing_count = cursor.fetchone()[0]
        
        # Solo insertar nuevos mensajes (los que no están en la BD)
        new_messages = state.messages[existing_count:]
        for msg in new_messages:
            cursor.execute("""
                INSERT INTO messages (conversation_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
            """, (
                state.conversation_id,
                msg.get('role', 'user'),
                msg.get('content', ''),
                datetime.now().isoformat()
            ))
        
        conn.commit()
        conn.close()
    
    def load_conversation(self, conversation_id: str) -> Optional[ChatState]:
        """Carga una conversación desde la base de datos"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Cargar datos de la conversación
        cursor.execute("""
            SELECT current_step, encrypted_personal_data, encrypted_car_data,
                   personal_data_confirmed, car_data_confirmed, eligibility_result,
                   created_at, updated_at
            FROM conversations
            WHERE conversation_id = ?
        """, (conversation_id,))
        
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return None
        
        (current_step, encrypted_personal, encrypted_car, personal_confirmed,
         car_confirmed, eligibility_json, created_at, updated_at) = row
        
        # Cargar mensajes desde la tabla separada
        cursor.execute("""
            SELECT role, content 
            FROM messages 
            WHERE conversation_id = ? 
            ORDER BY message_id ASC
        """, (conversation_id,))
        
        messages_rows = cursor.fetchall()
        messages = [
            {"role": role, "content": content}
            for role, content in messages_rows
        ]
        
        conn.close()
        
        # Descifrar datos PII
        personal_data = PersonalData(**self._decrypt_pii(encrypted_personal)) if encrypted_personal else PersonalData()
        car_data = CarData(**self._decrypt_pii(encrypted_car)) if encrypted_car else CarData()
        
        # Deserializar elegibilidad
        eligibility = json.loads(eligibility_json) if eligibility_json else None
        
        return ChatState(
            conversation_id=conversation_id,
            current_step=ConversationState(current_step),
            personal_data=personal_data,
            car_data=car_data,
            messages=messages,
            personal_data_confirmed=bool(personal_confirmed),
            car_data_confirmed=bool(car_confirmed),
            eligibility_result=eligibility,
            created_at=datetime.fromisoformat(created_at),
            updated_at=datetime.fromisoformat(updated_at)
        )
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Elimina una conversación de la base de datos
        
        Args:
            conversation_id: ID de la conversación a eliminar
            
        Returns:
            True si se eliminó correctamente, False en caso contrario
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Eliminar mensajes relacionados (cascada)
            cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            
            # Eliminar conversación
            cursor.execute("DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,))
            
            deleted = cursor.rowcount > 0
            conn.commit()
            conn.close()
            
            return deleted
        except Exception:
            return False