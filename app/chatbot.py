import uuid
from typing import Optional
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from app.models import ChatState, ConversationState, PersonalData, CarData
from app.interfaces import ConversationRepository, LLMProvider
from app.graph import EligibilityGraph, GraphState


class EligibilityChatbot:
    """Chatbot que guía al usuario por el flujo de elegibilidad usando LangGraph"""
    
    def __init__(
        self,
        repository: ConversationRepository,
        llm_provider: LLMProvider,
        use_llm_responses: bool = False
    ):
        """
        Inicializa el chatbot con dependencias inyectadas
        
        Args:
            repository: Repositorio para persistencia de conversaciones
            llm_provider: Proveedor de modelo de lenguaje
            use_llm_responses: Si True, usa LLM para generar respuestas personalizadas (por defecto False)
        """
        self.repository = repository
        self.use_llm_responses = use_llm_responses
        # Inicializar el grafo de LangGraph
        eligibility_graph = EligibilityGraph(llm_provider, use_llm_responses=use_llm_responses)
        self.graph = eligibility_graph.graph
    
    def _chat_state_to_graph_state(self, chat_state: ChatState) -> GraphState:
        """Convierte ChatState (Pydantic) a GraphState (TypedDict) para LangGraph"""
        # Convertir mensajes a formato LangChain
        messages: list[BaseMessage] = []
        for msg in chat_state.messages:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        
        return {
            "messages": messages,
            "conversation_id": chat_state.conversation_id,
            "current_step": chat_state.current_step.value,
            "personal_data": chat_state.personal_data.model_dump(),
            "car_data": chat_state.car_data.model_dump(),
            "personal_data_confirmed": chat_state.personal_data_confirmed,
            "car_data_confirmed": chat_state.car_data_confirmed,
            "eligibility_result": chat_state.eligibility_result,
            "last_response": ""
        }
    
    def _graph_state_to_chat_state(self, graph_state: GraphState, original_chat_state: ChatState) -> ChatState:
        """Convierte GraphState (TypedDict) de LangGraph a ChatState (Pydantic)"""
        # Convertir mensajes de LangChain a formato dict
        messages = []
        for msg in graph_state["messages"]:
            if isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"role": "assistant", "content": msg.content})
        
        # Convertir datos personales
        personal_data = PersonalData(**graph_state["personal_data"])
        
        # Convertir datos del auto
        car_data = CarData(**graph_state["car_data"])
        
        # Obtener respuesta del grafo
        response = graph_state.get("last_response", "")
        if response:
            messages.append({"role": "assistant", "content": response})
        
        return ChatState(
            conversation_id=graph_state["conversation_id"],
            current_step=ConversationState(graph_state["current_step"]),
            personal_data=personal_data,
            car_data=car_data,
            messages=messages,
            personal_data_confirmed=graph_state["personal_data_confirmed"],
            car_data_confirmed=graph_state["car_data_confirmed"],
            eligibility_result=graph_state["eligibility_result"],
            created_at=original_chat_state.created_at,
            updated_at=datetime.now()
        )
    
    def process_message(self, user_input: str, conversation_id: str = None) -> tuple[str, str]:
        """
        Procesa un mensaje del usuario usando LangGraph y retorna la respuesta
        
        Args:
            user_input: Mensaje del usuario
            conversation_id: ID de la conversación (se crea si no existe)
            
        Returns:
            Tupla con (respuesta, conversation_id)
        """
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        
        # Cargar o crear estado desde el repositorio
        chat_state = self.repository.load_conversation(conversation_id)
        if not chat_state:
            chat_state = ChatState(conversation_id=conversation_id)
            chat_state.current_step = ConversationState.GREETING
        
        # Agregar mensaje del usuario al estado
        chat_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Convertir ChatState a GraphState para LangGraph
        graph_state = self._chat_state_to_graph_state(chat_state)
        
        # Ejecutar el grafo de LangGraph
        # El grafo procesará según el estado actual y las transiciones
        final_graph_state = self.graph.invoke(graph_state)
        
        # Convertir GraphState de vuelta a ChatState
        updated_chat_state = self._graph_state_to_chat_state(final_graph_state, chat_state)
        
        # Guardar estado actualizado
        self.repository.save_conversation(updated_chat_state)
        
        # Extraer respuesta del estado final
        response = final_graph_state.get("last_response", "No se pudo generar una respuesta.")
        
        return response, conversation_id
