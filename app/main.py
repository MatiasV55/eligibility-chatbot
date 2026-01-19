#!/usr/bin/env python3
import sys
import uuid
from app.factory import ChatbotFactory


def main():
    print("=" * 60)
    print("Chatbot de Elegibilidad - KoolKars")
    print("=" * 60)
    print("\nEscribe 'salir' o 'exit' para terminar la conversación.\n")
    
    # Inicializar chatbot usando la factory (usa variables de entorno si están definidas)
    try:
        chatbot = ChatbotFactory.create_from_env()
    except Exception as e:
        print(f"Error al inicializar el chatbot: {e}")
        print("\nAsegúrate de que Ollama esté corriendo:")
        print("  - Si usas Docker: docker-compose up -d ollama")
        print("  - Si Ollama está local: ollama serve")
        print("\nY que el modelo esté disponible:")
        print("  - Docker: docker exec ollama ollama pull llama3.2")
        print("  - Local: ollama pull llama3.2")
        sys.exit(1)
    
    conversation_id = str(uuid.uuid4())
    print("Conversación iniciada. Escribe 'Hola' para comenzar.\n")
    
    while True:
        try:
            user_input = input("Usuario: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['salir', 'exit', 'quit']:
                print("\n¡Hasta luego!")
                break
            
            response, conversation_id = chatbot.process_message(user_input, conversation_id)
            
            print(f"Chatbot: {response}\n")
            
            state = chatbot.repository.load_conversation(conversation_id)
            if state and state.current_step.value == "completed":
                print("\n" + "=" * 60)
                print("Conversación finalizada. Gracias por usar nuestro servicio.")
                print("=" * 60)
                break
                
        except KeyboardInterrupt:
            print("\n\n¡Hasta luego!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Por favor, intenta de nuevo.\n")


if __name__ == "__main__":
    main()
