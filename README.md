# eligibility-chatbot

**Criterios de elegibilidad:**
- Cliente mayor de 18 años
- Auto del 2015 o posterior
- Kilometraje menor a 100,000 km

---

## 1. Configuración Inicial

### Requisitos
- Python 3.9+
- Docker (para Ollama)

### Instalación

```bash
# 1. Crear y activar entorno virtual
python3 -m venv venv
source venv/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Iniciar Ollama en Docker
docker-compose up -d ollama

# 4. Descargar modelo (solo la primera vez)
docker exec ollama ollama pull llama3.2
```

### Variables de Entorno (opcionales)

| Variable | Descripción | Default |
|----------|-------------|---------|
| `OLLAMA_MODEL` | Modelo a usar | `llama3.2` |
| `OLLAMA_BASE_URL` | URL de Ollama | `http://localhost:11434` |
| `USE_LLM_RESPONSES` | Usar LLM para respuestas | `false` |
| `DB_PATH` | Ruta de la base de datos | `data/conversations.db` |

---

## 2. Uso

### Ejecutar el chatbot

```bash
source venv/bin/activate
python -m app.main
```

### Comandos para salir
- `salir`, `exit` o `quit`

---

## 3. Troubleshooting

### "Connection refused" - Ollama no responde

```bash
# Verificar que Ollama está corriendo
docker ps | grep ollama

# Si no está, iniciarlo
docker-compose up -d ollama

# Verificar conexión
curl http://localhost:11434/api/tags
```

### "Model not found" - Modelo no disponible

```bash
# Descargar el modelo
docker exec ollama ollama pull llama3.2

# Ver modelos disponibles
docker exec ollama ollama list
```

### Limpiar todo y empezar de nuevo

```bash
# Detener y eliminar contenedor
docker-compose down -v

# Eliminar base de datos
rm -rf data/

# Reiniciar
docker-compose up -d ollama
docker exec ollama ollama pull llama3.2
```

---

## Estructura del Proyecto

```
eligibility-chatbot/
├── app/
│   ├── main.py           # Interfaz de terminal
│   ├── chatbot.py        # Lógica principal
│   ├── graph.py          # Flujo de estados (LangGraph)
│   ├── tools.py          # Tool de elegibilidad
│   ├── responses.py      # Respuestas pregeneradas
│   ├── database.py       # Persistencia con cifrado
│   ├── models.py         # Modelos de datos
│   ├── interfaces.py     # Interfaces abstractas
│   ├── repositories.py   # Repositorio SQLite
│   ├── llm_providers.py  # Proveedor Ollama
│   └── factory.py        # Factory de chatbot
├── data/                 # BD y claves (auto-generados)
├── docker-compose.yml
├── requirements.txt
└── README.md
```