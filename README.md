# multimodal_rag

## Descripción

**multimodal_rag** es una aplicación que permite realizar preguntas y respuestas sobre documentos PDF utilizando técnicas de Recuperación Aumentada por Generación (RAG) y modelos de lenguaje. El sistema cuenta con una interfaz web para cargar documentos y chatear, y un backend que procesa los PDFs, extrae información y responde consultas utilizando embeddings y una base vectorial.

## Estructura del proyecto

```
multimodal_rag/
├── backend/           # Backend en FastAPI para procesamiento y RAG
│   ├── src/
│   │   ├── main.py
│   │   ├── agent.py
│   │   ├── chains.py
│   │   ├── from_parser.py
│   │   ├── models.py
│   │   ├── pdf_parser.py
│   │   └── utils.py
│   └── requirements.txt
├── frontend/          # Frontend Streamlit para interfaz de usuario
│   ├── src/
│   │   └── app.py
│   └── requirements.txt
├── docker-compose.yaml
└── README.md
```

## Requisitos

- Docker y Docker Compose (recomendado)
- O bien, Python 3.9+ para ejecución local

## Instalación y ejecución

### Opción 1: Usando Docker Compose (recomendado)

1. Clona el repositorio y navega a la carpeta del proyecto.
2. Crea un archivo `.env` en la carpeta `backend/` con tu clave de OpenAI:

   ```env
   OPENAI_API_KEY=tu_clave_openai
   MAX_TOKENS=1000
   ```

3. Ejecuta:

   ```bash
   docker-compose up --build
   ```

4. Accede a la interfaz web en [http://localhost:8501](http://localhost:8501)

### Opción 2: Ejecución local (desarrollo)

#### Backend

1. Ve a la carpeta `backend/`:
   ```bash
   cd backend
   ```
2. Instala dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Crea el archivo `.env` con tu clave de OpenAI.
4. Ejecuta el backend:
   ```bash
   uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
   ```

#### Frontend

1. Ve a la carpeta `frontend/`:
   ```bash
   cd frontend
   ```
2. Instala dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Ejecuta la app:
   ```bash
   streamlit run src/app.py
   ```
4. Accede a [http://localhost:8501](http://localhost:8501)

## Uso

1. Sube uno o varios archivos PDF desde la barra lateral.
2. Una vez procesados, puedes comenzar a chatear y hacer preguntas sobre el contenido de los documentos.
3. El sistema utiliza RAG para buscar información relevante y generar respuestas contextuales.

### API: Endpoint `/predict`

Para interactuar programáticamente con el backend, puedes utilizar el endpoint `/predict`.

**Método:** `POST`

**Body (JSON):**

```json
{
  "question": "string",
  "chat_history": [
    {
      "role": "user",
      "content": "string"
    }
  ],
  "summary": ""
}
```

- `question`: Pregunta actual del usuario.
- `chat_history`: Historial de mensajes previos. Cada mensaje debe tener un `role` (puede ser `user`, `assistant` o `system`) y un `content` (texto del mensaje).
- `summary`: Resumen actual de la conversación (puede ser vacío o el último resumen recibido).

**Respuesta:**
- Devuelve la respuesta generada, el historial actualizado de chats y el resumen actualizado.

## Principales dependencias

### Backend
- FastAPI
- LangChain y LangChain-OpenAI
- ChromaDB
- PyMuPDF
- Unstructured
- Uvicorn

### Frontend
- Streamlit

## Notas
- Es necesario contar con una clave de OpenAI válida para el funcionamiento del backend.
- Los documentos PDF se almacenan y procesan localmente en la base vectorial (ChromaDB).