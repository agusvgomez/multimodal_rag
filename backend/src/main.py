from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.models import *
from src.from_parser import extraer_campos_formulario
from fastapi import FastAPI, UploadFile, File, HTTPException,Request
from src.pdf_parser import parse_pdf
from contextlib import asynccontextmanager
import logging
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_chroma import Chroma
import os
from src.agent import make_retrieve_tool, get_graph
from src.utils import to_langchain_messages, from_langchain_messages
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


origins = [
    "*"
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading vectorstore and embeddings...")

    embeddings = OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"])

    vectorstore = Chroma(
        collection_name="multivector_chunks",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db"
    )
    app.state.vectorstore = vectorstore
    logger.info("Vectorstore ready.")

    # Crear retrieve tool + grafo y guardarlo
    retrieve_tool = make_retrieve_tool(vectorstore.as_retriever()) 
    graph = get_graph(retrieve_tool)
    app.state.graph = graph

    logger.info("Agent graph loaded.")

    yield

    logger.info("Shutting down app...")


app = FastAPI(lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["POST"])


@app.post("/predict", response_model=Response)
def predict(request: PredictRequest, fastapi_request: Request):
    graph = fastapi_request.app.state.graph

    messages = to_langchain_messages(request.chat_history)
    messages.append(HumanMessage(content=request.question))

    state = {
        "messages": messages,
        "summary": request.summary
    }

    final_state = graph.invoke(state)
    updated_messages = final_state["messages"]
    answer_msg = updated_messages[-1]

    return Response(
        answer=answer_msg.content,
        documents="",
        chat_history=from_langchain_messages([msg for msg in updated_messages]),
        summary=final_state.get("summary", "") 
    )


@app.post("/upload_pdfs")
async def upload_pdfs(request: Request, files: List[UploadFile] = File(...)):
    vectorstore = request.app.state.vectorstore
    success_files = []

    for file in files:
        if file.content_type != "application/pdf":
            continue  # skip non-pdf files

        pdf_bytes = await file.read()
        try:
            logger.info(f"Processing {file.filename}")
            parse_pdf(pdf_bytes, vectorstore=vectorstore)
            success_files.append(file.filename)
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {str(e)}")

    return {
        "status": "completed",
        "processed_files": success_files,
        "skipped": len(files) - len(success_files)
    }

@app.post("/upload_form")
async def upload_form(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        return {"error": "Solo se permiten archivos PDF"}

    pdf_bytes = await file.read()
    try:
        logger.info(f"Procesando {file.filename}")
        output = extraer_campos_formulario(pdf_bytes)
        return {
            "status": "success",
            "filename": file.filename,
            "data": output
        }
    except Exception as e:
        logger.error(f"Error procesando {file.filename}: {str(e)}")
        return {
            "status": "error",
            "filename": file.filename,
            "error": str(e)
        }