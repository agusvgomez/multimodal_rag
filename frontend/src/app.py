import streamlit as st
import logging
import requests
import json
import io

BACKEND_URL = "http://backend:8000"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('chatbot')

def call_bot(input, session_messages):
    data = {
        "question": input,
        "chat_history": session_messages,
        "summary": st.session_state.summary 
    }
    payload = json.dumps(data)
    logger.info(f"Making POST request to {BACKEND_URL} with data: {data}")
    response = requests.post(f"{BACKEND_URL}/predict", data=payload)

    if response.status_code == 200:
        logger.info("Received successful response from backend")
        response_data = response.json()
        logger.info(f"Response {response_data}")

        # Guarda el nuevo resumen en la sesión
        st.session_state.summary = response_data.get("summary", "")

        return response_data
    else:
        logger.error(f"Failed to fetch data from backend. Status code: {response.status_code}")
        return {"answer": f"Failed to fetch data from backend. Status code: {response.status_code}", 
                "documents": [], "summary": ""}

def main():
    st.set_page_config(page_title=" RAG Bot")

    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Hola! ¿Cómo puedo ayudarte hoy?"
        }]

    if "summary" not in st.session_state:
        st.session_state.summary = ""

    if "pdfs_uploaded" not in st.session_state:
        st.session_state.pdfs_uploaded = False

    with st.sidebar:
        st.title('RAG Chatbot')

        if not st.session_state.pdfs_uploaded:
            uploaded_files = st.file_uploader(
                "Sube uno o varios PDFs", type=["pdf"], accept_multiple_files=True
            )

            if uploaded_files:
                files = [
                    ("files", (file.name, io.BytesIO(file.read()), "application/pdf"))
                    for file in uploaded_files
                ]
                response = requests.post(f"{BACKEND_URL}/upload_pdfs", files=files)

                if response.status_code == 200:
                    st.success("PDFs procesados exitosamente.")
                    st.session_state.pdfs_uploaded = True
                else:
                    st.error(f"Error al procesar PDFs: {response.status_code}")
        else:
            st.success("PDFs ya fueron cargados. Puedes comenzar a chatear.")

        if st.button("Limpiar historial"):
            st.session_state.messages = [{
                "role": "assistant",
                "content": "Hola! ¿Cómo puedo ayudarte hoy?"
            }]
            st.session_state.pdfs_uploaded = False

    # Mostrar historial de mensajes
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Armar el mensaje para pasar
    # def generate_response(prompt_input):
    #     output = call_bot(string_dialogue, st.session_state.messages)['answer']
    #     return output

    # Entrada de usuario
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Respuesta del asistente
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                response = call_bot(prompt, st.session_state.messages)['answer']
                placeholder = st.empty()
                full_response = ""
                for chunk in response:
                    full_response += chunk
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

         # Mostrar el resumen actual del asistente
        with st.expander("🧠 Memoria resumida del asistente", expanded=False):
            st.markdown(st.session_state.summary or "_(sin resumen todavía)_")


if __name__ == "__main__":
    main()
