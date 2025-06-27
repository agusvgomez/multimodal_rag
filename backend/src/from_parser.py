import fitz  # PyMuPDF
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Optional
import os
import json

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

class FormularioPersona(BaseModel):
    nombre: Optional[str] = Field(description="Nombre de la persona")
    apellidos: Optional[str] = Field(description="Apellidos de la persona")
    fecha_nacimiento: Optional[str] = Field(description="Fecha de nacimiento en formato YYYY-MM-DD")
    direccion: Optional[str] = Field(description="Dirección de residencia")
    telefono: Optional[str] = Field(description="Número de teléfono")

def extraer_texto_pdf(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    texto = "\n".join([page.get_text() for page in doc][:5])  # Limita a 5 primeras páginas
    doc.close()
    return texto[:15000]  # corta caracteres

def get_extraction_chain():
    model = ChatOpenAI(temperature=0, api_key=OPENAI_API_KEY)
    parser = JsonOutputParser(pydantic_object=FormularioPersona)

    prompt = PromptTemplate(
        template=(
            "Extrae la siguiente información de una persona a partir del texto. "
            "Devuelve solo JSON con esta estructura:\n"
            "{format_instructions}\n\n"
            "Texto del documento:\n{query}\n"
        ),
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | model | parser
    return chain

def extraer_campos_formulario(pdf_bytes: bytes) -> dict:
    chain = get_extraction_chain()
    texto = extraer_texto_pdf(pdf_bytes)

    try:
        salida = chain.invoke(texto)
        return salida  
    except Exception as e:
        return {"error": f"No se pudo extraer o estructurar los datos: {str(e)}",
                "raw_output": str(salida)}

# def clasificar_pagina(self, texto: str) -> str:
#     prompt = f"""
#     Dado el siguiente fragmento de un documento, responde con una sola palabra:
#     - "formulario" si contiene campos tipo clave: valor
#     - "texto_libre" si es un texto narrativo

#     Texto:
#     {texto}
#     Respuesta:
#     """
#     salida = self.llm(prompt, max_new_tokens=5)[0]["generated_text"].lower()
#     return "formulario" if "formulario" in salida else "texto_libre"


