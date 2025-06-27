from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from src.chains import get_image_chain, get_text_chain
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain.schema import Document
import uuid
import os
import logging
from dotenv import load_dotenv
load_dotenv()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

image_chain = get_image_chain()
text_chain = get_text_chain()
table_chain = get_text_chain()

text_categories = ['FigureCaption',
 'Footer',
 'Formula',
 'Header',
 'ListItem',
 'NarrativeText',
 'Title',
 'UncategorizedText']

def extract_and_categorize_content(elements):
    """Separate elements by type for different RAG processing"""
    
    text_elements = []
    table_elements = []
    image_elements = []
    
    for element in elements:
        if "Table" in element.category:
            table_elements.append(element)
        elif hasattr(element.metadata, 'image_base64') and element.metadata.image_base64:
            image_elements.append(element)
        elif element.category in text_categories:
            text_elements.append(element)
    
    return text_elements, table_elements, image_elements

def create_rag_chunks(text_elements, table_elements, image_elements):
    """Create optimized chunks for different content types"""

    all_chunks = []

    # TEXT CHUNKS title-based chunking
    text_chunks = chunk_by_title(
        text_elements,
        max_characters=1000,
        new_after_n_chars=750,
        combine_text_under_n_chars=250,
        overlap=50,
        include_orig_elements=True,
        multipage_sections=True
    )

    for chunk in text_chunks:
        chunk.metadata.content_type = "text"
        all_chunks.append(chunk)

    #  TABLE CHUNKS
    for table in table_elements:
        page = getattr(table.metadata, "page_number", "unknown")
        html = getattr(table.metadata, "text_as_html", "")
        
        if html:
            table_text = f"Table from page {page}:\n{html}"
            table.text = table_text
        table.metadata.content_type = "table"
        all_chunks.append(table)

    #  IMAGE CHUNKS
    for image in image_elements:
        page = getattr(image.metadata, "page_number", "unknown")
        
        # Inicializa el texto base
        image_text = f"Image from page {page}"
        
        # Agrega texto extraído si existe
        if image.text and image.text.strip():
            image_text += f":\nExtracted text: {image.text.strip()}"
        else:
            image_text += " [No extracted text available]"

        image.text = image_text
        image.metadata.content_type = "image"
        all_chunks.append(image)

    return all_chunks


def enhance_chunks_with_summaries(chunks, text_chain=None, table_chain=None, image_chain=None):
    """Add summaries/descriptions to existing chunks using batch processing"""
    
    # Separate chunks by type for batch processing
    image_chunks = [chunk for chunk in chunks if chunk.metadata.content_type == "image"]

    # Process image chunks
    if image_chain and image_chunks:
        # Extract image data for batch processing
        image_data = []
        for chunk in image_chunks:
            if hasattr(chunk.metadata, 'image_base64'):
                image_data.append(chunk.metadata.image_base64)
            else:
                image_data.append(chunk.text)  # Fallback to text
        
        image_descriptions = image_chain.batch(image_data)
        
        for chunk, description in zip(image_chunks, image_descriptions):
            chunk.metadata.description = description
            chunk.text = f"Image Description: {description}\n\n{chunk.text}"
    
    # Process table chunks
    table_chunks = [chunk for chunk in chunks if chunk.metadata.content_type == "table"]
    if table_chain and table_chunks:
        table_contents = [chunk.text for chunk in table_chunks]
        table_summaries = table_chain.batch(table_contents)
        
        for chunk, summary in zip(table_chunks, table_summaries):
            # chunk.metadata.summary = summary
            chunk.text = f"Table Summary: {summary}\n\n{chunk.text}"
    
    # # Process text chunks
    # text_chunks = [chunk for chunk in chunks if chunk.metadata.content_type == "text"]
    # if text_chain and text_chunks:
    #     text_contents = [chunk.text for chunk in text_chunks]
    #     text_summaries = text_chain.batch(text_contents)
        
    #     for chunk, summary in zip(text_chunks, text_summaries):
    #         chunk.metadata.summary = summary
    #         # chunk.text = f"Summary: {summary}\n\nFull Content:\n{chunk.text}"
    
    return chunks


def save_to_multivectorstore(enhanced_chunks, vectorstore):
    store = InMemoryStore()

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key="doc_id"
    )

    summary_docs = []
    full_docs = []
    doc_ids = []

    for chunk in enhanced_chunks:
        doc_id = str(uuid.uuid4())
        doc_ids.append(doc_id)

        content_type = getattr(chunk.metadata, "content_type", "unknown").lower()
        page_number = getattr(chunk.metadata, "page_number", -1)
        filename = getattr(chunk.metadata, "filename", "unknown")

        doc = Document(
            page_content=chunk.text,
            metadata={
                "doc_id": doc_id,
                "content_type": content_type,
                "page_number": page_number,
                "filename": filename
            }
        )

        summary_docs.append(doc)
        full_docs.append(doc)

    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, full_docs)))



def parse_pdf(pdf_bytes, vectorstore):
    logger.info("Starting PDF parsing process...")

    # Guardar archivo temporal
    with open("temp.pdf", "wb") as f:
        f.write(pdf_bytes)
    logger.debug("PDF saved to temp.pdf")
    try:
        # Extraer elementos con Unstructured
        logger.info("Partitioning PDF content...")
        elements = partition_pdf(
            "temp.pdf",
            strategy="hi_res",
            extract_image_block_types=["Image", "Table"],
            extract_image_block_to_payload=True,
            infer_table_structure=True,
            include_page_breaks=True,
        )
        logger.info(f"Partitioned {len(elements)} elements.")

        # Clasificar
        text_elements, table_elements, image_elements = extract_and_categorize_content(elements)
        logger.info(
            f"Extracted: {len(text_elements)} text | {len(table_elements)} tables | {len(image_elements)} images"
        )

        # Chunking
        rag_chunks = create_rag_chunks(text_elements, table_elements, image_elements)
        logger.info(f"Created {len(rag_chunks)} RAG chunks")

        # Enriquecimiento con resumen/descripciones
        logger.info("Enhancing chunks with summaries/descriptions...")
        enhanced_chunks = enhance_chunks_with_summaries(
            rag_chunks,
            text_chain=text_chain,
            table_chain=table_chain,
            image_chain=image_chain,
        )
        logger.info("Chunks enhanced.")

        # Guardar en vectorstore
        logger.info("Saving chunks to vectorstore...")
        save_to_multivectorstore(enhanced_chunks, vectorstore=vectorstore)
        logger.info("Chunks saved successfully.")
    finally:
        if os.path.exists("temp.pdf"):
            os.remove("temp.pdf")
            logger.debug("Temporary file temp.pdf removed.")
