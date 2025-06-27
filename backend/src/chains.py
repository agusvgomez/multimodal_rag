from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def get_image_chain():
    prompt_template = """Describe the image in detail"""
    messages = [
        (
            "user",
            [
                {"type": "text", "text": prompt_template},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{image}"},
                },
            ],
        )
    ]

    prompt = ChatPromptTemplate.from_messages(messages)

    image_chain = prompt | ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, max_tokens=300) | StrOutputParser()
    return image_chain

def get_text_chain():
    prompt_text = """
    You are an assistant tasked with summarizing tables and text.
    Give a concise summary of the table or text.

    Respond only with the summary, no additionnal comment.
    Do not start your message by saying "Here is a summary" or anything like that.
    Just give the summary as it is.

    Table or text chunk: {element}

    """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    text_chain = prompt | ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, max_tokens=150) | StrOutputParser()
    return text_chain