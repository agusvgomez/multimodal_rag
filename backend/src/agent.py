import os
from langgraph.graph import StateGraph, MessagesState, END
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool, Tool
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI
from langchain_core.messages.utils import count_tokens_approximately
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load API key from env
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MAX_TOKENS = os.environ.get("MAX_TOKENS")

# LLM setup
llm = ChatOpenAI(model="gpt-4", temperature=0, api_key=OPENAI_API_KEY)


class State(MessagesState):
    summary: str

def summarize_conversation(state: State):
    summary_so_far = state.get("summary", "")
    messages = state["messages"]

    prompt = (
        f"This is summary of the conversation to date: {summary_so_far}\n\n"
        "Extend the summary by taking into account the new messages above:"
        if summary_so_far
        else "Create a summary of the conversation above:"
    )

    messages_for_summary = messages + [HumanMessage(content=prompt)]
    response = llm.invoke(messages_for_summary)

    # Mantener solo los últimos 2 mensajes relevantes
    new_messages = messages[-2:] 

    return {"summary": response.content, "messages": new_messages}


def memory_check_and_summarize(state: State):
    messages = state["messages"]
    total_tokens = count_tokens_approximately(messages)

    if total_tokens > int(MAX_TOKENS):
        logger.info("Resumiendo conversación por exceso de tokens...")
        return summarize_conversation(state)
    
    return {"messages": messages, "summary": state.get("summary", "")}

def make_retrieve_tool(retriever):
    @tool(response_format="content_and_artifact")
    def retrieve(query: str):
        """Retrieve information related to a query."""
        docs = retriever.get_relevant_documents(query)
        serialized = "\n\n".join(
            f"Source: {doc.metadata}\nContent: {doc.page_content}" for doc in docs
        )
        return serialized, docs

    return retrieve


python_repl = PythonREPL()
repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Input should be a valid Python command. Use print() to show output.",
    func=python_repl.run,
)


# Decide tool or respond
def make_query_or_respond(retrieve_tool):
    def query_or_respond(state: State):  
        summary = state.get("summary", "")
        messages = state["messages"]


        if summary:
            relevant_messages = state["messages"][-2:]  
            relevant_messages = [SystemMessage(content=f"Summary of previous conversation:\n{summary}")] + relevant_messages

        llm_with_tools = llm.bind_tools([retrieve_tool, repl_tool])
        response = llm_with_tools.invoke(messages)

        # Log tools
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_names = [call["name"] for call in response.tool_calls]
            logger.info(f"LLM is invoking tools: {tool_names}")
        else:
            logger.info("LLM is responding directly (no tool used).")

        return {"messages": [response]}
    return query_or_respond


# Generate final answer
def generate(state: State):
    summary = state.get("summary", "")
    messages = state["messages"]

    # tomar outputs de herramientas recientes
    recent_tool_messages = [
        msg for msg in reversed(messages) if msg.type == "tool"
    ][::-1]

    context_blocks = [f"Tool output:\n{msg.content}" for msg in recent_tool_messages]
    tool_context = "\n\n".join(context_blocks)

    # Inyectar resumen si existe
    system_prompts = []

    if summary:
        system_prompts.append(
            SystemMessage(content=f"Summary of previous conversation:\n{summary}")
        )

    if tool_context:
        system_prompts.append(
            SystemMessage(content=f"You are a helpful assistant. Use the following context:\n\n{tool_context}")
        )

    # Extraer solo mensajes relevantes
    history = [
        msg for msg in messages
        if msg.type in ("human", "system") or (msg.type == "ai" and not msg.tool_calls)
    ]

    prompt = system_prompts + history
    response = llm.invoke(prompt)

    return {"messages": [response]}


def get_graph(retrieve_tool):
    query_or_respond = make_query_or_respond(retrieve_tool)
    tools = ToolNode([retrieve_tool, repl_tool])

    graph_builder = StateGraph(State)

    graph_builder.add_node("memory_check", memory_check_and_summarize)
    graph_builder.add_node("query_or_respond", query_or_respond)
    graph_builder.add_node("tools", tools)
    graph_builder.add_node("generate", generate)

    graph_builder.set_entry_point("memory_check")
    graph_builder.add_edge("memory_check", "query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"}
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    return graph_builder.compile()