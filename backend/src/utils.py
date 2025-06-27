from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from src.models import HistoryEntry, RoleEnum

def to_langchain_messages(chat_history: list[HistoryEntry]):
    lc_messages = []
    for entry in chat_history:
        if entry.role == "user":
            lc_messages.append(HumanMessage(content=entry.content))
        elif entry.role == "assistant":
            lc_messages.append(AIMessage(content=entry.content))
        elif entry.role == "system":
            lc_messages.append(SystemMessage(content=entry.content))
    return lc_messages

def from_langchain_messages(messages: list[BaseMessage]) -> list[HistoryEntry]:
    entries = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            entries.append(HistoryEntry(role=RoleEnum.human, content=msg.content))
        elif isinstance(msg, AIMessage):
            entries.append(HistoryEntry(role=RoleEnum.ai, content=msg.content))
        elif isinstance(msg, SystemMessage):
            entries.append(HistoryEntry(role=RoleEnum.system, content=msg.content))
    return entries