from typing import List, Dict, Any
from .agents import Agents


def summarizer(messages: list, agents: Agents, keep_last: int = 2) -> list:
    """
    Compresses a conversation by summarizing older messages,
    keeping only the most recent ones intact.

    Returns a new messages list: [summary_message, ...last_messages]
    """
    if len(messages) <= keep_last:
        return messages

    to_summarize = messages[:-keep_last]
    recent = messages[-keep_last:]

    conversation_text = ""
    for m in to_summarize:
        conversation_text += f"{m['role'].upper()}: {m['content']}\n\n"

    prompt = f"""
    The following is a conversation between a user and an assistant.
    Summarize it concisely, preserving all key facts, decisions, and context
    needed to continue the conversation coherently.

    CONVERSATION:
    {conversation_text}
    """

    summary = agents.chat(
        messages=[
            {"role": "system", "content": "You are a concise conversation summarizer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    summary_message = {
        "role": "system",
        "content": f"[Previous conversation summary]: {summary}"
    }

    return [summary_message] + recent