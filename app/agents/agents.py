from langchain.chat_models import init_chat_model

from typing import List, Dict

class Agents:
    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini", temperature: float = 0, 
                 top_p: float = 1, max_tokens: int = None):
        
        self.llm = init_chat_model(
            model,
            model_provider=provider,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    def invoke(self, messages: List[Dict[str, str]]) -> str:
        return self.llm.invoke(messages).content