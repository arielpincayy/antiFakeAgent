from typing import List, Dict, TypedDict

from .agents import Agents, router_bases
from .ner import NERModel

OPENAI = ["gpt-4-turbo", "gpt-4o-mini", "gpt-3.5-turbo"]
GROQ = ["openai/gpt-oss-20b"]
OLLAMA = ["qwen3.5:0.8b", "gemma3:1b"]

class AgentState(TypedDict):
    claim: str
    queries: List[str]
    results: List[Dict]
    iteration: int
    route: Dict
    enough: bool

class ResearchAgent:
    def __init__(self):
        self.agent = Agents(provider="openai", model=OPENAI[0])
        self.router_agent = Agents(provider="groq", model=GROQ[0])
        self.memory = []
        self.ner_model = NERModel()

    def router_bases(self, user_topic: str) -> Dict:
        """
        Determina a qué bases de datos o fuentes de información debe consultar el agente para investigar el tema del usuario.
        Utiliza un agente de enrutamiento para analizar el tema y decidir las fuentes más relevantes
        para la investigación.
        """
        route = router_bases(user_topic, self.router_agent)
        return route
    
    def generate_queries(self, news: str) -> List[str]:
        queries, _ = self.ner_model.extract_entities(news)
        return queries
    

    def run(self, user_topic: str):

        response = ""

        return response