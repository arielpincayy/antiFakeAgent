from typing import List, Dict, Any

from .agents import router, summarizer, Agents, generate_search_queries, analyzer, memory_readery
from .scrapping import Scrapping

OPENAI = ["gpt-4-turbo-preview"]
GROQ = ["openai/gpt-oss-20b"]
OLLAMA = ["qwen3.5:0.8b", "gemma3:1b"]


class ResearchAgent:
    def __init__(self):
        self.agent = Agents(provider="openai", model=OPENAI[0])
        self.router_agent = Agents(provider="groq", model=GROQ[0])
        self.scrapping = Scrapping()
        self.memory = []

    def generate_search_queries(self, query: str) -> List[str]:
        """
        Transforma la duda del usuario en términos de búsqueda optimizados,
        limpiando el formato Markdown si el modelo lo incluye.
        """
        return generate_search_queries(query, self.agent)

    def collect_results(self, query: str, limit_per_source: int = 5) -> List[Dict[str, Any]]:
        print(f"Refining search terms for: {query}...")
        search_terms = self.generate_search_queries(query)
        
        all_results = []
        for term in search_terms:
            print(f"Searching for: {term}")
            results = self.scrapping.collect_results(term, limit_per_source)
            all_results.extend(results)

        unique_results = {p['link']: p for p in all_results}.values()
        return list(unique_results)

    def analyze_relevance(self, query: str, papers: List[Dict[str, Any]]) -> str:
        '''
        Analayzes the relevance of each article
        '''
        return analyzer(query, papers, self.agent)

    def handle_memory(self, max_entries: int = 5):
        mem = summarizer(self.memory, self.agent, max_entries)
        self.memory = mem

    def route_query(self, query: str) -> str:
        """
        Decides whether to use memory or conduct new research.
        Returns: 'memory' or 'research'
        """
        return router(query, self.memory, self.router_agent)

    def answer_from_memory(self, query: str) -> str:
        """
        Read info from memory and answer based on that information.
        """
        return memory_readery(query, self.memory, self.agent)

    def run(self, user_topic: str):
        mode = self.route_query(user_topic)

        if mode == "memory":
            print("Using previous knowledge...")
            return self.answer_from_memory(user_topic)

        print("Starting deep research...")
        raw_data = self.collect_results(user_topic)

        if not raw_data:
            return "No results found on any platform."

        response = self.analyze_relevance(user_topic, raw_data)
        self.memory.append((user_topic, raw_data, response))
        self.handle_memory()

        return response